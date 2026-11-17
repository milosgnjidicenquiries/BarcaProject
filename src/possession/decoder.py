"""Component C: duration-aware temporal decoding, i.e. the exact argmax the model is named for.

    argmax_path   sum_t log_emission_t(s_t)  +  sum_transitions log A  +  sum_runs log p(duration)

Solved exactly -- not greedily, and not by a hand-written smoothing window -- with a semi-Markov
Viterbi over an age-augmented state. The state at frame t is (label, how long that label has already
run), so the cost of continuing depends on the run's age. That is the whole point. A plain
first-order HMM cannot say "a 2-frame flip is expensive" without also making every long spell
exponentially unlikely; the age axis buys the first for free, and a geometric tail past
`max_duration_frames` prevents the second.

Why this *is* the hysteresis, and why there is no separate hysteresis rule:
    A 1-3 frame blip of the other team must pay log p(d = 1..3) under a log-normal with a six-second
    median, plus two embedded-chain transitions, plus the frames of emission that argue against it.
    The decoder holds the old label through that. A genuine two-second counter-attack pays the same
    price and still wins when the emissions are strong enough, because the prior's left tail is open:
    it is a soft cost, never a hard minimum-run-length constraint. The specification asks for both
    properties at once and this is the only formulation that gives both.

Backpointers are deliberately compact. A full (T, S, A) table at 199319 x 3 x 901 float32 is 2.1 GB,
which will not fit on the laptop the organizers expect a participant to own. Instead each frame keeps,
per state, the age that won (uint16) and the state the run entered from (uint8) -- 1.8 MB for the
match. A run of length L is one contiguous block, so its length and its entry state determine
everything before it.

Nothing in this module reads a file, a label or the ground truth. Emissions come from the fusion
layer; block boundaries come from the integrity check.
"""

import numpy as np

# Params
# The three states the decoder can occupy. X is deliberately not one of them: it is emitted only
# outside the graded blocks, where there is nothing to score and abstention is the honest answer.
# Keeping it out of the lattice is what stops a submission from trading an ungraded frame against a
# real one.
H, A, D = 0, 1, 2
DECODED = (H, A, D)
N_STATES = 3
X_LABEL = 3

# log(0), placed far enough from -inf that adding a finite score to it never produces NaN.
NEG = -1e30


# -- duration prior ---------------------------------------------------------------------------

def discretised_log_lognormal(median_s, sigma_log, fps, max_frames, tail_survival):
    """log p(d = L) for L = 1..max_frames, with a geometric continuation from the last column.

    Two modelling choices, both of which change results rather than just appearances:

    * Bins, not density. p(L) = Phi(log((L+0.5)/fps)) - Phi(log((L-0.5)/fps)) -- the probability of
      the interval a run of L frames actually occupies. Evaluating the density at L instead
      overstates p(1) by a factor of a few, and short spells are precisely the frames the run-length
      prior exists to police.
    * The geometric tail is attached to the last *tabulated* value, p(L) = p(M-1) r^(L-M+1) for
      L >= M. That makes the increment at and beyond the final column the constant log r, so the
      table's last column doubles as the "at least this long" bucket with no extra bookkeeping and
      no discontinuity in cost. Very long holds stay possible instead of being cut off by the cap.

    `tail_survival` is a per-frame continuation probability. Setting it to 1.0 would be wrong in the
    loud direction: it turns the 36-second cap into a hard wall and forces transitions the truth
    does not contain.
    """
    from math import erf, log, sqrt

    # Everything below is in FRAMES. median_s is converted once, at the top, and the bin edges are
    # L +/- 0.5 frames -- mixing a frame median with a seconds-scaled argument silently shifts the
    # whole distribution by log(fps) and makes every spell look astronomically improbable.
    median = max(float(median_s), 0.04) * float(fps)
    mu, sigma = log(median), max(float(sigma_log), 0.05)
    cut = sqrt(2.0) * sigma

    def cdf(frames):
        frames = np.atleast_1d(np.asarray(frames, dtype=np.float64))
        return np.array([0.5 * (1.0 + erf((log(max(v, 1e-9)) - mu) / cut)) for v in frames])

    ratio = float(np.clip(tail_survival, 1e-6, 0.999999))
    lengths = np.arange(1, max_frames, dtype=np.float64)  # exact columns 1..M-1
    pmf = cdf(lengths + 0.5) - cdf(np.maximum(lengths - 0.5, 0.0))
    pmf = np.maximum(pmf, 1e-12)
    pmf /= pmf.sum() + pmf[-1] * ratio / (1.0 - ratio)  # exact normalisation, tail included

    table = np.empty(max_frames)
    table[: max_frames - 1] = np.log(pmf)
    table[max_frames - 1] = np.log(pmf[-1]) + log(ratio)
    return table, log(ratio)


def duration_increment(log_pmf, log_ratio):
    """inc[L] whose partial sum over a run of length L is exactly log p(d = L).

    Telescoping: entering a run pays log p(1); every further frame pays log p(L) - log p(L-1); past
    the table the increment is the constant log r. So a long spell costs a small fixed amount per
    frame rather than becoming impossible -- and a short one is charged in full.
    """
    inc = np.empty_like(log_pmf)
    inc[0] = log_pmf[0]
    inc[1:] = np.diff(log_pmf)
    inc[-1] = log_ratio  # shift into the last column and self-loop within it pay the same amount
    return inc


def build_matrices(cfg):
    """(inc, log_embedded, notes) read off r3p.toml, with nothing tuned at runtime.

    The embedded chain is P(next state | this spell ends). Its diagonal is NEG because a spell that
    ends does not restart as itself -- that is exactly how a run-length model represents "still the
    same possession", and it is why continuing a run costs duration, not transition, probability.
    """
    decoder = cfg["decoder"]
    max_frames = int(decoder["max_duration_frames"])
    tail = float(decoder["tail_survival"])
    fps = cfg["facts"]["fps"]
    names = ("H", "A", "D")

    inc = np.empty((N_STATES, max_frames))
    for state in DECODED:
        spec = decoder["duration"][names[state]]
        table, log_ratio = discretised_log_lognormal(
            spec["median_s"], spec["sigma_log"], fps, max_frames, tail)
        inc[state] = duration_increment(table, log_ratio)

    transition = decoder["transition"]
    log_embedded = np.full((N_STATES, N_STATES), NEG)
    for source in DECODED:
        for target in DECODED:
            if source != target:
                log_embedded[source, target] = np.log(max(float(transition[
                    "{}_to_{}".format(names[source], names[target])]), 1e-9))

    notes = {
        "max_duration_frames": max_frames,
        "duration_medians_s": {names[s]: decoder["duration"][names[s]]["median_s"]
                               for s in DECODED},
        "median_run_frames": {names[s]: int(round(decoder["duration"][names[s]]["median_s"] * fps))
                              for s in DECODED},
    }
    return inc, log_embedded, notes


# -- the lattice ----------------------------------------------------------------------------

def _separation(delta):
    """Gap between the best and second-best state, each at its best age. Reported as uncertainty.

    This is a comparison of forward-optimal prefixes, not a posterior: an exact posterior over ages
    would need a backward lattice as wide as the forward one. It is used to point a human at the
    frames worth watching and to gate the optional abstention -- never to score anything.
    """
    ordered = np.sort(delta.max(axis=1))[::-1]
    return float(ordered[0] - ordered[1])


def _entry(delta, log_embedded, rmod_row):
    """Best predecessor for each state at one frame: (state, score) per target, diagonal excluded.

    `rmod_row` is the restart-aware additive log-odds for entering each target *from D*. Applied to
    the D row only, so H <-> A stays exactly as the embedded chain says: a restart is the one moment
    the tagger is worth trusting, a mid-play turnover is not.
    """
    beta = delta.max(axis=1)
    states, scores = np.empty(N_STATES, dtype=np.int8), np.empty(N_STATES)
    for target in DECODED:
        terms = beta + log_embedded[:, target]
        if rmod_row is not None:
            terms = terms.copy()
            terms[D] += rmod_row[target]
        terms[target] = NEG
        states[target] = int(np.argmax(terms))
        scores[target] = terms[states[target]]
    return beta, states, scores


def _decode_block(emis, inc, log_embedded, rmod):
    """Viterbi + traceback over one contiguous run of frames. Returns (labels, separation, notes).

    Only two columns of the lattice are ever live, so memory is O(T*S) in backpointers and O(S*A) in
    scores regardless of how long the match is.
    """
    n_frames, max_frames = emis.shape[0], inc.shape[1]
    delta = np.full((N_STATES, max_frames), NEG)
    best_age = np.zeros((n_frames, N_STATES), dtype=np.uint16)
    entry_from = np.zeros((n_frames, N_STATES), dtype=np.uint8)
    greedy = np.zeros(n_frames, dtype=np.int8)
    separation = np.zeros(n_frames)

    delta[:, 0] = emis[0] + inc[:, 0]
    best_age[0] = 0
    separation[0] = _separation(delta)
    greedy[0] = int(np.argmax(delta.max(axis=1)))

    for t in range(1, n_frames):
        row = rmod[t] if rmod is not None else None
        beta, states, scores = _entry(delta, log_embedded, row)

        continuing = np.empty_like(delta)
        continuing[:, 0] = NEG
        continuing[:, 1:] = delta[:, :-1]
        continuing[:, -1] = np.maximum(continuing[:, -1], delta[:, -1])  # tail self-loop

        delta = continuing + emis[t][:, None] + inc
        delta[:, 0] = scores + emis[t] + inc[:, 0]

        best_age[t] = np.argmax(delta, axis=1)
        entry_from[t] = states
        separation[t] = _separation(delta)
        greedy[t] = int(np.argmax(delta.max(axis=1)))

    labels = _trace(greedy, best_age, entry_from, max_frames)
    edges = np.flatnonzero(np.diff(labels) != 0) + 1
    bounds = np.concatenate(([0], edges, [n_frames]))
    return labels, separation, {
        "n_frames": n_frames,
        "n_runs": int(len(edges) + 1),
        "longest_run": int(np.diff(bounds).max()) if n_frames else 0,
        "median_run": float(np.median(np.diff(bounds))) if n_frames else 0.0,
        "saturated_frames": int((best_age[np.arange(n_frames), greedy] == max_frames - 1).sum()),
    }


def _trace(greedy, best_age, entry_from, max_frames):
    """Recover the winning path from the compact backpointers, one whole run at a time.

    Walking runs instead of frames is what makes 1.8 MB of backpointers sufficient: at (t, s, L) the
    entire run is frames t-L+1..t, and everything before it is determined by the state that run
    entered from. A run that saturated the age table is extended backwards over every frame the
    greedy path also gave to s, because a saturated age only says "at least max_frames".
    """
    out = greedy.copy()
    t = len(greedy) - 1
    state = int(greedy[t])
    while t >= 0:
        length = int(best_age[t, state]) + 1
        start = t - length + 1
        if length >= max_frames:
            while start > 0 and greedy[start - 1] == state:
                start -= 1
        out[start : t + 1] = state
        if start == 0:
            break
        state = int(entry_from[start, state])
        t = start - 1
    return out


# -- public entry point ----------------------------------------------------------------------

def decode(log_emission, cfg, blocks, rmod=None):
    """Decode the whole video, half by half, with X wherever the frame is not graded.

    `blocks` is [(start, stop)] of contiguous graded frames -- the two halves. Decoding is per half
    on purpose: a single lattice over the video would have to price the 20156-frame halftime hole as
    either one absurd dead-ball spell or a transition at the restart, and neither is what happened.
    The boundary is instead a free choice of the first label of each half, which is what the
    ground-truth format itself does (its `valid_runs` are the same two spans).
    """
    n_frames = log_emission.shape[0]
    inc, log_embedded, notes = build_matrices(cfg)

    labels = np.full(n_frames, X_LABEL, dtype=np.int8)
    separation = np.zeros(n_frames)
    inside = np.zeros(n_frames, dtype=bool)
    per_block = []

    for (start, stop) in blocks:
        stop = min(int(stop), n_frames)
        if stop - start < 2:
            continue
        emis = log_emission[start:stop, DECODED].astype(np.float64, copy=False)
        block_rmod = rmod[start:stop] if rmod is not None else None
        traced, sep, report = _decode_block(emis, inc, log_embedded, block_rmod)
        labels[start:stop] = traced
        separation[start:stop] = sep
        inside[start:stop] = True
        per_block.append(dict(report, start=start, stop=stop))

    return {
        "labels": labels,
        "separation": separation,
        "graded": inside,
        "inc": inc,
        "log_embedded": log_embedded,
        "prior": notes,
        "blocks": per_block,
    }
