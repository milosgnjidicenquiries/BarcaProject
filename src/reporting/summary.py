"""What a label sequence actually looks like: counts, spells, transitions, uncertainty.

Deliberately not a re-implementation of the scoreboard. The official scorer in evaluation/ owns every
number that could be compared with another team; this module owns the descriptive ones the dashboard
and the run manifest need and the scorer does not export -- per-half label shares, spell durations in
seconds, where the decoder was least sure, and how many frames each evidence channel was allowed to
speak on.

Runs are encoded inside each graded block rather than across the whole array, matching the scorer's
valid_runs(). The two halves are 20156 frames apart and both sit on D at the boundary, so a
whole-array run-length encode merges them and undercounts the segments; that error is invisible in a
frame-level metric and very visible in a spell table.
"""

from __future__ import annotations

import numpy as np

from .. import official
from ..possession import decoder, evidence

LABEL_NAMES = ("H", "A", "D", "X")


def clip(text, width):
    """One line, no wider than the table it is going into. Nothing else, because a truncated value
    that does not say it was truncated is worse than a wrapped one."""
    value = "-" if text is None else str(text)
    return value if len(value) <= width else value[: width - 1] + "…"


def graded_blocks(graded):
    """Maximal runs of graded frames as half-open [start, stop): our view of valid_runs().

    Derived from the labels the organizers shipped alongside the tracking, never taken from a
    constant, so a short or shifted export changes what is decoded rather than silently decoding
    frames that are not there.
    """
    inside = graded.astype(np.int8)
    edges = np.diff(np.concatenate(([0], inside, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def runs(labels, graded):
    """[(start, stop, code)] over the graded blocks only. Half-open, as everywhere else here."""
    out = []
    for start, stop in graded_blocks(graded):
        window = labels[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            out.append((start + int(bounds[i]), start + int(bounds[i + 1]),
                        int(window[bounds[i]])))
    return out


def transition_table(labels, graded, fps, limit=None):
    """Every change the decoder made: frame, the two labels, and how sure it was there.

    `frame` is the first frame of the new label, so it is usable directly as a seek target -- which is
    how this table earns its place in the dashboard. A negative margin means the decoder would rather
    have been elsewhere; those are the rows to watch.
    """
    rows = []
    previous = None
    for start, stop, code in runs(labels, graded):
        if previous is not None:
            rows.append({"frame": start,
                         "from": LABEL_NAMES[previous], "to": LABEL_NAMES[code],
                         "seconds_since_previous": (start - previous_start) / float(fps)})
        previous, previous_start = code, start
        if limit is not None and len(rows) >= limit:
            break
    return rows


def label_table(labels, graded, halves):
    """Per label: frames overall, frames graded, share of graded, share of each half's alive time.

    The two "share of alive" columns are the honest version of a possession statistic. A share of all
    frames is dominated by dead time and by the ungraded stretch, so it moves for reasons that have
    nothing to do with who had the ball.
    """
    alive = graded & ((labels == decoder.H) | (labels == decoder.A))
    rows = []
    for code, name in enumerate(LABEL_NAMES):
        match = labels == code
        row = {
            "label": name,
            "frames": int(match.sum()),
            "graded_frames": int((match & graded).sum()),
            "share_of_graded": float((match & graded).sum()) / max(int(graded.sum()), 1),
        }
        for half in (1, 2):
            in_half = halves == half
            row["half_{}_share".format(half)] = float((match & in_half).sum()) / max(
                int((graded & in_half).sum()), 1)
        row["share_of_alive"] = float((match & alive).sum()) / max(int(alive.sum()), 1)
        rows.append(row)
    return rows


def spell_table(labels, graded, fps, limit=None):
    """Possession spells only: alive, single-team runs, with their duration in seconds.

    D is excluded on purpose. A stoppage is not a possession, and mixing the two into one "segments"
    table is how a dead-ball-heavy submission ends up looking like it produced a lot of football.
    """
    out = []
    for start, stop, code in runs(labels, graded):
        if code not in (decoder.H, decoder.A):
            continue
        out.append({"start_frame": start, "end_frame": stop - 1,
                    "seconds": (stop - start) / float(fps),
                    "label": LABEL_NAMES[code]})
        if limit is not None and len(out) >= limit:
            break
    return out


def spell_summary(labels, graded, fps):
    """Counted summary of the spells: how many, how long, who. The headline block of the report.

    Compared with the scorer's own spell_f1 only in shape -- these numbers are computed on the
    submission alone and use no ground truth of any kind, which is exactly why they can be published
    before a score exists.
    """
    table = spell_table(labels, graded, fps)
    durations = np.array([row["seconds"] for row in table]) if table else np.array([])
    home = sum(row["seconds"] for row in table if row["label"] == "H")
    away = sum(row["seconds"] for row in table if row["label"] == "A")
    summary = {
        "n_possessions": len(table),
        "n_home": sum(1 for row in table if row["label"] == "H"),
        "n_away": sum(1 for row in table if row["label"] == "A"),
        "mean_seconds": float(durations.mean()) if durations.size else None,
        "median_seconds": float(np.median(durations)) if durations.size else None,
        "p90_seconds": float(np.percentile(durations, 90)) if durations.size else None,
        "longest_seconds": float(durations.max()) if durations.size else None,
        "shortest_seconds": float(durations.min()) if durations.size else None,
        "alive_seconds": float(durations.sum()) if durations.size else 0.0,
        "home_share_of_alive": home / (home + away) if home + away else None,
        # The fraction of possessions under one second is the jitter readout: a real match has a few,
        # a frame-level classifier has many, and the run-length prior is what separates them.
        "under_1s": int((durations < 1.0).sum()) if durations.size else 0,
        "under_2s": int((durations < 2.0).sum()) if durations.size else 0,
        # Dead balls counted as spells, not as frames: 60 000 frames of stoppage and 40 stoppages are
        # different models of the same match, and only the second one is a coach's sentence.
        "n_dead_balls": sum(1 for _, _, code in runs(labels, graded) if code == decoder.D),
    }
    dead = np.array([(stop - start) / float(fps) for start, stop, code in runs(labels, graded)
                     if code == decoder.D])
    summary["mean_dead_ball_seconds"] = float(dead.mean()) if dead.size else None
    return summary


def uncertainty_table(labels, separation, graded, top=12):
    """The frames a human should look at first: graded, and where the decoder nearly did not choose.

    Read-only by construction -- nothing here can change a label, and the dashboard is pointed at the
    matching video frame so the inspection ends where it started, with an opinion about the model
    rather than an edit to the data.
    """
    masked = np.where(graded, separation, np.inf)
    order = np.argsort(masked)
    rows = []
    for frame in order:
        if not graded[frame]:
            continue
        rows.append({"frame": int(frame), "label": LABEL_NAMES[int(labels[frame])],
                     "margin_nats": float(separation[frame])})
        if len(rows) >= top:
            break
    return rows


def channel_usage(features, graded, cfg, q, p_dead):
    """Per evidence channel: weight, how many frames it spoke on, and its raw scale.

    This is the audit trail for the no-double-counting rule. If a channel's contribution to the fused
    score cannot be separated from another's, the run is not inspectable, and the fix is to change the
    parameterisation rather than to lower a weight until the number looks right.
    """
    rows = []
    blocks = dict(features["blocks"])
    blocks.update(features["dead"])
    weights = dict(cfg["fusion"]["weights"])
    coefficients = evidence.dead_coefficients(cfg)
    for name, raw in blocks.items():
        rows.append({
            "channel": name,
            "weight": weights.get(name, coefficients.get(name, 0.0)),
            "frames_used": int(np.count_nonzero(np.isfinite(raw) & graded)),
            "median": float(np.median(raw[graded & np.isfinite(raw)]))
                      if np.any(graded & np.isfinite(raw)) else None,
            "mad": float(np.median(np.abs(raw[graded & np.isfinite(raw)] -
                                          np.median(raw[graded & np.isfinite(raw)]))))
                   if np.any(graded & np.isfinite(raw)) else None,
        })
    return {
        "channels": rows,
        "temperature": None,
        "q_median": float(np.median(q[graded])) if graded.any() else None,
        "p_dead_median": float(np.median(p_dead[graded])) if graded.any() else None,
    }
