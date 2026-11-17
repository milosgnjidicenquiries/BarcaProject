# R3P — Technical document

**R3P** = *Reliability-weighted, Restart-aware, Run-length-regularized Possession Model*.
It reads the official Metrica tracking (ATD), homography and Smart Tagging files for game
`BAR-ATH` and writes one label per video frame — `H`, `A`, `D`, `X` — in the contract
`SUBMISSION.md` asks for.

Nothing in this file is a performance claim about the restricted match. Section 7 states exactly
which numbers exist and which do not.

---

## 1. Input, as the organizers deliver it

| file | loader used | what R3P takes from it |
|---|---|---|
| `atd/<game>_tactical.csv` | `starter/load_atd.py`, called unchanged | `frame_id`, per-slot `x,y,s` in normalised units, scaled by the loader to metres |
| `homography/…` | `starter/load_homography.py`, called unchanged | world→pitch transform, and the sanity check that tracking really is in metres |
| `smart_tagging/…` | `starter/load_smart_tagging.py` via `src/data/tagging.py` | kickoffs, set pieces, restart team — **weak evidence only** (§3) |

Coordinates are metric in the loader, so every distance in R3P is metres. `NaN` means *not
observed*, never *zero*: the ball column is missing on ~41 % of graded frames, and treating that
as a ball lying still is the single most damaging mistake available on this dataset. `video_frame
= frame_id - 1`; Smart Tagging seconds map to frames by `round(seconds × 25)`.

Published geometry, from `src/official.py` / `config/r3p.toml` → `facts` (105.3 × 68.0 m, 25 fps,
199,319 video frames, graded halves at video frames 11185–90651 and 110808–190197 = 158,857
frames). All of it is read from the organizers' files, not restated.

## 2. Component A — reliability-weighted fusion

Five kinds of per-frame evidence are computed independently, then combined. The control distance
is metric, exactly as the rules spell it out:

```
d_i  = sqrt((105.3 · dx_i)^2 + (68.0 · dy_i)^2)          metres, pitch-scaled
S    = mean_i exp(-d_i^2 / (2 σ^2))                       σ = 3.0 m controlling side
                                                        σ = 7.0 m supporting side
```

Channels and their fixed weights (`config/r3p.toml` → `fusion.weights`):

| channel | weight | what it measures |
|---|---|---|
| `nearest` | 1.00 | which side owns the player closest to the ball, in metres |
| `soft_proximity` | 0.70 | `S` above, computed over the `near_pairs_k = 2` nearest per side |
| `support` | 0.35 | the same kernel at the wider support radius |
| `ball_kinematics` | 0.55 | ball speed and the direction change after a kick |
| `territory` | 0.30 | attacking direction, guarded so an empty half cannot invent a direction |
| `continuity` | 0.45 | persistence of *observations*, not of previous predictions |
| `tagging` | 0.40 | Smart Tagging, present at a fraction of the geometric weights |

Each channel is then made comparable by robust statistics, over graded frames only:

```
z_c = clip((x_c − median_c) / max(MAD_c · 1.4826, mad_floor), −4, +4)
score = Σ_c w_c · z_c
q     = sigmoid(score / T) · ρ
```

`ρ` is per-frame reliability (`fusion.reliability`): the share of slots actually tracked, a
grace of `ball_gap_max_frames = 25` on the ball column, a geometric-imputation floor of
`geo_impute_rho = 0.35`, ghost-tracklet detection inside `duplicate_slot_radius_m = 1.5` m, and a
refusal to speak at all below `min_players_for_evidence = 8` players. Smoothed over
`temporal_smooth_frames = 12` frames.

**No double counting.** The dead-ball cues never enter `score`; `continuity` reads observations
rather than its own earlier output; `nearest` and `soft_proximity` are separate channels with
separate weights and separate medians, and `tests/test_pipeline.py` asserts that every configured
weight names a channel that exists, spoke, and was standardised.

## 3. Component B — dead ball and team possession, as two separate heads

One score cannot express "the ball is on the ground at the corner flag and nobody owns it", so the
model keeps two quantities apart:

```
P(D) = p_dead                     from the dead-ball cues alone
P(H) = (1 − p_dead) · q           q from Component A
P(A) = (1 − p_dead) · (1 − q)
```

`p_dead = sigmoid(logit_scale · Σ_c w_c · z_c + prior_logodds)` with signed weights
(`config/r3p.toml` → `deadball`):

| cue | weight | why this sign and size |
|---|---|---|
| `static` | +2.20 | ball speed < `static_speed_mps = 1.2` sustained over `static_window_frames = 25` |
| `moving` | −2.00 | a travelling ball argues *against* a stoppage; the sign is the point |
| `oob` | +1.80 | ball centre past the touchline by more than `oob_margin_m = 1.8` |
| `missing` | +0.55 | unobserved for more than `missing_gap_frames = 40` — the **weakest** cue |
| `collapse` | +0.80 | median player speed drops; median over *visible* slots only |
| `set piece` | +0.75 | Smart Tagging — weak evidence, deliberately below the geometric cues |
| `wall` | +0.45 | `wall_cluster_players = 6` inside `wall_cluster_radius_m = 9.0` m |

Rules this encodes directly:

- **Missing ball ≠ dead ball.** `missing` is the lowest-weight cue and a gap under 40 frames is
  imputed geometrically instead of being labelled. `tests/test_deadball.py::
  MissingBallIsNotDeadTest` pins that a dropped ball column moves `p_dead` less than a stationary
  ball does, in both the static and the moving case.
- **Touch / deflection / rebound ≠ turnover.** A deflection flips `nearest` but leaves
  `continuity` high; `TurnoverIsNotPredictedTest` asserts exactly that.
- **Tagging is weak.** `TaggingIsWeakEvidenceTest` builds a SET-PIECE-only tag and requires it to
  raise `p_dead` by less than a stationary ball does, so a mistimed tag cannot flip a frame.
- **Two heads stay separable.** `TwoHeadsTest` requires a fixture that swings `p_dead` by more
  than 0.45 while swinging `q` by less than a quarter of that.

## 4. Component C — duration-aware semi-Markov decoding

The label path is not chosen frame by frame. Per half, R3P maximises

```
argmax_path   Σ_t log emission_t(s_t)  +  Σ_t log A(s_{t−1}, s_t)  +  Σ_runs log p(d = L)
```

with `s_t ∈ {H, A, D}`, solved exactly by an **age-expanded Viterbi**: state = (label, run-age),
so the cost of continuing a spell is the duration model, not a self-transition.

- **Duration prior.** Bin-discretised log-normal in frames, per state, with a geometric tail
  (`decoder.duration`: median 6.0 s / 6.0 s / 7.0 s, `sigma_log` 0.9 / 0.9 / 0.85;
  `max_duration_frames = 900`, `tail_survival = 0.99`). `inc` telescopes, so
  `cumsum(inc)[L−1] = log p(d = L)` — `tests/test_decoder.py` checks the normalisation integrates
  to 1 including the tail.
- **Embedded transitions.** `A` has **no diagonal**: a run continuing is charged to duration, a run
  ending is charged to `A` (`H_to_A = 0.45`, `H_to_D = 0.55`, `A_to_H = 0.45`, `A_to_D = 0.55`,
  `D_to_H = 0.5`, `D_to_A = 0.5`). That is what makes the model run-length-aware instead of
  merely smoothed.
- **Hysteresis is the prior, not a window.** There is no median filter, no minimum-length clamp and
  no post-hoc relabelling step in the pipeline. `BlipTest` shows a 3-frame blip being held *and* a
  3-frame genuine counter-attack being taken when the emission pays for `log p(d)` — short
  possessions remain possible.
- **Restart-aware transitions.** `evidence.restart_modulation` adds `+strength (1.1)` log-odds to
  `D → H` / `D → A` at the tagged restart team, `−0.5 × strength` to the other, over
  `window_frames = 50`, and touches the `D` row only, so `H ↔ A` stays as the embedded chain says.
  It is a **one-time entry bonus**, which is why a sustained per-frame deficit can still beat it —
  `RestartAwareTransitionTest` is built around that distinction, and returns all-zero modulation
  when tagging is absent.
- **Per-half decoding.** Each graded block and each ungraded gap is decoded as its own block;
  `HalfSeparationTest` pins that no spell crosses the half break and that exactly one graded
  transition exists per half boundary.
- **`X` only outside graded periods.** `decoder.abstain.enabled = false`: inside a half every frame
  gets an answer. Outside, the label is `X` by construction, not by inference
  (`output.abstain_outside_graded = X`).

## 5. What a run writes

| artifact | path | in the repository? |
|---|---|---|
| submission CSV | `out/<TEAM_NAME>.csv` | **no** — git-ignored |
| manifest | `out/<TEAM_NAME>_manifest.json` | no |
| transitions | `out/<TEAM_NAME>_transitions.csv` | no |
| uncertainty | `out/<TEAM_NAME>_uncertainty.csv` | no |

The manifest is the audit trail: config digest, seed, Python/numpy/platform, the shapes actually
read, every integrity check with its status, every parameter *estimated* from unlabelled frames,
and the sha256 of the CSV that came out. `src/reporting/manifest.py::technical_document` renders
the run one-pager **from that same dict**, so a reported number cannot drift from an applied one.

The writer (`src/reporting/submission.py`) is deliberately dumb: header `frame,label`, one row per
video frame in order, only `H`/`A`/`D`/`X`, explicit LF on every platform, and it **raises** rather
than write a frame it has no answer for.

## 6. Provenance of every number

| kind | where it comes from |
|---|---|
| geometry, frame count, half boundaries | the organizers' published files, read through their own loaders |
| weights, sigmas, thresholds, transition and duration priors | `config/r3p.toml`, fixed, versioned, digest `eed08b805830d316` |
| channel medians and MADs | computed from the unlabelled frames of the match being processed |
| temperature `T` | solved on unlabelled frames so median confidence hits `target_median_confidence = 0.72` |
| attacking direction, geometric ball frames | estimated from unlabelled frames |
| decoded run-length medians | measured from the decoder's own runs on unlabelled frames |
| the public five-minute ground truth | **never read by the pipeline.** Only `tests/test_alignment.py` and `src/evaluation/scoring.py` touch it, for alignment verification, and no code path feeds it into training, tuning, selection or optimisation |

## 7. Evidence that exists, and the claim that does not

**Tests.** 80 unit tests, stdlib `unittest`, fixed seeds, run with
`python -W error::RuntimeWarning -m unittest discover -s tests -t .` — **all pass**, ~30 s on an
idle machine (~70 s under load).

| file | pins |
|---|---|
| `test_alignment.py` | published frame counts, half blocks, `frame_id → video_frame`, tag second→frame mapping |
| `test_evidence.py` | metric distances in metres, missing observations, ghost tracklets, reliability, no double counting |
| `test_deadball.py` | two-head factorisation, static/moving signs, missing ≠ dead, tagging as weak evidence, deflection ≠ turnover |
| `test_decoder.py` | the exact decoding objective, blip holding, half separation, restart modulation, determinism |
| `test_submission.py` | the CSV contract, validator PASS/FAIL cases, byte-identical rewrites |
| `test_pipeline.py` | end-to-end execution, integrity failure modes, reproducibility, the technical document |
| `test_document.py` | the submission PDF's source: the six required headings, table shape, wrapped bullets, bold containing code, no raw markdown surviving |

Three real model bugs were found by those tests and fixed rather than the tests being loosened:
`_duplicate_pairs` was re-scaling metres by the pitch (ghost tracklets were undetectable), the
attack-direction guard produced `NaN` on an empty half, and the `collapse` cue fed missing slots
through as zeros, which faked "everybody standing still" in most frames of a real match.

**Synthetic self-check.** A full-length (199,319-frame) bundle generated by
`src/data/synthetic.py`, with labels being that simulator's own latent state:

```
exit 0 · 18 integrity checks: 0 fail, 3 warn · official validator PASS at published length
H 53379 · A 49590 · D 55888 · X 40462
604 possessions · mean 6.82 s · median 4.82 s · longest 46.72 s · 18 under 1 s
232 dead-ball spells · 835 transitions · temperature 2.01
sha256 fd5866e0296ea3b084585fe770f062a7f55f8dedf465e67ea4831169cb2f2379
against the simulator's latent state:
  spell_f1_mean 0.7689 · macro_f1 0.9569 · accuracy 0.9579 · alive/dead 0.9944
  fragmentation 1.229 · spells 449 (latent) vs 604 (decoded) · mean spell 9.21 s vs 6.82 s
```

**These numbers measure code correctness, not football.** The truth is generated by our own
simulator, so it is a home game: it proves alignment, wiring, determinism and that the decoder
optimises what we wrote down. It says nothing about `BAR-ATH`.

**The claim we are not making.** R3P has never been run on the restricted match, so it is not
established that R3P beats the official baseline. No score against the organizers' baseline exists
in this repository, and we will not invent one.

## 8. Known limitations

1. The gated match assets are not present in this checkout (`data/` holds only its own README), so
   no real submission CSV has been produced here. Running `python run_pipeline.py --data-dir data
   --team-name "MilosG"` against an unpacked bundle is the remaining step, and it is the step
   whose output the organizers actually read.
2. On our own simulator the decoder **over-segments**: 604 decoded spells against 449 latent ones,
   mean 6.82 s against 9.21 s. The honest response was to fix the three real bugs and stop; further
   tuning against the simulator would be tuning against a fiction.
3. `config/r3p.toml` thresholds are physically reasoned (metres, seconds, frame counts) rather than
   fitted. That is defensible without ground truth and is exactly what a real run would need to be
   checked against.
4. The Streamlit dashboard (`app.py`) is compile-verified only: `streamlit` is not installed in this
   environment, so it has not been opened in a browser here.
