# R3P — technical document

**Team:** MilosG · **Model:** R3P (Reliability-weighted, Restart-aware, Run-length-regularized
Possession Model) · **Game:** BAR-ATH, tactical source · **Repository:**
`https://github.com/milosgnjidicenquiries/BarcaProject`

Written to the six headings that `SUBMISSION.md` asks for. Longer version:
`docs/TECHNICAL_DOCUMENT.md`; clause-by-clause rules mapping: `docs/COMPLIANCE.md`.

---

## 1. The approach

R3P does not classify frames. It decodes a whole path, because the thing being labelled — a
possession — is a property of an interval, not of a frame. A frame-level classifier is forced to
guess at every tracking blip and then smoothed afterwards, which is where short real possessions
are lost and fake ones are invented. The pipeline therefore has three stages, and only the last
one emits labels.

**A — reliability-weighted fusion.** Seven per-frame evidence channels are computed in metres from
the ATD and combined after robust standardisation:

```
d_i = sqrt((105.3*dx_i)^2 + (68.0*dy_i)^2)              metric pitch distance
S   = mean_i exp(-d_i^2 / (2 sigma^2))                  sigma = 3.0 m control, 7.0 m support
z_c = clip((x_c - median_c) / (1.4826*MAD_c), -4, 4)    robust, over graded frames only
score = sum_c w_c z_c        q = sigmoid(score / T) * rho
```

Channels: `nearest` (1.00), `soft_proximity` (0.70), `ball_kinematics` (0.55), `continuity` (0.45),
`tagging` (0.40), `support` (0.35), `territory` (0.30). `rho` is per-frame reliability — share of
slots tracked, a 25-frame grace on the ball column, ghost-tracklet detection inside 1.5 m, silence
below 8 visible players — smoothed over 12 frames. `T` is solved so that median confidence lands on
0.72. Why robust statistics rather than mean/std: one bad tracking outage per half would otherwise
move the scale of a whole channel.

**B — dead ball and team possession are separate inferences.** One score cannot express "the ball
is standing at the corner flag and nobody owns it". So the model keeps two heads apart:

```
P(D) = p_dead          from dead-ball cues only
P(H) = (1 - p_dead) q  from the fusion in A
P(A) = (1 - p_dead)(1 - q)
```

`p_dead` is a signed logistic over: ball standing still (+2.2), ball travelling (**−2.0**), ball out
of play (+1.8), ball missing (+0.55), squad-wide speed collapse (+0.8), tagged set piece (+0.75),
players massed in a wall (+0.45).

**C — duration-aware semi-Markov decoding, per half.** The winning label path maximises

```
argmax_path  sum_t log emission_t(s_t) + sum_t log A(s_{t-1},s_t) + sum_runs log p(d = L)
```

solved exactly by an age-expanded Viterbi (state = label + run-age), with a bin-discretised
log-normal run prior (median 6.0 s / 6.0 s / 7.0 s for H / A / D), a 900-frame explicit cap and a
geometric tail beyond it. `A` has no diagonal: a run *continuing* is charged to the duration model,
a run *ending* is charged to the transition matrix. That single choice is what makes the model
run-length-regularized instead of smoothed — hysteresis is learned structure, not a filter, and a
genuine two-second counter-attack can still win if the emission pays for `log p(d=50)`.

## 2. Which assets were used, and how

| asset | used | how |
|---|---|---|
| ATD tracking (tactical) | **yes** | read through the organizers' `starter/load_atd.py`, unmodified; per-slot `x,y,s` in metres; `video_frame = frame_id - 1` |
| Homography | **yes** | via `starter/load_homography.py`; used to confirm the coordinates really are metric and to place the ball against the touchline |
| ATD meta | **yes** | team names and `team_of_slot`, so `H`/`A` are never guessed from a slot index |
| Smart Tagging | **yes, weakly** | kickoffs and set pieces → `p_dead` cue at +0.75 and restart-aware `D → H`/`D → A` modulation at 1.1 log-odds over 50 frames; fusion weight 0.40. Never allowed to decide a frame alone |
| Public 5-minute ground truth | **verification only** | frame-alignment checks in `tests/test_alignment.py` and post-hoc reporting in `src/evaluation/scoring.py`. It is read by no inference, tuning, selection or optimisation path — the duration prior a real run uses is measured from the decoder's own output on unlabelled frames, not from the GT's mean spell length |
| Broadcast tracking | **not used** | one recording is graded; the tactical stream is the one submissions index to, so the second view would have added a fusion channel whose reliability is not measured the same way |
| Video files | **not used at all** | no frame of video was decoded or inspected. There is no `cv2`/`ffmpeg` dependency in the project, and the rules forbid turning visual observation into frame labels, so a path that could not be audited was not built |

## 3. Training data

**There is none.** Nothing in R3P is fitted, and no labelled dataset of any kind was used:

- No private or full-match ground truth was ever read.
- The public five-minute file was not used for training, tuning, parameter selection or
  optimisation (see the row above).
- No external labelled data from other matches was used — therefore there is no annotator,
  labelling method or quantity to document.
- No model weights, no pre-trained embeddings, no optimiser, no validation split. Every number in
  `config/r3p.toml` is a physically reasoned constant in metres, seconds or frames, versioned in
  the repository.

The only quantity *learned* at run time is unsupervised and comes from the unlabelled frames of the
match being processed: channel medians and MADs, the temperature, the attacking direction, the
ball positions imputed geometrically, and the decoded run-length medians. The manifest lists each
of them under `estimated_from_unlabelled_data`, so a reviewer can separate "chosen by a human" from
"measured by the run" without reading source.

## 4. The possession rules actually implemented

| situation | what the model does | why |
|---|---|---|
| **Loose ball** | `q` is damped: both sides score low, `continuity` carries the phase, and `nearest` alone (a single closest player) cannot flip ownership | one metric distance is not a possession |
| **Challenge / 50-50** | ownership follows the *weighted* side, not the nearest slot; support at 7 m and territory both vote | a duel is a distribution, and the decoder holds whichever side the distribution prefers until it stops preferring it |
| **Touch, deflection, rebound** | changes `nearest` and the ball kinematics; does **not** change `continuity`, so no turnover is emitted | the rules require this explicitly; `tests/test_deadball.py::TurnoverIsNotPredictedTest` pins it |
| **Ball missing from ATD (~41 % of graded frames)** | imputed geometrically, `rho` reduced to a 0.35 floor, `missing` the weakest dead-ball cue, gaps under 40 frames ignored | a dropout is an occlusion, not the referee stopping play |
| **Dead ball** | `p_dead` from the physical cues; a travelling ball argues against stoppage with a minus sign; a set-piece tag raises it but cannot dominate it | the two-head split is the reason the D class is not a by-product of the team class |
| **Restart** | one-time additive log-odds on `D → H` or `D → A` for the tagged restart team, over 50 frames, applied only to the `D` row; a sustained per-frame deficit still beats it | a kickoff tells you who *starts*, not who keeps it for the next minute |
| **Half-time** | the ungraded gap is a separate decode block; no spell crosses the break | a possession does not survive the whistle |
| **Outside graded periods** | `X`, by construction, never by inference; abstention inside a half is switched off | the contract reserves `X` for what is not graded |
| **Sub-second jitter** | suppressed by the duration prior, not by a filter or a minimum-length clamp; `under_1s` is reported as a diagnostic | a clamp would be a manual correction of the model's output |

## 5. Limitations

1. **The model has never been run on the graded recording.** The match bundle is gated and was not
   available in the environment where this was built; `data/` in the repository holds only its own
   README. Everything below was measured on a full-length bundle generated by
   `src/data/synthetic.py`, whose truth is our simulator's latent state.
2. **No baseline claim is made.** `spellF1` and `macroF1` against the organizers' baseline for
   BAR-ATH do not exist in this repository, because the data that would produce them was never
   here. We do not claim to beat the baseline.
3. **Over-segmentation on our own simulator.** 604 decoded possessions against 449 latent ones;
   mean spell 6.82 s against 9.21 s latent. The three bugs found while investigating this were
   real and were fixed in the model (a double pitch scaling that made ghost tracklets invisible, a
   `NaN` attacking direction on an empty half, and a `collapse` cue that read missing slots as
   zeros). We deliberately stopped there: tuning further against data we invented would be fitting
   a fiction, and would show up as a number we cannot defend.
4. **Fixed, un-fitted thresholds.** σ, the 25-frame ball grace, the 1.5 m ghost radius and the
   duration medians are physically reasoned. On the real match some of them may be wrong by a
   factor, and the honest way to find that out is one run plus the integrity and uncertainty
   tables — not a tuning loop against the public sample.
5. **`nearest` and `soft_proximity` are correlated by construction.** They are kept separate and
   separately standardised, and no dead-ball cue ever enters the team score, but the fusion is not
   an orthogonal decomposition of possession.
6. **The dashboard is compile-verified only.** `streamlit` is not installed in the build
   environment, so `app.py` has not been opened in a browser.
7. **With more time:** one supervised pass on a *different* Metrica game for the fusion weights,
   a second ball-visibility model to raise `rho` during broadcast occlusions, and an evaluation of
   the restart window against the tagger's own latency.

## 6. Reproduction

The bar is the same file, not an approximate one.

```bat
git clone https://github.com/milosgnjidicenquiries/BarcaProject
cd BarcaProject
py -3.13 -m venv .venv && .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-r3p.txt
python -W error::RuntimeWarning -m unittest discover -s tests -t .
```

Then unpack the match bundle Metrica sent you into `data/` (so `data\atd\`, `data\homography\` and
`data\smart_tagging\` exist), and:

```bat
python run_pipeline.py --data-dir data --team-name "MilosG"
```

This writes `out\MilosG.csv`, `out\MilosG_manifest.json`, `out\MilosG_transitions.csv` and
`out\MilosG_uncertainty.csv`, then validates with the organizers' own validator. Exit codes: `0`
ok, `2` data missing, `3` integrity failure under `--strict`, `4` validator rejected the file, `5`
self-check requested against a non-fixture, `6` no graded frames in the bundle. `data/` and `out/`
are git-ignored, so neither the bundle nor the submission is ever published.

Determinism: fixed seed 20271117, no stochastic component remains in the pipeline, and every run
records the sha256 of the CSV plus the config digest in its manifest. The current config digest is
`eed08b805830d316`; the digest changed only when the team name was written into the config, and the
team name is metadata — two runs that differ in nothing else produce byte-identical CSVs.

### Evidence that exists

80 unit tests pass in about 30 s on an idle machine (~70 s under load) on stdlib `unittest`, with
`RuntimeWarning` promoted to an error, no network and no match data: alignment and
`frame_id → video_frame`, metric distances in metres,
missing observations, ghost tracklets, tagging interpretation, the two-head factorisation,
dead-ball cues, controlled turnovers, half separation, the exact decoding objective, restart
modulation, the CSV contract, end-to-end execution, integrity failure modes and byte-level
reproducibility.

A full-length synthetic run (`python run_pipeline.py --data-dir <fixture> --selfcheck <latent>`):
`0` integrity failures, official validator **PASS** at the published length, 199,319 rows
(H 53 379 · A 49 590 · D 55 888 · X 40 462), sha256
`fd5866e0296ea3b084585fe770f062a7f55f8dedf465e67ea4831169cb2f2379`, and against the simulator's own
latent state `spell_f1_mean` 0.7689, `macro_f1` 0.9569, `alive_dead_f1` 0.9944, fragmentation 1.229.
**These numbers validate the wiring, not the football** — the truth in them was generated by this
repository, and none of it describes BAR-ATH.
