# R3P — Reliability-weighted, Restart-aware, Run-length-regularized Possession Model

An entry for Metrica Sports' **More Than a Hack 2027** (`frame,label` possession labelling of the
BAR–ATH recording). This repository is built **on top of** the organizers' repository: their
`evaluation/`, `starter/`, `games.py`, `DATA.md`, `RULES.md` and the rest are unmodified, and R3P
lives in `src/`, `config/`, `tests/`, `run_pipeline.py` and `app.py`.

The organizers' own README is preserved at [`docs/OFFICIAL_README.md`](docs/OFFICIAL_README.md).
The rules that bind this entry are in [`RULES.md`](RULES.md); how this code meets them is in
[`docs/COMPLIANCE.md`](docs/COMPLIANCE.md); what the model is, in
[`docs/TECHNICAL_DOCUMENT.md`](docs/TECHNICAL_DOCUMENT.md).

## One-paragraph description

R3P never classifies a frame. It builds seven per-frame evidence channels in metres from the ATD —
the ball's distance to each side under the real pitch metric, a Gaussian control kernel, a support
count, ball kinematics, territory, a temporal continuity vote, and Smart Tagging as a deliberately
weak prior — weights them into a single signed log-odds after reliability weighting and robust
(median/MAD) standardisation, splits the problem into **is the ball dead** and, separately, **which
team has it**, and then decodes the whole match with an exact semi-Markov Viterbi over an
age-augmented state, so the winning path maximises emission + transition + *run-length* probability.
Hysteresis is not a rule bolted on top; it is the duration prior. A three-frame tracking blip has to
pay `log p(d = 3)` under a six-second-median log-normal and loses, while a real two-second
counter-attack pays the same price and wins if the evidence is strong enough. Possessions are decoded
per half, and `X` is emitted only outside the graded window.

## Install (Windows)

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-r3p.txt
```

`requirements-r3p.txt` pins numpy 2.2.6 / pandas 2.2.3 / streamlit 1.67.0. The model itself imports
only numpy and pandas. Python 3.10+ is fine; nothing here needs a GPU, torch, or kloppy.

## Run

```bat
python run_pipeline.py --data-dir data --team-name "MilosG"
```

That single line reads the tactical bundle from `data/`, runs the integrity checks, infers, writes
`out/MilosG.csv` plus a manifest and two diagnostics CSVs, and validates the file with the
organizers' own `evaluation/validate_submission.py` logic. Add `--reproduce-baseline` to also rebuild
the official smooth-50 baseline from the ATD and compare bytes, `--public-check` to score against the
published five-minute sample (alignment verification only — see the rules), `--strict` to make an
integrity failure stop the run.

Exit codes: `0` ok, `2` data missing, `3` integrity failure under `--strict`, `4` the validator
rejected the file, `5` a self-check was requested against something that is not a development
fixture, `6` the bundle contains no graded frames.

## The submission document

```bat
python docs\build_pdf.py
```

Renders [`docs/SUBMISSION_DOCUMENT.md`](docs/SUBMISSION_DOCUMENT.md) into `MilosG.pdf` in the
repository root — the file the submission form asks for, named after the team because the board is
labelled by file name. The markdown is the single source and is committed; the PDF is git-ignored
(`*.pdf`), so the repository carries readable text and the form gets the upload. The builder needs
no new dependency: it renders the markdown subset itself and prints through the Edge or Chrome that
is already installed. `tests/test_document.py` checks the rendered structure, so a construct the
converter does not understand fails a test instead of quietly vanishing from the PDF.

## Dashboard (read-only)

```bat
python -m streamlit run app.py
```

Integrity table, live progress, the algorithm and its provenance, the timeline, the H/A/D/X
distribution, transitions with their uncertainty, and a button that writes the submission and runs
the validator. It has **no** annotation, editing or label-correction capability: the rules allow
charts and video inspection for understanding, and forbid turning them into labels.

## Tests

```bat
python -m unittest discover -s tests -t .
```

80 tests, about 30 seconds on an idle machine, no network, no match data. They use fixed seeds and
hand-built tracking arrays plus a synthetic bundle written in the delivered file formats:

| file | what it pins down |
| --- | --- |
| `tests/test_alignment.py` | `video_frame = frame_id - 1`, the 0-1 → metre scaling, the two graded blocks (158 857 frames), per-half decoding masks, a short file being a row-count failure rather than NaN padding |
| `tests/test_evidence.py` | the anisotropic pitch metric (a 2.20 m x-gap losing to a 2.10 m y-gap), missing observations, ghost/duplicate tracklets, the tagging reader's `round(s × 25)` and team-name matching |
| `tests/test_deadball.py` | the two-head factorisation, a moving ball counted with a **minus** sign, a ball dropout not being a stationary ball, the wall/oob/collapse cues, Smart Tagging capped below the physical cues, one deflection not winning a phase |
| `tests/test_decoder.py` | the decoded path beating the frame-wise argmax and hand-built alternatives **on the documented objective**, the duration median landing on 6 s in frames, the halftime hole, restart modulation, `-1e30` vs `-inf`, determinism |
| `tests/test_submission.py` | the official validator accepting our bytes and rejecting blanks, a one-based frame column, unknown letters and a short file; LF-only bytes; refusing to write unlabeled frames |
| `tests/test_pipeline.py` | end-to-end on a real bundle on disk, byte-identical reruns, controlled turnovers recovered, the integrity layer catching a wrong-length input, and the exported technical document containing no number the run did not produce |
| `tests/test_document.py` | the submission PDF's own source: the six headings `SUBMISSION.md` demands, tables keeping their rows and columns, wrapped bullets staying one item, bold containing inline code rendering as both, and no raw markdown surviving into the uploaded file |

## What is in the repository

```
config/r3p.toml            every parameter, each tagged [physics] [derived] [format] [model]
src/official.py            the bridge to the organizers' code (their files are never edited)
src/data/                  ATD streaming reader, Smart Tagging, homography, integrity, synthetic bundle
src/possession/evidence.py components A (reliability-weighted fusion) and B (dead-ball head)
src/possession/decoder.py  component C (semi-Markov, age-expanded Viterbi, per half)
src/possession/model.py    the three stages, the abstention policy, the run report
src/pipeline.py            read -> check -> infer -> write -> validate
src/reporting/             the submission CSV, the manifest, the human-readable summary
src/evaluation/scoring.py  calls the organizers' scorer; never re-implements it
run_pipeline.py            the CLI
app.py                     the read-only Streamlit dashboard
tests/                     80 unittest cases, fixed seeds, no match data
```

`data/`, `out/`, `scratch/`, every bundle, every CSV and every log are git-ignored. **The submission
CSV is not in this repository** — it is derived from restricted match assets, and publishing it would
publish a reconstruction of the match.

## Honest status of the claims in this README

* The match assets are gated by the organizers and are **not present** on the machine this was built
  on. The model has therefore never been run against the real recording. Everything above is
  implemented and executed against a development bundle generated by `src/data/synthetic.py`, which
  writes the same file formats and is marked `SYNTHETIC_BUNDLE.json` so no run can mistake it for the
  real thing.
* On that synthetic bundle, the whole pipeline runs at match length (199 319 frames) and produces a
  CSV that passes the organizers' validator, with the official metrics against the simulator's own
  latent state reported in `docs/TECHNICAL_DOCUMENT.md` — labelled there, and in the manifest, as a
  code-correctness check and nothing else.
* No claim is made that R3P beats the official baseline. That comparison can only be made on the
  real data, by the organizers, and it has not been made here.
