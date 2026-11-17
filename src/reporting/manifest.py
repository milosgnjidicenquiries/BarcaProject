"""The run manifest: what happened, in machine-readable form, and the same thing as a one-pager.

Reproducibility is a claim about bytes, so the manifest records the things that decide bytes: the
config digest, the seed, the numpy and Python versions, the shape of the data actually read, the
integrity results, the parameters that were *estimated* from unlabelled frames rather than chosen, and
the hash of the CSV that came out. Two runs over the same bundle with the same digest must produce the
same file; if they do not, this is where the difference is visible.

Nothing here is a performance claim. Scores appear in the manifest only when a scorer was actually run
and the ground truth it was given is named. The synthetic self-check says "synthetic" in its key name
for exactly that reason.
"""

from __future__ import annotations

import datetime
import json
import os
import platform

import numpy as np

from .. import official, settings
from ..possession import decoder
from . import submission, summary


def _json_safe(value):
    """numpy scalars and arrays to plain python, so json.dump never dies halfway through a run."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def build(cfg, bundle, labels, report, csv_path=None, validator=None, selfcheck=None,
          timings=None):
    facts = cfg["facts"]
    codes = np.asarray(labels, dtype=np.int8)
    graded = report["arrays"]["graded"]
    prior = report["duration_prior"]

    return {
        "team_name": cfg["run"]["team_name"],
        "model": "R3P",
        "model_version": cfg["meta"]["model_version"],
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                   .replace(microsecond=0).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "provenance": {
            "config_digest": settings.config_digest(),
            "seed": int(cfg["meta"]["seed"]),
            "data_dir": cfg["run"]["data_dir"],
            "source": cfg["run"]["source"],
            "game": cfg["run"]["game"],
            "official_repository": "metrica-sports/more-than-a-hack-2027",
            "development_bundle": bool(bundle.development_bundle),
            "bundle_paths": {name: os.path.relpath(path, official.ROOT).replace(os.sep, "/")
                             for name, path in bundle.paths.items()},
        },
        "data": {
            "frames_read": int(len(bundle.tracking)),
            "frames_expected": int(facts["n_video_frames"]),
            "fps": facts["fps"],
            "pitch_m": [facts["pitch_length_m"], facts["pitch_width_m"]],
            "graded_frames": report["graded_frames"],
            "ungraded_frames": int(len(codes) - report["graded_frames"]),
            "graded_blocks": report["graded_blocks"],
            "halves": facts["halves"],
            "integrity": dict(report["bundle"], rows=bundle.checks),
        },
        "output": {
            "path": (os.path.relpath(csv_path, official.ROOT).replace(os.sep, "/")
                     if csv_path else None),
            "sha256": submission.sha256_file(csv_path) if csv_path else None,
            "rows": int(len(codes)),
            "header": "frame,label",
            "label_counts": {name: int((codes == index).sum())
                             for index, name in enumerate(summary.LABEL_NAMES)},
            "x_only_outside_graded": report["outside_graded_are_X"],
            "abstained_inside_graded": report["abstained_inside_graded"],
            "validator": validator,
        },
        "model_parameters": {
            "fusion_weights": dict(cfg["fusion"]["weights"]),
            "reliability": dict(cfg["fusion"]["reliability"]),
            "sigma_control_m": cfg["fusion"]["sigma_control_m"],
            "sigma_support_m": cfg["fusion"]["sigma_support_m"],
            "temporal_smooth_frames": cfg["fusion"]["temporal_smooth_frames"],
            "deadball": dict(cfg["deadball"]),
            "decoder": {
                "max_duration_frames": int(cfg["decoder"]["max_duration_frames"]),
                "tail_survival": cfg["decoder"]["tail_survival"],
                "duration_medians_s": prior["duration_medians_s"],
                "median_run_frames": prior["median_run_frames"],
                "embedded_transition": dict(cfg["decoder"]["transition"]),
                "restart": dict(cfg["decoder"]["restart"]),
                "abstain": dict(cfg["decoder"]["abstain"]),
            },
            "channel_usage": report["channel_usage"],
            # Everything in this block was fitted on unlabelled frames of the match itself. Listing it
            # apart is the point: these are the only numbers a run invents as it goes, and a reviewer
            # should be able to see them without reading the source.
            "estimated_from_unlabelled_data": {
                "temperature": report["temperature"],
                "attacking_direction_home": report["direction_home"],
                "geometric_ball_frames": report["geometric_frames"],
                "standardisation": report["standardisation"],
                "median_q_alive": report["q_median"],
                "median_p_dead": report["p_dead_median"],
                "median_frame_reliability": report["reliability_median"],
                "median_state_separation_nats": report["separation_median"],
            },
        },
        "temporal_structure": {
            "decoder_blocks": report["decoder_blocks"],
            "n_transitions": report["n_transitions"],
            "spells": report["spells"],
            "label_table": report["label_table"],
        },
        "timing_seconds": timings or {},
        "selfcheck_on_synthetic_latent_state": selfcheck,
        "compliance": {
            "labels_generated_algorithmically": True,
            "hand_labelled_or_corrected_frames": 0,
            "private_ground_truth_used": False,
            "public_ground_truth_used_for": (
                "nothing in this run; the pipeline may verify alignment with it, and that "
                "verification never feeds training, tuning, model selection or optimisation"),
            "public_gt_mean_spell_seconds_used_as_a_prior": False,
            "external_labelled_data_used": False,
            "video_inspection_used_for": (
                "understanding and debugging only; no visual observation was turned into a "
                "frame-specific label, correction or hard-coded rule"),
            "manual_annotation_or_correction_implemented": False,
            "restricted_match_assets_in_repository": False,
            "submission_csv_committed": False,
        },
    }


def technical_document(cfg, bundle, labels, report, csv_path=None, validator=None,
                       selfcheck=None, timings=None):
    """The one-pager, rendered from the manifest dict rather than written by hand.

    There is exactly one place a number can come from: `build()`. A document that recomputed its own
    summary would be a second source of truth, and the second source is the one that drifts.
    """
    manifest = _json_safe(build(cfg, bundle, labels, report, csv_path=csv_path,
                                validator=validator, selfcheck=selfcheck, timings=timings))
    out = manifest["output"]
    data = manifest["data"]
    facts = cfg["facts"]
    fusion = cfg["fusion"]
    dead = cfg["deadball"]
    dec = cfg["decoder"]
    prior = report["duration_prior"]
    usage = report["channel_usage"]["channels"]

    lines = []
    add = lines.append
    add("# R3P -- technical document for run `{}`".format(manifest["team_name"]))
    add("")
    add("Generated {} from config digest `{}` and seed {}. Python {}, numpy {}. "
        "Every number below is copied from the run manifest, not restated by hand.".format(
            manifest["generated_utc"], manifest["provenance"]["config_digest"],
            manifest["provenance"]["seed"], manifest["environment"]["python"],
            manifest["environment"]["numpy"]))
    add("")
    add("## What the model is")
    add("")
    add("R3P decodes one label per video frame -- `H`, `A`, `D`, `X` -- from the official ATD tracking "
        "and Smart Tagging files. It has three parts, and the third is where the labels are actually "
        "made:")
    add("")
    add("1. **Reliability-weighted fusion.** %d evidence channels are combined per frame after being "
        "made comparable by robust (median/MAD) standardisation, then weighted by how much of the frame "
        "was observable. The distance that drives the team-control channel is metric:"
        % len(fusion["weights"]))
    add("")
    add("   `d = sqrt((%.1f*dx)^2 + (%.1f*dy)^2)` over the pitch, "
        "`S_team = mean_i exp(-d_i^2 / (2*sigma^2))`." % (facts["pitch_length_m"],
                                                          facts["pitch_width_m"]))
    add("   sigma = %.1f m for the controlling side and %.1f m for the supporting side; "
        "the per-frame reliability is the share of slots actually tracked, smoothed over %d frames."
        % (fusion["sigma_control_m"], fusion["sigma_support_m"], fusion["temporal_smooth_frames"]))
    add("")
    add("2. **Restart-aware separation of dead ball from team possession.** `p_dead` and `q` "
        "(who owns it, given that somebody does) are two heads, not one score: "
        "`P(D) = p_dead`, `P(H) = (1 - p_dead) * q`, `P(A) = (1 - p_dead) * (1 - q)`. A missing ball is "
        "imputed geometrically and never read as a dead ball; Smart Tagging enters as weak evidence "
        "with weight %.2f, below the geometric cues; a touch, deflection or rebound changes `nearest` "
        "without changing `continuity`, so it is not a turnover."
        % dead["tagging_weight"])
    add("")
    add("3. **Duration-aware semi-Markov decoding.** For each half independently:")
    add("")
    add("   `argmax_path [ sum_t log emission_t(s_t) + sum_t log transition(s_{t-1}, s_t) "
        "+ sum_spells log duration(s, L) ]`")
    add("")
    add("   solved by an age-expanded Viterbi over %d states with a maximum explicit spell length of "
        "%d frames and a geometric tail beyond it (%.4f survival per frame). The duration prior was "
        "*estimated from the unlabelled frames of this match*: median run length %s frames and %s s "
        "per state. Restart-aware transition modulation applies only when the previous spell is "
        "`D`. Hysteresis lives in the transition matrix, not in a post-processing filter."
        % (decoder.N_STATES, dec["max_duration_frames"], dec["tail_survival"],
           _fmt_value(prior["median_run_frames"]), _fmt_value(prior["duration_medians_s"])))
    add("")
    add("`X` is emitted only outside the graded periods; inside a half every frame is decided.")
    add("")
    add("## The data actually read")
    add("")
    add("| | |")
    add("|---|---|")
    add("| frames read | %d (expected %d) |" % (data["frames_read"], data["frames_expected"]))
    add("| rate | %d fps |" % data["fps"])
    add("| pitch | %.1f x %.1f m |" % (facts["pitch_length_m"], facts["pitch_width_m"]))
    add("| graded frames | %d |" % data["graded_frames"])
    add("| ungraded frames (X) | %d |" % data["ungraded_frames"])
    add("| halves | %s |" % data["halves"])
    add("| development bundle | %s |" % manifest["provenance"]["development_bundle"])
    add("")
    add("### Integrity checks")
    add("")
    counts = {}
    for row in data["integrity"]["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    add("{} checks: {}.".format(len(data["integrity"]["rows"]),
                                ", ".join("{} {}".format(n, s) for s, n in sorted(counts.items()))))
    for row in data["integrity"]["rows"]:
        if row["status"] != "ok":
            add("- **%s** `%s`: expected %s, observed %s -- %s" % (
                row["status"], row["name"], row["expected"], row["observed"], row["note"]))
    add("")
    add("## Output contract")
    add("")
    add("- Path: `%s`" % out["path"])
    add("- Rows: %d, header `%s`, no index column, no blank cells, LF newlines." % (out["rows"],
                                                                                    out["header"]))
    add("- sha256: `%s`" % out["sha256"])
    add("- Official validator: %s" % _validator_line(out["validator"]))
    add("- Label counts: %s" % out["label_counts"])
    add("- X only outside graded periods: %s; abstentions inside a half: %d"
        % (out["x_only_outside_graded"], out["abstained_inside_graded"]))
    add("")
    add("## Temporal structure of the decoded path")
    add("")
    add("- Decoder blocks (one per half, plus the ungraded gaps): %s" % manifest["temporal_structure"]
        ["decoder_blocks"])
    add("- Transitions: %d" % manifest["temporal_structure"]["n_transitions"])
    add("- Possessions: %s" % _spell_line(report["spells"]))
    add("")
    add("### Blocks the decoder solved independently")
    add("")
    add("| frames | runs | median run | longest run | saturated at the cap |")
    add("|---|---|---|---|---|")
    for row in report["decoder_blocks"]:
        add("| {}–{} | {} | {} | {} | {} |".format(row["start"], row["stop"], row["n_runs"],
                                                    row["median_run"], row["longest_run"],
                                                    row["saturated_frames"]))
    add("")
    add("### Estimated from unlabelled frames of this match")
    add("")
    add("These are the only quantities a run invents as it goes. None of them is fitted against "
        "ground truth; the public five-minute file is never read by this code path.")
    add("")
    for key, value in manifest["model_parameters"]["estimated_from_unlabelled_data"].items():
        if isinstance(value, dict) and any(isinstance(v, dict) for v in value.values()):
            add("- `%s`:" % key)
            for name, row in value.items():
                add("  - %s: %s" % (name, _fmt_value(row)))
        else:
            add("- `%s`: %s" % (key, _fmt_value(value)))
    add("")
    add("### Evidence channels")
    add("")
    add("| channel | weight | median | MAD | frames used |")
    add("|---|---|---|---|---|")
    for row in usage:
        add("| %s | %.3f | %s | %s | %s |" % (row["channel"], row["weight"],
                                               _fmt_value(row.get("median")),
                                               _fmt_value(row.get("mad")), row.get("frames_used")))
    add("")
    add("## What this run does *not* claim")
    add("")
    if selfcheck:
        add("A self-check against the latent state of our own synthetic bundle is included below. It "
            "tests that the pipeline is wired correctly on data whose truth we generated; it is not a "
            "measurement of performance on the restricted match.")
        add("")
        add("```json")
        add(json.dumps(selfcheck, indent=2))
        add("```")
    else:
        add("No ground truth was scored in this run, so this document contains no performance numbers "
            "at all -- not for the official baseline, and not for R3P. A comparison that was never run "
            "is not a result.")
    add("")
    add("The official baseline was not beaten by any evidence available to this run, and no claim that "
        "it was is made here.")
    add("")
    add("## Compliance")
    add("")
    for key, value in manifest["compliance"].items():
        add("- `%s`: %s" % (key, _fmt_value(value)))
    add("")
    return "\n".join(lines)


def _validator_line(validator):
    """The validator's own verdict, flattened to one sentence.

    A dict dumped with repr() puts absolute temporary paths and doubled backslashes into a document
    that is meant to be read, and a reader cannot tell PASS from a dict that happens to say PASS.
    """
    if validator is None:
        return "not run in this call"
    if not isinstance(validator, dict):
        return str(validator)
    parts = ["**PASS**" if validator.get("valid") else "**FAIL**"]
    if validator.get("reason"):
        parts.append(str(validator["reason"]))
    if "rows_written" in validator and "expected_rows" in validator:
        parts.append("{} rows written against the published {}".format(
            validator["rows_written"], validator["expected_rows"]))
    round_trip = validator.get("round_trip")
    if isinstance(round_trip, (list, tuple)) and len(round_trip) == 2:
        parts.append("re-reads {}".format("identically" if round_trip[0]
                                          else "differently: {}".format(round_trip[1])))
    return " -- ".join(parts)


def _fmt_value(value):
    if isinstance(value, float):
        return "{:.4g}".format(value)
    if isinstance(value, dict):
        return ", ".join("{} = {}".format(k, _fmt_value(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_value(v) for v in value)
    return "null" if value is None else str(value)


def _spell_line(spells):
    if not spells:
        return "no alive spells decoded"
    parts = []
    for key in ("n_possessions", "mean_seconds", "median_seconds", "p90_seconds",
                "longest_seconds", "shortest_seconds", "under_1s", "under_2s"):
        if key in spells:
            parts.append("{} = {}".format(key, _fmt_value(spells[key])))
    return "; ".join(parts)


def write(manifest, path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(manifest), handle, indent=2)
        handle.write("\n")
    return path


def transitions_csv(manifest, report, path):
    """frame,from,to,seconds_after_previous for every decoded change.

    Written from the decoder's own runs rather than re-derived from the CSV, because the CSV has no
    margin column by contract and a table rebuilt from it could only ever report the frames.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    rows = report["transitions"]
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,from,to,seconds_after_previous\n")
        for row in rows:
            handle.write("{},{},{},{:.2f}\n".format(
                row["frame"], row["from"], row["to"], row["seconds_since_previous"]))
    return path


def uncertainty_csv(report, path):
    """The frames the model was least sure about, with the label it settled on.

    Read-only by construction: this is a list of places to look, and nothing downstream consumes it as
    an instruction.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    graded = report["arrays"]["graded"]
    separation = report["arrays"]["separation"]
    labels = np.asarray(report["labels"], dtype=np.int8)
    order = np.argsort(np.where(graded, separation, np.inf))
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,label,separation_nats\n")
        for frame in order:
            if not graded[frame]:
                continue
            handle.write("{},{},{:.4f}\n".format(int(frame),
                                                 summary.LABEL_NAMES[int(labels[frame])],
                                                 float(separation[frame])))
    return path
