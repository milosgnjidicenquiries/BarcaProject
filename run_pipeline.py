#!/usr/bin/env python
"""R3P command line: read the bundle, prove what it contains, infer, write, validate.

    python run_pipeline.py --data-dir data --team-name "TEAM_NAME"

That single form is what the technical document promises, so it is what works with no other arguments:
the tactical recording, the parameters in config/r3p.toml, output in out/, and the integrity checks
run first. Everything else is an opt-in.

What the run prints, in order:

  1. the integrity table -- what the files are, whether they agree with the published figures, and
     whether this is the restricted match or a development fixture;
  2. the model summary -- label shares, possessions and their durations, the temperature that was
     solved on unlabelled frames, the run-length medians;
  3. the submission path, its sha256, and the result of the organizers' own validator over the bytes;
  4. the manifest and the two CSVs that make the run checkable.

Scores are not printed unless a ground truth is named, because a number with no denominator is exactly
the kind of claim the rules forbid. --public-check names the five-minute sample and is for verifying
that the pipeline and the frame alignment work; it is never a tuning signal.

Exit codes: 0 fine, 2 data missing, 3 an integrity check failed under --strict, 4 the validator
rejected the file, 5 a self-check was asked for against something that is not a development fixture,
6 the bundle holds no graded frames (so a submission from it would be all X).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src import official, pipeline, settings  # noqa: E402
from src.evaluation import scoring  # noqa: E402
from src.reporting import manifest, submission, summary  # noqa: E402

RULE = "=" * 78
DEFAULT_OUT = os.path.join(HERE, "out")


def print_checks(rows, quiet=False):
    if quiet or not rows:
        return
    print(RULE)
    print("integrity of the input")
    print("{:<34} {:>10}  {}".format("check", "status", "what was seen"))
    for row in rows:
        print("{:<34} {:>10}  {}".format(row["name"][:34], row["status"],
                                         summary.clip(row["observed"], 30)))
    n_fail = sum(1 for row in rows if row["status"] == "fail")
    n_warn = sum(1 for row in rows if row["status"] == "warn")
    print("\n{} checks, {} failed, {} warned".format(len(rows), n_fail, n_warn))


def print_model(run, quiet=False):
    if quiet:
        return
    report, cfg = run.report, run.cfg
    print(RULE)
    print("R3P {}  config {}".format(cfg["meta"]["model_version"],
                                     settings.config_digest()))
    print("  frames read    {:>9}   graded {:>9}   ungraded {:>9}".format(
        report["n_frames"], report["graded_frames"],
        report["n_frames"] - report["graded_frames"]))
    print("  labels         " + "  ".join(
        "{} {:>8}".format(name, count) for name, count in run.counts().items()))
    spells = report["spells"]
    print("  possessions    {:>9}   mean {:>6} s   median {:>6} s   longest {:>7} s".format(
        spells["n_possessions"], _fmt(spells["mean_seconds"]), _fmt(spells["median_seconds"]),
        _fmt(spells["longest_seconds"])))
    print("  dead balls     {:>9}   transitions {:>6}".format(
        spells.get("n_dead_balls", 0), report["n_transitions"]))
    print("  under 1 s      {:>9}   under 2 s  {:>6}".format(
        spells["under_1s"], spells["under_2s"]))
    print("  temperature    {:>9}   solved on unlabelled frames, median confidence {}".format(
        _fmt(report["temperature"]), cfg["calibration"]["target_median_confidence"]))
    prior = report["duration_prior"]["median_run_frames"]
    print("  run-length     " + "  ".join(
        "{} {} frames ({} s)".format(name, frames, cfg["decoder"]["duration"][name]["median_s"])
        for name, frames in prior.items()))
    print("  home attacks   {:>9}   geometric-ball frames {:>8}".format(
        "left->right" if report["direction_home"] > 0 else "right->left",
        report["geometric_frames"]))
    if report["abstained_inside_graded"]:
        print("  abstained inside graded frames: {}".format(report["abstained_inside_graded"]))


def _fmt(value, digits=2):
    return "-" if value is None else "{:.{}f}".format(value, digits)


def baseline_replay(data_dir, source, out_dir, quiet=False):
    """Recompute the organizers' nearest-player baseline and prove it is their published file.

    The check is the label column, with byte equality reported separately. Byte equality is the
    strongest statement available without a ground truth -- it shows the loader, the frame alignment
    and the team mapping are all read the way the organizers read them. Labels are compared first
    because the official writer opens its file in text mode, so on Windows its LF becomes CRLF and a
    byte mismatch there can be a newline convention rather than a bug. The two claims are worth
    keeping apart.
    """
    produced = os.path.join(out_dir, "baseline_replay_smooth50.csv")
    published = os.path.join(official.EVALUATION, "results", "baseline_smooth50.csv")

    codes = official.official_baseline.build(data_dir=data_dir)
    labels = official.official_baseline.smooth(codes, 50)
    official.official_baseline.write_submission(produced, labels)

    if not os.path.isfile(published):
        if not quiet:
            print("baseline replay: wrote {}, but the published file is missing -- nothing to "
                  "compare against.".format(os.path.relpath(produced, HERE)))
        return None

    # Both sides go through the organizers' own reader, so a writer difference cannot be mistaken
    # for a data difference.
    theirs = official.pm.label_codes(submission.read_submission(published))
    ours = official.pm.label_codes(submission.read_submission(produced))
    same_labels = np.array_equal(theirs, ours)
    same_bytes = submission.sha256_file(published) == submission.sha256_file(produced)
    if not quiet:
        counts = {name: int((ours == index).sum()) for index, name in enumerate(official.pm.LABELS)}
        print(RULE)
        print("baseline replay -- starter/baseline.py at window 50, re-run in this repository")
        print("  labels identical to published: {}".format(same_labels))
        print("  bytes identical to published:  {}{}".format(
            same_bytes, "" if same_bytes else "  ({} vs {} line endings; judge the labels)".format(
                _newline_of(published), _newline_of(produced))))
        print("  replayed counts: " + "  ".join("{:>1} {:>7}".format(k, v)
                                                for k, v in sorted(counts.items())))
    if not same_labels:
        print("baseline replay: MISMATCH -- something here reads the tracking differently than the "
              "organizers do. Investigate before trusting any other number.")
    return same_labels


def _newline_of(path):
    with open(path, "rb") as handle:
        head = handle.read(4096)
    return "CRLF" if b"\r\n" in head else "LF"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data",
                        help="where the unpacked match bundle lives (default: data)")
    parser.add_argument("--team-name", default=None,
                        help="registered team name; becomes the CSV file name (default: config)")
    parser.add_argument("--source", default=official.DEFAULT_SOURCE, choices=sorted(official.SOURCES),
                        help="which recording to read (default: tactical, the one submissions index to)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where artifacts go (default: out)")
    parser.add_argument("--config", default=None, help="alternate r3p.toml")
    parser.add_argument("--public-check", action="store_true",
                        help="score against the public five-minute sample (pipeline verification only)")
    parser.add_argument("--selfcheck", metavar="GT_CSV", default=None,
                        help="score against a synthetic latent-state CSV (development fixtures only)")
    parser.add_argument("--reproduce-baseline", action="store_true",
                        help="recompute the official baseline and compare it with the published file")
    parser.add_argument("--strict", action="store_true",
                        help="stop if an integrity check fails instead of carrying on and recording it")
    parser.add_argument("--skip-checks", action="store_true",
                        help="do not gate on integrity (dashboard fast mode); recorded in the manifest")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    cfg = settings.resolve(settings.load_config(args.config), data_dir=data_dir,
                           team_name=args.team_name, source=args.source)
    team_name = cfg["run"]["team_name"]
    if not team_name or team_name == "REPLACE_WITH_REGISTERED_TEAM_NAME":
        print("A team name is required: --team-name \"...\" (it must be the registered name, and it "
              "is also the CSV file name). Config placeholder is not an answer.")
        return 2

    os.makedirs(args.out, exist_ok=True)
    if not quiet_started(args.quiet):
        print(RULE)
        print("R3P -- Reliability-weighted, Restart-aware, Run-length-regularized Possession Model")
        print("team {!r}   source {}   data {}".format(team_name, args.source, data_dir))

    def progress(done, total, message):
        if not args.quiet:
            print("  [{:>6}/{}]  {}".format(done, total, message))

    try:
        run = pipeline.execute(data_dir=data_dir, source=args.source, cfg=cfg, progress=progress,
                               skip_checks=args.skip_checks)
    except IOError as error:
        print("\n{}\n{}".format(RULE, error))
        return 2

    print_checks(run.checks(), args.quiet)
    if args.strict and run.failures():
        print("\nStopping: {} integrity check(s) failed. Re-run without --strict to carry on and "
              "record them.".format(len(run.failures())))
        for row in run.failures():
            print("  {}  {}".format(row["name"], row["note"] or row["observed"]))
        return 3

    print_model(run, args.quiet)

    if run.report["graded_frames"] == 0:
        print(RULE)
        print("STOPPED: none of the {} frames read from {} fall inside the blocks the organizers\n"
              "score ({}). Every frame would be an abstention, so the output would be all X -- and\n"
              "an all-X file passes the format validator, which is exactly why it must not be\n"
              "written quietly. A short development fixture needs the full-length synthetic bundle;\n"
              "the restricted match needs the match data.".format(
                  run.n_frames, data_dir, cfg["facts"]["halves"]))
        print("\nFor a code-correctness run, use a bundle that spans both halves and re-run.")
        return 6

    if args.reproduce_baseline:
        baseline_replay(data_dir, args.source, args.out, args.quiet)

    written = run.to_csv(args.out)
    validator = run.validate(n_expected=run.n_frames)
    if not args.quiet:
        print(RULE)
        print("submission  {}".format(os.path.relpath(written["path"], HERE)))
        print("  rows {:>9}   sha256 {}".format(written["rows"], validator["sha256"]))
        print("  official validator: {}  ({})".format(
            "PASS" if validator["valid"] else "FAIL", validator["reason"]))
        print("  re-reads as the labels that were written: {}".format(validator["round_trip"]))

    selfcheck = None
    if args.selfcheck:
        if not run.bundle.development_bundle:
            print("\n--selfcheck was given a ground-truth path but this is not a development fixture. "
                  "Refusing to score anything else that way.")
            return 5
        selfcheck = scoring.score(run.labels, args.selfcheck, fps=run.bundle.fps,
                                  n_frames=run.n_frames)
        if not args.quiet:
            print(RULE)
            print("SELF-CHECK on a synthetic bundle -- not a competition result")
            print("  ground truth  {}".format(selfcheck["gt_path"]))
            print("  spell_f1_mean {:.4f}   macro_f1 {:.4f}   accuracy {:.4f}   alive/dead {:.4f}".format(
                selfcheck["spell_f1_mean"], selfcheck["macro_f1"], selfcheck["accuracy"],
                selfcheck["alive_dead_f1"]))
            print("  spells  ground truth {:>6}   predicted {:>6}   error {:>7}   fragmentation {:.3f}"
                  .format(selfcheck["n_gt_spells"], selfcheck["n_pred_spells"],
                          selfcheck["spell_count_error"], selfcheck["fragmentation"]))
            print("  mean spell  ground truth {:.2f} s   predicted {:.2f} s".format(
                selfcheck["gt_mean_spell_duration_s"] or float("nan"),
                selfcheck["mean_spell_duration_s"] or float("nan")))
            print("  the latent state that generated this bundle is not a human annotation of a real "
                  "match, so these numbers describe code correctness only.")

    if args.public_check:
        result = run.score()
        if not args.quiet:
            print(RULE)
            print("PUBLIC WINDOW -- pipeline verification, never tuning")
            print("  {}".format(result["gt_note"]))
            print("  rows scored {:>9} of {}   spell_f1_mean {:.4f}   macro_f1 {:.4f}".format(
                result["n_graded"], result["n_frames"], result["spell_f1_mean"], result["macro_f1"]))
            print("  published floor for reference: run  python evaluation/run_evaluation.py")
            print("  Five minutes of a 75-minute match is not a result, and R3P was not fitted to "
                  "it. Do not put this number on a leaderboard.")

    data, path = run.manifest(args.out, validator=validator, selfcheck=selfcheck)
    if not args.quiet:
        print(RULE)
        print("wrote")
        for key in ("manifest", "transitions", "uncertainty"):
            print("  {}".format(os.path.relpath(data["artifacts"][key], HERE)))
        print("\nNext: python evaluation/validate_submission.py {}  -- the organizers' own check, "
              "run independently.".format(os.path.relpath(written["path"], HERE)))

    if run.failures():
        print("\n{} integrity check(s) FAILED -- see the table above and the manifest.".format(
            len(run.failures())))
    if not validator["valid"]:
        return 4
    return 0


def quiet_started(quiet):
    return quiet


if __name__ == "__main__":
    sys.exit(main())
