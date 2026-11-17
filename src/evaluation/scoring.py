"""Scoring a submission against a ground truth, using only the organizers' own scorer.

This directory exists so that the numbers printed by the dashboard, the CLI and the tests are the same
numbers `evaluation/run_evaluation.py` would print: the functions below are thin calls into it. There
is no reimplementation of spellF1 or macroF1 anywhere in this repository, and that is a compliance
property, not a convenience -- a private copy of the metric is the easiest way in the world to report a
score that does not mean anything.

Two separate things happen here and must not be confused:

* **validation** asks whether the file is well-formed, which the official validator answers exactly;
* **scoring** asks whether the labels are right, which needs a ground truth the participant does not
  have. `gt_path` is therefore always explicit, and when it points at the public five-minute file the
  result is labelled as an alignment check, because that is all it is permitted to be.
"""

from __future__ import annotations

import os

import numpy as np

from .. import official
from ..possession import decoder
from ..reporting import summary

PUBLIC_GT = os.path.join(official.EVALUATION, "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS_DIR = os.path.join(official.EVALUATION, "results")
TAUS = official.pm.TAUS
BASELINE_FILES = ("baseline_raw.csv", "baseline_smooth25.csv",
                  "baseline_smooth50.csv", "baseline_smooth125.csv")


def load_labels(path):
    """(codes, n_frames) from any frame,label CSV, through the official reader."""
    table = official.pm.read_label_csv(path)
    return official.pm.label_codes(table), len(table)


def score(pred_codes, gt_path, fps=25, bootstrap_n=None, n_frames=None):
    """The official metrics, plus the name of the ground truth they were computed against.

    A score without a named GT is the shape of claim the rules forbid, so the path is part of the
    result rather than an argument that disappears.
    """
    gt_codes, gt_frames = load_labels(gt_path)
    if n_frames is not None and gt_frames != n_frames:
        raise ValueError("ground truth has {} rows, the run has {}".format(gt_frames, n_frames))
    result = official.pm.evaluate(gt_codes, pred_codes[:len(gt_codes)],
                                  taus=TAUS, bootstrap_n=bootstrap_n or official.pm.BOOTSTRAP_N)
    result["gt_path"] = os.path.relpath(gt_path, official.ROOT).replace(os.sep, "/")
    result["gt_rows"] = gt_frames
    result["public_window_alignment_check_only"] = os.path.abspath(gt_path) == os.path.abspath(PUBLIC_GT)
    if result["public_window_alignment_check_only"]:
        # Stated in the payload, not only in the manual, because this is the file a reviewer reads.
        result["gt_note"] = ("the five-minute public sample: usable for verifying that the pipeline "
                             "and the frame alignment work, never for training, tuning, model "
                             "selection or optimisation")
    return result


def label_distribution(pred_codes, gt_codes=None):
    """Per-class precision/recall/F1 and the confusion matrix, as rows for a table."""
    if gt_codes is None:
        raise ValueError("a distribution needs a ground truth")
    cm = official.pm.confusion(gt_codes, pred_codes)
    cm3 = official.pm.scored_confusion(cm)
    rows = []
    for index, name in enumerate(summary.LABEL_NAMES[:decoder.N_STATES]):
        precision, recall, f1 = official.pm.per_class_prf(cm3, index)
        rows.append({"label": name, "support": int(cm3[index, :].sum()),
                     "precision": precision, "recall": recall, "f1": f1})
    return {"per_class": rows, "macro_f1": official.pm.macro_f1(cm3),
            "accuracy": official.pm.accuracy(cm3), "confusion": cm.tolist()}


def validate_csv(path, n_frames=None):
    """Run the organizers' validator over the file and return (ok, message)."""
    if not os.path.isfile(path):
        return False, "no such file: {}".format(path)
    try:
        table = official.pm.read_label_csv(path)
    except Exception as error:
        return False, "could not be read: {}".format(error)
    reason = official.pm.validate_submission(table, n_frames if n_frames is not None else len(table))
    return reason is None, reason or "valid"


def baseline_reference(gt_path=None, taus=TAUS, bootstrap_n=0):
    """The published baselines, scored now by the shipped scorer.

    This is the floor R3P has to be compared against, and it is computed here rather than quoted from
    EVALUATION.md so that "we beat the baseline" can never become an unverified claim. It needs the
    ground truth; without one the function returns nothing rather than guessing.
    """
    gt_path = gt_path or PUBLIC_GT
    if not os.path.isfile(gt_path):
        return None
    gt_codes, _ = load_labels(gt_path)
    entries = []
    for name in BASELINE_FILES:
        path = os.path.join(RESULTS_DIR, name)
        if not os.path.isfile(path):
            continue
        codes, rows = load_labels(path)
        result = official.pm.evaluate(gt_codes, codes[:len(gt_codes)], taus=taus,
                                      bootstrap_n=bootstrap_n)
        entries.append({"name": os.path.splitext(name)[0], "rows": rows,
                        "spell_f1_mean": result["spell_f1_mean"],
                        "macro_f1": result["macro_f1"],
                        "accuracy": result["accuracy"],
                        "alive_dead_f1": result["alive_dead_f1"],
                        "n_possessions": result["n_gt_spells"],
                        "transition_f1_tight": result["transitions"][taus[0]]["f1"],
                        "transition_f1_loose": result["transitions"][taus[-1]]["f1"]})
    if entries:
        entries[0]["gt_path"] = os.path.relpath(gt_path, official.ROOT).replace(os.sep, "/")
    return entries


def compare_with_baselines(pred_codes, gt_path=None, bootstrap_n=0):
    """Our labels and each published baseline under one scorer, sorted by the ranking metric.

    The point of doing it in a single table is that a claim about beating the floor then has a visible
    denominator: same GT, same scorer, same tolerances, same bootstrap seed.
    """
    gt_codes, _ = load_labels(gt_path or PUBLIC_GT)
    ours = official.pm.evaluate(gt_codes, pred_codes[:len(gt_codes)], taus=TAUS,
                                bootstrap_n=bootstrap_n)
    reference = baseline_reference(gt_path, bootstrap_n=bootstrap_n) or []
    rows = [{"name": "R3P", "spell_f1_mean": ours["spell_f1_mean"],
             "macro_f1": ours["macro_f1"], "accuracy": ours["accuracy"],
             "alive_dead_f1": ours["alive_dead_f1"],
             "n_possessions": ours["n_gt_spells"],
             "transition_f1_tight": ours["transitions"][TAUS[0]]["f1"],
             "transition_f1_loose": ours["transitions"][TAUS[-1]]["f1"]}]
    rows.extend(reference)
    rows.sort(key=lambda row: (-row["spell_f1_mean"], -row["macro_f1"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"rows": rows, "headline": ours,
            "gt_path": os.path.relpath(gt_path or PUBLIC_GT, official.ROOT).replace(os.sep, "/")}
