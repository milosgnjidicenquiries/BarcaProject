"""R3P as one callable: reliability-weighted fusion, split dead-ball head, duration-aware decoding.

    labels, report = run(bundle, cfg)

Three things this file is responsible for, and nothing else:

* ordering the stages so each one's output is the next one's input, with no stage reaching into a
  file or a label;
* the abstention policy -- where X may appear and where it may not;
* the run report the dashboard and the manifest are built from, so the two cannot disagree about
  what happened.

The modelling lives in evidence.py (components A and B) and decoder.py (component C). The descriptive
counting lives in reporting/summary.py. Scoring lives in the organizers' evaluation/ directory, which
this only calls.
"""

from __future__ import annotations

import numpy as np

from ..reporting import summary
from . import decoder, evidence


def _abstain(labels, separation, graded, cfg):
    """The optional X policy, applied only inside the graded frames.

    Off by default, and the reason is in the scorer rather than in taste: a predicted X on a graded
    frame is a guaranteed macro-F1 miss -- it lands in the false-negative column of every scored
    class -- while buying nothing on spellF1 unless the frame really sits mid-possession. The gap
    threshold exists so that switching it on cannot produce single-frame islands of abstention, which
    the scorer's forward-fill would erase anyway.
    """
    abstain = cfg["decoder"]["abstain"]
    if not bool(abstain["enabled"]):
        return labels, 0

    below = float(abstain["below_prob"])
    minimum_gap = int(abstain["min_gap_frames"])
    want = graded & (separation < below)

    out, flipped = labels.copy(), 0
    index = 0
    while index < len(want):
        if not want[index]:
            index += 1
            continue
        stop = index
        while stop < len(want) and want[stop]:
            stop += 1
        if stop - index >= minimum_gap:
            out[index:stop] = decoder.X_LABEL
            flipped += stop - index
        index = stop
    return out, flipped


def run(bundle, cfg, progress=None):
    """Full inference over an already-loaded bundle. Returns (int8 labels, report dict)."""
    tracking = bundle.tracking
    n_frames = len(tracking)
    graded = bundle.graded_mask()
    halves = bundle.half_of()
    fps = cfg["facts"]["fps"]

    if progress:
        progress(0, n_frames, "ball channel")
    # Trimmed to the tracking length rather than trusted to equal it: the integrity check reports a
    # mismatch loudly, and inference only has to survive one.
    homography_confidence = (bundle.homography.confidence()[:n_frames]
                             if bundle.homography is not None else None)
    ball = evidence.ball_channel(tracking, cfg, graded, homography_confidence)

    if progress:
        progress(0, n_frames, "evidence")
    features = evidence.evidence(tracking, cfg, graded, ball, bundle.tagging, homography_confidence)
    fused = evidence.fuse(cfg, features, graded)

    if progress:
        progress(0, n_frames, "decoding")
    rmod = evidence.restart_modulation(bundle.tagging, n_frames, cfg)
    blocks = summary.graded_blocks(graded)
    decoded = decoder.decode(fused["log_emission"], cfg, blocks, rmod)

    labels, abstained = _abstain(decoded["labels"], decoded["separation"], graded, cfg)
    q, p_dead = fused["q"], fused["p_dead"]

    return labels, {
        "n_frames": n_frames,
        "graded_frames": int(graded.sum()),
        "abstained_inside_graded": abstained,
        "outside_graded_are_X": bool(np.all(labels[~graded] == decoder.X_LABEL)),
        "temperature": fused["temperature"],
        "q_median": float(np.median(q[graded])) if graded.any() else None,
        "p_dead_median": float(np.median(p_dead[graded])) if graded.any() else None,
        "reliability_median": float(np.median(features["rho"][graded])) if graded.any() else None,
        "separation_median": float(np.median(decoded["separation"][graded]))
                              if graded.any() else None,
        "duration_prior": decoded["prior"],
        "decoder_blocks": decoded["blocks"],
        "labels": labels,
        "spells": summary.spell_summary(labels, graded, fps),
        "label_table": summary.label_table(labels, graded, halves),
        "n_transitions": len(summary.transition_table(labels, graded, fps)),
        "transitions": summary.transition_table(labels, graded, fps),
        "channel_usage": summary.channel_usage(features, graded, cfg, q, p_dead),
        "uncertainty": summary.uncertainty_table(labels, decoded["separation"], graded),
        "graded_blocks": blocks,
        "direction_home": features["meta"]["direction_home"],
        "geometric_frames": features["meta"]["geometric_frames"],
        "standardisation": fused["standardisation"],
        "features": features,
        "fused": fused,
        "decoded": decoded,
        "ball": ball,
        "arrays": {"q": q, "p_dead": p_dead, "delta": fused["delta"],
                   "separation": decoded["separation"], "rho": features["rho"],
                   "nearest_team": features["nearest_team"], "margin": features["margin"],
                   "ball_rho": ball["rho"], "graded": graded, "halves": halves},
        "bundle": bundle.integrity_summary(),
    }
