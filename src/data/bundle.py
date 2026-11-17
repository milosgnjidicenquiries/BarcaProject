"""One object holding everything a run reads: tracking, tagging, homography, and the checks."""

from __future__ import annotations

import os

import numpy as np

from .. import official
from . import atd, homography as homography_module, integrity, tagging


class Bundle:
    def __init__(self, tracking, tagging_features, homography_quality, checks, paths,
                 data_dir, source, development_bundle):
        self.tracking = tracking
        self.tagging = tagging_features
        self.homography = homography_quality
        self.checks = checks
        self.paths = paths
        self.data_dir = data_dir
        self.source = source
        self.development_bundle = development_bundle

    @property
    def n_frames(self):
        return len(self.tracking)

    @property
    def fps(self):
        return self.tracking.fps

    @property
    def pitch(self):
        return self.tracking.pitch

    def graded_mask(self, game=official.GAME):
        """Frames the scorer can see. Outside it, an answer is ignored, so R3P abstains."""
        mask = np.zeros(self.n_frames, dtype=bool)
        for _, first, last in official.half_video_blocks(game):
            mask[max(0, int(first)):min(self.n_frames, int(last) + 1)] = True
        return mask

    def half_of(self, game=official.GAME):
        """0 outside play, 1 or 2 inside. Decoding is per half, so the model needs this per frame."""
        out = np.zeros(self.n_frames, dtype=np.int8)
        for half, first, last in official.half_video_blocks(game):
            out[max(0, int(first)):min(self.n_frames, int(last) + 1)] = half
        return out

    def integrity_summary(self):
        return {
            "n_checks": len(self.checks),
            "n_fail": sum(1 for row in self.checks if row["status"] == "fail"),
            "n_warn": sum(1 for row in self.checks if row["status"] == "warn"),
            "development_bundle": self.development_bundle,
        }


def load_bundle(data_dir=None, cfg=None, source="tactical", with_tagging=True,
                with_homography=True, progress=None, game=official.GAME):
    """Read one recording's assets. Raises IOError naming every missing file if the bundle is not
    there -- which, for the gated match data, is the normal state of a fresh clone (data/README.md).
    """
    paths = official.asset_paths(game, data_dir, source)
    present = {name: os.path.isfile(path) for name, path in paths.items()}
    needed = ["atd_meta", "atd_raw"]
    if with_tagging and source == "tactical":
        needed.append("smart_tagging")
    if with_homography:
        needed.append("homography")
    missing = [paths[name] for name in needed if not present.get(name)]
    if missing:
        raise IOError(
            "match data not found:\n  {}\n\nThe data is gated -- see DATA.md. Unpack the bundle "
            "into\n  {}\nor point --data-dir / MTAH_DATA_DIR at wherever you put it."
            .format("\n  ".join(missing), data_dir or official.ROOT))

    def note(done, total, message):
        if progress:
            progress(done, total, message)

    note(0, 1, "reading tracking")
    tracking = atd.load_tracking(data_dir=data_dir, source=source, progress=progress)

    tagging_features = None
    if with_tagging and source == "tactical":
        note(0, 1, "reading smart tagging")
        names = official.team_names(tracking.paths["atd_meta"])
        tagging_features = tagging.load_tagging(
            data_dir=data_dir, cfg=cfg, n_frames=len(tracking), source=source,
            home_name=names[0], away_name=names[1])

    homography_quality = None
    if with_homography:
        note(0, 1, "reading homography")
        homography_quality = homography_module.load_homography(data_dir=data_dir, source=source,
                                                               n_frames=len(tracking))

    note(0, 1, "integrity checks")
    checks, _, _ = integrity.run_integrity(tracking, tagging_features, homography_quality,
                                           data_dir=data_dir, source=source, game=game)
    return Bundle(tracking, tagging_features, homography_quality, checks, paths, data_dir, source,
                  integrity.is_development_bundle(data_dir))
