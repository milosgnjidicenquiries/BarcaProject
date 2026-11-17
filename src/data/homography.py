"""Homography: quality columns only.

R3P needs no pixel<->pitch transform. The ATD already reports normalised pitch coordinates, so
every distance in the model is metric directly; projecting to the image would be needed only to
draw on the video. What IS worth having is the two reliability columns, because they measure the
camera's own confidence in the frame:

* ``valid``      -- does a transform exist at all (tactical: yes everywhere, per DATA.md);
* ``reprojection_error`` -- how far the fitted matrix misses. Median 0.24 during play, about 1.5
  across the half-time break, when the camera is not pointed at the pitch.

So the error is used as a weak observation-reliability term and as an independent check on the
half boundaries, which is precisely what ``integrity`` does with it. The reader is the official
one; the 3x3 matrices are loaded and dropped rather than held for 199,319 frames.
"""

from __future__ import annotations

import numpy as np

from .. import official


class HomographyQuality:
    def __init__(self, error, valid, n_frames, alignment):
        self.error = error                  # (T,) float64, NaN where invalid
        self.valid = valid                  # (T,) bool
        self.n_frames = n_frames
        self.alignment = alignment

    def __len__(self):
        return len(self.error)

    def summary(self):
        usable = self.error[self.valid] if self.valid.any() else np.array([np.nan])
        return {
            "rows": int(len(self.error)),
            "one_row_per_video_frame": bool(self.alignment["agrees"]),
            "valid_fraction": float(np.mean(self.valid)),
            "median_reprojection_error": float(np.nanmedian(usable)),
            "p99_reprojection_error": float(np.nanpercentile(usable, 99)),
        }

    def confidence(self, reference_error=None):
        """1 on clean frames, falling towards 0 as the camera loses the pitch.

        ``reference_error`` is the match's own median (a [derived] quantity computed on unlabelled
        data), so the curve adapts to the recording instead of to a constant we guessed.
        """
        error = np.where(self.valid, self.error, np.nan)
        if reference_error is None:
            finite = error[np.isfinite(error)]
            reference_error = float(np.median(finite)) if finite.size else 1.0
        reference_error = max(reference_error, 1e-3)
        # A frame fitting 4x worse than the match median is a quarter of the trust; 100x worse is
        # effectively none. Squared ratio, clipped, so the tail cannot dominate.
        ratio = np.nan_to_num(error / reference_error, nan=100.0)
        return np.clip(1.0 / (1.0 + (ratio / 4.0) ** 2), 0.0, 1.0).astype(np.float32)


def load_homography(data_dir=None, source="tactical", n_frames=None, keep_matrices=False):
    matrices, error, valid = official.official_homography.load(source=source,
                                                               data_dir=data_dir)
    alignment = official.official_homography.check_alignment(source=source, data_dir=data_dir)
    if not keep_matrices:
        del matrices
    n_frames = n_frames or official.source_frames(source)
    # A row-count mismatch is reported, not raised: the dashboard has to be able to open a bundle
    # that is broken and say which part, and `integrity` turns this into a hard fail row.
    alignment["expected_rows"] = int(n_frames)
    alignment["row_count_matches"] = bool(len(error) == n_frames)
    return HomographyQuality(error.astype(np.float64), valid.astype(bool), n_frames, alignment)
