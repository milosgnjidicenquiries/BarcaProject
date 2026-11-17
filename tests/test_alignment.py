"""Frame alignment, units and the two-half structure: the parts that are contracts, not choices.

These tests are the cheapest place to catch the worst class of bug. A one-frame shift moves every
transition the scorer measures, and a normalised coordinate read as a metre changes which player is
nearest -- neither shows up as an exception, both show up as a submission that is quietly wrong.
"""

from __future__ import annotations

import os
import shutil
import unittest

import numpy as np

from tests import helpers as h

from src import official
from src.data import atd, bundle as bundle_module, synthetic


def write_minimal_raw(path, frame_ids, ball_values):
    """A syntactically valid ATD with 46 slots parked at (0.5, 0.5) and a ball per caller.

    Hand-written rather than simulated, because the point is to know what every field must be.
    """
    filler = ";".join("0.500000,0.500000,0.000000" for _ in range(synthetic.N_SLOTS))
    with open(path, "w", newline="\n") as handle:
        for frame_id, (bx, by) in zip(frame_ids, ball_values):
            handle.write("{}:{}:{},{},1.0\n".format(frame_id, filler, bx, by))


class AlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = h.config()
        cls.temp = h.TempDir()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frame_id_minus_one_is_the_row(self):
        """video_frame = frame_id - 1, asserted against a file whose ids do not start at 1."""
        directory = self.temp.new()
        meta = os.path.join(directory, "meta.xml")
        raw = os.path.join(directory, "raw.txt")
        synthetic.write_atd_meta(meta, ("Barcelona", "Atletic Club"),
                                 np.zeros(synthetic.N_SLOTS, dtype=np.int8))
        # ids 10..14: rows 9..13 must be filled and rows 0..8 must be empty.
        write_minimal_raw(raw, range(10, 15), [(0.10, 0.20)] * 5)

        tracking = atd.load_tracking(meta_path=meta, raw_path=raw, n_frames=20)
        # 5 records with ids 10..14 fill rows 9..13. The loader trims to the last row it actually
        # saw rather than padding to the 20 asked for, so a short export is a short array.
        self.assertEqual(len(tracking), 14)
        self.assertTrue(np.all(np.isnan(tracking.ball_x[:9])),
                        "rows before the first frame_id were invented out of nothing")
        self.assertAlmostEqual(float(tracking.ball_x[9]), 0.10 * h.LENGTH, places=4)
        self.assertAlmostEqual(float(tracking.ball_y[9]), 0.20 * h.WIDTH, places=4)
        self.assertEqual(tracking.anomalies["frame_id_first"], 10)
        self.assertEqual(tracking.anomalies["n_frames"], 14)

    def test_normalised_coordinates_become_metres(self):
        """The pitch is not square, so the two axes are scaled by different numbers."""
        directory = self.temp.new()
        meta = os.path.join(directory, "meta.xml")
        raw = os.path.join(directory, "raw.txt")
        synthetic.write_atd_meta(meta, ("Barcelona", "Atletic Club"),
                                 np.zeros(synthetic.N_SLOTS, dtype=np.int8))
        write_minimal_raw(raw, [1, 2], [(0.0, 1.0), (0.5, 0.5)])

        tracking = atd.load_tracking(meta_path=meta, raw_path=raw, n_frames=2)
        self.assertAlmostEqual(float(tracking.ball_x[0]), 0.0, places=4)
        self.assertAlmostEqual(float(tracking.ball_y[0]), h.WIDTH, places=4)
        self.assertAlmostEqual(float(tracking.px[1, 0]), 0.5 * h.LENGTH, places=4)
        self.assertAlmostEqual(float(tracking.py[1, 0]), 0.5 * h.WIDTH, places=4)

    def test_loader_matches_the_official_alignment_check(self):
        """Round trip through the delivered formats: our arrays against the file and the official
        streaming check, at random frames rather than at the ones the writer happened to like."""
        directory, paths, sim = h.simulate_to_dir(n_frames=1200, seed=101)
        try:
            tracking = atd.load_tracking(data_dir=directory, n_frames=1200)
            report = official.official_atd.check_alignment_raw(data_dir=directory)
            self.assertEqual(report["n_records"], len(tracking))
            self.assertEqual(report["frame_id_range"], (1, len(tracking)))
            self.assertEqual(report["first_video_frame"], 0)
            self.assertTrue(report["contiguous"])

            # Independent re-parse of the same lines, straight from the text.
            with open(paths["atd_raw"]) as handle:
                lines = handle.read().splitlines()
            rng = np.random.default_rng(4)
            for row in rng.integers(0, len(tracking), size=40):
                parts = lines[int(row)].split(":")
                self.assertEqual(int(parts[0]), int(row) + 1)
                ball = parts[2].split(",")
                if ball[0][0] in "Nn":                      # the file's own missing marker
                    self.assertTrue(np.isnan(tracking.ball_x[int(row)]),
                                    "a NaN ball became a number")
                    self.assertTrue(np.isnan(tracking.ball_y[int(row)]))
                else:
                    self.assertAlmostEqual(float(tracking.ball_x[int(row)]),
                                           float(ball[0]) * h.LENGTH, places=3)
                    self.assertAlmostEqual(float(tracking.ball_y[int(row)]),
                                           float(ball[1]) * h.WIDTH, places=3)
                slot = parts[1].split(";")[7].split(",")
                if slot[0][0] not in "Nn":
                    self.assertAlmostEqual(float(tracking.px[int(row), 7]),
                                           float(slot[0]) * h.LENGTH, places=3)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_short_file_is_a_row_count_mismatch_not_nan_padding(self):
        """A truncated export must show up as fewer rows, not as 199k silent frames."""
        directory = self.temp.new()
        meta = os.path.join(directory, "meta.xml")
        raw = os.path.join(directory, "raw.txt")
        synthetic.write_atd_meta(meta, ("Barcelona", "Atletic Club"),
                                 np.zeros(synthetic.N_SLOTS, dtype=np.int8))
        write_minimal_raw(raw, range(1, 31), [(0.5, 0.5)] * 30)
        tracking = atd.load_tracking(meta_path=meta, raw_path=raw, n_frames=199319)
        self.assertEqual(len(tracking), 30)
        self.assertEqual(tracking.anomalies["frames_missing_from_tracking"], 0)


class HalfStructureTest(unittest.TestCase):
    def test_graded_mask_is_exactly_the_two_official_blocks(self):
        """11185..90651 and 110808..190197, nothing else, and 158857 frames in total."""
        n = official.N_VIDEO_FRAMES
        tracking = h.tracking(np.full((n, 1), np.nan), np.full((n, 1), np.nan),
                              np.full(n, np.nan), np.full(n, np.nan))
        loaded = bundle_module.Bundle(tracking, None, None, [], {}, None, "tactical", False)
        graded = loaded.graded_mask()
        self.assertEqual(int(graded.sum()), 79467 + 79390)
        self.assertFalse(graded[11184])
        self.assertTrue(graded[11185])
        self.assertTrue(graded[90651])
        self.assertFalse(graded[90652])
        self.assertFalse(graded[110807])
        self.assertTrue(graded[110808])
        self.assertEqual(int((~graded).sum()), n - 158857)

    def test_halftime_is_not_in_either_half(self):
        loaded = bundle_module.Bundle(
            h.tracking(np.full((official.N_VIDEO_FRAMES, 1), np.nan),
                       np.full((official.N_VIDEO_FRAMES, 1), np.nan),
                       np.zeros(official.N_VIDEO_FRAMES), np.zeros(official.N_VIDEO_FRAMES)),
            None, None, [], {}, None, "tactical", False)
        halves = loaded.half_of()
        self.assertEqual(halves[100000], 0)
        self.assertEqual(halves[50000], 1)
        self.assertEqual(halves[150000], 2)
        self.assertEqual(int((halves == 0).sum()), official.N_VIDEO_FRAMES - 158857)

    def test_prematch_and_post_match_frames_exist(self):
        """The graded window does not start at frame 0 and does not end at the last frame."""
        blocks = official.half_video_blocks(official.GAME)
        self.assertEqual(blocks[0][1], 11185)
        self.assertLess(blocks[1][2], official.N_VIDEO_FRAMES - 1)


if __name__ == "__main__":
    unittest.main()
