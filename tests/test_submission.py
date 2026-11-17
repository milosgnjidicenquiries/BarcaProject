"""The submission file: the only artifact the organizers actually read.

SUBMISSION.md is the contract -- header `frame,label`, one row per video frame from 0, labels drawn
from H/A/D/X, no blanks, no index column. Everything here goes through the *official* reader and the
*official* validator (`evaluation/possession_metrics.py`), unmodified, because a writer and a reader
written by the same author can agree with each other and be wrong together.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from tests import helpers as h

from src import official
from src.evaluation import scoring
from src.reporting import submission

N = 200


def write(codes, directory, name="out.csv"):
    path = os.path.join(directory, name)
    return submission.write_submission(codes, path), path


class WriterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = h.TempDir()
        self.dir = self.tmp.new()

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_through_the_official_reader(self):
        codes = (np.arange(N) % 4).astype(np.int8)
        _, path = write(codes, self.dir)
        ok, reason = submission.verify_round_trip(codes, path)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "identical")

    def test_the_official_validator_passes_and_the_hash_matches_the_bytes(self):
        codes = np.zeros(N, dtype=np.int8)
        codes[10:20] = 1
        codes[50:70] = 2
        codes[150:] = 3
        info, path = write(codes, self.dir)
        ok, message = scoring.validate_csv(path, N)
        self.assertTrue(ok, message)
        self.assertEqual(info["sha256"], submission.sha256_file(path))
        self.assertEqual(info["rows"], N)
        self.assertEqual(info["label_counts"], {"H": N - 80, "A": 10, "D": 20, "X": 50})

    def test_newlines_are_lf_on_every_platform(self):
        """Byte-for-byte reproducibility is a stated requirement, and CRLF would break it."""
        _, path = write(np.zeros(N, dtype=np.int8), self.dir)
        with open(path, "rb") as handle:
            raw = handle.read()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.startswith(b"frame,label\n0,H\n"))
        self.assertEqual(raw.count(b"\n"), N + 1)

    def test_the_frame_column_is_the_video_frame_zero_based(self):
        _, path = write(np.zeros(N, dtype=np.int8), self.dir)
        table = submission.read_submission(path)
        self.assertEqual(list(table.columns), ["frame", "label"])
        self.assertTrue(np.array_equal(table["frame"].values, np.arange(N)))
        self.assertEqual(list(table["label"].head(3)), ["H", "H", "H"])

    def test_a_second_write_of_the_same_codes_is_byte_identical(self):
        codes = (np.arange(N) // 7 % 4).astype(np.int8)
        first, path_a = write(codes, self.dir, "a.csv")
        second, path_b = write(codes, self.dir, "b.csv")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(submission.sha256_file(path_a), submission.sha256_file(path_b))

    def test_it_refuses_to_write_frames_it_has_no_answer_for(self):
        """An unset -1 must not become "X": a bug cannot be allowed to look like an abstention."""
        codes = np.zeros(N, dtype=np.int8)
        codes[40] = -1
        with self.assertRaises(ValueError):
            write(codes, self.dir)
        with self.assertRaises(ValueError):
            write(np.zeros(0, dtype=np.int8), self.dir)

    def test_the_full_match_length_is_writeable_and_valid(self):
        """199319 rows is the real artifact size; the writer must survive it and validate."""
        codes = np.full(official.source_frames("tactical"), 3, dtype=np.int8)
        info, path = write(codes, self.dir, "full.csv")
        self.assertEqual(info["rows"], 199319)
        self.assertEqual(submission.codes_to_names(np.array([0, 1, 2, 3])).tolist(),
            list(official.pm.LABELS))
        ok, message = scoring.validate_csv(path, 199319)
        self.assertTrue(ok, message)


class RejectionTest(unittest.TestCase):
    """What the validator is for: these are the ways a plausible-looking file is still wrong."""

    def setUp(self):
        self.tmp = h.TempDir()
        self.dir = self.tmp.new()

    def tearDown(self):
        self.tmp.cleanup()

    def bad(self, body, name):
        path = os.path.join(self.dir, name)
        with open(path, "w", newline="\n", encoding="ascii") as handle:
            handle.write(body)
        return path

    def test_a_label_outside_hadwx_is_rejected(self):
        body = "frame,label\n" + "".join("{},{}\n".format(i, "H" if i else "P")
                                         for i in range(N))
        path = self.bad(body, "letters.csv")
        ok, message = scoring.validate_csv(path, N)
        self.assertFalse(ok)
        self.assertIn("labels outside", message)

    def test_a_blank_cell_is_rejected_rather_than_treated_as_x(self):
        body = "frame,label\n0,H\n1,\n2,A\n"
        ok, message = scoring.validate_csv(self.bad(body, "blank.csv"), 3)
        self.assertFalse(ok)
        self.assertIn("empty label", message)

    def test_a_short_file_is_rejected(self):
        codes = np.zeros(120, dtype=np.int8)
        _, path = write(codes, self.dir, "short.csv")
        ok, message = scoring.validate_csv(path, N)
        self.assertFalse(ok)
        self.assertIn("row count", message)

    def test_a_frame_column_that_skips_is_rejected(self):
        frames = np.arange(N)
        frames[80] = 79                                  # a duplicate, so no longer 0..n-1
        body = "frame,label\n" + "".join("{},H\n".format(f) for f in frames)
        ok, message = scoring.validate_csv(self.bad(body, "dupe.csv"), N)
        self.assertFalse(ok)
        self.assertIn("not 0..", message)

    def test_a_one_based_frame_column_is_rejected(self):
        """The ATD counts from 1 and the submission counts from 0; off-by-one must be an error."""
        body = "frame,label\n" + "".join("{},H\n".format(i + 1) for i in range(N))
        ok, message = scoring.validate_csv(self.bad(body, "one_based.csv"), N)
        self.assertFalse(ok)
        self.assertIn("not 0..", message)

    def test_an_extra_column_is_rejected_by_the_reader(self):
        body = "frame,label,confidence\n" + "".join("{},H,0.9\n".format(i) for i in range(N))
        path = self.bad(body, "extra.csv")
        table = official.pm.read_label_csv(path)          # reader is tolerant of extra columns...
        self.assertIn("confidence", table.columns)
        # ...but the pipeline never writes one, and the frame column alone still has to be right.
        self.assertEqual(submission.read_submission(path)["label"].nunique(), 1)

    def test_a_lowercase_file_is_accepted_because_the_validator_uppercases(self):
        """Not our contract to enforce, but worth knowing: the scorer is case-insensitive."""
        body = "frame,label\n" + "".join("{},h\n".format(i) for i in range(N))
        ok, message = scoring.validate_csv(self.bad(body, "lower.csv"), N)
        self.assertTrue(ok, message)
        codes = official.pm.label_codes(official.pm.read_label_csv(
            os.path.join(self.dir, "lower.csv")))
        self.assertTrue(np.all(codes == 0))


if __name__ == "__main__":
    unittest.main()
