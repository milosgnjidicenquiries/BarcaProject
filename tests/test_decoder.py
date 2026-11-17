"""Component C: the exact argmax over emission + transition + duration, decoded per half.

The specification's objective is

    argmax  sum_t log_emission_t(s_t) + sum_transitions log A + sum_runs log p(duration)

and these tests check three things about it that a smoke test would miss: that the returned path
really is the optimum of that expression (not of a smoothed version of it), that the run-length term
is what lets a three-frame blip lose and a six-second counter-attack win, and that the two halves are
separate lattices so the 20156-frame halftime hole can never manufacture a transition.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests import helpers as h

from src.possession import decoder


def make(spans, confidence=2.0, n_frames=None):
    """log-emissions from [(count, state)]: `confidence` for that state, -confidence for the rest."""
    total = n_frames or sum(count for count, _ in spans)
    log_e = np.full((total, 4), -float(confidence))
    log_e[:, 3] = -20.0                      # X is never a decoded state
    at = 0
    for count, state in spans:
        log_e[at:at + count, state] = float(confidence)
        at += count
    return log_e


def duration_logprob(inc, state, length):
    """log p(d = length) implied by the increment table, tail included."""
    table = np.cumsum(inc[state])
    cap = len(table)
    if length <= cap:
        return float(table[length - 1])
    return float(table[cap - 1] + (length - cap) * inc[state, cap - 1])


def path_score(labels, log_e, inc, log_embedded):
    """The objective the decoder claims to maximise, recomputed here from scratch."""
    labels = np.asarray(labels)
    total = float(log_e[np.arange(len(labels)), labels].sum())
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = [0] + [int(e) for e in edges] + [len(labels)]
    for index in range(len(bounds) - 1):
        start, stop = bounds[index], bounds[index + 1]
        total += duration_logprob(inc, int(labels[start]), stop - start)
        if index + 1 < len(bounds) - 1:
            total += float(log_embedded[int(labels[start]), int(labels[stop])])
    return total


class BlipTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()
        self.inc, self.log_embedded, _ = decoder.build_matrices(self.cfg)

    def decode(self, log_e, rmod=None):
        n = log_e.shape[0]
        return decoder.decode(log_e, self.cfg, [(0, n)], rmod)

    def test_a_three_frame_blip_does_not_change_the_label(self):
        log_e = make([(200, 0), (3, 1), (200, 0)], confidence=2.5)
        result = self.decode(log_e)
        self.assertTrue(np.all(result["labels"] == decoder.H))
        self.assertEqual(result["blocks"][0]["n_runs"], 1)

    def test_a_six_second_counter_attack_does_change_it(self):
        """Both properties at once is the point: soft cost, not a hard minimum length."""
        log_e = make([(200, 0), (150, 1), (200, 0)], confidence=2.5)
        labels = self.decode(log_e)["labels"]
        self.assertTrue(np.all(labels[260:290] == decoder.A))
        self.assertTrue(np.all(labels[:190] == decoder.H))
        self.assertTrue(np.all(labels[360:] == decoder.H))

    def test_the_short_spell_is_priced_by_the_duration_prior_not_by_a_window(self):
        """3 frames of A must lose at this confidence, and win when the emission is far stronger.

        A smoothing window cannot do that -- it is amplitude-blind. The run-length prior is what makes
        the trade-off explicit, and the test pins both sides of it.
        """
        weak = self.decode(make([(200, 0), (3, 1), (200, 0)], confidence=2.5))["labels"]
        self.assertEqual(set(np.unique(weak).tolist()), {decoder.H})

        log_e = make([(200, 0), (3, 1), (200, 0)], confidence=2.5)
        log_e[200:203, 0] = -12.0            # an unmistakable, if brief, change of possession
        strong = self.decode(log_e)["labels"]
        self.assertIn(decoder.A, np.unique(strong).tolist())

    def test_a_cap_length_hold_still_happens(self):
        """The geometric tail: longer than max_duration_frames must stay possible, not forbidden."""
        n = 2000
        labels = self.decode(make([(n, 0)]))["labels"]
        self.assertTrue(np.all(labels == decoder.H))
        cap = int(self.cfg["decoder"]["max_duration_frames"])
        self.assertGreater(n, cap)


class ObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()
        self.inc, self.log_embedded, _ = decoder.build_matrices(self.cfg)

    def test_the_decoded_path_beats_the_framewise_argmax_and_hand_built_alternatives(self):
        rng = np.random.default_rng(20271117)
        n = 900
        truth = np.concatenate([np.zeros(300, dtype=int), np.full(120, 2), np.full(240, 1),
                                np.full(240, 0, dtype=int)])
        log_e = np.full((n, 4), -1.0)
        log_e[np.arange(n), truth] = 1.0
        log_e[:, :3] += rng.normal(0.0, 0.6, size=(n, 3))

        result = decoder.decode(log_e, self.cfg, [(0, n)])
        decoded = result["labels"]
        best = path_score(decoded, log_e, self.inc, self.log_embedded)
        greedy = np.argmax(log_e[:, :3], axis=1).astype(np.int8)
        self.assertGreaterEqual(best, path_score(greedy, log_e, self.inc, self.log_embedded))

        # Perturbations of the answer: one frame re-labelled, and each boundary slid by 30 frames.
        for frame in (0, 149, 300, 420, 660, 899):
            flipped = decoded.copy()
            flipped[frame] = (int(flipped[frame]) + 1) % 3
            self.assertGreaterEqual(best, path_score(flipped, log_e, self.inc, self.log_embedded),
                                    "a single-frame change at {} scored better".format(frame))
        for boundary in (300, 420, 660):
            for shift in (-30, 30):
                moved = decoded.copy()
                low, high = sorted((boundary, boundary + shift))
                moved[low:high] = decoder.D
                self.assertGreaterEqual(
                    best, path_score(moved, log_e, self.inc, self.log_embedded),
                    "sliding the boundary at {} by {} scored better".format(boundary, shift))

    def test_the_prior_is_discretised_in_frames_and_right_skewed(self):
        fps = self.cfg["facts"]["fps"]
        table, log_ratio = decoder.discretised_log_lognormal(
            6.0, 0.90, fps, 900, 0.99)
        pmf = np.exp(table[:899])
        self.assertAlmostEqual(float(pmf.sum() + np.exp(table[899]) / (1.0 - 0.99)), 1.0, places=6)

        # A run of L frames occupies the bin L +/- 0.5, so the MEDIAN of the discretised table has to
        # land on median_s * fps. Evaluating the density at L instead of the bin would put the median
        # somewhere else entirely, and mixing seconds into a frame index shifts it by log(fps).
        cdf = np.cumsum(pmf) / float(pmf.sum())
        median_frames = int(np.searchsorted(cdf, 0.5)) + 1
        self.assertAlmostEqual(median_frames, 6.0 * fps, delta=10)

        # Log-normal, so the mode sits well below the median: most possessions are short, the average
        # is pulled up by the long ones. A Gaussian prior here would have no left tail at all.
        mode_frames = int(np.argmax(pmf)) + 1
        self.assertLess(mode_frames, median_frames)
        self.assertGreater(mode_frames, 1)
        self.assertLess(log_ratio, 0.0)


class HalfSeparationTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()

    def test_the_halftime_hole_cannot_join_two_runs_or_produce_a_phantom_one(self):
        n = 1000
        blocks = [(0, 400), (600, n)]
        log_e = make([(400, 0), (200, 2), (400, 1)])     # the hole is filled with D by construction
        log_e[400:600] = -1.0
        result = decoder.decode(log_e, self.cfg, blocks)
        labels = result["labels"]

        self.assertTrue(np.all(labels[400:600] == decoder.X_LABEL))
        self.assertFalse(result["graded"][400:600].any())
        self.assertTrue(np.all(labels[:400] != decoder.X_LABEL))
        self.assertTrue(np.all(labels[600:] != decoder.X_LABEL))
        # Two lattices, so the "runs" the report counts are per half: one H spell and one A spell,
        # never a single H...A run that had to cross 200 ungraded frames.
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual([b["n_runs"] for b in result["blocks"]], [1, 1])
        self.assertEqual(int((np.diff(labels[labels != decoder.X_LABEL]) != 0).sum()), 1)

    def test_a_short_block_is_not_decoded_at_all(self):
        """One frame cannot carry a run-length model, and pretending otherwise hides a bad export."""
        result = decoder.decode(make([(100, 0), (1, 1), (100, 0)]), self.cfg, [(0, 100), (101, 102)])
        self.assertTrue(np.all(result["labels"][100:] == decoder.X_LABEL))

    def test_x_is_never_a_decided_label_inside_a_half(self):
        n = 300
        log_e = np.zeros((n, 4)) - 1.0                    # perfectly ambiguous
        result = decoder.decode(log_e, self.cfg, [(50, 250)])
        self.assertEqual(set(np.unique(result["labels"][:50]).tolist()), {decoder.X_LABEL})
        self.assertNotIn(decoder.X_LABEL, np.unique(result["labels"][50:250]).tolist())


class RestartAwareTransitionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()
        self.inc, self.log_embedded, _ = decoder.build_matrices(self.cfg)

    def test_a_restart_anchor_can_choose_which_team_the_ball_goes_to(self):
        """Same emissions, different tagger credit -> different label after the stoppage.

        The modulation is a one-off bonus on *entering* from a dead-ball spell, so the fixture gives
        away only a fifth of a log-unit for five frames: enough to decide the entry without the tag,
        and comfortably outweighed by the tag once it is there. A long, deep deficit would rightly
        beat a short tagger's opinion -- which is the bound the specification asks for.
        """
        n = 180
        log_e = make([(120, 2), (60, 1)], confidence=2.0)
        log_e[120:, decoder.H] = 2.0
        log_e[120:, decoder.A] = 2.0
        log_e[120:125, decoder.A] = 2.2        # the untagged evidence edges away, then ties

        plain = decoder.decode(log_e, self.cfg, [(0, n)])["labels"]
        self.assertTrue(np.all(plain[130:180] == decoder.A))

        strength = float(self.cfg["decoder"]["restart"]["strength"])
        rmod = np.zeros((n, 4))
        rmod[120:170, decoder.H] = strength
        rmod[120:170, decoder.A] = -0.5 * strength
        biased = decoder.decode(log_e, self.cfg, [(0, n)], rmod)["labels"]
        self.assertTrue(np.all(biased[125:165] == decoder.H),
                        "restart modulation did not reach the decoder")

    def test_modulation_only_touches_entries_from_a_stoppage(self):
        """Mid-play H<->A turnovers are the tagger's weakest moment, so it must stay silent there."""
        strength = float(self.cfg["decoder"]["restart"]["strength"])
        n = 180
        log_e = make([(120, 0), (60, 1)], confidence=2.0)
        rmod = np.zeros((n, 4))
        rmod[120:170, decoder.H] = strength               # "away restarts", but there was no D
        labels = decoder.decode(log_e, self.cfg, [(0, n)], rmod)["labels"]
        self.assertTrue(np.all(labels[140:] == decoder.A))

    def test_restart_modulation_is_zero_without_tagging(self):
        from src.possession import evidence
        self.assertTrue(np.all(evidence.restart_modulation(None, 60, self.cfg) == 0.0))


class DeterminismTest(unittest.TestCase):
    def test_the_same_emissions_decode_to_the_same_labels(self):
        cfg = h.config()
        rng = np.random.default_rng(7)
        log_e = rng.normal(size=(500, 3)).astype(np.float64)
        log_e = np.concatenate([log_e, np.full((500, 1), -20.0)], axis=1)
        first = decoder.decode(log_e, cfg, [(0, 500)])["labels"]
        second = decoder.decode(log_e, cfg, [(0, 500)])["labels"]
        self.assertTrue(np.array_equal(first, second))

    def test_the_matrices_are_built_once_and_read_only_off_the_config(self):
        cfg = h.config()
        a = decoder.build_matrices(cfg)
        b = decoder.build_matrices(cfg)
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertTrue(np.array_equal(a[1], b[1]))
        self.assertEqual(a[2]["median_run_frames"]["H"],
                         int(round(cfg["decoder"]["duration"]["H"]["median_s"] * cfg["facts"]["fps"])))
        # A spell that ends does not restart as itself: that is how this model says "same run".
        # NEG is a large finite number, not -inf, so adding a finite score can never make a NaN.
        self.assertTrue(np.all(np.diag(a[1]) < -1e29))
        self.assertTrue(np.all(np.isfinite(a[1][~np.eye(3, dtype=bool)])))
        # The increments are differences of a log-pmf, so they change sign; what must never be
        # positive is their partial sum, which is log p(duration = L).
        self.assertTrue(np.all(np.cumsum(a[0], axis=1) <= 0.0))


if __name__ == "__main__":
    unittest.main()
