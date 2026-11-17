"""Component A: metric geometry, missing observations, reliability, and what tagging may say.

Each test here is a statement about football or about the file format, checked on arrays built by
hand so the answer is not a matter of taste. The recurring failure mode this file guards against is
a distance or a weight that is *plausible everywhere and wrong somewhere*: a normalised distance
picks the wrong player near the touchline, a dropout becomes a stoppage, an automatic tag becomes a
label.
"""

from __future__ import annotations

import shutil
import unittest

import numpy as np

from tests import helpers as h

from src.data import tagging as tagging_module
from src.possession import evidence

H, A, D = evidence.H, evidence.A, evidence.D


def run_components(tracking, cfg, graded, tagging_features=None, homography_confidence=None):
    ball = evidence.ball_channel(tracking, cfg, graded, homography_confidence)
    features = evidence.evidence(tracking, cfg, graded, ball, tagging_features,
                                 homography_confidence)
    return ball, features, evidence.fuse(cfg, features, graded)


class MetricDistanceTest(unittest.TestCase):
    """d = sqrt((105.3 dx)^2 + (68.0 dy)^2) -- and why the normalised version answers a different
    question."""

    def setUp(self):
        self.cfg = h.config()

    def test_anisotropy_is_not_optional(self):
        n = 4
        px, py = np.full((n, h.N_SLOTS), np.nan), np.full((n, h.N_SLOTS), np.nan)
        bx, by = 50.0, 30.0
        # Home 2.20 m away along the long axis, away 2.10 m along the short axis.
        px[0, 0], py[0, 0] = bx + 2.20, by
        px[0, 23], py[0, 23] = bx, by + 2.10
        for frame in range(n):
            px[frame, 0], py[frame, 0] = bx + 2.20, by
            px[frame, 23], py[frame, 23] = bx, by + 2.10
        tracking = h.tracking(px, py, np.full(n, bx), np.full(n, by),
                              ball_speed=np.full(n, 8.0))
        graded = np.ones(n, dtype=bool)
        _, features, fused = run_components(tracking, self.cfg, graded)

        # In normalised coordinates home is nearer (0.0209 < 0.0309). In metres it is not, and a
        # negative `nearest` is the away side winning -- the two must agree with each other.
        self.assertEqual(int(features["nearest_team"][1]), A,
                         "the nearest player was chosen in normalised space, not metric space")
        self.assertLess(float(features["blocks"]["nearest"][1]), 0.0)
        self.assertAlmostEqual(float(features["blocks"]["nearest"][1]),
                               np.tanh((2.10 - 2.20) / (2.0 * self.cfg["fusion"]["sigma_control_m"])),
                               places=5, msg="the metric distances did not reach the channel")

    def test_symmetry_flips_the_side_not_the_magnitude(self):
        n = 6
        bx, by = 50.0, 30.0
        px = np.full((n, h.N_SLOTS), np.nan)
        py = np.full((n, h.N_SLOTS), np.nan)
        for frame in range(n):
            near, far = (1.0, 8.0) if frame < n // 2 else (8.0, 1.0)
            px[frame, 0], py[frame, 0] = bx + near, by
            px[frame, 23], py[frame, 23] = bx - far, by
        tracking = h.tracking(px, py, np.full(n, bx), np.full(n, by), ball_speed=np.full(n, 6.0))
        graded = np.ones(n, dtype=bool)
        _, features, _ = run_components(tracking, self.cfg, graded)
        nearest = features["blocks"]["nearest"]
        self.assertGreater(nearest[0], 0.2)
        self.assertLess(nearest[-1], -0.2)
        self.assertAlmostEqual(abs(nearest[0]), abs(nearest[-1]), places=6,
                               msg="mirroring the scene changed the evidence strength, not its sign")

    def test_sigma_control_is_the_kernel_radius(self):
        """At exactly sigma_control_m the Gaussian is exp(-0.5): the kernel radius is the config
        number, not a value the code forgot to pass."""
        sigma = float(self_cfg_sigma())
        n = 3
        px, py = np.full((n, h.N_SLOTS), np.nan), np.full((n, h.N_SLOTS), np.nan)
        bx, by = 50.0, 30.0
        for frame in range(n):
            px[frame, 0], py[frame, 0] = bx + sigma, by          # home at exactly sigma
            px[frame, 23], py[frame, 23] = bx + 3.0 * sigma, by  # away, far
        tracking = h.tracking(px, py, np.full(n, bx), np.full(n, by), ball_speed=np.full(n, 6.0))
        _, features, _ = run_components(tracking, h.config(), np.ones(n, dtype=bool))
        self.assertGreater(features["blocks"]["nearest"][1], 0.0)
        self.assertAlmostEqual(float(features["margin"][1]), 2.0 * sigma, places=2,
                               msg="the nearest-distance margin is not the metric difference "
                                   "it claims to be")


def self_cfg_sigma():
    return h.config()["fusion"]["sigma_control_m"]


class MissingObservationTest(unittest.TestCase):
    def test_reliability_decays_then_stops(self):
        reliability = h.config()["fusion"]["reliability"]
        tau = float(reliability["ball_gap_tau_frames"])
        max_gap = int(reliability["ball_gap_max_frames"])
        n = max_gap + 20
        px = np.full((n, h.N_SLOTS), 50.0)
        py = np.full((n, h.N_SLOTS), 30.0)
        ball_x = np.full(n, np.nan)
        ball_y = np.full(n, np.nan)
        ball_x[0], ball_y[0] = 50.0, 30.0
        ball_x[n - 1], ball_y[n - 1] = 60.0, 34.0     # observed at both ends, so gaps bridge
        tracking = h.tracking(px, py, ball_x, ball_y)
        graded = np.ones(n, dtype=bool)
        ball = evidence.ball_channel(tracking, h.config(), graded, None)

        self.assertAlmostEqual(float(ball["rho"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(ball["rho"][5]), np.exp(-5.0 / tau), places=5)
        self.assertEqual(float(ball["rho"][max_gap + 5]), 0.0,
                         "extrapolation past ball_gap_max_frames was still trusted")
        self.assertTrue(np.all(ball["rho"][1:max_gap] <= 1.0))
        self.assertTrue(np.all(np.diff(ball["rho"][1:max_gap]) <= 1e-9),
                         "trust rose while the ball was further away in time")

    def test_a_ball_dropout_is_not_a_stationary_ball(self):
        """The regression this repository actually produced: averaging NaN as zero invented
        stoppages out of occlusion, on the frames where the ball is missing most."""
        cfg = h.config()
        window = int(cfg["deadball"]["static_window_frames"])
        n = 20 * window
        rng = np.random.default_rng(3)
        px = np.full((n, h.N_SLOTS), np.nan)
        py = np.full((n, h.N_SLOTS), np.nan)
        for frame in range(n):
            for index in range(11):                    # two shaped teams, a few metres apart
                px[frame, index] = 40.0 + 4.0 * (index % 4) + 0.5 * (frame % 3)
                py[frame, index] = 12.0 + 5.0 * (index // 4)
                px[frame, 23 + index] = 62.0 + 4.0 * (index % 4)
                py[frame, 23 + index] = 14.0 + 5.0 * (index // 4)
        # The ball moves up to 1.25 m per frame (~31 m/s) when seen -- plainly alive -- but is
        # reported on only 55% of frames, which is worse than the 41% DATA.md publishes.
        moving_x = 50.0 + 25.0 * np.sin(np.arange(n) / 20.0)
        seen_mask = rng.random(n) < 0.55
        ball_x = np.where(seen_mask, moving_x, np.nan).astype(np.float32)
        ball_y = np.full(n, 30.0, dtype=np.float32)
        # 2 m/s is what a team in open play looks like. Left at the helper's default of zero it
        # would be a scene of twenty-two people standing still, and the motion-collapse cue would
        # be right to say something about it.
        tracking = h.tracking(px, py, ball_x, ball_y, ps=np.full((n, h.N_SLOTS), 2.0))
        _, features, fused = run_components(tracking, cfg, np.ones(n, dtype=bool))

        dead_cues = features["dead"]
        self.assertLess(float(np.median(dead_cues["static"])), 0.35,
                        "a moving ball seen half the time was read as a stationary ball")
        self.assertGreater(float(np.median(dead_cues["moving"])), 0.65)
        self.assertLess(float(np.median(dead_cues["collapse"])), 0.2)
        self.assertLess(float(np.median(fused["p_dead"])), 0.45,
                        "p_dead sat above a coin flip with no dead cue to justify it")

    def test_frames_with_nothing_visible_have_no_opinion(self):
        cfg = h.config()
        n = 30
        px = np.full((n, h.N_SLOTS), np.nan)
        py = np.full((n, h.N_SLOTS), np.nan)
        tracking = h.tracking(px, py, np.full(n, np.nan), np.full(n, np.nan))
        graded = np.ones(n, dtype=bool)
        _, features, fused = run_components(tracking, cfg, graded)
        self.assertTrue(np.all(features["nearest_team"] == -1))
        self.assertTrue(np.all(np.isnan(features["margin"])))
        self.assertEqual(float(np.max(np.abs(fused["delta"]))), 0.0,
                         "an empty frame produced evidence anyway")


class DuplicateTrackletTest(unittest.TestCase):
    def test_a_ghost_slot_lowers_reliability_but_not_the_side(self):
        cfg = h.config()
        n = 12
        bx, by = 50.0, 30.0

        def build(with_duplicate):
            px = np.full((n, h.N_SLOTS), np.nan)
            py = np.full((n, h.N_SLOTS), np.nan)
            for frame in range(n):
                px[frame, 0], py[frame, 0] = bx + 1.0, by            # carrier
                px[frame, 23], py[frame, 23] = bx + 6.0, by          # nearest opponent
                px[frame, 5], py[frame, 5] = 20.0, 10.0              # a teammate far away
                if with_duplicate:
                    px[frame, 6], py[frame, 6] = bx + 1.2, by + 0.2  # the same person, twice
                else:
                    px[frame, 6], py[frame, 6] = 25.0, 12.0          # an ordinary fourth player
                px[frame, 7], py[frame, 7] = 30.0, 20.0              # five visible either way,
                px[frame, 8], py[frame, 8] = 35.0, 22.0              # so n_visible is not the variable
                px[frame, 9], py[frame, 9] = 15.0, 50.0
            return h.tracking(px, py, np.full(n, bx), np.full(n, by), ball_speed=np.full(n, 7.0))

        _, single, _ = run_components(build(False), cfg, np.ones(n, dtype=bool))
        _, doubled, _ = run_components(build(True), cfg, np.ones(n, dtype=bool))
        self.assertLess(float(np.median(doubled["rho"])), float(np.median(single["rho"])),
                        "a duplicated tracklet did not reduce the frame's reliability")


class TaggingInterpretationTest(unittest.TestCase):
    """Smart Tagging is automatic, timed in seconds, and names clubs. All three of those are
    places where a silent misreading is possible, so all three are asserted."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = h.config()
        cls.directory, cls.paths, cls.sim = h.simulate_to_dir(n_frames=1200, seed=2027)
        cls.events = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_seconds_become_frames_the_way_the_official_reader_does(self):
        features = tagging_module.load_tagging(
            data_dir=self.directory, cfg=self.cfg, n_frames=1200,
            home_name="Barcelona", away_name="Atletic Club")
        set_piece = [event for event in features.events
                     if (event["code"] or "").strip().lower() == "set pieces"]
        self.assertTrue(set_piece, "the fixture produced no SET PIECES interval to check")
        event = set_piece[0]
        start_frame = int(round(float(event["start_s"]) * h.FPS))
        self.assertEqual(int(event["start_frame"]), start_frame,
                         "the frame mapping is not round(seconds x fps)")
        inside = np.flatnonzero(features.setpiece)
        self.assertTrue(inside.min() <= start_frame <= inside.max())
        self.assertTrue(bool(features.setpiece[start_frame]))

    def test_club_spellings_and_missing_teams(self):
        side_of, _ = tagging_module.build_team_lookup(self.cfg, "Barcelona", "Atletic Club")
        self.assertEqual(side_of("Barcelona"), tagging_module.HOME)
        self.assertEqual(side_of("barcelona"), tagging_module.HOME)
        self.assertEqual(side_of("Atletic Club"), tagging_module.AWAY)
        self.assertEqual(side_of("Athletic Club"), tagging_module.AWAY)   # the corrected spelling
        self.assertEqual(side_of("ATH"), tagging_module.AWAY)
        for blank in ("N/A", "na", "", None):
            self.assertEqual(side_of(blank), tagging_module.UNKNOWN,
                             "{!r} became a side instead of no information".format(blank))
        self.assertEqual(side_of("Manchester City"), tagging_module.UNKNOWN)

    def test_an_unknown_team_cannot_favour_anybody(self):
        n = 40
        features = tagging_module.TaggingFeatures(
            [], np.full(n, tagging_module.UNKNOWN, dtype=np.int8),
            np.zeros(n, dtype=bool), np.full(n, tagging_module.UNKNOWN, dtype=np.int8),
            np.zeros(n, dtype=np.int8), np.full(n, tagging_module.UNKNOWN, dtype=np.int8),
            np.zeros(n, dtype=np.float32), {}, {})
        difference = evidence.tagging_difference(features, n)
        self.assertTrue(np.allclose(difference, 0.0))

    def test_tagging_is_a_weak_channel_by_configuration(self):
        weight = float(self.cfg["fusion"]["weights"]["tagging"])
        clipped = 4.0                                   # fuse() clips every standardised block
        self.assertLessEqual(weight, 0.5,
                             "tagging is configured as a strong channel; it is automatic data")
        self.assertLess(weight * clipped, 2.0,
                        "one channel can swing the fused score by {} nats/frame".format(
                            weight * clipped))

    def test_a_phase_tag_alone_does_not_decide_a_possession(self):
        """Everything neutral, tagging insisting on HOME for a whole minute."""
        cfg = self.cfg
        n = 1500
        px, py, ball_x, ball_y = [], [], [], []
        for frame in range(n):
            row_x = np.full(h.N_SLOTS, np.nan)
            row_y = np.full(h.N_SLOTS, np.nan)
            for index in range(11):                    # both teams, identically placed
                row_x[index] = 40.0 + index
                row_y[index] = 30.0
                row_x[23 + index] = 65.0 - index
                row_y[23 + index] = 30.0
            px.append(row_x)
            py.append(row_y)
            ball_x.append(52.5)
            ball_y.append(30.0)
        tracking = h.tracking(np.array(px), np.array(py), np.array(ball_x, dtype=np.float32),
                              np.array(ball_y, dtype=np.float32),
                              ball_speed=np.full(n, 6.0, dtype=np.float32))
        features = tagging_module.TaggingFeatures(
            [], np.full(n, tagging_module.HOME, dtype=np.int8), np.zeros(n, dtype=bool),
            np.full(n, tagging_module.UNKNOWN, dtype=np.int8), np.zeros(n, dtype=np.int8),
            np.full(n, tagging_module.UNKNOWN, dtype=np.int8), np.zeros(n, dtype=np.float32),
            {}, {})
        _, _, fused = run_components(tracking, cfg, np.ones(n, dtype=bool), features)
        q = fused["q"][100:1400]
        self.assertLess(float(np.max(q)), 0.95,
                        "the tagger alone produced a near-certain home possession")
        self.assertGreater(float(np.min(q)), 0.05)


if __name__ == "__main__":
    unittest.main()
