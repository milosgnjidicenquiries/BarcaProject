"""Component B: dead-ball and team-possession are two separate inferences.

Every test here states one clause of the specification and checks the code implements it:

    "Model dead-ball and team-possession as separate but interacting variables."
    "Do not treat a missing ball as automatically a dead ball."
    "Smart Tagging is weak evidence only."
    "A touch, deflection or rebound is not automatically a turnover."

The fixtures are hand-built tracking arrays in metres -- the same units ``src.data.atd`` produces --
so a cue's value here is traceable to a physical statement about the frame, not to a random number.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests import helpers as h

from src.possession import evidence


def infer(px, py, ball_x, ball_y, cfg, ps=None, team_of_slot=None):
    """The two heads over one hand-built clip: (features, fused)."""
    n_frames = px.shape[0]
    tr = h.tracking(px, py, ball_x, ball_y, ps=ps, team_of_slot=team_of_slot)
    graded = np.ones(n_frames, dtype=bool)
    ball = evidence.ball_channel(tr, cfg, graded)
    features = evidence.evidence(tr, cfg, graded, ball, None, None)
    return features, evidence.fuse(cfg, features, graded)


def flat(n_frames, x, y, slots, side_of_slot=0, n_slots=h.N_SLOTS):
    """The same few players at the same spots for every frame (a frozen picture)."""
    px, py = h.slots_at({}, n_frames, n_slots)
    for offset, slot in enumerate(slots):
        px[:, slot] = x + 1.1 * offset
        py[:, slot] = y + 0.9 * offset
    return px, py


def wander(n_frames, anchors, seed=20271117, amplitude=3.0, n_slots=h.N_SLOTS):
    """{slot: (x, y)} with a smooth per-frame drift added.

    The fusion standardises every block by its own median and MAD over the graded frames, so a
    channel that is *constant* has MAD 0 and contributes exactly nothing. A frozen fixture therefore
    proves nothing about the side head; drifting anchors keep the football statement true (who is
    near the ball) while giving the channels a distribution to be scaled by.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(anchors))
    rates = rng.uniform(0.4, 1.6, size=len(anchors))
    px, py = h.slots_at({}, n_frames, n_slots)
    t = np.arange(n_frames) / float(n_frames)
    for index, (slot, (x, y)) in enumerate(anchors.items()):
        px[:, slot] = x + amplitude * np.sin(2.0 * np.pi * rates[index] * t + phases[index])
        py[:, slot] = y + amplitude * np.cos(2.0 * np.pi * rates[index] * t + phases[index])
    return px, py


def players_around(bx, by, home_slots=None, away_slots=None, gap_home=1.0, gap_away=7.0,
                   n_frames=None, n_slots=h.N_SLOTS):
    """Ten of each side, positioned against a per-frame ball path: `bx`/`by` may be arrays.

    Home holds the ball and away presses from behind, so the geometry says "home in possession" for
    every frame, while the absolute positions keep changing.
    """
    bx = np.atleast_1d(np.asarray(bx, dtype=float))
    by = np.atleast_1d(np.asarray(by, dtype=float))
    n_frames = n_frames or len(bx)
    if len(bx) == 1:
        bx = np.full(n_frames, bx[0])
        by = np.full(n_frames, by[0])
    home_slots = list(range(0, 10)) if home_slots is None else list(home_slots)
    away_slots = list(range(23, 33)) if away_slots is None else list(away_slots)
    px, py = h.slots_at({}, n_frames, n_slots)
    for offset, slot in enumerate(home_slots):
        px[:, slot] = bx + gap_home + 1.1 * offset
        py[:, slot] = by + 0.9 * offset
    for offset, slot in enumerate(away_slots):
        px[:, slot] = bx - gap_away - 1.1 * offset
        py[:, slot] = by + 0.7 * offset
    return px, py


class TwoHeadsTest(unittest.TestCase):
    """The factorisation P(H) = (1-p_dead) q, P(A) = (1-p_dead)(1-q), P(D) = p_dead."""

    def setUp(self):
        self.cfg = h.config()

    def test_the_two_heads_multiply_out_to_the_emission_matrix(self):
        n = 40
        bx, by = h.standing_ball(n)
        px, py = flat(n, h.LENGTH / 2, h.WIDTH / 2, list(range(8)))
        features, fused = infer(px, py, bx, by, self.cfg)
        emission = np.exp(fused["log_emission"][:, :3])
        p_dead, q = fused["p_dead"], fused["q"]
        self.assertTrue(np.allclose(emission[:, 2], p_dead))
        self.assertTrue(np.allclose(emission[:, 0], (1.0 - p_dead) * q))
        self.assertTrue(np.allclose(emission[:, 1], (1.0 - p_dead) * (1.0 - q)))
        # A distribution, not three independent scores: the three columns sum to one.
        self.assertTrue(np.allclose(emission.sum(axis=1), 1.0, atol=1e-9))

    def test_a_dead_ball_barely_moves_the_side_head(self):
        """Same geometry, opposite stoppage: mostly p_dead, a little q.

        "Separate but interacting", quantified. Both clips have the players in the same places
        relative to the same route, so the only difference is whether the ball travels that route or
        sits in the middle of it. The alive/dead head should swing; the home/away head should barely
        notice, because the team evidence -- who is nearest, who supports -- did not change.
        """
        n = 120
        fps = self.cfg["facts"]["fps"]
        t = np.arange(n) / fps
        route_x = 30.0 + 14.0 * np.sin(2.0 * np.pi * 0.05 * t)
        route_y = 30.0 + 10.0 * np.sin(2.0 * np.pi * 0.03 * t + 1.0)
        moving_x, moving_y = route_x, route_y
        still_x = np.full(n, route_x.mean())
        still_y = np.full(n, route_y.mean())

        # Players are placed against the *route*, identically in both clips.
        px, py = players_around(route_x, route_y)
        home_only = np.array([0] * (h.N_SLOTS // 2) + [1] * (h.N_SLOTS // 2), dtype=np.int8)

        _, dead_head = infer(px, py, still_x, still_y, self.cfg, ps=np.zeros((n, h.N_SLOTS)),
                             team_of_slot=home_only)
        _, alive_head = infer(px, py, moving_x, moving_y,
                              self.cfg, ps=np.full((n, h.N_SLOTS), 6.0), team_of_slot=home_only)

        p_dead_swing = abs(float(np.median(dead_head["p_dead"])) -
                           float(np.median(alive_head["p_dead"])))
        q_swing = abs(float(np.median(dead_head["q"])) - float(np.median(alive_head["q"])))
        self.assertGreater(p_dead_swing, 0.45)
        self.assertLess(q_swing, 0.25 * p_dead_swing)
        self.assertGreater(float(np.median(alive_head["q"])), 0.5, "home holds the route")


class StationaryBallTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()

    def test_a_ball_lying_still_amid_a_crowd_is_dead(self):
        n = 80
        bx, by = h.standing_ball(n, 30.0, 20.0)
        px, py = flat(n, 29.0, 19.5, list(range(10)))
        features, fused = infer(px, py, bx, by, self.cfg, ps=np.zeros((n, h.N_SLOTS)))
        self.assertGreater(float(features["dead"]["static"][-1]), 0.9)
        self.assertLess(float(features["dead"]["moving"][-1]), 0.1)
        self.assertGreater(float(features["dead"]["wall"][-1]), 0.0)
        self.assertGreater(float(fused["p_dead"][-1]), 0.85)

    def test_a_ball_traveling_at_15_m_per_s_is_in_play(self):
        n = 80
        speed = 15.0
        frames = np.arange(n) / self.cfg["facts"]["fps"]
        bx = 20.0 + speed * frames
        by = np.full(n, 30.0)
        px, py = flat(n, 25.0, 30.0, list(range(10)))
        features, fused = infer(px, py, bx, by, self.cfg, ps=np.full((n, h.N_SLOTS), 7.0))
        self.assertLess(float(features["dead"]["static"][-1]), 0.1)
        self.assertGreater(float(features["dead"]["moving"][-1]), 0.9)
        self.assertLess(float(fused["p_dead"][-1]), 0.2)

    def test_the_alive_cue_enters_the_dead_logit_with_a_minus_sign(self):
        """The regression that produced one D for the whole match.

        Without a signed alive term the head is a one-sided accumulator of dead cues over a negative
        prior, so p_dead never leaves the prior and D wins on emission everywhere.
        """
        coefficients = evidence.dead_coefficients(self.cfg)
        self.assertAlmostEqual(coefficients["moving"],
                               -float(self.cfg["deadball"]["alive_motion_weight"]))
        self.assertLess(coefficients["moving"], 0.0)
        for name in ("static", "oob", "missing", "collapse", "setpiece", "wall"):
            self.assertGreater(coefficients[name], 0.0, name)

    def test_the_applied_coefficients_are_the_reported_ones(self):
        n = 40
        bx, by = h.standing_ball(n)
        px, py = flat(n, 50.0, 30.0, list(range(8)))
        features, fused = infer(px, py, bx, by, self.cfg)
        coefficients = fused["dead_coefficients"]
        self.assertEqual(coefficients, evidence.dead_coefficients(self.cfg))
        z = np.full(n, float(self.cfg["deadball"]["prior_logodds"]))
        for name, coefficient in coefficients.items():
            z = z + coefficient * np.nan_to_num(features["dead"][name]) * \
                float(self.cfg["deadball"]["logit_scale"])
        self.assertTrue(np.allclose(fused["p_dead"], 1.0 / (1.0 + np.exp(-z)), atol=1e-9))


class MissingBallIsNotDeadTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()

    def test_the_missing_cue_is_the_weakest_dead_cue(self):
        deadball = self.cfg["deadball"]
        missing = float(deadball["missing_weight"])
        for name in ("static_weight", "oob_weight", "alive_motion_weight"):
            self.assertLess(missing, float(deadball[name]), name)

    def test_a_dropout_weakens_dead_evidence_but_never_outweighs_a_stationary_ball(self):
        """DATA.md: the ball is unobserved on 41% of graded frames, most of them live play."""
        n = 240
        bx, by = h.standing_ball(n, 40.0, 25.0)
        px, py = flat(n, 40.0, 25.0, list(range(10)))
        # Lost from frame 60 on, while the players keep running.
        bx[60:] = np.nan
        by[60:] = np.nan
        features, fused = infer(px, py, bx, by, self.cfg, ps=np.full((n, h.N_SLOTS), 6.0))
        self.assertGreater(float(features["dead"]["missing"][-1]), 0.5)
        self.assertEqual(float(features["dead"]["static"][-1]), 0.0)
        self.assertEqual(float(features["dead"]["moving"][-1]), 0.0)

        still_x, still_y = h.standing_ball(n, 40.0, 25.0)
        _, dead_features = infer(px, py, still_x, still_y, self.cfg,
                                 ps=np.zeros((n, h.N_SLOTS)))
        self.assertGreater(float(np.median(dead_features["p_dead"])),
                           float(np.median(fused["p_dead"])))

    def test_a_ball_never_seen_is_trusted_only_as_a_guess(self):
        n = 120
        bx = np.full(n, np.nan)
        by = np.full(n, np.nan)
        px, py = flat(n, 50.0, 30.0, list(range(10)))
        features, fused = infer(px, py, bx, by, self.cfg)
        # The ball is gone, so the best available position is the geometric guess, and a guess is
        # worth geo_impute_rho, never 1.0. Compare with the identical clip where the ball IS seen.
        seen_x, seen_y = h.standing_ball(n, 50.0, 30.0)
        observed, _ = infer(px, py, seen_x, seen_y, self.cfg)
        ceiling = float(self.cfg["fusion"]["reliability"]["geo_impute_rho"])
        self.assertLessEqual(float(np.max(features["rho"])), ceiling + 1e-9)
        self.assertGreater(float(np.median(observed["rho"])), float(np.max(features["rho"])))
        self.assertEqual(int(features["ball"]["seen"].sum()), 0)
        self.assertEqual(float(features["dead"]["static"][-1]), 0.0,
                         "a lost ball is not a stationary ball")


class CueSpecificityTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()

    def test_the_wall_cue_needs_a_real_cluster(self):
        needed = int(self.cfg["deadball"]["wall_cluster_players"])
        n = 40
        bx, by = h.standing_ball(n, 50.0, 30.0)
        px, py = flat(n, 50.0, 30.0, list(range(needed - 3)))
        few, _ = infer(px, py, bx, by, self.cfg)
        px, py = flat(n, 50.0, 30.0, list(range(needed + 3)))
        many, _ = infer(px, py, bx, by, self.cfg)
        self.assertEqual(float(few["dead"]["wall"][-1]), 0.0)
        self.assertGreater(float(many["dead"]["wall"][-1]), 0.0)

    def test_a_ball_over_the_touchline_is_out_of_play(self):
        n = 40
        bx, by = h.standing_ball(n, 50.0, -6.0)          # 6 m past the near touchline
        px, py = flat(n, 50.0, 5.0, list(range(10)))
        features, fused = infer(px, py, bx, by, self.cfg, ps=np.full((n, h.N_SLOTS), 6.0))
        self.assertEqual(float(features["dead"]["oob"][-1]), 1.0)

        inside_x, inside_y = h.standing_ball(n, 50.0, 5.0)
        inside, inside_fused = infer(px, py, inside_x, inside_y, self.cfg,
                                     ps=np.full((n, h.N_SLOTS), 6.0))
        self.assertEqual(float(inside["dead"]["oob"][-1]), 0.0)
        self.assertGreater(float(fused["p_dead"][-1]), float(inside_fused["p_dead"][-1]))

    def test_everybody_walking_is_a_collapse_but_a_missing_slot_is_not(self):
        n = 60
        bx, by = h.standing_ball(n, 50.0, 30.0)
        px, py = flat(n, 50.0, 30.0, list(range(10)))
        running, _ = infer(px, py, bx, by, self.cfg, ps=np.full((n, h.N_SLOTS), 6.0))
        walking, _ = infer(px, py, bx, by, self.cfg, ps=np.full((n, h.N_SLOTS), 0.4))
        self.assertGreater(float(walking["dead"]["collapse"][-1]),
                           float(running["dead"]["collapse"][-1]))
        self.assertGreater(float(walking["dead"]["collapse"][-1]), 0.5)

        # Most tracklets down, the rest running at full speed: the median is a runner. Feeding the
        # missing slots through as 0 instead puts "everybody is standing still" into the median of
        # every poorly-tracked frame, which is most of a match.
        sparse_ps = np.full((n, h.N_SLOTS), 6.0)
        sparse_ps[:, 12:] = np.nan
        sparse, _ = infer(px, py, bx, by, self.cfg, ps=sparse_ps)
        self.assertAlmostEqual(float(sparse["dead"]["collapse"][-1]),
                               float(running["dead"]["collapse"][-1]), places=3)


class TaggingIsWeakEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.cfg = h.config()

    def test_a_set_piece_tag_raises_the_dead_head_without_dominating_it(self):
        n = 60
        bx, by = h.standing_ball(n, 50.0, 30.0)
        px, py = flat(n, 50.0, 30.0, list(range(4)))
        # A ball in motion, so the static cue is off and the tag is the only dead evidence.
        frames = np.arange(n) / self.cfg["facts"]["fps"]
        moving_x = 20.0 + 12.0 * frames
        moving_y = np.full(n, 30.0)

        plain = _tag_free_features(px, py, moving_x, moving_y, self.cfg, n)
        tagged = _tag_set_piece(px, py, moving_x, moving_y, self.cfg, n)
        rise = float(np.median(tagged["p_dead"])) - float(np.median(plain["p_dead"]))
        static_only = float(np.median(_tag_free_features(
            px, py, bx, by, self.cfg, n, ps=np.zeros((n, h.N_SLOTS)))["p_dead"])) \
            - float(np.median(plain["p_dead"]))
        self.assertGreater(rise, 0.0)
        self.assertLess(rise, static_only,
                        "a SET PIECES interval outranks a stationary ball; it must not")
        self.assertLess(float(self.cfg["deadball"]["tagging_weight"]),
                        float(self.cfg["deadball"]["static_weight"]))


class TurnoverIsNotPredictedTest(unittest.TestCase):
    """A touch, deflection or rebound is not automatically a turnover.

    Component B cannot know this on its own -- it is exactly why the side head is a soft `q` and the
    run structure belongs to component C. What is testable here is that a single frame of inverted
    geometry is not enough to move the emission past the point where a one-frame flip is free.
    """

    def test_one_frame_of_opposite_geometry_does_not_win_the_phase(self):
        """A deflection is one frame; the channel with a memory still says who was on top of it.

        The instantaneous blocks see the deflection and change their answer. `continuity`, the only
        block with a backwards window, must not -- and the decoded "one frame is not a change of
        possession" is component C's job, tested in test_decoder.
        """
        cfg = h.config()
        n = 61
        fps = cfg["facts"]["fps"]
        t = np.arange(n) / fps
        route_x = 45.0 + 9.0 * np.sin(2.0 * np.pi * 0.04 * t)
        route_y = 34.0 + 7.0 * np.sin(2.0 * np.pi * 0.06 * t + 0.7)
        px, py = players_around(route_x, route_y, gap_away=5.0 + 4.0 * np.abs(np.sin(t)))
        px[30, 0], py[30, 0] = route_x[30] + 9.0, route_y[30] + 4.0    # carrier knocked away
        px[30, 23], py[30, 23] = route_x[30] + 0.6, route_y[30] + 0.4  # opponent pounces

        tr = h.tracking(px, py, route_x, route_y, ps=np.full((n, h.N_SLOTS), 6.0))
        graded = np.ones(n, dtype=bool)
        ball = evidence.ball_channel(tr, cfg, graded)
        features = evidence.evidence(tr, cfg, graded, ball, None, None)

        nearest = features["blocks"]["nearest"]
        self.assertGreater(float(np.median(nearest[:20])), 0.3, "home holds the ball")
        self.assertLess(float(nearest[30]), 0.0, "the flip frame must register as away")
        self.assertGreater(float(features["blocks"]["continuity"][30]), 0.3,
                           "continuity followed one deflection and changed its mind")
        self.assertEqual(int(features["nearest_team"][30]), 1)
        self.assertEqual(int(np.median(features["nearest_team"][:20])), 0)
        # ...and one frame later the memory channel is essentially undisturbed.
        self.assertGreater(float(features["blocks"]["continuity"][31]), 0.5)


def _tag_set_piece(px, py, bx, by, cfg, n):
    """A clip where the tagger says SET PIECES for every frame, but names no team and no restart."""
    from src.data import tagging
    features = tagging.TaggingFeatures(
        events=[],
        phase_team=np.full(n, tagging.UNKNOWN, dtype=np.int8),
        setpiece=np.ones(n, dtype=bool),
        setpiece_team=np.full(n, tagging.UNKNOWN, dtype=np.int8),
        setpiece_type=np.full(n, -1, dtype=np.int8),
        restart_team=np.full(n, tagging.UNKNOWN, dtype=np.int8),
        restart_strength=np.zeros(n, dtype=np.float32),
        mapping={"team_tag_missing": 0}, meta={})
    return _fuse_with_tagging(px, py, bx, by, cfg, n, features)[1]


def _tag_free_features(px, py, bx, by, cfg, n, ps=None):
    tr = h.tracking(px, py, bx, by, ps=ps)
    graded = np.ones(n, dtype=bool)
    ball = evidence.ball_channel(tr, cfg, graded)
    features = evidence.evidence(tr, cfg, graded, ball, None, None)
    return evidence.fuse(cfg, features, graded)


def _fuse_with_tagging(px, py, bx, by, cfg, n, tagging_features):
    tr = h.tracking(px, py, bx, by)
    graded = np.ones(n, dtype=bool)
    ball = evidence.ball_channel(tr, cfg, graded)
    features = evidence.evidence(tr, cfg, graded, ball, tagging_features, None)
    return features, evidence.fuse(cfg, features, graded)


if __name__ == "__main__":
    unittest.main()
