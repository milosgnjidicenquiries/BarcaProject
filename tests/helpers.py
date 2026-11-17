"""Shared fixtures for the R3P tests: controlled tracking, small synthetic bundles, fixed seeds.

Everything here builds data in the delivered formats or as arrays shaped exactly like the loader's
output, and every generator takes an explicit seed. No test reads the restricted match, and no test
reads the public ground truth either -- the fixtures are the only labels any test is allowed to know
about, which is what makes them usable for a correctness check rather than a score.

Run from the repository root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import official, settings  # noqa: E402
from src.data import atd, synthetic  # noqa: E402

PITCH = official.PITCH                  # (105.3, 68.0) metres, from games.py
LENGTH, WIDTH = PITCH
FPS = official.FPS
N_SLOTS = synthetic.N_SLOTS


def config():
    """The shipped config, resolved, so a test exercises the numbers that actually run."""
    return settings.resolve(settings.load_config(), data_dir=None, team_name="R3P-TEST")


def tracking(px, py, ball_x, ball_y, ball_speed=None, ps=None, team_of_slot=None,
             fps=FPS, pitch=PITCH):
    """A Tracking object over hand-built metric arrays -- no files, no parser, no guessing.

    Positions are METRES, because that is what the loader hands out; a test that passed normalised
    coordinates here would be testing a unit convention the production code does not use.
    """
    n_frames, n_slots = px.shape
    px = np.ascontiguousarray(px, dtype=np.float32)
    py = np.ascontiguousarray(py, dtype=np.float32)
    ps = np.zeros((n_frames, n_slots), dtype=np.float32) if ps is None else np.asarray(
        ps, dtype=np.float32)
    ball_x = np.asarray(ball_x, dtype=np.float32)
    ball_y = np.asarray(ball_y, dtype=np.float32)
    ball_speed = np.zeros(n_frames, dtype=np.float32) if ball_speed is None else np.asarray(
        ball_speed, dtype=np.float32)
    if team_of_slot is None:
        team_of_slot = np.concatenate([np.zeros(N_SLOTS // 2, dtype=np.int8),
                                       np.ones(N_SLOTS // 2, dtype=np.int8)])
    return atd.Tracking(px, py, ps, ball_x, ball_y, ball_speed,
                        np.asarray(team_of_slot, dtype=np.int8), fps, pitch, {}, {})


def standing_ball(n_frames, x=LENGTH / 2, y=WIDTH / 2):
    return np.full(n_frames, x), np.full(n_frames, y)


def slots_at(positions_by_slot, n_frames, n_slots=N_SLOTS):
    """(n_frames, n_slots, 2) from {slot: (x, y)}, NaN everywhere else."""
    px = np.full((n_frames, n_slots), np.nan)
    py = np.full((n_frames, n_slots), np.nan)
    for slot, (x, y) in positions_by_slot.items():
        px[:, slot] = x
        py[:, slot] = y
    return px, py


def possession_frame(ball_xy, carrier_xy, support_xy=None, n_slots=N_SLOTS,
                     carrier_side=0, by=WIDTH / 2):
    """One frame: a carrier of one side near the ball, its team nearby, the other side far away.

    Everyone else is put on a ring well outside sigma_control_m, so what the channels see is exactly
    the football statement being tested and nothing else.
    """
    px = np.full(n_slots, np.nan)
    py = np.full(n_slots, np.nan)
    home_slots = np.arange(0, N_SLOTS // 2)
    away_slots = np.arange(N_SLOTS // 2, N_SLOTS)
    bx, by_ = ball_xy

    # A neutral spread for both teams, in their own half, so territory and support start flat.
    for side, slots in ((0, home_slots), (1, away_slots)):
        anchor = 0.25 * LENGTH if side == 0 else 0.75 * LENGTH
        for index, slot in enumerate(slots[:10]):
            px[slot] = anchor + 8.0 * (index % 3)
            py[slot] = 6.0 + 5.0 * index

    own = home_slots if carrier_side == 0 else away_slots
    px[own[0]], py[own[0]] = carrier_xy
    if support_xy is not None:
        px[own[1]], py[own[1]] = support_xy
    return px, py


def sequence(items, n_slots=N_SLOTS):
    """[(n_frames, kwargs_for_possession_frame)] -> (px, py) over the whole span."""
    px, py = [], []
    for count, kwargs in items:
        single_x, single_y = possession_frame(**kwargs)
        px.append(np.tile(single_x, (count, 1)))
        py.append(np.tile(single_y, (count, 1)))
    return np.vstack(px), np.vstack(py)


def graded_mask(n_frames, blocks):
    """A graded mask over given half-open blocks, in the shape bundle.graded_mask returns."""
    mask = np.zeros(n_frames, dtype=bool)
    for first, last in blocks:
        mask[int(first):int(last)] = True
    return mask


def half_blocks(n_frames):
    """The official two halves, clipped to a short array: [(first, stop)] half-open."""
    out = []
    for _, first, last in official.half_video_blocks(official.GAME):
        out.append((int(max(0, first)), int(min(n_frames, last + 1))))
    return out


def simulate_to_dir(n_frames=1200, seed=20271117, blocks=None, keep=None):
    """A real bundle on disk in the delivered formats, in a temporary directory.

    The default is short, and the default blocks are two small halves, so the whole write-parse-
    infer round trip costs a second rather than a minute. Returns (data_dir, paths, sim); the
    caller is responsible for the directory, which ``tempdir`` hands back for cleanup.
    """
    cfg = config()
    if blocks is None:
        first = max(1, n_frames // 10)
        mid = n_frames // 2
        blocks = [(1, first, mid), (2, mid + 10, n_frames)]
    sim = synthetic.simulate(cfg, n_frames=n_frames, seed=seed, blocks=blocks)
    directory = keep or tempfile.mkdtemp(prefix="r3p_test_")
    paths = synthetic.write_bundle(directory, sim, overwrite=True)
    return directory, paths, sim


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TempDir:
    """Temporary directories that are removed on exit, so a failed test cannot litter the repo."""

    def __init__(self):
        self.paths = []

    def new(self):
        path = tempfile.mkdtemp(prefix="r3p_test_")
        self.paths.append(path)
        return path

    def cleanup(self):
        for path in self.paths:
            shutil.rmtree(path, ignore_errors=True)
        self.paths = []
