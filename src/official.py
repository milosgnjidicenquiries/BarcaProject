"""Bridge to the official repository. Imports it, never edits it.

Everything R3P shares with the organisers -- asset paths, frame alignment, the EPTS team map, the
Smart Tagging reader, the homography reader, the scorer and the validator -- is already written and
already reviewed in ``more-than-a-hack-2027``. Re-implementing it here would be a second place for
the same fact to be wrong, so this module only exposes it.

The rule for anything added below: call the official function, do not copy it.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER = os.path.join(ROOT, "starter")
EVALUATION = os.path.join(ROOT, "evaluation")

for path in (ROOT, STARTER, EVALUATION):
    if path not in sys.path:
        sys.path.insert(0, path)

import games  # noqa: E402  (official)

# ``starter/baseline.py`` holds the two things R3P genuinely wants from it: the EPTS team map, and
# the reference algorithm the board is ranked against. Importing it is also what guarantees the
# baseline we report is the published one rather than a re-implementation of it.
import baseline as official_baseline  # noqa: E402
import load_atd as official_atd  # noqa: E402
import load_homography as official_homography  # noqa: E402
import load_smart_tagging as official_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
import run_evaluation as official_run_evaluation  # noqa: E402
import validate_submission as official_validator  # noqa: E402

GAME = games.GAME
GAMES = games.GAMES
SOURCES = games.SOURCES
DEFAULT_SOURCE = getattr(games, "DEFAULT_SOURCE", "tactical")
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]
N_VIDEO_FRAMES = GAMES[GAME]["n_video_frames"]

asset_paths = games.asset_paths
require = games.require
source_frames = games.source_frames
half_video_blocks = games.half_video_blocks
half_of_video_frame = games.half_of_video_frame
video_frame_of = official_atd.video_frame_of
frame_id_of = official_atd.frame_id_of

team_of_each_track = official_baseline.team_of_each_track
team_names = official_baseline.team_names

__all__ = [
    "GAME", "GAMES", "FPS", "PITCH", "N_VIDEO_FRAMES", "ROOT", "EVALUATION", "STARTER",
    "games", "pm", "official_baseline", "official_tagging", "official_homography",
    "official_atd", "official_run_evaluation", "official_validator",
    "asset_paths", "require", "source_frames", "half_video_blocks", "half_of_video_frame",
    "video_frame_of", "frame_id_of", "team_of_each_track", "team_names",
]
