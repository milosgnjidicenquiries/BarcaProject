"""Load ``config/r3p.toml`` and resolve it against the per-game constants in ``games.py``.

Nothing else in the code base is allowed to carry a magic number. ``games.py`` stays the single
authority for the pitch size, frame count, fps and half boundaries -- so if the organisers correct
one of those, R3P follows it rather than arguing with it.
"""

from __future__ import annotations

import copy
import hashlib
import os
import sys
import tomllib
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, SOURCES, half_video_blocks, source_frames  # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT, "config", "r3p.toml")


class Config(dict):
    """A dict you can walk with dots, so ``cfg.fusion.weights.nearest`` reads like the TOML."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as error:
            raise AttributeError(name) from error
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted, default=None):
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path=None) -> Config:
    with open(path or DEFAULT_CONFIG_PATH, "rb") as handle:
        raw = tomllib.load(handle)
    return Config(raw)


def config_digest(path=None) -> str:
    """Hash of the exact bytes of the config file, recorded in every run manifest."""
    with open(path or DEFAULT_CONFIG_PATH, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def game_facts(game=GAME) -> dict:
    """The facts a run depends on that come from the official repository, not from our config."""
    length, width = GAMES[game]["pitch"]
    return {
        "game": game,
        "n_video_frames": GAMES[game]["n_video_frames"],
        "fps": GAMES[game]["fps"],
        "pitch_length_m": length,
        "pitch_width_m": width,
        "halves": [(h, int(a), int(b)) for h, a, b in half_video_blocks(game)],
        "sources": sorted(SOURCES),
        "tactical_asset_count": len(SOURCES["tactical"]["assets"]),
    }


def resolve(cfg: Config, game=GAME, data_dir=None, team_name=None, source="tactical") -> Config:
    """Freeze the run: config + game facts + the data directory actually used.

    Returns a deep copy, so a dashboard that re-runs with a different team name cannot mutate the
    config another panel is still reading.
    """
    out = copy.deepcopy(dict(cfg))
    out["facts"] = game_facts(game)
    out["run"] = {
        "team_name": team_name or cfg["meta"]["team_name"],
        "data_dir": os.path.abspath(data_dir) if data_dir else None,
        "source": source,
        "game": game,
        "game_facts": game_facts(game),
        "n_frames": source_frames(source, game),
    }
    return Config(out)
