"""Smart Tagging as frame-indexed evidence, not as labels.

The reader itself is the official one (``starter/load_smart_tagging.py``): the XML is parsed there,
and ``<start>``/``<end>`` are seconds from the first frame of the tactical video, so
``round(seconds * 25)`` is a submission row. What this module adds is the conversion from 638
intervals to a per-frame array, and the discipline around it.

Three properties of this asset decide how it is used (DATA.md states all three):

* it is **automatic**, so a tag is evidence with a false-positive rate, never an answer;
* ``Team`` is sometimes ``N/A`` -- including every throw-in, exactly where it would be worth most;
* the intervals are **phases**, not possessions. A Build Up can span a possession, stop short of
  one, or cover two. Painting an interval as a label imports its errors wholesale.

So R3P takes two things and only two from it: a weak dead-ball prior around set pieces, and a team
identity for restarts. The *timing* of every boundary comes from the tracking, never from a tag.
"""

from __future__ import annotations

import numpy as np

from .. import official

# Per-frame int8 codes for restart type. Order is the config's, so `type_index` and the config list
# have to stay in step -- asserted in tests/test_tagging.py rather than assumed.
TYPE_NONE = -1
TYPE_OTHER = 0

HOME, AWAY, UNKNOWN = 0, 1, -1


class TaggingFeatures:
    """Per-frame views of the Smart Tagging intervals."""

    def __init__(self, events, phase_team, setpiece, setpiece_team, setpiece_type,
                 restart_team, restart_strength, mapping, meta):
        self.events = events
        self.phase_team = phase_team            # (T,) int8 team tag of the interval covering t
        self.setpiece = setpiece                # (T,) bool  inside a SET PIECES interval
        self.setpiece_team = setpiece_team      # (T,) int8 team tag of that set piece, or UNKNOWN
        self.setpiece_type = setpiece_type      # (T,) int8 index into the config restart types
        self.restart_team = restart_team        # (T,) int8 team credited with the restart
        self.restart_strength = restart_strength  # (T,) float32 0..1 weight of that credit
        self.mapping = mapping
        self.meta = meta

    def __len__(self):
        return len(self.phase_team)

    def summary(self):
        return {
            "n_instances": len(self.events),
            "n_frames_with_phase_team": int(np.sum(self.phase_team != UNKNOWN)),
            "n_frames_in_set_piece": int(np.sum(self.setpiece)),
            "n_restart_anchors": int(np.sum(self.restart_team != UNKNOWN)),
            "n_unnamed_teams": int(self.mapping["team_tag_missing"]),
            "n_kickoff_tags": self.meta.get("n_kickoff_tags"),
            "time_base_agrees": self.meta.get("agrees"),
        }


def _normalise(token):
    return (token or "").strip().lower()


def build_team_lookup(cfg, home_name, away_name):
    """Club name -> side. The XML names clubs; the submission wants H or A.

    The configured lists are tried first, then a substring test against the team names actually in
    the ATD metadata, so a spelling change in the export degrades to UNKNOWN rather than silently
    to the wrong side.
    """
    home_values = [_normalise(v) for v in cfg["tagging"]["home_values"]]
    away_values = [_normalise(v) for v in cfg["tagging"]["away_values"]]
    na_values = [_normalise(v) for v in cfg["tagging"]["na_values"]]

    def side_of(raw):
        value = _normalise(raw)
        if value in na_values:
            return UNKNOWN
        if value in home_values:
            return HOME
        if value in away_values:
            return AWAY
        for known, side in ((_normalise(home_name), HOME), (_normalise(away_name), AWAY)):
            if known and value and (known in value or value in known):
                return side
        return UNKNOWN

    return side_of, {"home_values": home_values, "away_values": away_values,
                     "matched_by": {"home": home_name, "away": away_name}}


def load_tagging(data_dir=None, cfg=None, n_frames=None, source="tactical",
                 home_name=None, away_name=None):
    """Read the pattern file and expand it to per-frame evidence arrays."""
    if source != "tactical":
        raise ValueError("Smart Tagging exists only for the tactical recording (games.py SOURCES)")
    events = official.official_tagging.load(data_dir=data_dir)
    n_frames = n_frames or official.N_VIDEO_FRAMES
    fps = official.FPS
    side_of, mapping = build_team_lookup(cfg, home_name, away_name)

    phase_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    setpiece = np.zeros(n_frames, dtype=bool)
    setpiece_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    setpiece_type = np.full(n_frames, TYPE_NONE, dtype=np.int8)

    type_index = {name: i + 1 for i, name in enumerate(cfg["tagging"]["restart_types"])}
    set_piece_codes = {_normalise(c) for c in cfg["tagging"]["set_piece_codes"]}
    n_missing_team = 0

    for event in events:
        start = max(0, int(event["start_frame"]))
        end = min(n_frames, int(event["end_frame"]))
        if end <= start:
            continue
        code = _normalise(event["code"])
        team = side_of(event["tags"].get("Team"))
        if team == UNKNOWN:
            n_missing_team += 1
        phase_team[start:end] = team
        if code in set_piece_codes:
            setpiece[start:end] = True
            setpiece_team[start:end] = team
            setpiece_type[start:end] = type_index.get(
                _normalise(event["tags"].get("Type")), TYPE_OTHER)

    restart_team, restart_strength = _restart_credit(events, side_of, n_frames, fps, cfg)
    mapping["team_tag_missing"] = n_missing_team

    meta = official.official_tagging.check_time_base(events)
    return TaggingFeatures(events, phase_team, setpiece, setpiece_team, setpiece_type,
                           restart_team, restart_strength, mapping,
                           {"n_kickoff_tags": meta["n_kickoff_tags"],
                            "agrees": bool(meta["agrees"]),
                            "kickoff_rows": meta["halves"]})


def _restart_credit(events, side_of, n_frames, fps, cfg):
    """Which team is expected to come into possession around each restart, and how strongly.

    A set-piece interval brackets the stoppage AND the kick, and DATA.md says it is accurate to
    about a second -- so the interval does not tell us the frame the ball comes back to life. What
    it does tell us reliably is *who takes it*. That is the piece R3P keeps: the identity, offered
    to the decoder as a prior on the D -> H/A transition, with a strength that peaks just before
    the nominal end of the interval and decays after it. The instant itself is decided by the
    tracking, which is where the timing information actually is.
    """
    team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    weight = np.zeros(n_frames, dtype=np.float32)
    window = float(cfg["tagging"]["phase_window_frames"])
    set_piece_codes = {_normalise(c) for c in cfg["tagging"]["set_piece_codes"]}

    for event in events:
        if _normalise(event["code"]) not in set_piece_codes:
            continue
        side = side_of(event["tags"].get("Team"))
        if side == UNKNOWN:
            continue                      # every throw-in lands here: no identity, no credit
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        if end <= start:
            continue
        # Ramp across the last second of the interval, hold one second past it.
        lo = max(0, int(start + (end - start) - window))
        hi = min(n_frames, int(end + window))
        if hi <= lo:
            continue
        span = float(hi - lo)
        ramp = np.clip(1.0 - np.abs((np.arange(lo, hi) - end) + 0.5) / max(span, 1.0), 0.0, 1.0)
        take = weight[lo:hi] < ramp
        team[lo:hi][take] = side
        weight[lo:hi] = np.maximum(weight[lo:hi], ramp)
    return team, weight
