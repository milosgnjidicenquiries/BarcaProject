"""A synthetic bundle in the delivered file formats, for tests and for development.

WHY THIS EXISTS
---------------
The match assets are gated: a fresh clone has ``data/README.md`` and nothing else. Without a
stand-in there is no way to exercise the parser, no way to run the pipeline, and nothing to show an
organiser except code that has never executed. This module writes files the official readers accept,
shaped like the delivered tactical recording: 199,319 frames, 46 track slots, the ball missing on
~41% of graded frames, two graded halves with a 20,156-frame break between them.

WHAT IT IS NOT
--------------
It is not a model of the real match, and its labels are not ground truth of any kind. They are the
latent variables of a simulation this repository wrote. Recovering them proves the code computes what
the documentation claims; it says nothing about accuracy on BAR-ATH. Numbers produced from this
bundle are never presented as competition results, and the output directory carries a
``SYNTHETIC_BUNDLE.json`` marker which ``integrity.run_integrity`` turns into a loud warning on
every run that reads it, so a development number cannot quietly travel into a submission.

The simulation is deliberately built *pessimistic* relative to R3P's assumptions, because a friendly
simulator would test nothing: deflections that put an opponent nearest the ball for a few frames
inside a live possession, scrambles where neither team is clearly closer, duplicated tracklets,
wrong-sided team tags, unnamed throw-ins, and ball dropouts concentrated in exactly the crowded
phases where they matter.

COORDINATES
-----------
The delivered ATD stores positions normalised to 0-1 of the pitch; the loader turns them back into
metres. This module generates metres for legibility and divides on the way out, so the round trip is
covered by tests/test_alignment.py.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from .. import official
from .integrity import SYNTHETIC_MARKER

HOME_TEAM_NAME = "Barcelona"
AWAY_TEAM_NAME = "Atletic Club"          # the export's own spelling, one 'h', per DATA.md
N_SLOTS = 46
SLOTS_PER_TEAM = N_SLOTS // 2
ON_PITCH_PER_TEAM = 11

H, A, D = 0, 1, 2
MISSING_TOKEN = "NaN"                     # what the delivered file writes, and what baseline.py tests


def _lognormal_frames(rng, median_s, sigma_log, floor_s=0.4, fps=25):
    median = median_s * fps
    return float(np.clip(rng.lognormal(np.log(median), sigma_log), floor_s * fps, None))


def _state_sequence(rng, n_frames, fps, graded, cfg):
    """Latent H/A/D per video frame, plus who restarts after every dead-ball run."""
    state = np.full(n_frames, -1, dtype=np.int8)
    restart = np.full(n_frames, -1, dtype=np.int8)
    prior = cfg["decoder"]["duration"]

    inside = np.flatnonzero(graded)
    if not len(inside):
        return state, restart
    spans = np.split(inside, np.flatnonzero(np.diff(inside) != 1) + 1)

    for span in spans:
        t, end = int(span[0]), int(span[-1]) + 1
        current = D                                  # a half opens with a kickoff
        restarting = int(rng.integers(0, 2))
        while t < end:
            entry = prior["D"] if current == D else prior["H" if current == H else "A"]
            length = _lognormal_frames(rng, entry["median_s"], entry["sigma_log"], fps=fps)
            if current == D:
                length = max(length, 2.0 * fps)
            stop = min(end, t + int(length))
            if stop <= t:                            # a run of at least one frame, always
                stop = t + 1
            state[t:stop] = current
            if current == D:
                restart[t:stop] = restarting
                current = restarting
            elif rng.random() < 0.52:
                current = D
                restarting = int(rng.integers(0, 2))
            else:
                current = A if current == H else H
            t = stop
    return state, restart


def _ball_trajectory(rng, n_frames, state, restart, pitch, fps, graded):
    """Ball position and velocity in metres.

    The one rule that R3P's dead-ball head leans on is encoded here because it is a rule of the
    game, not a fitted constant: while the ball is out of play it does not move. Everything else is
    a random walk with heading changes on the scale of a pass.
    """
    length, width = pitch
    position = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_velocity = np.zeros((n_frames, 2), dtype=np.float32)

    inside = np.flatnonzero(graded)
    if not len(inside):
        return position, velocity
    spans = np.split(inside, np.flatnonzero(np.diff(inside) != 1) + 1)
    dt = 1.0 / fps
    carrier = np.array([length / 2.0, width / 2.0])

    for span in spans:
        t, end = int(span[0]), int(span[-1]) + 1
        for frame in range(t, end):
            code = int(state[frame])
            if code == D:
                if frame == t:                       # the ball comes to rest at the stoppage spot
                    if rng.random() < 0.55:
                        side = int(rng.integers(0, 4))
                        carrier = np.array([1.0 if side == 2 else (length - 1.0 if side == 3
                                                                   else carrier[0]),
                                            0.5 if side == 0 else (width - 0.5 if side == 1
                                                                   else carrier[1])])
                    else:
                        carrier = np.array([rng.uniform(10, length - 10),
                                            rng.uniform(5, width - 5)])
                position[frame] = carrier + rng.normal(0.0, 0.04, size=2)
                ball_velocity[frame] = 0.0
                continue

            direction = 1.0 if code == H else -1.0   # home attacks toward +x
            vel = ball_velocity[frame - 1].astype(np.float64) if frame > 0 else np.zeros(2)
            if frame == t or not np.isfinite(carrier[0]) or not np.any(vel):
                vel = np.array([direction * rng.uniform(4, 12), rng.normal(0, 4)])
            if rng.random() < 0.05:                  # a pass, a cut, a shot off the dribble
                vel = np.array([direction * rng.uniform(2, 16), rng.normal(0, 7)])
            vel = vel + rng.normal(0, 0.9, size=2)
            carrier = carrier + vel * dt
            if carrier[0] < 1.0:
                carrier[0], vel[0] = 1.0, abs(vel[0])
            elif carrier[0] > length - 1.0:
                carrier[0], vel[0] = length - 1.0, -abs(vel[0])
            if carrier[1] < 1.0:
                carrier[1], vel[1] = 1.0, abs(vel[1])
            elif carrier[1] > width - 1.0:
                carrier[1], vel[1] = width - 1.0, -abs(vel[1])
            speed = float(np.linalg.norm(vel))
            if speed > 26.0:                         # 94 km/h is about the fastest a ball travels
                vel = vel * (26.0 / speed)
            position[frame] = carrier
            ball_velocity[frame] = vel

    # Contested touches: an opponent is nearest for a moment inside a live possession. A touch is
    # not a takeover -- see CHALLENGE.md, "what does not count as a change of possession".
    alive = np.flatnonzero(graded & (state != D))
    for _ in range(max(1, len(alive) // 900)):
        if not len(alive):
            break
        index = int(rng.choice(alive))
        run = int(rng.integers(1, 4))
        for offset in range(run):
            frame = index + offset
            if frame < n_frames and graded[frame] and state[frame] != D:
                position[frame] = position[frame] + rng.normal(0, 0.5, size=2)
    return position, ball_velocity


def _formation(count, length, width):
    """Two 11-player shapes in metres, one per side, mirrored."""
    xs = np.linspace(0.10, 0.62, count)
    ys = np.tile(np.array([0.12, 0.30, 0.44, 0.56, 0.70, 0.88]), count // 6 + 1)[:count]
    home = np.stack([xs * length, ys * width], axis=1)
    away = np.stack([length - home[:, 0], home[:, 1]], axis=1)
    return home, away


def _players(rng, n_frames, state, restart, ball, pitch, fps, graded):
    """46 slots of positions in metres, with tracklet fragmentation and duplicated people.

    Slots are tracklets, not persons (DATA.md), and the simulation says so the way the real file
    does: a slot's identity is its column index, columns disappear for tens of seconds, and two
    columns sometimes stand on top of each other.
    """
    length, width = pitch
    px = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    py = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    team_of_slot = np.concatenate([np.zeros(SLOTS_PER_TEAM, dtype=np.int8),
                                   np.ones(SLOTS_PER_TEAM, dtype=np.int8)])
    base_home, base_away = _formation(ON_PITCH_PER_TEAM, length, width)

    # Which slots are on the pitch changes over the match (subs, re-acquisitions, identity loss).
    block = max(1, int(45 * fps))
    live = np.zeros((n_frames, N_SLOTS), dtype=bool)
    for start in range(0, n_frames, block):
        stop = min(n_frames, start + block)
        for side in (0, 1):
            slots = np.flatnonzero(team_of_slot == side)
            live[start:stop, rng.choice(slots, size=ON_PITCH_PER_TEAM, replace=False)] = True

    home_slots = np.flatnonzero(team_of_slot == 0)
    away_slots = np.flatnonzero(team_of_slot == 1)

    for frame in range(n_frames):
        if not graded[frame] or not np.isfinite(ball[frame, 0]):
            continue
        code = int(state[frame])
        bx, by = float(ball[frame, 0]), float(ball[frame, 1])
        owner = code if code != D else int(restart[frame]) if restart[frame] >= 0 else -1

        for side, base, slots in ((0, base_home, home_slots), (1, base_away, away_slots)):
            active = slots[live[frame, slots]]
            n = len(active)
            if not n:
                continue
            if code == D:
                # Everyone gathers: a wall, a short kick, a throw-in crowd.
                spread_x, spread_y, pull = 0.30, 0.45, 0.35
            else:
                spread_x, spread_y, pull = (0.36, 0.62, 0.52) if side == owner else (0.40, 0.66, 0.70)
            centre_x = bx * pull + (0.0 if side == 0 else length) * (1 - pull)
            centre_y = by * 0.6 + (width / 2) * 0.4
            xs = centre_x + (base[:n, 0] - (0.0 if side == 0 else length)) * spread_x
            ys = centre_y + (base[:n, 1] - width / 2) * spread_y
            px[frame, active] = np.clip(xs, 0.5, length - 0.5)
            py[frame, active] = np.clip(ys, 0.5, width - 0.5)

            # The player nearest the ball for this team, at a distance that encodes the state.
            if code == D:
                radius = 1.0 if side == owner else 2.8
            elif side == owner:
                radius = abs(float(rng.normal(1.5, 0.8))) + 0.3
            else:
                radius = abs(float(rng.normal(3.8, 1.8))) + 0.6
            pick = int(rng.integers(0, n))
            angle = float(rng.uniform(0, 2 * np.pi))
            px[frame, active[pick]] = bx + radius * np.cos(angle)
            py[frame, active[pick]] = by + radius * np.sin(angle)
            if code != D and side == owner and n > 2:
                # a support option a short pass away: possession is a shape, not just a touch
                other = int(rng.integers(0, n))
                if other != pick:
                    px[frame, active[other]] = bx + 6.5 * np.cos(angle + 1.3)
                    py[frame, active[other]] = by + 6.5 * np.sin(angle + 1.3)

    return px, py, team_of_slot, live


def _crowd(px, py, ball, limit=6.0):
    """Tracked players within ``limit`` metres of the ball, per frame."""
    dx = px - ball[:, 0][:, None]
    dy = py - ball[:, 1][:, None]
    return np.nansum((dx * dx + dy * dy) <= limit * limit, axis=1).astype(np.int16)


def simulate(cfg, n_frames=None, seed=None, blocks=None, progress=None):
    """Every array a bundle needs, plus the latent labels for self-consistency testing only.

    ``blocks`` overrides the graded half boundaries so unit tests can run a small simulation; a
    ``None`` value takes them from ``games.py``, which is the real contract.
    """
    game = official.GAME
    fps = official.FPS
    pitch = official.PITCH
    n_frames = n_frames or official.N_VIDEO_FRAMES
    seed = cfg["meta"]["seed"] if seed is None else int(seed)
    rng = np.random.default_rng(seed)

    if blocks is None:
        blocks = [(h, int(a), int(b)) for h, a, b in official.half_video_blocks(game)]
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[max(0, first):min(n_frames, last + 1)] = True

    def report(part, message):
        if progress:
            progress(int(n_frames * part), n_frames, message)

    report(0.05, "simulating state sequence")
    state, restart = _state_sequence(rng, n_frames, fps, graded, cfg)
    report(0.25, "simulating ball trajectory")
    ball, ball_velocity = _ball_trajectory(rng, n_frames, state, restart, pitch, fps, graded)
    report(0.45, "simulating players")
    px, py, team_of_slot, live = _players(rng, n_frames, state, restart, ball, pitch, fps, graded)
    report(0.85, "degrading the observations")

    # Ball dropout, concentrated where the tracker genuinely loses it (crowds, and the standing
    # stillness around a restart) rather than spread evenly. DATA.md: 41% of graded frames.
    finite_ball = np.nan_to_num(ball, nan=-1e6)
    crowd = _crowd(px, py, finite_ball)
    p_missing = np.clip(0.24 + 0.045 * crowd + 0.10 * (state == D), 0.0, 0.80)
    drop = (rng.random(n_frames) < p_missing) & graded
    missing = drop | ~graded
    bx = np.where(missing, np.nan, ball[:, 0]).astype(np.float32)
    by = np.where(missing, np.nan, ball[:, 1]).astype(np.float32)
    bs = np.where(missing, np.nan,
                  np.linalg.norm(ball_velocity, axis=1)).astype(np.float32)

    visible = rng.random((n_frames, N_SLOTS)) < np.where(graded[:, None] & live, 0.955, 0.0)
    px = np.where(visible, px, np.nan).astype(np.float32)
    py = np.where(visible, py, np.nan).astype(np.float32)

    # Duplicated tracklets: the tracker splitting one person in two, which DATA.md warns about.
    for start in range(0, n_frames, max(1, int(25 * fps))):
        stop = min(n_frames, start + int(rng.uniform(0.2, 0.9) * 25 * fps))
        pair = int(rng.integers(0, SLOTS_PER_TEAM))
        clone = pair + (SLOTS_PER_TEAM if pair < SLOTS_PER_TEAM // 2 else 0)
        ok = ~np.isnan(px[start:stop, pair])
        px[start:stop, clone] = np.where(ok, px[start:stop, pair] + 0.3, np.nan)
        py[start:stop, clone] = np.where(ok, py[start:stop, pair] + 0.25, np.nan)

    step = np.hypot(np.diff(px, axis=0, prepend=px[:1]),
                    np.diff(py, axis=0, prepend=py[:1])) * fps
    ps = np.where(np.isnan(px), np.nan, step).astype(np.float32)

    labels = np.where(~graded, "X",
                      np.where(state == D, "D", np.where(state == H, "H", "A")))
    report(1.0, "simulated")
    return {
        "n_frames": int(n_frames), "fps": int(fps), "pitch": tuple(pitch), "seed": int(seed),
        "blocks": blocks, "graded": graded, "state": state, "restart": restart,
        "team_of_slot": team_of_slot, "team_names": (HOME_TEAM_NAME, AWAY_TEAM_NAME),
        "px": px, "py": py, "ps": ps, "ball_x": bx, "ball_y": by, "ball_speed": bs,
        "labels": labels, "crowd": crowd, "n_slots": N_SLOTS,
    }


# ---------------------------------------------------------------------------------------------
# Writing the bundle in the delivered formats
# ---------------------------------------------------------------------------------------------

def write_atd_meta(path, team_names, team_of_slot):
    """FIFA/EPTS metadata, in the element shape ``starter/baseline.py`` reads.

    Exactly two ``Team`` elements, home first -- the convention ``team_of_each_track`` depends on,
    and the reason a fourth one appearing anywhere in the file is a hard error rather than a
    warning. This mirrors the fields the official code needs; it is not a byte-level replica of a
    Metrica header, because no Metrica file is in this repository to copy.
    """
    root = ET.Element("Stats", {"version": "epts.1.1.0", "timeunit": "40"})
    sequence = ET.SubElement(root, "Sequence", {"id": "1", "name": "BAR-ATH", "location": "synthetic"})
    metadata = ET.SubElement(sequence, "MetaData")
    ET.SubElement(metadata, "FrameRate", {"num": "25", "denom": "1"})
    ET.SubElement(metadata, "Order", {"num": "1", "totalCount": "1"})
    ET.SubElement(metadata, "Company", {"id": "", "name": "src/data/synthetic.py"})
    ET.SubElement(metadata, "Competition", {"id": "1", "name": "Synthetic", "seasonYear": "2024"})
    device = ET.SubElement(metadata, "Device", {"id": "1", "type": "OPTRACKING", "name": "synthetic",
                                                "periodCount": "2"})
    ET.SubElement(device, "ardi", {"id": "0", "version": "1", "length": "105", "width": "68"})
    ET.SubElement(device, "Referee", {"id": "0", "FName": "Syn", "LName": "Thetic"})
    for index, name in enumerate(team_names):
        team = ET.SubElement(device, "Team", {"id": str(100 + index), "Type": "1"})
        ET.SubElement(team, "Name").text = name
    for index in range(2):
        period = ET.SubElement(device, "Period", {"No": str(index + 1), "time": "2700"})
        ET.SubElement(period, "Start").text = "0"
        ET.SubElement(period, "End").text = "67500"
    players = ET.SubElement(device, "Players")
    for slot, team_index in enumerate(team_of_slot):
        ET.SubElement(players, "Player", {
            "id": str(slot + 1), "teamId": str(100 + int(team_index)),
            "ShirtNumber": str(slot + 1), "position": "GK"})
        ET.SubElement(players[-1], "Name").text = "Track {}".format(slot + 1)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _columns(sim):
    """Normalised 0-1 columns, the way the delivered file stores them."""
    length, width = sim["pitch"]
    return (sim["px"] / length).astype(np.float64), (sim["py"] / width).astype(np.float64), \
        sim["ps"].astype(np.float64), \
        (sim["ball_x"] / length).astype(np.float64), (sim["ball_y"] / width).astype(np.float64), \
        sim["ball_speed"].astype(np.float64)


def write_atd_raw(path, sim, progress=None, chunk=4000):
    """``frame:x,y,s;x,y,s;...:x,y,s`` with ``frame`` counting from 1. Missing is ``NaN``."""
    px, py, ps, bx, by, bs = _columns(sim)
    n_frames, n_slots = px.shape
    started = time.time()
    with open(path, "w", newline="\n") as handle:
        for base in range(0, n_frames, chunk):
            stop = min(n_frames, base + chunk)
            rows = int(stop - base)
            # One string per (frame, slot) triple, then joins: the formatter is the cost, so it
            # runs over whole columns rather than inside a nested Python loop.
            cells_x = _format_column(px[base:stop])
            cells_y = _format_column(py[base:stop])
            cells_s = _format_column(ps[base:stop])
            groups = ["{},{},{}".format(cells_x[r, s], cells_y[r, s], cells_s[r, s])
                      for r in range(rows) for s in range(n_slots)]
            view = np.array(groups, dtype=object).reshape(rows, n_slots)
            ball_columns = [_format_column(values[base:stop]).ravel()
                            for values in (bx, by, bs)]
            ball = ["{},{},{}".format(*triple) for triple in zip(*ball_columns)]
            handle.write("".join(
                "{}:{}:{}\n".format(base + r + 1, ";".join(view[r]), ball[r]) for r in range(rows)))
            if progress:
                progress(stop, n_frames, "writing tracking")
    return round(time.time() - started, 1)


def _format_column(values):
    """Vectorised "%.6f" with the file's own ``NaN`` token."""
    rows, cols = values.shape if values.ndim == 2 else (values.shape[0], 1)
    flat = values.reshape(rows * cols)
    text = [MISSING_TOKEN if not np.isfinite(v) else "{:.6f}".format(float(v)) for v in flat]
    return np.array(text, dtype=object).reshape(rows, cols)


def write_homography(path, sim, progress=None, chunk=50000):
    """One row per video frame, valid everywhere (tactical, per DATA.md).

    The nine matrix columns are a real normalised-pitch-to-1920x1080 map with a slow pan, so
    ``starter/load_homography.project`` returns sane pixels if anyone uses it for debugging. The
    reliability columns are the ones R3P actually reads: median 0.24 during play, about 1.5 across
    the break, which is the published behaviour and what the integrity check compares against.
    """
    n_frames = sim["n_frames"]
    rng = np.random.default_rng(sim["seed"] + 7)
    t = np.arange(n_frames, dtype=np.float64)
    pan = 20.0 * np.sin(2 * np.pi * t / (30.0 * sim["fps"]))
    error = np.where(sim["graded"], rng.lognormal(np.log(0.24), 0.35, n_frames),
                     rng.lognormal(np.log(1.5), 0.5, n_frames))
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for base in range(0, n_frames, chunk):
            stop = min(n_frames, base + chunk)
            lines = []
            for frame in range(base, stop):
                offset = pan[frame]
                lines.append("{},1920,0,{:.4f},0,1080,{:.4f},0,0,1,{:.7f},1\n".format(
                    frame, offset, offset * 0.5, error[frame]))
            handle.write("".join(lines))
            if progress:
                progress(stop, n_frames, "writing homography")


def write_smart_tagging(path, sim, progress=None):
    """Phases and set pieces in SportsCode shape, timed in seconds from frame 0.

    Deliberately imperfect, because the delivered tagging is automatic (DATA.md): intervals jitter
    by up to a second around the real stoppage, roughly one phase in twelve names the wrong side,
    and every throw-in carries ``N/A``. ``starter/load_smart_tagging.py`` reads it unchanged.
    """
    fps = sim["fps"]
    graded, state = sim["graded"], sim["state"]
    rng = np.random.default_rng(sim["seed"] + 11)
    instances = []

    root = ET.Element("Content", {"version": "1.0"})
    coverage = ET.SubElement(root, "Coverage")
    placards = ET.SubElement(coverage, "placards")
    placard = ET.SubElement(placards, "placard", {"id": "1"})
    playlist = ET.SubElement(placard, "playlist", {"id": "1", "name": "BAR-ATH"})
    plays = ET.SubElement(playlist, "plays")

    def emit(start_frame, end_frame, code, tags):
        play = ET.SubElement(plays, "play")
        instance = ET.SubElement(play, "instance", {"id": str(len(instances) + 1)})
        ET.SubElement(instance, "start").text = "{:.3f}".format(start_frame / fps)
        ET.SubElement(instance, "end").text = "{:.3f}".format(end_frame / fps)
        ET.SubElement(instance, "code").text = code
        for group, value in tags.items():
            label = ET.SubElement(instance, "label")
            ET.SubElement(label, "group").text = group
            ET.SubElement(label, "text").text = value
        instances.append((int(start_frame), int(end_frame), code, dict(tags)))

    def half_name(frame):
        for _, first, last in sim["blocks"]:
            if first <= frame <= last:
                return "1st Half" if first == sim["blocks"][0][1] else "2nd Half"
        return "1st Half"

    inside = np.flatnonzero(graded)
    if len(inside):
        for span in np.split(inside, np.flatnonzero(np.diff(inside) != 1) + 1):
            edges = [int(span[0])]
            edges += [int(e) for e in
                      np.flatnonzero(np.diff(state[span[0]:span[-1] + 1]) != 0) + span[0] + 1]
            edges.append(int(span[-1]) + 1)
            for index in range(len(edges) - 1):
                start, stop = edges[index], edges[index + 1]
                if stop - start < max(2, int(0.4 * fps)):
                    continue
                code = int(state[start])
                half = half_name(start)
                if code == D:
                    restarting = int(sim["restart"][start])
                    team = (HOME_TEAM_NAME if restarting == H else
                            AWAY_TEAM_NAME if restarting == A else "N/A")
                    kind = str(rng.choice(["Throw-in", "Corner Kick", "Goal Kick", "Free Kick"]))
                    if kind == "Throw-in":
                        team = "N/A"              # as in the real file, where it matters most
                    elif rng.random() < 0.12:
                        team = "N/A"
                    jitter = int(rng.normal(0, 0.4 * fps))
                    emit(max(0, start + jitter), max(1, stop + jitter), "SET PIECES",
                         {"Team": team, "Half": half, "Type": kind, "Side": "Central"})
                else:
                    team = HOME_TEAM_NAME if code == H else AWAY_TEAM_NAME
                    if rng.random() < 0.08:
                        team = AWAY_TEAM_NAME if code == H else HOME_TEAM_NAME
                    emit(start, stop, str(rng.choice(["BUILD UP", "PROGRESSION",
                                                      "BALL IN FINAL THIRD", "PLAYERS IN THE BOX"])),
                         {"Team": team, "Half": half})

        for _, first, _ in sim["blocks"]:
            emit(max(0, first - int(2.5 * fps)), first + int(2.5 * fps), "SET PIECES",
                 {"Team": HOME_TEAM_NAME, "Half": half_name(first), "Type": "Kick Off (Start)",
                  "Side": "Central"})

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    if progress:
        progress(len(instances), len(instances), "writing smart tagging")
    return instances


def write_bundle(out_dir, sim, progress=None, overwrite=False):
    """Write the tactical bundle, the development marker, and the latent labels for self-checks."""
    tactical = os.path.join(out_dir, "tactical")
    os.makedirs(tactical, exist_ok=True)
    stem = official.SOURCES["tactical"]["stem"] if hasattr(official, "SOURCES") else "BAR-ATH_tactical"
    paths = {
        "atd_meta": os.path.join(tactical, stem + "_atd.xml"),
        "atd_raw": os.path.join(tactical, stem + "_atd.txt"),
        "homography": os.path.join(tactical, stem + "_homography.csv"),
        "smart_tagging": os.path.join(tactical, stem + "_smart_tagging.xml"),
    }
    for path in paths.values():
        if os.path.exists(path) and not overwrite:
            raise SystemExit("refusing to overwrite {} -- pass --overwrite".format(path))

    started = time.time()
    write_atd_meta(paths["atd_meta"], sim["team_names"], sim["team_of_slot"])
    seconds = write_atd_raw(paths["atd_raw"], sim, progress)
    write_homography(paths["homography"], sim, progress)
    instances = write_smart_tagging(paths["smart_tagging"], sim, progress)

    counts = {}
    for label in ("H", "A", "D", "X"):
        counts[label] = int(np.sum(sim["labels"] == label))
    marker = {
        "synthetic": True,
        "generated_by": "src/data/synthetic.py",
        "seed": int(sim["seed"]),
        "n_frames": int(sim["n_frames"]),
        "n_smart_tagging_instances": len(instances),
        "write_tracking_seconds": seconds,
        "label_counts": counts,
        "note": ("Simulated match. No human labels and no relation to the real BAR-ATH recording. "
                 "Output computed from this bundle is a code-correctness check, never a "
                 "competition result, and must not be submitted."),
        "elapsed_s": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, SYNTHETIC_MARKER), "w") as handle:
        json.dump(marker, handle, indent=2)

    selfcheck = os.path.join(out_dir, "synthetic_selfcheck")
    os.makedirs(selfcheck, exist_ok=True)
    gt_path = os.path.join(selfcheck, "synthetic_latent_gt.csv")
    with open(gt_path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(sim["labels"]))
    paths["selfcheck_gt"] = gt_path
    return paths
