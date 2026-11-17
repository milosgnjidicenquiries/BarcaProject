"""Read the ATD (Metrica FIFA/EPTS tracking) into metric arrays.

The delivered file is 199,319 lines of the shape

    frame_id:x,y,s;x,y,s;...;x,y,s:x,y,s

where the leading group is the frame's own counter, the ``;``-separated groups are the 46 track
slots in metadata order, and the trailing group is the ball. ``s`` is speed in m/s. Missing
observations are the literal string ``NaN``. Coordinates are normalised to 0-1 of the pitch, so
every distance R3P uses is scaled by ``games.GAMES["BAR-ATH"]["pitch"]`` = 105.3 x 68.0 m --
skipping that stretches distances along the short axis and picks the wrong nearest player near the
touchlines.

Alignment, per DATA.md and ``starter/load_atd.py``: the ATD is already on the video timeline and
counts from 1 where a submission counts from 0, so ``video_frame = frame_id - 1``. That mapping is
imported from the official module rather than restated.

Why not kloppy: ``dataset.to_df()`` on this file is minutes and gigabytes for four numeric columns
per slot, and the tactical bundle must be readable on a laptop. This loader streams it in chunks.
``src.data.integrity`` cross-checks the result against the official ``check_alignment_raw``.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np

from .. import official

CHUNK_LINES = 8192

# Column layout once the separators are normalised to commas.
BALL_FIELDS = 3


class Tracking:
    """Frame-indexed tracking arrays. Row ``t`` is submission row ``t`` is video frame ``t``.

    Kept as plain float32 arrays rather than a DataFrame: 199,319 x 46 x 3 floats is ~110 MB, and
    the same shape as a pandas object is several times that with no benefit, because every consumer
    here is vectorised numpy.
    """

    def __init__(self, px, py, ps, ball_x, ball_y, ball_speed, team_of_slot, fps, pitch,
                 paths, anomalies):
        self.px = px
        self.py = py
        self.ps = ps
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.ball_speed = ball_speed
        self.team_of_slot = team_of_slot
        self.fps = fps
        self.pitch = pitch
        self.paths = paths
        self.anomalies = anomalies

    def __len__(self):
        return self.px.shape[0]

    @property
    def n_slots(self):
        return self.px.shape[1]

    @property
    def ball_visible(self):
        return ~np.isnan(self.ball_x)

    @property
    def home_slot(self):
        return self.team_of_slot == official.official_baseline.HOME

    @property
    def away_slot(self):
        return self.team_of_slot == official.official_baseline.AWAY

    def visible_mask(self, slots=None):
        mask = ~np.isnan(self.px)
        if slots is not None:
            mask = mask[:, slots]
        return mask

    def slot_speed(self):
        """Per-slot speed as the tracker reported it, NaN where the slot is not seen."""
        return self.ps


def _expected_fields(n_slots):
    return 1 + BALL_FIELDS * (n_slots + 1)


def _slot_columns(n_slots):
    offset = 1
    index = np.arange(n_slots)
    return offset + 3 * index, offset + 3 * index + 1, offset + 3 * index + 2


def _parse_chunk(lines, expected, chunk_start):
    """(n_rows, expected) float64 block, or None when the chunk is ragged.

    A ragged chunk is not an error to guess at -- it is the signal that the file is not what
    DATA.md describes, and the caller falls back to a slower per-line parse that records exactly
    which lines were odd instead of silently dropping them.
    """
    text = "\n".join(lines)
    try:
        block = np.loadtxt(io.StringIO(text), delimiter=",", dtype=np.float64)
    except ValueError:
        return None
    if block.ndim == 1:
        block = block.reshape(1, -1)
    if block.shape[1] != expected:
        return None
    return block


def _parse_line_fallback(line, expected):
    """Per-line parse of an ALREADY separator-normalised line, used only when a chunk is ragged.

    Slow but exact: it records which fields were unreadable instead of guessing, and returns None
    so the caller can count the line as an anomaly rather than drop it silently.
    """
    tokens = line.strip().split(",")
    if len(tokens) != expected:
        return None
    row = np.full(expected, np.nan, dtype=np.float64)
    for index, token in enumerate(tokens):
        if token and token[0] not in "Nn":          # "NaN" is the file's missing marker
            try:
                row[index] = float(token)
            except ValueError:
                return None
    return row


def load_tracking(meta_path=None, raw_path=None, data_dir=None, source="tactical",
                  n_frames=None, fps=None, pitch=None, progress=None):
    """Stream the ATD into frame-indexed metric arrays.

    ``progress`` is an optional ``callback(done_rows, total_rows, message)`` -- the dashboard draws
    a bar from it, and the CLI prints a line every chunk. It never changes the result.
    """
    if meta_path is None or raw_path is None:
        meta_path, raw_path = official.require(["atd_meta", "atd_raw"],
                                               data_dir=data_dir, source=source)
    n_frames = n_frames or official.source_frames(source)
    pitch = pitch or official.PITCH
    fps = fps or official.FPS
    length, width = pitch

    team = official.team_of_each_track(meta_path)
    n_slots = int(team.shape[0])
    expected = _expected_fields(n_slots)
    col_x, col_y, col_s = _slot_columns(n_slots)

    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_s = np.full(n_frames, np.nan, dtype=np.float32)

    anomalies = {"ragged_lines": 0, "out_of_range_frames": 0, "duplicate_frame_ids": 0,
                 "lines": 0, "n_slots": n_slots, "expected_fields": expected,
                 "frame_id_first": None, "frame_id_last": None}
    seen = np.zeros(n_frames, dtype=bool)

    started = time.time()
    lines_buffer = []
    consumed = 0

    def flush(force_fallback=False):
        nonlocal lines_buffer
        if not lines_buffer:
            return
        block = None
        if not force_fallback:
            block = _parse_chunk(lines_buffer, expected, consumed)
        if block is None:
            rows = []
            for line in lines_buffer:
                row = _parse_line_fallback(line, expected)
                if row is None:
                    anomalies["ragged_lines"] += 1
                else:
                    rows.append(row)
            if not rows:
                lines_buffer = []
                return
            block = np.vstack(rows)

        ids = block[:, 0].astype(np.int64)
        index = ids - 1                                # video_frame = frame_id - 1
        in_range = (index >= 0) & (index < n_frames)
        anomalies["out_of_range_frames"] += int(np.sum(~in_range))
        index = index[in_range]
        block = block[in_range]

        repeats = seen[index]
        anomalies["duplicate_frame_ids"] += int(np.sum(repeats))

        px[index] = block[:, col_x].astype(np.float32) * length
        py[index] = block[:, col_y].astype(np.float32) * width
        ps[index] = block[:, col_s].astype(np.float32)
        ball_x[index] = block[:, -3].astype(np.float32) * length
        ball_y[index] = block[:, -2].astype(np.float32) * width
        ball_s[index] = block[:, -1].astype(np.float32)
        seen[index] = True
        lines_buffer = []

    with open(raw_path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            stripped = line.strip().replace(";", ",").replace(":", ",")
            if not lines_buffer:
                anomalies["frame_id_first"] = int(float(stripped.split(",", 1)[0]))
            lines_buffer.append(stripped)
            anomalies["lines"] += 1
            if len(lines_buffer) >= CHUNK_LINES:
                flush()
                consumed += CHUNK_LINES
                if progress:
                    progress(min(consumed, n_frames), n_frames, "reading tracking")

    flush()
    if progress:
        progress(n_frames, n_frames, "reading tracking")

    last_index = int(np.flatnonzero(seen)[-1]) if seen.any() else -1
    anomalies["frame_id_last"] = last_index + 1
    anomalies["frames_missing_from_tracking"] = int(n_frames - seen.sum())
    anomalies["seconds"] = round(time.time() - started, 2)
    anomalies["allocated_frames"] = int(n_frames)

    # Trim to the frames that exist rather than padding to the length the metadata claims. A short
    # file then cannot manufacture 193k silent frames inside the model; it becomes a row-count
    # mismatch the integrity check reports and every downstream array agrees about.
    rows = max(int(anomalies["lines"]), last_index + 1)
    if 0 < rows < n_frames:
        px, py, ps = px[:rows], py[:rows], ps[:rows]
        ball_x, ball_y, ball_s = ball_x[:rows], ball_y[:rows], ball_s[:rows]
        seen = seen[:rows]
        anomalies["frames_missing_from_tracking"] = 0

    anomalies["n_frames"] = int(len(seen))
    anomalies["ball_presence"] = float(seen.mean()) if len(seen) else 0.0

    return Tracking(px, py, ps, ball_x, ball_y, ball_s, team, fps, pitch,
                    {"atd_meta": meta_path, "atd_raw": raw_path}, anomalies)


def ball_presence(tracking):
    """Fraction of frames where the tracking reports a ball position (DATA.md: 56.9% overall)."""
    return float(np.mean(tracking.ball_visible))
