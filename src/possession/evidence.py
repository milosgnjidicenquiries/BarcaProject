"""R3P component A (reliability-weighted evidence fusion) and component B (dead-ball head).

Two decisions are kept separate on purpose:

* **who has it**, from five blocks of contemporaneous evidence about the ball and the shape of the
  two teams;
* **whether the ball is in play at all**, from its own cues.

Emissions are then ``P(D) = p_dead``, ``P(H) = (1 - p_dead) q``, ``P(A) = (1 - p_dead)(1 - q)``.
Conflating them is what makes a missing ball read as a dead ball, and a missing ball happens on 41%
of graded frames (DATA.md) -- so the separation is not elegance, it is survival.

Nothing here reads a label. The only quantities estimated from data are the median/MAD
standardisation statistics and the temperature, both computed on the unlabelled match.
"""

from __future__ import annotations

import numpy as np

from .. import official

H, A, D, X = 0, 1, 2, 3
CHUNK = 4096
EPS = 1e-9
# The baseline's third team value: a slot that is neither side, usually an official or a steward.
# Never counted as a duplicate of a player, and never an answer to "who has it".
NOBODY = -1


def _duplicate_pairs(x_window, y_window, visible, side, radius):
    """Near-duplicate same-side slot pairs per frame -- the tracking's own self-reported doubt.

    A duplicated tracklet is the one anomaly that changes a decision rather than only weakening it:
    two records of one player double that side's vote in `support` and halve its nearest distance for
    free. Comparing slots with each other, and only within a side, is what makes the count mean
    duplication rather than "two opponents are near one another", which is a duel and is evidence.

    ``x_window``/``y_window`` are metres -- ``src.data.atd`` scales the file's 0-1 coordinates by the
    pitch when it loads them, so applying the pitch again here would look for duplicates within a
    centimetre and find none.
    """
    delta_x = x_window[:, :, None] - x_window[:, None, :]
    delta_y = y_window[:, :, None] - y_window[:, None, :]
    within = (delta_x * delta_x + delta_y * delta_y) <= radius * radius
    same_side = side[:, None] == side[None, :]
    paired = (within & same_side & (side[:, None] != NOBODY)
              & visible[:, :, None] & visible[:, None, :])
    counts = paired.sum(axis=2) - 1          # a slot is never a duplicate of itself
    return np.maximum(counts, 0).sum(axis=1) / 2.0


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _rollmean(values, window, valid=None):
    """Centred moving average that ignores missing frames instead of treating them as zero.

    A ball the tracker lost is not a ball with speed 0; averaging it as one would manufacture
    dead-ball evidence out of an occlusion.
    """
    window = max(1, int(window))
    mask = np.isfinite(values) if valid is None else (valid & np.isfinite(values))
    x = np.where(mask, values, 0.0).astype(np.float64)
    w = mask.astype(np.float64)
    top = np.concatenate(([0.0], np.cumsum(x)))
    bottom = np.concatenate(([0.0], np.cumsum(w)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window // 2)
    high = np.minimum(len(values), index + window // 2 + 1)
    numerator = top[high] - top[low]
    denominator = bottom[high] - bottom[low]
    out = np.full(len(values), np.nan)
    seen = denominator > 0
    out[seen] = numerator[seen] / denominator[seen]
    return out


def _gap_since(seen):
    """Frames since the last True. inf before the first observation."""
    index = np.arange(len(seen), dtype=np.float64)
    last = np.maximum.accumulate(np.where(seen, index, -1.0))
    return np.where(last < 0, np.inf, index - last)


def _robust_standardise(values, inside):
    """(x - median) / (1.4826 * MAD), measured over ``inside`` and only over finite frames.

    Median and MAD because these distributions are heavy-tailed by nature: one duplicated tracklet
    or ball mis-detect would otherwise set the scale of the whole feature, and a scale set by
    outliers carries no meaning. Frames with no value are set to 0 -- "this channel is silent here"
    -- but only AFTER the statistics, so a NaN never counts as a zero and never moves the median.
    """
    finite = np.isfinite(values) & inside
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64), 0.0, 1.0
    sample = values[finite].astype(np.float64)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median))) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-9:
        mad = float(np.std(sample)) or 1.0
    standard = (values - median) / mad
    return np.where(np.isfinite(standard), standard, 0.0), median, mad


def _interpolate_ball(x, y, seen, max_gap):
    """Bridge short observation gaps linearly, and say how far each frame is from real data.

    Only gaps with observations at both ends are bridged. A gap at the end of a half is left
    unestimated rather than projected off a cliff.
    """
    n = len(x)
    ex, ey = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    index = np.flatnonzero(seen)
    if index.size < 2:
        return ex, ey

    grid = np.arange(n, dtype=np.float64)
    filled_x = np.interp(grid, index, x[index])
    filled_y = np.interp(grid, index, y[index])
    gap = _gap_since(seen)

    # Nearest observation ahead of each frame. A suffix minimum of "own index where observed, n
    # elsewhere" is that in one pass; the naive version that takes the minimum over the observed
    # indices alone collapses every frame to the first observation and bridges unbridgable gaps.
    ahead = np.full(n, float(n))
    ahead[index] = index
    ahead = np.minimum.accumulate(ahead[::-1])[::-1]

    bridgeable = ~seen & (gap <= max_gap) & ((ahead - grid) <= max_gap)
    ex = np.where(bridgeable, filled_x, x)
    ey = np.where(bridgeable, filled_y, y)
    return ex, ey


# ---------------------------------------------------------------------------------------------
# component A
# ---------------------------------------------------------------------------------------------

def ball_channel(tracking, cfg, graded, homography_confidence=None):
    """Effective ball position and, more importantly, how much to trust it per frame."""
    reliability = cfg["fusion"]["reliability"]
    bx = np.where(np.isnan(tracking.ball_x), np.nan, tracking.ball_x).astype(np.float64)
    by = np.where(np.isnan(tracking.ball_y), np.nan, tracking.ball_y).astype(np.float64)
    seen = np.isfinite(bx) & np.isfinite(by)
    max_gap = float(reliability["ball_gap_max_frames"])
    ex, ey = _interpolate_ball(bx, by, seen, max_gap)
    gap = _gap_since(seen)

    rho = np.where(seen, 1.0, np.exp(-np.minimum(gap, 1e6) / float(reliability["ball_gap_tau_frames"])))
    rho = np.where(gap > max_gap, 0.0, rho)
    if homography_confidence is not None:
        confidence = np.asarray(homography_confidence)[:len(rho)]
        rho = rho * np.clip(confidence, 0.0, 1.0) ** 0.25
    return {"x": ex, "y": ey, "seen": seen, "gap": gap, "rho": rho,
            "raw_x": bx, "raw_y": by}


def _geometric_ball(tr, ball, cfg, rows_slice):
    """Where the ball probably is when the tracker has lost it: the duel.

    Two opposing players standing on top of one another are almost always competing for the ball,
    so the closest home-away pair brackets it. This is the only channel available on frames where
    ``rho`` has decayed to zero, which is a large share of the 41% the ball is missing from, and it
    is deliberately trusted at ``geo_impute_rho`` rather than as an observation.
    """
    home = tr.home_slot
    away = tr.away_slot
    px = tr.px[rows_slice]
    py = tr.py[rows_slice]
    visible = np.isfinite(px) & np.isfinite(py)
    home_xy = np.stack([np.where(visible & home, px, np.nan),
                        np.where(visible & home, py, np.nan)], axis=-1)
    away_xy = np.stack([np.where(visible & away, px, np.nan),
                        np.where(visible & away, py, np.nan)], axis=-1)
    h_index = np.flatnonzero(home)
    a_index = np.flatnonzero(away)
    hx = home_xy[:, h_index, :]
    ax = away_xy[:, a_index, :]
    dx = hx[:, :, None, :] - ax[:, None, :, :]
    distance = np.sqrt(np.nansum(dx * dx, axis=-1))
    distance = np.where(np.isfinite(distance), distance, np.inf)
    flat = distance.reshape(distance.shape[0], -1)
    best = np.argmin(flat, axis=1)
    rows = np.arange(flat.shape[0])
    pair_h = best // ax.shape[1]
    pair_a = best % ax.shape[1]
    nearest_home = hx[rows, pair_h]
    nearest_away = ax[rows, pair_a]
    closest = np.nanmin(flat, axis=1)
    mid = 0.5 * (nearest_home + nearest_away)
    usable = np.isfinite(closest) & (closest < 12.0)
    return mid[:, 0], mid[:, 1], usable, closest


def evidence(tracking, cfg, graded, ball, tagging=None, homography_confidence=None,
             progress=None):
    """Every per-frame feature, as a signed quantity where positive favours the home side."""
    fusion = cfg["fusion"]
    reliability = fusion["reliability"]
    n_frames = len(tracking)
    length, width = tracking.pitch
    fps = tracking.fps
    sigma = float(fusion["sigma_control_m"])
    sigma_support = float(fusion["sigma_support_m"])

    px, py, ps = tracking.px, tracking.py, tracking.ps
    home, away = tracking.home_slot, tracking.away_slot
    bx, by, rho = ball["x"], ball["y"], ball["rho"]

    blocks = {name: np.zeros(n_frames) for name in
              ("nearest", "soft_proximity", "support", "ball_kinematics", "territory",
               "continuity", "tagging")}
    dead = {name: np.zeros(n_frames) for name in
            ("static", "moving", "oob", "missing", "collapse", "setpiece", "wall")}
    n_visible = np.zeros(n_frames)
    duplicates = np.zeros(n_frames)
    nearest_team = np.full(n_frames, -1, dtype=np.int8)
    margin = np.zeros(n_frames)

    # ---- ball kinematics -------------------------------------------------------------------
    seen = ball["seen"]
    index = np.flatnonzero(seen)
    ball_vx = np.zeros(n_frames)
    ball_vy = np.zeros(n_frames)
    if len(index) > 2:
        earlier, later = index[:-1], index[1:]
        dt = np.maximum(later - earlier, 1).astype(np.float64) / fps
        vx = np.clip((ball["raw_x"][later] - ball["raw_x"][earlier]) / dt, -40, 40)
        vy = np.clip((ball["raw_y"][later] - ball["raw_y"][earlier]) / dt, -40, 40)
        ball_vx[later] = vx
        ball_vy[later] = vy
    # Averaged over observed frames only: a frame the ball was lost on is not a frame the ball
    # stood still on, and letting the zeros into the mean manufactures stoppages out of occlusions.
    ball_speed = np.hypot(_rollmean(ball_vx, 5, valid=seen), _rollmean(ball_vy, 5, valid=seen))

    # ---- which way each team attacks, read off the unlabelled tracking ----------------------
    # A team's centroid sits in the half it defends most of the time, so the direction it attacks
    # is away from there. [derived]: no labels, and no assumption about which side is which.
    def centroid(side):
        values = np.full(n_frames, np.nan)
        for base in range(0, n_frames, CHUNK):
            stop = min(n_frames, base + CHUNK)
            mask = np.isfinite(px[base:stop]) & side[None, :]
            totals = np.where(mask, px[base:stop], 0.0).sum(axis=1)
            counts = np.maximum(mask.sum(axis=1), 1)
            values[base:stop] = np.where(mask.any(axis=1), totals / counts, np.nan)
        return values

    home_centre, away_centre = centroid(home), centroid(away)

    def _side_mean(centres):
        values = np.where(graded, centres, np.nan) if graded.any() else np.full(n_frames, np.nan)
        # An empty slice is not a direction, it is no information: fall back to the middle line
        # rather than letting nanmean warn and return NaN, which then compares False everywhere.
        return float(np.nanmean(values)) if np.isfinite(values).any() else length / 2

    home_mean, away_mean = _side_mean(home_centre), _side_mean(away_centre)
    direction_home = 1.0 if home_mean < away_mean else -1.0

    # ---- per-frame proximity ----------------------------------------------------------------
    for base in range(0, n_frames, CHUNK):
        stop = min(n_frames, base + CHUNK)
        rows = slice(base, stop)
        x_window, y_window = px[rows], py[rows]
        visible = np.isfinite(x_window) & np.isfinite(y_window)

        # d = sqrt((105.3*dx)^2 + (68.0*dy)^2). dx, dy are already normalised here only if the
        # loader left them so; this loader stores metres, which is the same product.
        dx = x_window - bx[rows, None]
        dy = y_window - by[rows, None]
        distance = np.where(visible & np.isfinite(dx) & np.isfinite(dy),
                            np.sqrt(dx * dx + dy * dy), np.inf)
        kernel = np.exp(-distance * distance / (2.0 * sigma * sigma))
        kernel = np.where(np.isfinite(distance), kernel, 0.0)

        side = np.where(home, 0, np.where(away, 1, NOBODY))
        home_visible = visible & home[None, :]
        away_visible = visible & away[None, :]
        soft_home = (kernel * home[None, :]).sum(axis=1) / np.maximum(home_visible.sum(axis=1), 1)
        soft_away = (kernel * away[None, :]).sum(axis=1) / np.maximum(away_visible.sum(axis=1), 1)
        nearest_home = np.where(home_visible, distance, np.inf).min(axis=1)
        nearest_away = np.where(away_visible, distance, np.inf).min(axis=1)
        support_home = np.maximum((home_visible & (distance <= sigma_support)).sum(axis=1) - 1, 0)
        support_away = np.maximum((away_visible & (distance <= sigma_support)).sum(axis=1) - 1, 0)

        # A side with nobody visible gets a finite sentinel rather than inf, or the subtraction
        # below becomes inf - inf and the whole run is one long numpy warning.
        sentinel = 4.0 * length
        near_home = np.where(np.isfinite(nearest_home), nearest_home, sentinel)
        near_away = np.where(np.isfinite(nearest_away), nearest_away, sentinel)
        both_sides = np.isfinite(nearest_home) & np.isfinite(nearest_away)
        blocks["nearest"][rows] = np.tanh((near_away - near_home) / (2.0 * sigma))
        # log-ratio, not difference: S_home and S_away share the same kernel, so the ratio cancels
        # the part of them that is "the ball was found near somebody" and keeps "near how many".
        blocks["soft_proximity"][rows] = np.clip(np.log((soft_home + EPS) / (soft_away + EPS)),
                                                 -4.0, 4.0)
        blocks["support"][rows] = np.tanh((support_home - support_away) / 2.0)

        along = ball_vx[rows] * direction_home
        in_motion = 1.0 - np.exp(-np.nan_to_num(ball_speed[rows]) / 3.0)
        blocks["ball_kinematics"][rows] = np.tanh(along / 12.0) * in_motion
        blocks["territory"][rows] = np.clip(
            (bx[rows] - length / 2) / (length / 2), -1, 1) * direction_home

        n_visible[rows] = visible.sum(axis=1)
        duplicates[rows] = _duplicate_pairs(x_window, y_window, visible, side,
                                            float(reliability["duplicate_slot_radius_m"]))
        nearest_team[rows] = np.where(both_sides, np.where(near_home < near_away, 0, 1),
                                      -1).astype(np.int8)
        margin[rows] = np.where(both_sides, np.abs(near_home - near_away), np.nan)

        inside_pitch = ((ball["raw_x"][rows] > -cfg["deadball"]["oob_margin_m"]) &
                        (ball["raw_x"][rows] < length + cfg["deadball"]["oob_margin_m"]) &
                        (ball["raw_y"][rows] > -cfg["deadball"]["oob_margin_m"]) &
                        (ball["raw_y"][rows] < width + cfg["deadball"]["oob_margin_m"]))
        dead["oob"][rows] = np.where(seen[rows], (~inside_pitch).astype(float), 0.0)
        crowd = ((distance <= cfg["deadball"]["wall_cluster_radius_m"]) & visible).sum(axis=1)
        dead["wall"][rows] = np.clip((crowd - cfg["deadball"]["wall_cluster_players"] + 2) / 5.0,
                                     0.0, 1.0)

    # ---- geometric recovery of frames with no ball at all ------------------------------------
    starved = np.flatnonzero((rho <= 1e-3) & graded & (n_visible >= 4))
    if len(starved):
        for base in range(0, len(starved), CHUNK):
            picked = starved[base:base + CHUNK]
            mid_x, mid_y, usable, _ = _geometric_ball(tracking, ball, cfg, picked)
            bx[picked] = np.where(usable, mid_x, bx[picked])
            by[picked] = np.where(usable, mid_y, by[picked])
            rho[picked] = np.where(usable, float(reliability["geo_impute_rho"]), rho[picked])
        # proximity has to be recomputed on the imputed positions for exactly those frames
        for base in range(0, len(starved), CHUNK):
            picked = starved[base:base + CHUNK]
            rows = picked
            x_window, y_window = px[rows], py[rows]
            visible = np.isfinite(x_window) & np.isfinite(y_window)
            dx = x_window - bx[rows, None]
            dy = y_window - by[rows, None]
            distance = np.where(visible, np.sqrt(np.where(np.isfinite(dx * dx + dy * dy),
                                                          dx * dx + dy * dy, np.inf)), np.inf)
            home_visible = visible & home[None, :]
            away_visible = visible & away[None, :]
            nearest_home = np.where(home_visible, distance, np.inf).min(axis=1)
            nearest_away = np.where(away_visible, distance, np.inf).min(axis=1)
            kernel = np.exp(-distance * distance / (2.0 * sigma * sigma))
            kernel = np.where(np.isfinite(distance), kernel, 0.0)
            soft_home = (kernel * home[None, :]).sum(axis=1) / np.maximum(home_visible.sum(axis=1), 1)
            soft_away = (kernel * away[None, :]).sum(axis=1) / np.maximum(away_visible.sum(axis=1), 1)
            both_sides = np.isfinite(nearest_home) & np.isfinite(nearest_away)
            sentinel = 4.0 * length
            near_home = np.where(both_sides, nearest_home, sentinel)
            near_away = np.where(both_sides, nearest_away, sentinel)
            blocks["nearest"][rows] = np.where(both_sides,
                                               np.tanh((near_away - near_home) / (2.0 * sigma)), 0.0)
            blocks["soft_proximity"][rows] = np.clip(np.log((soft_home + EPS) / (soft_away + EPS)),
                                                    -4.0, 4.0)
            nearest_team[rows] = np.where(both_sides, np.where(near_home < near_away, 0, 1),
                                          -1).astype(np.int8)
            margin[rows] = np.where(both_sides, np.abs(near_home - near_away), np.nan)

    # ---- continuity: an observation statistic, never the decoded label ------------------------
    # Using the previous *prediction* here would double-count the temporal prior that component C
    # already applies, which is the classic way to build a segmenter that never changes its mind.
    vote = np.where(rho > 0.25, np.where(nearest_team == 0, 1.0,
                                         np.where(nearest_team == 1, -1.0, 0.0)), np.nan)
    blocks["continuity"] = np.nan_to_num(_rollmean(
        vote, fusion["temporal_smooth_frames"] * 2 + 1))

    for name in ("nearest", "soft_proximity", "support"):
        blocks[name] = blocks[name] * rho

    # ---- dead-ball cues -----------------------------------------------------------------------
    # Both motion cues are averaged over the frames where the ball was actually seen. Averaging over
    # all frames instead makes a dropout look like a stationary ball, which is the exact confusion
    # component B exists to avoid -- and it makes the two cues sum to less than one wherever tracking
    # is bad, so a badly-tracked live phase would read as neither dead nor alive.
    threshold = cfg["deadball"]["static_speed_mps"]
    window = cfg["deadball"]["static_window_frames"]
    still_flag = (np.nan_to_num(ball_speed) < threshold).astype(float)
    moving_flag = (np.nan_to_num(ball_speed) > threshold).astype(float)
    dead["static"] = np.nan_to_num(_rollmean(still_flag, window, valid=seen))
    dead["moving"] = np.nan_to_num(_rollmean(moving_flag, window, valid=seen))
    dead["missing"] = np.clip((ball["gap"] - cfg["deadball"]["missing_gap_frames"]) / 100.0, 0, 1)
    # Median over the slots that are actually seen (a position AND a reported speed), zeros and all:
    # nanmedian, not nan_to_num-then-median. Passing the missing slots through as 0 put a fake
    # "everybody is standing still" into the median of every frame with a few tracklets down, which
    # is most of a match. A frame with nothing seen stays NaN, and is dropped from the mean below
    # rather than being asked to vote -- nanmedian of an all-NaN row only warns and tells nothing.
    player_speed = np.where(np.isfinite(ps) & np.isfinite(px), ps, np.nan)
    collapse = np.full(n_frames, np.nan)
    for base in range(0, n_frames, CHUNK):
        stop = min(n_frames, base + CHUNK)
        window_rows = np.flatnonzero(np.isfinite(player_speed[base:stop]).any(axis=1))
        if len(window_rows):
            collapse[base + window_rows] = np.nanmedian(
                player_speed[base:stop][window_rows], axis=1)
    dead["collapse"] = np.clip(1.0 - _rollmean(collapse, window) / 2.0, 0.0, 1.0)
    dead["collapse"] = np.nan_to_num(dead["collapse"])
    if tagging is not None:
        dead["setpiece"] = np.where(tagging.setpiece, 1.0, 0.0)
    blocks["tagging"] = tagging_difference(tagging, n_frames)

    min_players = float(reliability["min_players_for_evidence"])
    frame_reliability = np.clip(
        rho * np.clip(n_visible / min_players, 0.0, 1.0) *
        (1.0 - 0.25 * np.clip(duplicates / 6.0, 0.0, 1.0)), 0.0, 1.0)

    if progress:
        progress(n_frames, n_frames, "fused evidence")
    return {"blocks": blocks, "dead": dead, "rho": frame_reliability, "ball_rho": rho,
            "nearest_team": nearest_team, "margin": margin, "ball_speed": ball_speed,
            "n_visible": n_visible, "ball": ball,
            "meta": {"direction_home": direction_home, "sigma_control_m": sigma,
                     "geometric_frames": int(len(starved))}}


def tagging_difference(tagging, n_frames):
    """Smart Tagging as a signed, bounded, deliberately weak prior."""
    if tagging is None:
        return np.zeros(n_frames)
    value = np.zeros(n_frames)
    value[tagging.phase_team == official.official_baseline.HOME] = 1.0
    value[tagging.phase_team == official.official_baseline.AWAY] = -1.0
    restart = np.zeros(n_frames)
    restart[tagging.restart_team == official.official_baseline.HOME] = 1.0
    restart[tagging.restart_team == official.official_baseline.AWAY] = -1.0
    anchored = tagging.restart_strength > 0.3
    return np.clip(0.6 * value + 0.4 * restart * anchored, -1.0, 1.0)


# ---------------------------------------------------------------------------------------------
# emissions
# ---------------------------------------------------------------------------------------------

def dead_coefficients(cfg):
    """Per dead-ball cue: its coefficient in the logit, signed. One table, used by fuse and by the
    channel audit, so a reported weight can never differ from an applied one."""
    deadball = cfg["deadball"]
    return {"static": float(deadball["static_weight"]),
            # `moving` enters with a minus: a ball that is travelling is in play, which is the only
            # positive evidence for the alive side this head has. Without it the head is a one-sided
            # accumulator of dead cues over a negative prior, so p_dead sits near 0.5 all match and D
            # wins on emission alone, everywhere.
            "moving": -float(deadball["alive_motion_weight"]),
            "oob": float(deadball["oob_weight"]),
            "missing": float(deadball["missing_weight"]),
            "collapse": float(deadball["motion_collapse_weight"]),
            "setpiece": float(deadball["tagging_weight"]),
            "wall": float(deadball["wall_weight"])}


def fuse(cfg, features, graded):
    """Weight the blocks, split alive/dead from home/away, and return log-emissions."""
    weights = cfg["fusion"]["weights"]
    blocks = features["blocks"]
    n_frames = len(graded)

    missing_blocks = [name for name in weights if name not in blocks]
    if missing_blocks:
        # A weight whose block does not exist is not a small weight, it is a channel that never
        # speaks -- and the run looks normal from the outside. Say so rather than summing without it.
        raise KeyError("fusion.weights names {}, which evidence does not produce; blocks are {}".format(
            sorted(missing_blocks), sorted(blocks)))

    delta = np.zeros(n_frames)
    standardisation = {}
    for name, weight in weights.items():
        raw = blocks[name]
        if not np.any(np.isfinite(raw)):
            continue
        standard, median, mad = _robust_standardise(raw, graded)
        delta += float(weight) * np.clip(standard, -4.0, 4.0)
        standardisation[name] = {"median": median, "mad": mad, "weight": float(weight)}

    dead = features["dead"]
    coefficients = dead_coefficients(cfg)
    unproduced = [name for name in coefficients if name not in dead]
    if unproduced:
        raise KeyError("dead-ball coefficients name {}, which evidence does not produce; cues are {}"
                       .format(sorted(unproduced), sorted(dead)))
    scale = float(cfg["deadball"]["logit_scale"])
    z_dead = np.full(n_frames, float(cfg["deadball"]["prior_logodds"]))
    for name, coefficient in coefficients.items():
        z_dead += coefficient * np.nan_to_num(dead[name]) * scale

    # An unobservable frame is not a frame with a weak opinion about the teams; it is a frame with
    # no opinion at all. Damping toward 0 keeps the decoder in charge of what to say there.
    delta = delta * (0.35 + 0.65 * features["rho"])

    temperature = _calibrate(delta, graded, cfg)
    q = 1.0 / (1.0 + np.exp(-delta / max(temperature, 1e-6)))
    p_dead = 1.0 / (1.0 + np.exp(-z_dead))

    log_emission = np.empty((n_frames, 4))
    log_emission[:, D] = np.log(np.clip(p_dead, EPS, 1.0))
    alive = np.clip(1.0 - p_dead, EPS, 1.0)
    log_emission[:, H] = np.log(alive) + np.log(np.clip(q, EPS, 1.0))
    log_emission[:, A] = np.log(alive) + np.log(np.clip(1.0 - q, EPS, 1.0))
    log_emission[:, X] = np.log(EPS)
    return {"log_emission": log_emission, "q": q, "p_dead": p_dead, "delta": delta,
            "temperature": temperature, "standardisation": standardisation,
            "dead_coefficients": coefficients}


def _calibrate(delta, graded, cfg):
    """Bisection for the temperature whose median confident frame reaches the configured one.

    Without it the fusion weights would secretly be deciding how often the decoder may change its
    mind, and that is a temporal question that belongs to component C.
    """
    calibration = cfg["calibration"]
    target = float(calibration["target_median_confidence"])
    sample = delta[graded]
    if not len(sample):
        return 1.0
    low, high = float(calibration["min_temperature"]), float(calibration["max_temperature"])

    def median_confidence(temperature):
        q = 1.0 / (1.0 + np.exp(-sample / temperature))
        return float(np.median(np.maximum(q, 1.0 - q)))

    if median_confidence(high) > target:
        return high
    if median_confidence(low) < target:
        return low
    for _ in range(40):
        middle = 0.5 * (low + high)
        if median_confidence(middle) < target:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def restart_modulation(tagging, n_frames, cfg):
    """log-odds added to the D -> H and D -> A transitions, per frame. All zero without tagging."""
    out = np.zeros((n_frames, 4))
    if tagging is None or not cfg["decoder"]["restart"]["enabled"]:
        return out
    strength = float(cfg["decoder"]["restart"]["strength"])
    weight = np.clip(tagging.restart_strength, 0.0, 1.0) * strength
    home = tagging.restart_team == official.official_baseline.HOME
    away = tagging.restart_team == official.official_baseline.AWAY
    out[home, H] += weight[home]
    out[home, A] -= 0.5 * weight[home]
    out[away, A] += weight[away]
    out[away, H] -= 0.5 * weight[away]
    return out
