"""Per-frame, identity-free features from a Tracking (metres, m/s, NaN-safe).

Slots are tracklets, not people, so every feature is collective: per-team minima, medians and
centroids over whatever slots are tracked on a frame. Nothing follows a slot through time.

Ball: on the real match the ball is missing on 41% of graded frames, in long gaps (median about
11 s). Gaps of up to BALL_INTERP_S are linearly interpolated (and flagged); longer gaps leave the
ball unknown, and the team call then rests on shape and on who touched the ball last.
"""

import warnings
from dataclasses import dataclass

import numpy as np

BALL_INTERP_S = 1.0
TOUCH_M = 1.5          # a player this close to the ball is touching it
TOUCH_MARGIN_M = 0.5   # ...and closer than the nearest opponent by this much
SMOOTH_S = 1.0


@dataclass
class Features:
    ball_xy: np.ndarray          # (n, 2) observed or interpolated, NaN if unknown
    ball_observed: np.ndarray    # (n,) bool
    ball_known: np.ndarray       # (n,) bool, observed or interpolated
    ball_speed: np.ndarray       # (n,) m/s, NaN if unknown
    ball_missing_s: np.ndarray   # (n,) seconds since the ball was last observed (0 if observed)
    d_home: np.ndarray           # (n,) nearest home player to the ball, m (NaN if unknown)
    d_away: np.ndarray
    out_dist: np.ndarray         # (n,) metres outside the pitch (negative inside), NaN unknown
    player_speed: np.ndarray     # (n,) median tracked-player speed, m/s, smoothed over SMOOTH_S
    last_touch: np.ndarray       # (n,) +1 home, -1 away, 0 none yet (forward-filled)
    since_touch_s: np.ndarray    # (n,) seconds since that touch (inf if none)
    shape_diff: np.ndarray       # (n,) home advance minus away advance, m (NaN if unknown)


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())


def interpolate_ball(ball: np.ndarray, observed: np.ndarray, max_gap: int):
    """Linear interpolation over gaps of at most max_gap frames with a seen ball on both sides."""
    out = ball.copy()
    interp = np.zeros(len(ball), dtype=bool)
    for start, stop in _runs(~observed):
        if start == 0 or stop >= len(ball) or stop - start > max_gap:
            continue
        a, b = ball[start - 1], ball[stop]
        w = (np.arange(1, stop - start + 1) / float(stop - start + 1))[:, None]
        out[start:stop] = a + (b - a) * w
        interp[start:stop] = True
    return out, interp


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average ignoring NaN (NaN where the whole window is NaN)."""
    if width <= 1:
        return x
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    kernel = np.ones(width)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def attack_directions(tracking, spec):
    """{(team, half): +1 if the team attacks towards larger x in that half, else -1}.

    A team's average centroid over a whole half lies in its own half, so it attacks the other way.
    """
    length = tracking.pitch[0]
    out = {}
    observed = tracking.player_observed
    for half, first, last in spec.halves:
        for team in (0, 1):
            cols = tracking.slot_team == team
            x = tracking.players_xy[first:last + 1][:, cols, 0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_x = np.nanmean(x) if observed[first:last + 1][:, cols].any() else length / 2
            out[(team, half)] = -1 if mean_x > length / 2 else 1
    return out


def compute(tracking, spec) -> Features:
    fps = spec.fps
    n = tracking.n_frames
    length, width = tracking.pitch
    observed = tracking.ball_observed
    ball, interp = interpolate_ball(tracking.ball_xy, observed, int(round(BALL_INTERP_S * fps)))
    known = observed | interp

    # ball speed from positions (the raw ball speed channel is unreliable), +/- 2 frame difference
    speed = np.full(n, np.nan)
    speed[2:-2] = np.hypot(ball[4:, 0] - ball[:-4, 0], ball[4:, 1] - ball[:-4, 1]) * fps / 4.0
    speed[~known] = np.nan

    last_seen = np.where(observed, np.arange(n), -1)
    last_seen = np.maximum.accumulate(last_seen)
    missing_s = np.where(last_seen >= 0, (np.arange(n) - last_seen) / fps, np.inf)

    # nearest player per team (chunked: 46 slots x 200k frames)
    d_team = np.full((2, n), np.nan)
    chunk = 20000
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        d = np.hypot(tracking.players_xy[lo:hi, :, 0] - ball[lo:hi, None, 0],
                     tracking.players_xy[lo:hi, :, 1] - ball[lo:hi, None, 1])
        d = np.where(np.isnan(d), np.inf, d)
        for team in (0, 1):
            m = d[:, tracking.slot_team == team].min(axis=1)
            d_team[team, lo:hi] = np.where(np.isfinite(m), m, np.nan)
    d_home, d_away = d_team

    with np.errstate(invalid="ignore"):
        out_dist = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1],
                                      ball[:, 1] - width])

    # players' median speed (positions, +/- 1 frame), smoothed over SMOOTH_S
    p = tracking.players_xy
    step = np.full(p.shape[:2], np.nan)
    step[1:-1] = np.hypot(p[2:, :, 0] - p[:-2, :, 0], p[2:, :, 1] - p[:-2, :, 1]) * fps / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(step, axis=1)
    player_speed = _smooth(med, int(round(SMOOTH_S * fps)))

    # last touch: a player within TOUCH_M of the ball and clearly nearer than any opponent
    with np.errstate(invalid="ignore"):
        home_touch = (d_home < TOUCH_M) & (d_home + TOUCH_MARGIN_M < np.nan_to_num(d_away, nan=np.inf))
        away_touch = (d_away < TOUCH_M) & (d_away + TOUCH_MARGIN_M < np.nan_to_num(d_home, nan=np.inf))
    touch = np.where(home_touch, 1, np.where(away_touch, -1, 0))
    idx = np.where(touch != 0, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    last_touch = np.where(idx >= 0, touch[np.maximum(idx, 0)], 0)
    since_touch = np.where(idx >= 0, (np.arange(n) - idx) / fps, np.inf)

    # shape: how far each team's centroid has pushed towards the goal it attacks
    shape_diff = np.full(n, np.nan)
    directions = attack_directions(tracking, spec)
    observed_p = tracking.player_observed
    for half, first, last in spec.halves:
        sl = slice(first, last + 1)
        adv = []
        for team in (0, 1):
            cols = tracking.slot_team == team
            obs = observed_p[sl][:, cols]
            count = obs.sum(axis=1)
            cx = np.where(obs, p[sl][:, cols, 0], 0.0).sum(axis=1) / np.maximum(count, 1)
            a = (cx - length / 2.0) * directions[(team, half)]
            adv.append(np.where(count >= 3, a, np.nan))
        shape_diff[sl] = _smooth(adv[0] - adv[1], int(round(SMOOTH_S * fps)))

    return Features(ball_xy=ball, ball_observed=observed, ball_known=known, ball_speed=speed,
                    ball_missing_s=missing_s, d_home=d_home, d_away=d_away, out_dist=out_dist,
                    player_speed=player_speed, last_touch=last_touch, since_touch_s=since_touch,
                    shape_diff=shape_diff)
