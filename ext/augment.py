"""Corrupt external tracking to look like the real ATD (M2: ball dropout only; M3 adds the rest).

Calibrated on label-free aggregates of the real match (tools/real_stats.py, graded frames):
  * the ball is lost in rare, long gaps: hazard ~0.0014 per frame with at most one player within
    3 m, 0.0024 with two, 0.0041 with three or more;
  * gap lengths follow the real histogram (bins below, in frames), mostly 5-30 s.
Gaps are inserted independently of the true state. On the real match the ball goes missing mostly
in live play, so a model must not learn "no ball = dead ball", which the Metrica data would teach.
Deterministic for a given seed.
"""

import numpy as np

HAZARD_BY_CROWD = (0.00135, 0.0014, 0.0024, 0.0041)       # 0, 1, 2, 3+ players within 3 m
# (low, high, count) of real missing-gap lengths in frames, graded frames of the real match
GAP_BINS = ((1, 1, 2), (2, 2, 9), (3, 5, 3), (6, 12, 2), (13, 25, 1), (51, 125, 1),
            (126, 250, 53), (251, 750, 60), (751, 2250, 19))


def ball_dropout(tracking, seed: int = 0):
    """In-place: blank the ball over gaps drawn from the real gap process. Returns the share of
    frames newly blanked."""
    rng = np.random.default_rng(seed)
    n = tracking.n_frames
    ball = tracking.ball_xy
    seen = tracking.ball_observed.copy()
    counts = np.array([c for _, _, c in GAP_BINS], dtype=float)
    probs = counts / counts.sum()
    draws = rng.random(n)
    blanked = np.zeros(n, dtype=bool)
    t = 0
    while t < n:
        if seen[t]:
            d = np.hypot(tracking.players_xy[t, :, 0] - ball[t, 0],
                         tracking.players_xy[t, :, 1] - ball[t, 1])
            crowd = min(3, int(np.sum(np.nan_to_num(d, nan=np.inf) < 3.0)))
            if draws[t] < HAZARD_BY_CROWD[crowd]:
                lo, hi, _ = GAP_BINS[rng.choice(len(GAP_BINS), p=probs)]
                length = int(rng.integers(lo, hi + 1))
                blanked[t:t + length] = True
                t += length
                continue
        t += 1
    added = blanked & seen
    ball[blanked] = np.nan
    tracking.ball_speed[blanked] = np.nan
    return float(added.mean())
