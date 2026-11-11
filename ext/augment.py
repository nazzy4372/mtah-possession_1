"""Corrupt external tracking to look like the real ATD, calibrated on label-free real aggregates.

Targets are the graded-frame aggregates of the real match (tools/real_stats.py, run by the human):

  ball      40.7% missing, in 150 long gaps (median ~11 s); loss hazard 0.0014/frame with <= 1
            player within 3 m, 0.0024 with 2, 0.0041 with 3+
  players   20.5 tracked per frame on average; about 5% of frames with 13 or fewer, 2.4% with 23-25
  tracklets 11.5 per minute in total, median length 888 frames (35 s), long tail to minutes
  slots     46 slots split between the teams (24 / 22 on the real file)
  duplicates 3.5% of frames have a same-team pair of tracks within 0.5 m

Pipeline of corruptions, each deterministic for a seed:
  1. view loss: episodes in which only players inside a band of the pitch around the play are
     tracked (gives the low-count tail);
  2. player dropouts: per-player gaps of 1-10 s;
  3. fragmentation: every observed run of a player becomes one or more tracklets (random cuts),
     each written to a free slot of its team's pool, so slot != person;
  4. duplicates: short episodes in which a second slot shadows a player;
  5. position jitter;
  6. ball dropout, independent of the true state. On the real match the ball goes missing mostly
     in live play, so a model must not learn "no ball = dead ball", which the Metrica data would
     teach.

``full(tracking, seed)`` returns a NEW Tracking with 46 slots. ``ball_dropout`` alone is the M2
condition, kept for comparison.
"""

from dataclasses import dataclass, replace

import numpy as np

from mtah.io_atd import Tracking

HAZARD_BY_CROWD = (0.00135, 0.0014, 0.0024, 0.0041)       # 0, 1, 2, 3+ players within 3 m
# (low, high, count) of real missing-gap lengths in frames, graded frames of the real match
GAP_BINS = ((1, 1, 2), (2, 2, 9), (3, 5, 3), (6, 12, 2), (13, 25, 1), (51, 125, 1),
            (126, 250, 53), (251, 750, 60), (751, 2250, 19))


@dataclass(frozen=True)
class AugmentParams:
    # Calibrated against the real aggregates with ext/augment_check (see docs/label_derivation.md).
    n_slots: int = 46
    view_loss_share: float = 0.12      # share of frames inside a view-loss episode
    view_loss_s: tuple = (2.0, 20.0)
    view_band_m: tuple = (25.0, 50.0)  # width (along x) of the band that stays tracked
    player_gap_share: float = 0.02     # share of player-frames lost to individual gaps
    player_gap_s: tuple = (2.0, 20.0)
    cut_every_s: float = 400.0         # mean time between random tracklet cuts per player
    duplicate_share: float = 0.03
    duplicate_s: tuple = (1.0, 5.0)
    duplicate_offset_m: float = 0.6
    jitter_m: float = 0.0              # clean external tracking already matches the real step size
    # Share of each dead stretch with a resting ball restored. 1.0 gives P(ball missing | dead)
    # 0.34-0.35 and P(missing | alive) 0.40-0.43 on Metrica, against about 0.34 / 0.45 estimated for
    # the real match, so missingness carries almost no dead-ball signal, as on the real match.
    dead_ball_fill: float = 1.0
    restart_jitter_m: float = 0.1


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _episodes(rng, n, share, length_frames):
    """Non-overlapping-ish random episodes covering about `share` of n frames."""
    lo, hi = length_frames
    mean = (lo + hi) / 2.0
    count = int(round(share * n / mean))
    starts = np.sort(rng.integers(0, max(1, n - hi), size=count))
    lengths = rng.integers(lo, hi + 1, size=count)
    mask = np.zeros(n, dtype=bool)
    for s, length in zip(starts, lengths):
        mask[s:s + length] = True
    return mask, list(zip(starts.tolist(), (starts + lengths).tolist()))


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


def restore_dead_ball(tracking: Tracking, rests, fraction: float, jitter_m: float, seed: int):
    """In-place: put back a resting ball for part of each dead stretch where the source blanked it.

    rests: [(start, stop, x, y)] in video frames and metres, from the converter. Each is a stretch
    over which the ball should be resting at (x, y): just outside the line after a ball out, or on
    the restart spot before a restart. Only frames where the ball is missing are filled, and only
    the given fraction of each stretch, nearest its own anchor."""
    rng = np.random.default_rng(seed)
    ball = tracking.ball_xy
    for start, stop, x, y, anchor in rests:
        length = int(round((stop - start) * fraction))
        if length <= 0:
            continue
        a, b = (stop - length, stop) if anchor == "end" else (start, start + length)
        missing = np.isnan(ball[a:b, 0])
        fill = np.array([x, y]) + rng.normal(0.0, jitter_m, size=(b - a, 2))
        ball[a:b][missing] = fill[missing]


def rests_from_labels(gt_codes, ball_xy, fps: int, dead_code: int = 2, lookahead_s: float = 1.0):
    """Resting-ball stretches for sources without event coordinates (SkillCorner).

    For each dead run, the restart spot is the first ball position observed within lookahead_s
    after the run ends. The ball is taken to rest there over the whole run. This uses the external
    game's labels and tracking only."""
    dead = gt_codes == dead_code
    seen = ~np.isnan(ball_xy[:, 0])
    out = []
    for start, stop in _runs(dead):
        after = np.flatnonzero(seen[stop:stop + int(lookahead_s * fps)])
        if len(after):
            x, y = ball_xy[stop + int(after[0])]
            out.append((start, stop, float(x), float(y), "end"))
    return out


def full(tracking: Tracking, seed: int = 0, params: AugmentParams = AugmentParams(),
         fps: int = 25, rests=()) -> Tracking:
    """A corrupted copy of `tracking` with params.n_slots tracklet slots (see module docstring).
    `rests` (from the converter's rests.json) restores a resting dead ball before corruption."""
    rng = np.random.default_rng(seed)
    n = tracking.n_frames
    players = tracking.players_xy.copy()
    team_of = tracking.slot_team.copy()
    ball_src = Tracking(players_xy=tracking.players_xy, players_speed=tracking.players_speed,
                        ball_xy=tracking.ball_xy.copy(), ball_speed=tracking.ball_speed.copy(),
                        slot_team=tracking.slot_team, pitch=tracking.pitch, meta=tracking.meta)
    if rests:
        restore_dead_ball(ball_src, rests, params.dead_ball_fill, params.restart_jitter_m, seed + 2)

    # 1. view loss: outside a band around the play, nobody is tracked
    _, episodes = _episodes(rng, n, params.view_loss_share,
                            (int(params.view_loss_s[0] * fps), int(params.view_loss_s[1] * fps)))
    for start, stop in episodes:
        xs = players[start, :, 0]
        if np.isnan(xs).all():
            continue
        centre = float(np.nanmean(xs)) + rng.uniform(-10, 10)
        band = rng.uniform(*params.view_band_m)
        x = players[start:stop, :, 0]
        outside = np.abs(x - centre) > band / 2.0
        players[start:stop][outside] = np.nan

    # 2. individual player gaps
    lo_g, hi_g = int(params.player_gap_s[0] * fps), int(params.player_gap_s[1] * fps)
    for k in range(players.shape[1]):
        mask, _ = _episodes(rng, n, params.player_gap_share, (lo_g, hi_g))
        players[mask, k] = np.nan

    # 3. fragmentation into tracklets on a slot pool per team
    pools = [list(range(0, params.n_slots // 2)), list(range(params.n_slots // 2, params.n_slots))]
    out = np.full((n, params.n_slots, 2), np.nan)
    slot_team = np.array([0] * (params.n_slots // 2) + [1] * (params.n_slots - params.n_slots // 2),
                         dtype=np.int8)
    fragments = []   # (start, stop, team, source column)
    for k in range(players.shape[1]):
        for start, stop in _runs(~np.isnan(players[:, k, 0])):
            cuts = [start]
            t = start
            while True:
                t += int(rng.exponential(params.cut_every_s * fps)) + 1
                if t >= stop:
                    break
                cuts.append(t)
            cuts.append(stop)
            fragments += [(a, b, int(team_of[k]), k) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    fragments.sort()
    busy_until = np.zeros(params.n_slots, dtype=np.int64)
    for start, stop, team, k in fragments:
        free = [s for s in pools[team] if busy_until[s] <= start]
        slot = int(rng.choice(free)) if free else \
            int(min(pools[team], key=lambda s: busy_until[s]))
        out[start:stop, slot] = players[start:stop, k]
        busy_until[slot] = stop

    # 4. duplicates: a free slot shadows a tracked player for a few seconds
    _, episodes = _episodes(rng, n, params.duplicate_share,
                            (int(params.duplicate_s[0] * fps), int(params.duplicate_s[1] * fps)))
    for start, stop in episodes:
        tracked = np.flatnonzero(~np.isnan(out[start, :, 0]))
        if not len(tracked):
            continue
        src = int(rng.choice(tracked))
        free = [s for s in pools[int(slot_team[src])] if busy_until[s] <= start
                and np.isnan(out[start:stop, s, 0]).all()]
        if not free:
            continue
        dst = int(rng.choice(free))
        out[start:stop, dst] = out[start:stop, src] + rng.normal(0.0, params.duplicate_offset_m,
                                                                 size=2)

    # 5. jitter
    out += rng.normal(0.0, params.jitter_m, size=out.shape)

    speed = np.full(out.shape[:2], np.nan)
    speed[1:] = np.hypot(out[1:, :, 0] - out[:-1, :, 0], out[1:, :, 1] - out[:-1, :, 1]) * fps
    result = Tracking(players_xy=out, players_speed=speed, ball_xy=ball_src.ball_xy,
                      ball_speed=ball_src.ball_speed, slot_team=slot_team,
                      pitch=tracking.pitch, meta=replace(tracking.meta, slot_team=slot_team))
    # 6. ball dropout
    ball_dropout(result, seed=seed + 1)
    return result
