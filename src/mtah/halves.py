"""Graded-frame policy and algorithmic kickoff detection (CLAUDE.md hard rule 7).

The published half blocks say which frames are graded, but not when the ball is first kicked. For
BAR-ATH the second half is graded about two minutes before its kickoff, and every frame in between
is dead ball. That kickoff is found here from the tracking alone, never hard-coded.

Methods are tried in order. The first that finds something inside the search window
``[block_start - LEAD_S, block_start + SEARCH_S]`` wins:

1. ``ball+shape``: the ball rests on the centre spot (all positions within ``MOVE_M`` of their
   median for ``MIN_REST_S``, inside ``CENTRE_RADIUS_M`` of the spot) while the teams stand in
   their own halves on at least half of the preceding ``SHAPE_LOOKBACK_S``. The kickoff is the
   frame after the last frame the ball was seen at rest.
2. ``ball``: the same rest on the spot, without the shape condition.
3. ``shape`` (HYPOTHESIS): the end of a run of at least ``MIN_SHAPE_S`` in kickoff formation.
4. ``data-start``: tracking begins within 1 s of the block start after at least
   ``DATA_START_S`` with none, and any formation run that starts with the data ends within
   ``DATA_START_MAX_SHAPE_S``. The half then starts with the data. This is the case for a first
   half that is graded from its own kickoff, with no tracked rest before it. A longer formation
   run from the start of the data is a line-up before the kickoff, so (3) handles it.
5. ``fallback``: nothing found. The published block start is used, the result is flagged loudly,
   and aggregate diagnostics are attached. It never raises, so a submission is still produced.

A detected frame that falls outside the search window raises ``KickoffNotFound``. That is the
consistency assertion against the published boundaries.

Thresholds are deliberately loose about position noise. The real ball track jitters by about 0.1 m
a frame, and players standing on the halfway line can be measured on either side of it.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

CENTRE_RADIUS_M = 3.0       # ball "on the spot" (calibration and jitter tolerance)
MOVE_M = 0.75               # ball "at rest": stays within this of its own median
MIN_REST_S = 0.5
MAX_HELD_GAP = 5            # frames of missing ball bridged while it rests
SHAPE_LOOKBACK_S = 2.0
MIN_SHAPE_S = 2.0
MIN_PLAYERS = 6             # per team
OWN_HALF_SHARE = 0.8
HALFWAY_TOL_M = 1.0         # within this of the halfway line counts as either half
KICKER_EXEMPT_M = 3.0
LEAD_S = 10.0               # a first half is graded from its kickoff, so look slightly before
SEARCH_S = 300.0            # kickoffs happen within 5 minutes of the published block start
DATA_START_S = 2.0          # 'no tracking just before the block' means none in this many s
DATA_START_MAX_SHAPE_S = 5.0  # a formation that starts with the data and ends sooner = kickoff

LABEL_X, LABEL_D = "X", "D"
DETECTED = ("ball+shape", "ball", "shape", "data-start")


class KickoffNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class Kickoff:
    half: int
    block_first: int
    block_last: int
    frame: int
    method: str
    diagnostics: Dict = field(default_factory=dict, compare=False)

    @property
    def detected(self) -> bool:
        return self.method in DETECTED

    def describe(self, fps: int) -> str:
        offset = (self.frame - self.block_first) / float(fps)
        if self.detected:
            return ("half {}: kickoff detected by {} at {:+.2f} s from the published block "
                    "start; consistent with published bounds: yes".format(
                        self.half, self.method, offset))
        return ("half {}: WARNING kickoff NOT DETECTED, using the published block start "
                "(see diagnostics)".format(self.half))


def _runs(mask: np.ndarray):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _longest(mask: np.ndarray) -> int:
    runs = _runs(mask)
    return max((b - a for a, b in runs), default=0)


def _hold_short_gaps(ball: np.ndarray, observed: np.ndarray, max_gap: int) -> np.ndarray:
    """Copy of ball with gaps of at most max_gap frames filled by the last seen position."""
    out = ball.copy()
    for start, stop in _runs(~observed):
        if start > 0 and stop - start <= max_gap:
            out[start:stop] = ball[start - 1]
    return out


def teams_separated(players_xy: np.ndarray, slot_team: np.ndarray, centre,
                    share: float = OWN_HALF_SHARE) -> np.ndarray:
    """Per frame: each team in its own half (kickoff formation), ignoring the kicker area and
    players within HALFWAY_TOL_M of the halfway line."""
    cx, cy = centre
    x = players_xy[..., 0]
    y = players_xy[..., 1]
    observed = ~np.isnan(x) & ~np.isnan(y)
    dx = np.where(observed, x - cx, 0.0)
    near_spot = np.hypot(dx, np.where(observed, y - cy, 0.0)) < KICKER_EXEMPT_M
    on_line = np.abs(dx) <= HALFWAY_TOL_M
    side = np.sign(dx)

    ok = np.ones(players_xy.shape[0], dtype=bool)
    team_sides = []
    for team in (0, 1):
        cols = slot_team == team
        obs = observed[:, cols]
        count = obs.sum(axis=1)
        centroid = np.where(obs, x[:, cols], 0.0).sum(axis=1) / np.maximum(count, 1)
        team_side = np.where(count > 0, np.sign(centroid - cx), 0.0)
        counted = obs & ~near_spot[:, cols]
        same = ((side[:, cols] == team_side[:, None]) | on_line[:, cols]) & counted
        n_counted = counted.sum(axis=1)
        frac = np.where(n_counted > 0, same.sum(axis=1) / np.maximum(n_counted, 1), 0.0)
        ok &= (count >= MIN_PLAYERS) & (frac >= share) & (team_side != 0)
        team_sides.append(team_side)
    ok &= team_sides[0] == -team_sides[1]
    return ok


def _rest_kicks(held: np.ndarray, observed: np.ndarray, centre, min_rest: int):
    """Kick frames (window-relative) after each rest of the ball on the centre spot."""
    with np.errstate(invalid="ignore"):
        on_spot = np.hypot(held[:, 0] - centre[0], held[:, 1] - centre[1]) < CENTRE_RADIUS_M
    kicks = []
    for start, stop in _runs(on_spot):
        i = start
        while i + min_rest <= stop:
            window = held[i:i + min_rest]
            anchor = np.median(window, axis=0)
            if np.all(np.hypot(*(window - anchor).T) <= MOVE_M):
                j = i + min_rest
                while j < stop and np.hypot(*(held[j] - anchor)) <= MOVE_M:
                    j += 1
                if j < len(held):  # the ball did leave its rest inside the window
                    seen = np.flatnonzero(observed[i:j])
                    kicks.append(i + int(seen[-1]) + 1)
                i = j
            else:
                i += 1
    return kicks


def kickoff_diagnostics(tracking, first: int, fps: int) -> Dict:
    """Aggregates only (counts and run lengths relative to the published block start), safe to
    print for the real match. Explains why a method did or did not fire."""
    length, width = tracking.pitch
    centre = (length / 2.0, width / 2.0)
    lead, ten = int(round(LEAD_S * fps)), 10 * fps
    windows = {"pre_10s": (first - lead, first), "first_10s": (first, first + ten),
               "search": (first - lead, first + int(round(SEARCH_S * fps)) + 1)}
    out = {}
    for name, (lo, hi) in windows.items():
        lo, hi = max(0, lo), min(tracking.n_frames, hi)
        players = tracking.players_xy[lo:hi]
        observed = ~np.isnan(players[..., 0])
        ball_obs = tracking.ball_observed[lo:hi]
        ball = tracking.ball_xy[lo:hi]
        with np.errstate(invalid="ignore"):
            dist = np.hypot(ball[:, 0] - centre[0], ball[:, 1] - centre[1])
        per_team = [observed[:, tracking.slot_team == t].sum(axis=1) for t in (0, 1)]
        out[name] = {
            "frames": hi - lo,
            "frames_any_player": int(observed.any(axis=1).sum()),
            "mean_players_home": round(float(per_team[0].mean()), 2) if hi > lo else 0,
            "mean_players_away": round(float(per_team[1].mean()), 2) if hi > lo else 0,
            "frames_ball_seen": int(ball_obs.sum()),
            "frames_ball_within_3m_of_spot": int(np.sum(dist < 3.0)),
            "frames_ball_within_10m_of_spot": int(np.sum(dist < 10.0)),
            "longest_run_ball_within_3m": _longest(dist < 3.0),
            "frames_shape_ok": int(teams_separated(players, tracking.slot_team, centre).sum()),
            "longest_run_shape_ok": _longest(teams_separated(players, tracking.slot_team, centre)),
            "longest_run_shape_ok_at_0.6": _longest(
                teams_separated(players, tracking.slot_team, centre, share=0.6)),
        }
    return out


def detect_kickoff(tracking, half: int, first: int, last: int, fps: int) -> Kickoff:
    length, width = tracking.pitch
    centre = (length / 2.0, width / 2.0)
    lo = max(0, first - int(round(LEAD_S * fps)))
    hi = min(tracking.n_frames, last + 1, first + int(round(SEARCH_S * fps)) + 1)

    def found(rel_frame, method):
        return _checked(Kickoff(half, first, last, lo + int(rel_frame), method), fps)

    if hi - lo >= 2:
        observed = tracking.ball_observed[lo:hi]
        held = _hold_short_gaps(tracking.ball_xy[lo:hi], observed, MAX_HELD_GAP)
        shape = teams_separated(tracking.players_xy[lo:hi], tracking.slot_team, centre)
        kicks = _rest_kicks(held, observed, centre, int(round(MIN_REST_S * fps)))
        lookback = int(round(SHAPE_LOOKBACK_S * fps))
        for kick in kicks:
            if shape[max(0, kick - lookback):kick].mean() >= 0.5:
                return found(kick, "ball+shape")
        if kicks:
            return found(kicks[0], "ball")
        data_begin = _data_begin_at_block(tracking, first, fps)
        shape_runs = _runs(shape)
        if data_begin is not None:
            # Formation in place when the data begins and gone within a few seconds: the data
            # began at the kickoff. (No formation at data start is NOT evidence: in a second half
            # tracking can resume while players are still walking out, minutes before kickoff.)
            opening = [(a, b) for a, b in shape_runs if abs(lo + a - data_begin) <= fps // 2]
            if opening and opening[0][1] - opening[0][0] < DATA_START_MAX_SHAPE_S * fps:
                return _checked(Kickoff(half, first, last, first, "data-start"), fps)
        min_shape = int(round(MIN_SHAPE_S * fps))
        for start, stop in shape_runs:
            if stop - start >= min_shape and stop < len(shape):
                return found(stop, "shape")

    return Kickoff(half, first, last, first, "fallback",
                   diagnostics=kickoff_diagnostics(tracking, first, fps))


def _data_begin_at_block(tracking, first: int, fps: int):
    """The frame tracking begins, if that is within 1 s of the block start after at least
    DATA_START_S with none; else None."""
    tracked = tracking.player_observed.any(axis=1)
    quiet = int(round(DATA_START_S * fps))
    for t in range(max(0, first - fps), min(tracking.n_frames, first + fps + 1)):
        if tracked[t]:
            return t if t - quiet >= 0 and not tracked[t - quiet:t].any() else None
    return None


def _checked(kickoff: Kickoff, fps: int) -> Kickoff:
    earliest = kickoff.block_first - int(round(LEAD_S * fps))
    latest = kickoff.block_first + int(round(SEARCH_S * fps))
    if not (earliest <= kickoff.frame <= latest and kickoff.frame <= kickoff.block_last):
        raise KickoffNotFound("half {}: detected kickoff inconsistent with published bounds"
                              .format(kickoff.half))
    return kickoff


def detect_kickoffs(tracking, spec) -> List[Kickoff]:
    return [detect_kickoff(tracking, half, first, last, spec.fps)
            for half, first, last in spec.halves]


def apply_policy(labels: np.ndarray, spec, kickoffs: List[Kickoff]) -> np.ndarray:
    """X outside the graded blocks, D from each block start up to its detected kickoff."""
    out = np.array(labels, dtype="<U1", copy=True)
    graded = spec.graded_mask()
    out[~graded] = LABEL_X
    for kickoff in kickoffs:
        if kickoff.frame > kickoff.block_first:
            out[kickoff.block_first:min(kickoff.frame, kickoff.block_last + 1)] = LABEL_D
    return out
