"""Graded-frame policy and algorithmic kickoff detection (CLAUDE.md hard rule 7).

The published half blocks say which frames are graded; they do not say when the ball is first
kicked. For BAR-ATH the second half is graded about two minutes before its kickoff, and every frame
in between is dead ball. That kickoff is found here from the tracking alone, never hard-coded.

Kickoff shape, per frame:
  * ball observed within ``CENTRE_RADIUS_M`` of the centre spot;
  * ball at rest: it moves less than ``REST_STEP_M`` from the previous frame;
  * each team has at least ``MIN_PLAYERS`` tracked players, and at least ``OWN_HALF_SHARE`` of
    them (ignoring anyone within ``KICKER_EXEMPT_M`` of the centre spot) stand in their own half,
    where a team's own half is the side its centroid is on, and the two centroids are on opposite
    sides.
A kickoff is the frame right after a run of at least ``MIN_REST_S`` of the ball resting on the spot
with the shape holding on most of the run: the first frame the ball has left its rest.

Fallback (HYPOTHESIS, used only when no ball-based kickoff is found, for instance because the ball
is not tracked on the spot): the frame right after a run of at least ``MIN_SHAPE_S`` with the teams
separated into their own halves.

The detected frame must lie inside ``[block_start - LEAD_S, block_start + SEARCH_S]``. That is the
consistency assertion against the published boundaries: anything else raises.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

CENTRE_RADIUS_M = 2.0
REST_STEP_M = 0.05          # per frame, i.e. 1.25 m/s at 25 fps
MIN_REST_S = 0.5
MAX_HELD_GAP = 5            # frames of missing ball bridged inside a rest on the spot
MIN_SHAPE_S = 2.0
MIN_SHAPE_SHARE = 0.8       # of the rest run
MIN_PLAYERS = 7
OWN_HALF_SHARE = 0.9
KICKER_EXEMPT_M = 3.0
LEAD_S = 10.0               # a first half is graded from its kickoff, so look slightly before
SEARCH_S = 300.0            # kickoffs happen within 5 minutes of the published block start

LABEL_X, LABEL_D = "X", "D"


class KickoffNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class Kickoff:
    half: int
    block_first: int
    block_last: int
    frame: int
    method: str  # "ball" or "shape"

    @property
    def offset_s(self):
        return None if self.frame is None else self.frame - self.block_first

    def describe(self, fps: int) -> str:
        return ("half {}: kickoff detected by {} at {:+.2f} s from the published block start; "
                "consistent with published bounds: yes".format(
                    self.half, self.method, (self.frame - self.block_first) / float(fps)))


def _hold_short_gaps(ball: np.ndarray, observed: np.ndarray, max_gap: int) -> np.ndarray:
    """Copy of ball with gaps of at most max_gap frames filled by the last seen position, so a
    momentary tracking dropout does not break a ball resting on the spot."""
    out = ball.copy()
    for start, stop in _runs(~observed):
        if start > 0 and stop - start <= max_gap:
            out[start:stop] = ball[start - 1]
    return out


def _runs(mask: np.ndarray):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def teams_separated(players_xy: np.ndarray, slot_team: np.ndarray, centre) -> np.ndarray:
    """Per frame: each team in its own half (kickoff formation), ignoring the kicker area."""
    cx, cy = centre
    x = players_xy[..., 0]
    y = players_xy[..., 1]
    observed = ~np.isnan(x) & ~np.isnan(y)
    near_spot = np.hypot(np.where(observed, x - cx, np.inf), np.where(observed, y - cy, np.inf)) \
        < KICKER_EXEMPT_M
    side = np.sign(np.where(observed, x - cx, 0.0))

    ok = np.ones(players_xy.shape[0], dtype=bool)
    team_sides = []
    for team in (0, 1):
        cols = slot_team == team
        obs = observed[:, cols]
        count = obs.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid = np.where(count > 0, np.where(obs, x[:, cols], 0.0).sum(axis=1)
                                / np.maximum(count, 1), np.nan)
        team_side = np.sign(centroid - cx)
        counted = obs & ~near_spot[:, cols]
        same = (side[:, cols] == team_side[:, None]) & counted
        n_counted = counted.sum(axis=1)
        share = np.where(n_counted > 0, same.sum(axis=1) / np.maximum(n_counted, 1), 0.0)
        ok &= (count >= MIN_PLAYERS) & (share >= OWN_HALF_SHARE) & (team_side != 0) \
            & ~np.isnan(team_side)
        team_sides.append(team_side)
    ok &= team_sides[0] == -team_sides[1]
    return ok


def detect_kickoff(tracking, half: int, first: int, last: int, fps: int) -> Kickoff:
    length, width = tracking.pitch
    centre = (length / 2.0, width / 2.0)
    lo = max(0, first - int(round(LEAD_S * fps)))
    hi = min(tracking.n_frames, last + 1, first + int(round(SEARCH_S * fps)) + 1)
    if hi - lo < 2:
        raise KickoffNotFound("half {}: empty search window".format(half))

    observed = tracking.ball_observed[lo:hi]
    ball = _hold_short_gaps(tracking.ball_xy[lo:hi], observed, MAX_HELD_GAP)
    shape = teams_separated(tracking.players_xy[lo:hi], tracking.slot_team, centre)

    with np.errstate(invalid="ignore"):
        on_spot = np.hypot(ball[:, 0] - centre[0], ball[:, 1] - centre[1]) < CENTRE_RADIUS_M
        step = np.full(len(ball), np.inf)
        step[1:] = np.hypot(np.diff(ball[:, 0]), np.diff(ball[:, 1]))
    step[np.isnan(step)] = np.inf
    resting = on_spot & (step < REST_STEP_M)
    resting[0] = on_spot[0]  # no previous frame; position alone

    min_rest = int(round(MIN_REST_S * fps))
    for start, stop in _runs(resting):
        if stop - start >= min_rest and shape[start:stop].mean() >= MIN_SHAPE_SHARE \
                and stop < len(ball):
            # The kick is the frame after the last frame the ball was actually seen at rest, not
            # after the end of a held (gap-filled) stretch.
            last_seen = start + int(np.flatnonzero(observed[start:stop])[-1])
            return _checked(Kickoff(half, first, last, lo + last_seen + 1, "ball"), fps)

    min_shape = int(round(MIN_SHAPE_S * fps))
    for start, stop in _runs(shape):
        if stop - start >= min_shape and stop < len(ball):
            return _checked(Kickoff(half, first, last, lo + stop, "shape"), fps)

    raise KickoffNotFound(
        "half {}: no kickoff shape within [{:+.0f} s, {:+.0f} s] of the published block start"
        .format(half, -LEAD_S, SEARCH_S))


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
