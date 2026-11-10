"""Graded-frame policy and algorithmic kickoff detection (CLAUDE.md hard rule 7).

The published half blocks say which frames are graded, but not when the ball is first kicked. For
BAR-ATH the second half is graded about two minutes before its kickoff, and every frame in between
is dead ball. The kickoff is found here from the tracking alone and is never hard-coded.

The signal is the kickoff FORMATION, not the ball. On the real match the ball is almost never
tracked while it rests on the centre spot; it is picked up once it moves. The players, though, stand
in their own halves before every kickoff and leave them within about a second of it. In open play
both teams almost never stand fully in their own halves for seconds at a time.

Per frame, the formation is in one of three states:
  OK        both teams have >= MIN_PLAYERS tracked, >= OWN_HALF_SHARE of each (ignoring the kicker
            area and anyone within HALFWAY_TOL_M of the halfway line) in its own half, and the
            two teams on opposite sides;
  VIOLATED  both teams tracked well enough, and the condition fails;
  UNKNOWN   a team has too few tracked players to tell. This is common around a restart, while the
            camera or the tracker has not caught up. It never ends a formation by itself.

A formation run starts at an OK frame and ends at the first VIOLATED frame that is followed by
VIOLATION_S with no OK frame. Short flickers and UNKNOWN stretches do not end it. The kickoff is
the end of the pre-kickoff line-up: the first run of at least MIN_FORMATION_S in the search window.
Its end is the first frame the formation has broken.

Refinement with the ball, inside the REFINE_S before the formation breaks:
  * a rest on the spot (the ball stays within MOVE_M of its median for MIN_REST_S, within
    CENTRE_RADIUS_M of the spot): the kickoff is the frame after the ball was last seen at rest;
  * else the first frame the ball is seen within NEAR_SPOT_M of the spot;
  * else the break itself.

When tracking begins at the block start (within 1 s, after DATA_START_S with none):
  * a formation already in place when the data begins counts as the line-up whatever its length
    (the data began during the line-up);
  * live-start: no formation, and players already running (median speed >= LIVE_SPEED_MS over
    LIVE_WINDOW_S). Tracking began after the kick, so the kickoff is the block start. Seen on
    external game 1 (2.4 m/s against 0-0.4 m/s at the five line-up starts). Players walking out
    before a second half are slow, so they do not trigger it.
fallback: nothing found, so the published block start is used, flagged, with aggregate
diagnostics.

Consistency (the assertion of rule 7): the kickoff must lie inside the search window. When the game
publishes kickoff anchors (BAR-ATH's games.py does), it must also lie within ANCHOR_TOL_S of the
anchor. The anchor checks the detector and never sets a label. A kickoff that fails the check is
marked inconsistent, and pipeline.run refuses to write a file.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# formation
MIN_PLAYERS = 6             # per team, below this the formation is UNKNOWN
OWN_HALF_SHARE = 0.8
HALFWAY_TOL_M = 1.0         # within this of the halfway line counts as either half
KICKER_EXEMPT_M = 3.0
VIOLATION_S = 0.5           # a violation must last this long (no OK frame) to end a run
MIN_FORMATION_S = 2.0
# ball refinement
REFINE_S = 3.0
CENTRE_RADIUS_M = 3.0
NEAR_SPOT_M = 5.0
MOVE_M = 0.75
MIN_REST_S = 0.5
MAX_HELD_GAP = 5
# windows and checks
LEAD_S = 10.0               # a first half is graded from its kickoff, so look slightly before
SEARCH_S = 300.0
DATA_START_S = 2.0          # tracking "begins" after at least this long without any
LIVE_WINDOW_S = 3.0
LIVE_SPEED_MS = 1.5         # players running at data start: play is already live
ANCHOR_TOL_S = 2.0

OK, VIOLATED, UNKNOWN = 1, -1, 0
LABEL_X, LABEL_D = "X", "D"
DETECTED = ("formation+ball", "formation", "live-start")


@dataclass(frozen=True)
class Kickoff:
    half: int
    block_first: int
    block_last: int
    frame: int
    method: str
    consistent: bool = True
    anchor_delta_s: Optional[float] = None
    diagnostics: Dict = field(default_factory=dict, compare=False)

    @property
    def detected(self) -> bool:
        return self.method in DETECTED

    def describe(self, fps: int) -> str:
        offset = (self.frame - self.block_first) / float(fps)
        if not self.detected:
            head = "half {}: WARNING kickoff NOT DETECTED, using the published block start".format(
                self.half)
        else:
            head = "half {}: kickoff detected by {} at {:+.2f} s from the published block " \
                   "start".format(self.half, self.method, offset)
        check = "; consistent with published bounds: {}".format("yes" if self.consistent else "NO")
        if self.anchor_delta_s is not None:
            check += " (published kickoff anchor {:+.2f} s away)".format(self.anchor_delta_s)
        return head + check


def _runs(mask: np.ndarray):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _longest(mask: np.ndarray) -> int:
    return max((b - a for a, b in _runs(mask)), default=0)


def _hold_short_gaps(ball: np.ndarray, observed: np.ndarray, max_gap: int) -> np.ndarray:
    """Copy of ball with gaps of at most max_gap frames filled by the last seen position."""
    out = ball.copy()
    for start, stop in _runs(~observed):
        if start > 0 and stop - start <= max_gap:
            out[start:stop] = ball[start - 1]
    return out


def formation_state(players_xy: np.ndarray, slot_team: np.ndarray, centre,
                    share: float = OWN_HALF_SHARE) -> np.ndarray:
    """Per frame OK / VIOLATED / UNKNOWN (see module docstring)."""
    cx, cy = centre
    x = players_xy[..., 0]
    y = players_xy[..., 1]
    observed = ~np.isnan(x) & ~np.isnan(y)
    dx = np.where(observed, x - cx, 0.0)
    near_spot = np.hypot(dx, np.where(observed, y - cy, 0.0)) < KICKER_EXEMPT_M
    on_line = np.abs(dx) <= HALFWAY_TOL_M
    side = np.sign(dx)

    known = np.ones(players_xy.shape[0], dtype=bool)
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
        known &= count >= MIN_PLAYERS
        ok &= (frac >= share) & (team_side != 0)
        team_sides.append(team_side)
    ok &= team_sides[0] == -team_sides[1]
    return np.where(known, np.where(ok, OK, VIOLATED), UNKNOWN).astype(np.int8)


def formation_runs(state: np.ndarray, fps: int):
    """[(start, stop, how)] formation runs; stop is the first frame of a sustained violation, or
    len(state) with how="open" if the run never breaks inside the window."""
    sustain = max(1, int(round(VIOLATION_S * fps)))
    is_ok = state == OK
    # next_ok[i]: index of the first OK frame at or after i (len if none)
    idx = np.where(is_ok, np.arange(len(state)), len(state))
    next_ok = np.minimum.accumulate(idx[::-1])[::-1]
    runs, i, n = [], 0, len(state)
    while i < n:
        if not is_ok[i]:
            i = int(next_ok[i])
            continue
        start, j = i, i + 1
        how = "open"
        while j < n:
            if state[j] == VIOLATED and next_ok[j] - j >= sustain:
                how = "violated"
                break
            j += 1
        runs.append((start, j, how))
        i = j if how == "open" else int(next_ok[j]) if next_ok[j] < n else n
    return runs


def _rest_kicks(held: np.ndarray, observed: np.ndarray, centre, min_rest: int):
    """Window-relative frames right after each rest of the ball on the centre spot."""
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
                if j < len(held):
                    seen = np.flatnonzero(observed[i:j])
                    kicks.append(i + int(seen[-1]) + 1)
                i = j
            else:
                i += 1
    return kicks


def _refine_with_ball(tracking, lo: int, brk: int, fps: int, centre):
    """(frame, used_ball) for a formation break at absolute frame brk."""
    a = max(lo, brk - int(round(REFINE_S * fps)))
    if brk <= a:
        return brk, False
    observed = tracking.ball_observed[a:brk + 1]
    held = _hold_short_gaps(tracking.ball_xy[a:brk + 1], observed, MAX_HELD_GAP)
    kicks = [k for k in _rest_kicks(held, observed, centre, int(round(MIN_REST_S * fps)))
             if a + k <= brk]
    if kicks:
        return a + kicks[-1], True
    with np.errstate(invalid="ignore"):
        dist = np.hypot(tracking.ball_xy[a:brk + 1, 0] - centre[0],
                        tracking.ball_xy[a:brk + 1, 1] - centre[1])
    near = np.flatnonzero(dist < NEAR_SPOT_M)
    if len(near):
        return a + int(near[0]), True
    return brk, False


def _players_running(tracking, start: int, fps: int) -> bool:
    """Median player speed over the first LIVE_WINDOW_S at least LIVE_SPEED_MS: play is live."""
    seg = tracking.players_xy[start:start + int(round(LIVE_WINDOW_S * fps))]
    if len(seg) < 2:
        return False
    step = np.hypot(np.diff(seg[..., 0], axis=0), np.diff(seg[..., 1], axis=0)) * fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames
        per_frame = np.nanmedian(step, axis=1)
    per_frame = per_frame[np.isfinite(per_frame)]
    return bool(len(per_frame)) and float(np.median(per_frame)) >= LIVE_SPEED_MS


def _data_begin_at_block(tracking, first: int, fps: int):
    """The frame tracking begins, if within 1 s of the block start after DATA_START_S with none."""
    tracked = tracking.player_observed.any(axis=1)
    quiet = int(round(DATA_START_S * fps))
    for t in range(max(0, first - fps), min(tracking.n_frames, first + fps + 1)):
        if tracked[t]:
            return t if t - quiet >= 0 and not tracked[t - quiet:t].any() else None
    return None


def _window(tracking, first: int, last: int, fps: int):
    lo = max(0, first - int(round(LEAD_S * fps)))
    hi = min(tracking.n_frames, last + 1, first + int(round(SEARCH_S * fps)) + 1)
    return lo, hi


def kickoff_diagnostics(tracking, first: int, fps: int) -> Dict:
    """Aggregates only (counts and run lengths relative to the published block start)."""
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
        ball = tracking.ball_xy[lo:hi]
        with np.errstate(invalid="ignore"):
            dist = np.hypot(ball[:, 0] - centre[0], ball[:, 1] - centre[1])
        per_team = [observed[:, tracking.slot_team == t].sum(axis=1) for t in (0, 1)]
        state = formation_state(players, tracking.slot_team, centre)
        out[name] = {
            "frames": hi - lo,
            "frames_any_player": int(observed.any(axis=1).sum()),
            "mean_players_home": round(float(per_team[0].mean()), 2) if hi > lo else 0,
            "mean_players_away": round(float(per_team[1].mean()), 2) if hi > lo else 0,
            "frames_ball_seen": int(tracking.ball_observed[lo:hi].sum()),
            "frames_ball_within_3m_of_spot": int(np.sum(dist < 3.0)),
            "frames_ball_within_10m_of_spot": int(np.sum(dist < 10.0)),
            "frames_formation_ok": int(np.sum(state == OK)),
            "frames_formation_violated": int(np.sum(state == VIOLATED)),
            "frames_formation_unknown": int(np.sum(state == UNKNOWN)),
            "longest_run_formation_ok": _longest(state == OK),
        }
    return out


def formation_run_table(tracking, first: int, last: int, fps: int, min_s: float = 1.0):
    """Formation runs of at least min_s in the search window, as offsets in seconds from the
    published block start. Event timings, no positions; for tools/kickoff_diag.py --runs."""
    length, width = tracking.pitch
    lo, hi = _window(tracking, first, last, fps)
    state = formation_state(tracking.players_xy[lo:hi], tracking.slot_team,
                            (length / 2.0, width / 2.0))
    rows = []
    for start, stop, how in formation_runs(state, fps):
        if stop - start >= min_s * fps:
            seg = state[start:stop]
            rows.append({"start_s": round((lo + start - first) / fps, 2),
                         "end_s": round((lo + stop - first) / fps, 2),
                         "length_s": round((stop - start) / fps, 2), "ended": how,
                         "unknown_share": round(float(np.mean(seg == UNKNOWN)), 2)})
    return rows


def detect_kickoff(tracking, half: int, first: int, last: int, fps: int,
                   anchor: Optional[int] = None) -> Kickoff:
    length, width = tracking.pitch
    centre = (length / 2.0, width / 2.0)
    lo, hi = _window(tracking, first, last, fps)

    def result(frame, method):
        return _checked(Kickoff(half, first, last, int(frame), method), fps, anchor)

    if hi - lo >= 2:
        state = formation_state(tracking.players_xy[lo:hi], tracking.slot_team, centre)
        runs = [r for r in formation_runs(state, fps) if r[2] == "violated"]
        data_begin = _data_begin_at_block(tracking, first, fps)

        if data_begin is not None and state[data_begin - lo] != OK \
                and _players_running(tracking, data_begin, fps):
            return result(first, "live-start")

        # A line-up qualifies by length, or by being in place the moment the data begins.
        min_run = int(round(MIN_FORMATION_S * fps))
        lineup = next(((a, b) for a, b, _ in runs if b - a >= min_run or (
            data_begin is not None and abs(lo + a - data_begin) <= fps // 2)), None)
        if lineup is not None:
            frame, used_ball = _refine_with_ball(tracking, lo, lo + lineup[1], fps, centre)
            return result(frame, "formation+ball" if used_ball else "formation")

    return _checked(Kickoff(half, first, last, first, "fallback",
                            diagnostics=kickoff_diagnostics(tracking, first, fps)), fps, anchor)


def _checked(kickoff: Kickoff, fps: int, anchor: Optional[int]) -> Kickoff:
    earliest = kickoff.block_first - int(round(LEAD_S * fps))
    latest = min(kickoff.block_last, kickoff.block_first + int(round(SEARCH_S * fps)))
    consistent = earliest <= kickoff.frame <= latest
    delta = None
    if anchor is not None:
        delta = (kickoff.frame - anchor) / float(fps)
        consistent = consistent and abs(delta) <= ANCHOR_TOL_S
    return Kickoff(kickoff.half, kickoff.block_first, kickoff.block_last, kickoff.frame,
                   kickoff.method, consistent, delta, kickoff.diagnostics)


def detect_kickoffs(tracking, spec) -> List[Kickoff]:
    anchors = dict(spec.kickoff_anchors)
    return [detect_kickoff(tracking, half, first, last, spec.fps, anchors.get(half))
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
