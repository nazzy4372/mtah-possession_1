"""Dead-ball evidence from a collective slowdown of the players.

When play stops, everybody slows down within a second or three. This module turns that into
EVIDENCE for the dead-ball state: a per-frame score that is added to the dead-ball emission and
then weighed by the decoder against everything else. It never writes a label.

1. Signal. Per frame, the mean of the `top_n` highest player speeds (speeds from positions,
   clipped at `max_speed_ms` against tracker jumps), smoothed with a centred rolling median. Its
   baseline is the rolling median over the previous `baseline_s` seconds, leaving out the most
   recent `baseline_skip_s`. A slowdown candidate is a run of at least `min_slow_s` seconds with
   smoothed / baseline <= `ratio_max`.
2. Exclusion. A candidate whose ball is inside either penalty area or its arc is dropped: a
   goalkeeper holding the ball, or a slow build-up near the box, slows everyone down while the
   ball is alive. The ball position is the tracked one, or the last one seen within
   `last_seen_s`; if it is older than that, the candidate is kept at lower confidence.
3. Timing. Players slow after the ball goes dead, so the dead stretch starts earlier: the search
   goes back up to `back_search_s` for the ball crossing a pitch line, then a sharp ball stop,
   then the ball disappearing. It ends at the restart: the ball moving off after resting, or the
   top speeds recovering to `recover_ratio` of the baseline at the onset.
4. Guard. Slow build-up also lowers speeds, so a candidate only counts with one more cue: ball
   outside the pitch, ball at rest for `still_min_s`, ball missing for `missing_min_s`, or a
   tagged set piece starting within `tag_window_s`.

Every number in DeadballParams is a HYPOTHESIS, to be tuned on external games only, never on the
provided match or its public ground truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# IFAB pitch markings, metres
BOX_DEPTH_M = 16.5
BOX_WIDTH_M = 40.32
PENALTY_SPOT_M = 11.0
ARC_RADIUS_M = 9.15

CUES = ("out", "still", "missing", "tag")
START_REASONS = ("line crossing", "ball stop", "ball disappears", "slowdown onset")


@dataclass(frozen=True)
class DeadballParams:
    top_n: int = 7                 # players whose speeds are averaged
    min_players: int = 10          # fewer tracked players than this: low confidence
    max_speed_ms: float = 12.0     # a faster "player" is a tracker jump
    smooth_s: float = 0.5          # centred rolling median
    baseline_s: float = 10.0       # baseline window ...
    baseline_skip_s: float = 1.0   # ... ending this long before the frame
    ratio_max: float = 0.60        # slowdown: smoothed / baseline at or below this
    min_slow_s: float = 1.0        # ... for at least this long
    box_exclusion: bool = True
    last_seen_s: float = 2.0       # an untracked ball keeps its last position this long
    back_search_s: float = 3.0     # how far before the slowdown the dead ball can start
    line_margin_m: float = 0.5
    stop_from_ms: float = 4.0      # sharp stop: from above this ...
    still_ms: float = 0.5          # ... to below this ...
    stop_within_s: float = 0.4     # ... within this
    still_min_s: float = 1.0
    missing_min_s: float = 1.0
    restart_speed_ms: float = 2.0  # ball moving off after resting
    recover_ratio: float = 0.8     # top speeds back to this share of the onset baseline
    tag_window_s: float = 3.0
    require_cue: bool = True
    max_dead_s: float = 120.0
    low_confidence: float = 0.5    # score of a low-confidence detection
    weight: float = 2.0            # nats per frame added to the dead-ball emission (rules v1)


@dataclass
class Interval:
    start: int                     # first dead frame
    end: int                       # first frame after the dead stretch
    onset: int                     # first frame of the slowdown
    reason: str                    # how the start was found (START_REASONS)
    cues: Tuple[str, ...]
    confidence: float


@dataclass
class Evidence:
    score: np.ndarray              # (n,) 0..1, the dead-ball evidence
    top_speed: np.ndarray          # (n,) smoothed mean of the top speeds, m/s
    baseline: np.ndarray           # (n,)
    ratio: np.ndarray              # (n,) top_speed / baseline, NaN where undefined
    candidate: np.ndarray          # (n,) bool, inside a slowdown candidate
    low_confidence: np.ndarray     # (n,) bool, too few players tracked
    in_box: np.ndarray             # (n,) bool, ball (or its last position) in a penalty area
    cue_out: np.ndarray            # (n,) bool, ball outside the pitch
    cue_still: np.ndarray          # (n,) bool, ball at rest for still_min_s
    cue_missing: np.ndarray        # (n,) bool, ball untracked for missing_min_s
    intervals: List[Interval] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


def _frames(seconds: float, fps: int) -> int:
    return max(1, int(round(seconds * fps)))


def _run_length(mask: np.ndarray) -> np.ndarray:
    """Per frame, how many consecutive True values end at it (0 where False)."""
    idx = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(~mask, idx, -1))
    return np.where(mask, idx - last_false, 0)


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())


def in_penalty_zone(x: np.ndarray, y: np.ndarray, pitch) -> np.ndarray:
    """True where (x, y) lies inside either penalty area or its arc (and inside the pitch)."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        inside = (x >= 0) & (x <= length) & (y >= 0) & (y <= width)
        depth = np.minimum(x, length - x)                    # distance from the nearer goal line
        box = (depth <= BOX_DEPTH_M) & (np.abs(y - width / 2.0) <= BOX_WIDTH_M / 2.0)
        arc = np.hypot(depth - PENALTY_SPOT_M, y - width / 2.0) <= ARC_RADIUS_M
        return inside & (box | arc)


def top_speed(tracking, fps: int, p: DeadballParams):
    """(mean of the top_n player speeds per frame, number of players with a speed)."""
    xy = tracking.players_xy
    step = np.full(xy.shape[:2], np.nan)
    step[1:-1] = np.hypot(xy[2:, :, 0] - xy[:-2, :, 0], xy[2:, :, 1] - xy[:-2, :, 1]) * fps / 2.0
    tracked = np.isfinite(step)
    filled = np.where(tracked, np.minimum(step, p.max_speed_ms), -1.0)
    k = min(p.top_n, filled.shape[1])
    top = -np.partition(-filled, k - 1, axis=1)[:, :k]
    valid = top >= 0
    count = valid.sum(axis=1)
    mean = np.where(count > 0, (top * valid).sum(axis=1) / np.maximum(count, 1), np.nan)
    return mean, tracked.sum(axis=1)


def compute(tracking, spec, params: DeadballParams = DeadballParams(),
            set_piece_starts: Optional[np.ndarray] = None) -> Evidence:
    """Dead-ball evidence for every frame. `set_piece_starts` are the video frames at which the
    Smart Tagging starts a set piece (optional, supporting evidence only)."""
    p = params
    fps = spec.fps
    n = tracking.n_frames
    length, width = tracking.pitch

    raw, n_tracked = top_speed(tracking, fps, p)
    width_smooth = _frames(p.smooth_s, fps) | 1
    smooth = pd.Series(raw).rolling(width_smooth, center=True, min_periods=1).median().to_numpy()
    low_conf = n_tracked < p.min_players

    win, skip = _frames(p.baseline_s, fps), _frames(p.baseline_skip_s, fps)
    baseline = np.full(n, np.nan)
    for _, first, last in spec.halves:        # never across the half-time gap
        seg = pd.Series(raw[first:last + 1])
        baseline[first:last + 1] = seg.rolling(win, min_periods=max(1, win // 4)).median() \
            .shift(skip + 1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(baseline > 0.05, smooth / baseline, np.nan)
        slow = np.nan_to_num(ratio, nan=np.inf) <= p.ratio_max

    # --- ball
    observed = tracking.ball_observed
    ball = tracking.ball_xy
    idx = np.arange(n)
    seen_at = np.maximum.accumulate(np.where(observed, idx, -1))
    age_s = np.where(seen_at >= 0, (idx - seen_at) / float(fps), np.inf)
    last_xy = ball[np.maximum(seen_at, 0)]
    position_known = age_s <= p.last_seen_s
    in_box = position_known & in_penalty_zone(last_xy[:, 0], last_xy[:, 1], tracking.pitch)

    speed = np.full(n, np.nan)
    speed[2:-2] = np.hypot(ball[4:, 0] - ball[:-4, 0], ball[4:, 1] - ball[:-4, 1]) * fps / 4.0
    with np.errstate(invalid="ignore"):
        m = p.line_margin_m
        outside = observed & ((ball[:, 0] < -m) | (ball[:, 0] > length + m)
                              | (ball[:, 1] < -m) | (ball[:, 1] > width + m))
        still = np.nan_to_num(speed, nan=np.inf) < p.still_ms
    still_run = _run_length(still)
    missing_run = _run_length(~observed)
    cue_still = still_run >= _frames(p.still_min_s, fps)
    cue_missing = missing_run >= _frames(p.missing_min_s, fps)

    crossing = outside & ~np.concatenate(([False], outside[:-1]))
    recent_fast = pd.Series(np.nan_to_num(speed, nan=0.0)).rolling(
        _frames(p.stop_within_s, fps), min_periods=1).max().to_numpy() >= p.stop_from_ms
    stop = still & recent_fast & ~np.concatenate(([False], still[:-1]))
    vanish = ~observed & np.concatenate(([False], observed[:-1]))
    rested = pd.Series(still_run).rolling(_frames(p.stop_within_s, fps) + 1,
                                           min_periods=1).max().to_numpy()
    moved_off = (np.nan_to_num(speed, nan=0.0) > p.restart_speed_ms) \
        & (rested >= _frames(p.still_min_s, fps))

    tags = np.sort(np.asarray(set_piece_starts if set_piece_starts is not None else [],
                              dtype=np.int64))

    # --- candidates -> dead stretches
    half_end = np.zeros(n, dtype=np.int64)       # 0 outside the graded halves
    half_first = np.zeros(n, dtype=np.int64)
    for _, first, last in spec.halves:
        half_end[first:last + 1] = last + 1
        half_first[first:last + 1] = first
    candidate = np.zeros(n, dtype=bool)
    stats = {"candidates": 0, "excluded_box": 0, "no_extra_cue": 0, "accepted": 0,
             "low_confidence": 0}
    stats.update({"cue_" + c: 0 for c in CUES})
    stats.update({"start_" + r.replace(" ", "_"): 0 for r in START_REASONS})
    intervals: List[Interval] = []
    back = _frames(p.back_search_s, fps)
    for onset, cand_end in _runs(slow):
        if cand_end - onset < _frames(p.min_slow_s, fps) or half_end[onset] == 0:
            continue
        candidate[onset:cand_end] = True
        stats["candidates"] += 1
        if p.box_exclusion and in_box[onset]:
            stats["excluded_box"] += 1
            continue

        lo = max(onset - back, int(half_first[onset]))
        window = slice(lo, onset + 1)
        start, reason = onset, START_REASONS[3]
        for mask, name in ((crossing, START_REASONS[0]), (stop, START_REASONS[1]),
                           (vanish, START_REASONS[2])):
            hits = np.flatnonzero(mask[window])
            if len(hits):
                start, reason = lo + int(hits[0] if name == START_REASONS[0] else hits[-1]), name
                break

        limit = min(int(half_end[onset]), onset + _frames(p.max_dead_s, fps))
        with np.errstate(invalid="ignore"):
            recovered = smooth[onset:limit] > p.recover_ratio * baseline[onset]
        ends = [limit]
        for mask in (recovered, moved_off[onset:limit]):
            hits = np.flatnonzero(mask)
            if len(hits):
                ends.append(onset + int(hits[0]))
        end = max(min(ends), onset + 1)

        cues = []
        if outside[start:end].any():
            cues.append("out")
        if cue_still[start:end].any():
            cues.append("still")
        if cue_missing[start:end].any():
            cues.append("missing")
        w = _frames(p.tag_window_s, fps)
        k = np.searchsorted(tags, start - w)
        if k < len(tags) and tags[k] <= end + w:
            cues.append("tag")
        if p.require_cue and not cues:
            stats["no_extra_cue"] += 1
            continue

        unsure = low_conf[onset:cand_end].mean() > 0.5 or not position_known[onset]
        confidence = p.low_confidence if unsure else 1.0
        intervals.append(Interval(start, end, onset, reason, tuple(cues), confidence))
        stats["accepted"] += 1
        stats["low_confidence"] += int(unsure)
        stats["start_" + reason.replace(" ", "_")] += 1
        for c in cues:
            stats["cue_" + c] += 1

    score = np.zeros(n)
    for iv in intervals:
        score[iv.start:iv.end] = np.maximum(score[iv.start:iv.end], iv.confidence)
    return Evidence(score=score, top_speed=smooth, baseline=baseline, ratio=ratio,
                    candidate=candidate, low_confidence=low_conf, in_box=in_box,
                    cue_out=outside, cue_still=cue_still, cue_missing=cue_missing,
                    intervals=intervals, stats=stats)


def add_evidence(logp: np.ndarray, evidence: Evidence, weight: float) -> np.ndarray:
    """logp (n, 3) over (H, A, D) with weight * score added to the dead-ball column. The decoder
    still decides: the evidence makes a dead ball cheaper where it fired, nothing more."""
    out = logp.copy()
    out[:, 2] += weight * evidence.score
    return out


FEATURES = ("db_ratio", "db_candidate", "db_box", "db_cue_out", "db_cue_still", "db_cue_missing",
            "db_score")


def features(evidence: Evidence) -> Dict[str, np.ndarray]:
    """The intermediate signals as per-frame features for a learned model. All are the same for
    both teams, so they need no mirroring. The ratio is scale-free across tracking sources."""
    return {"db_ratio": np.clip(evidence.ratio, 0.0, 3.0),
            "db_candidate": evidence.candidate.astype(float),
            "db_box": evidence.in_box.astype(float),
            "db_cue_out": evidence.cue_out.astype(float),
            "db_cue_still": evidence.cue_still.astype(float),
            "db_cue_missing": evidence.cue_missing.astype(float),
            "db_score": evidence.score}


def summary(evidence: Evidence, spec) -> List[Dict]:
    """Aggregates per half: intervals, durations, share. No frame numbers."""
    rows = []
    for half, first, last in spec.halves:
        durations = [(min(iv.end, last + 1) - max(iv.start, first)) / float(spec.fps)
                     for iv in evidence.intervals if first <= iv.onset <= last]
        rows.append({"half": half, "intervals": len(durations),
                     "median_s": float(np.median(durations)) if durations else 0.0,
                     "share": float(np.mean(evidence.score[first:last + 1] > 0))})
    return rows
