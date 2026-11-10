"""Boundary snapping after decoding: move each transition to the physical event that caused it.

The decoder places boundaries where the accumulated evidence tips over, which lags the real event.
spellF1 is sensitive to timing (8 frames costs about 0.1), so each boundary may move inside a
small window to:
  * H <-> A: the first frame a player of the new team is within ``r_m`` of the ball and nearer
    than the old team;
  * alive -> D: the first frame the ball is outside the pitch;
  * D -> alive: the first frame the ball moves faster than ``kick_ms`` (the restart kick).
A boundary never moves past its neighbours, and nothing moves when no such frame exists.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RefineParams:
    window_s: float = 1.5
    r_m: float = 1.5
    out_window_s: float = 2.0
    kick_ms: float = 3.0


def _boundaries(labels, first, last):
    seg = labels[first:last + 1]
    return [first + int(i) for i in np.flatnonzero(seg[1:] != seg[:-1]) + 1]


def refine(labels: np.ndarray, F, spec, params: RefineParams = RefineParams(), kickoffs=()) -> np.ndarray:
    out = labels.copy()
    fps = spec.fps
    w = int(round(params.window_s * fps))
    w_out = int(round(params.out_window_s * fps))
    fixed = {k.frame for k in kickoffs}
    d = {"H": F.d_home, "A": F.d_away}
    for half, first, last in spec.halves:
        bounds = _boundaries(labels, first, last)
        for i, b in enumerate(bounds):
            if b in fixed:
                continue
            lo_lim = (bounds[i - 1] + 1) if i > 0 else first + 1
            hi_lim = (bounds[i + 1] - 1) if i + 1 < len(bounds) else last
            old, new = labels[b - 1], labels[b]
            target = None
            if old in "HA" and new in "HA":
                lo, hi = max(lo_lim, b - w), min(hi_lim, b + w)
                with np.errstate(invalid="ignore"):
                    ok = (d[new][lo:hi + 1] < params.r_m) & (d[new][lo:hi + 1] < d[old][lo:hi + 1])
                hits = np.flatnonzero(ok)
                target = lo + int(hits[0]) if len(hits) else None
            elif old in "HA" and new == "D":
                lo, hi = max(lo_lim, b - w_out), min(hi_lim, b + w_out // 2)
                with np.errstate(invalid="ignore"):
                    hits = np.flatnonzero(F.out_dist[lo:hi + 1] > 0.0)
                target = lo + int(hits[0]) if len(hits) else None
            elif old == "D" and new in "HA":
                lo, hi = max(lo_lim, b - w), min(hi_lim, b + w)
                with np.errstate(invalid="ignore"):
                    hits = np.flatnonzero(F.ball_speed[lo:hi + 1] > params.kick_ms)
                target = lo + int(hits[0]) if len(hits) else None
            if target is None or target == b:
                continue
            if target < b:
                out[target:b] = new
            else:
                out[b:target] = old
    return out
