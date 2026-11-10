"""M1 placeholder labeller: the upstream baseline, vectorised over our parsed arrays.

Replaced by real emissions and the semi-Markov decoder in M2. It exists so that the pipeline
produces a genuine, valid file end to end, and so that tests/test_parity.py can prove that our
parser and frame indexing agree exactly with upstream ``starter/baseline.py``.

Upstream semantics preserved exactly: distances in metres from normalised coordinates times the
pitch size, nearest tracked player wins, ties go to the lowest slot index (strict ``<`` there,
first ``argmin`` here), and a frame with no ball has no answer. Smoothing is upstream's own
``baseline.smooth``, imported rather than copied.
"""

import numpy as np

from mtah.official import baseline

NOBODY = -1
SMOOTH_WINDOW = 50  # +/- frames, upstream's published floor (baseline_smooth50)


def nearest_player_codes(tracking, chunk: int = 20000) -> np.ndarray:
    """Per frame: 0 (home) / 1 (away) for the team of the nearest tracked player, -1 if none."""
    n = tracking.n_frames
    out = np.full(n, NOBODY, dtype=np.int8)
    ball_seen = ~np.isnan(tracking.ball_xy[:, 0])
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        dx = tracking.players_xy[lo:hi, :, 0] - tracking.ball_xy[lo:hi, None, 0]
        dy = tracking.players_xy[lo:hi, :, 1] - tracking.ball_xy[lo:hi, None, 1]
        dist = dx * dx + dy * dy
        player_missing = np.isnan(tracking.players_xy[lo:hi, :, 0])
        dist[player_missing] = np.inf
        best = np.argmin(dist, axis=1)
        found = np.isfinite(dist[np.arange(hi - lo), best]) & ball_seen[lo:hi]
        out[lo:hi][found] = tracking.slot_team[best[found]]
    return out


def placeholder_labels(tracking, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """H/A/D per frame: nearest player, majority vote over +/- window, D where no ball nearby."""
    return np.asarray(baseline.smooth(nearest_player_codes(tracking), window), dtype="<U1")
