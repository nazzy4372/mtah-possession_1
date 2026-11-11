"""Feature matrix for the learned emission model (v2): per frame, identity-free, team-swappable.

Built on features.Features plus a few counts. Design rules, from the M3 domain-gap check
(tools/feature_stats.py, real match vs external games):
  * player speed enters only RELATIVE to its own half (rank and robust z-score). The real match
    and Metrica share a speed scale but SkillCorner does not, so an absolute threshold would not
    transfer between sources;
  * "is the ball missing" is never a feature in its own right. Ball features are NaN when the ball
    is unknown, and the training data is augmented so that missingness carries almost no dead-ball
    signal (ext/augment.py), as on the real match;
  * every home/away pair can be swapped (see SWAP) so training can mirror H <-> A, which removes
    any home bias and doubles the data.

Rolling windows (centred unless named otherwise) give the model context: +/-0.5, 1, 2 and 5 s,
plus a 2 s look-back and look-ahead for the speed signal (offline labelling, so the future is
available).
"""

import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d

from mtah import features as feat

SPRINT_MS = 4.0
TOUCH_FADE_S = 8.0

# feature groups (for ablations)
GROUP_SHAPE = ("shape_diff", "shape_diff_m2", "adv_home", "adv_away", "spread_home",
               "spread_away")
GROUP_SPEED_RAW = ("ps_raw", "ps_raw_m2")


def _nanroll(x, half_width, mode="centred"):
    """NaN-ignoring moving average. mode: centred | past | future."""
    n = len(x)
    valid = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(n)
    if mode == "centred":
        lo, hi = idx - half_width, idx + half_width + 1
    elif mode == "past":
        lo, hi = idx - half_width, idx + 1
    else:
        lo, hi = idx, idx + half_width + 1
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _nanrollmax(x, half_width):
    filled = np.where(np.isfinite(x), x, -np.inf)
    out = maximum_filter1d(filled, size=2 * half_width + 1, mode="nearest")
    return np.where(np.isfinite(out), out, np.nan)


def _relative(x, spec):
    """(rank in [0, 1], robust z) of x within each graded half; NaN elsewhere."""
    rank = np.full(len(x), np.nan)
    z = np.full(len(x), np.nan)
    for _, first, last in spec.halves:
        seg = x[first:last + 1]
        valid = np.isfinite(seg)
        if valid.sum() < 10:
            continue
        vals = seg[valid]
        order = np.argsort(vals, kind="stable")
        r = np.empty(len(vals))
        r[order] = np.arange(len(vals)) / float(len(vals) - 1)
        rr = np.full(len(seg), np.nan)
        rr[valid] = r
        rank[first:last + 1] = rr
        med = np.median(vals)
        iqr = np.subtract(*np.percentile(vals, [75, 25])) or 1.0
        z[first:last + 1] = (seg - med) / iqr
    return rank, z


def build(tracking, spec, F=None):
    """(names, X float32 (n_frames, n_features)). Values outside the graded halves are NaN."""
    fps = spec.fps
    F = F if F is not None else feat.compute(tracking, spec)
    n = tracking.n_frames
    length, width = tracking.pitch
    s = lambda sec: int(round(sec * fps))  # noqa: E731
    cols = {}

    # --- stoppage: relative player speed, sprinting share
    ps_rank, ps_z = _relative(F.player_speed, spec)
    cols["ps_rank"], cols["ps_z"] = ps_rank, ps_z
    for name, w in (("m05", 0.5), ("m1", 1), ("m2", 2), ("m5", 5)):
        cols["ps_rank_" + name] = _nanroll(ps_rank, s(w))
    cols["ps_rank_past2"] = _nanroll(ps_rank, s(2), "past")
    cols["ps_rank_next2"] = _nanroll(ps_rank, s(2), "future")
    cols["ps_raw"] = F.player_speed
    cols["ps_raw_m2"] = _nanroll(F.player_speed, s(2))

    p = tracking.players_xy
    step = np.full(p.shape[:2], np.nan)
    step[1:-1] = np.hypot(p[2:, :, 0] - p[:-2, :, 0], p[2:, :, 1] - p[:-2, :, 1]) * fps / 2.0
    tracked = ~np.isnan(step)
    with np.errstate(invalid="ignore"):
        sprint = np.where(tracked.sum(1) > 0, (step > SPRINT_MS).sum(1) / np.maximum(tracked.sum(1), 1),
                          np.nan)
    cols["sprint_share"] = sprint
    cols["sprint_share_m2"] = _nanroll(sprint, s(2))

    # --- ball
    with np.errstate(invalid="ignore", divide="ignore"):
        bspeed = np.log1p(F.ball_speed)
    cols["ball_speed_log"] = bspeed
    cols["ball_speed_log_m1"] = _nanroll(bspeed, s(1))
    still = np.where(np.isfinite(F.ball_speed), (F.ball_speed < 0.5).astype(float), np.nan)
    cols["ball_still_m1"] = _nanroll(still, s(1))
    cols["ball_still_m5"] = _nanroll(still, s(5))
    cols["out_dist"] = np.clip(F.out_dist, -40, 10)
    cols["out_max_m2"] = np.clip(_nanrollmax(F.out_dist, s(2)), -40, 10)
    bx, by = F.ball_xy[:, 0], F.ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        cols["line_dist_y"] = np.minimum(by, width - by)            # to the nearest touchline
        cols["line_dist_x"] = np.minimum(bx, length - bx)           # to the nearest goal line
        cols["centre_dist"] = np.hypot(bx - length / 2.0, by - width / 2.0)

    # --- proximity
    cols["d_home"] = np.clip(F.d_home, 0, 40)
    cols["d_away"] = np.clip(F.d_away, 0, 40)
    margin = np.clip(F.d_away - F.d_home, -30, 30)
    cols["margin"] = margin
    for name, w in (("m05", 0.5), ("m1", 1), ("m3", 3)):
        cols["margin_" + name] = _nanroll(margin, s(w))
    for r in (2.0, 5.0):
        d = np.hypot(p[..., 0] - F.ball_xy[:, None, 0], p[..., 1] - F.ball_xy[:, None, 1])
        close = np.nan_to_num(d, nan=np.inf) < r
        known = F.ball_known
        for team, tname in ((0, "home"), (1, "away")):
            cnt = close[:, tracking.slot_team == team].sum(1).astype(float)
            cols["n_{}_{}".format(tname, int(r))] = np.where(known, cnt, np.nan)

    # --- last touch
    fade = np.exp(-np.where(np.isfinite(F.since_touch_s), F.since_touch_s, 1e9) / TOUCH_FADE_S)
    cols["touch_signed"] = F.last_touch * fade
    cols["since_touch"] = np.clip(np.where(np.isfinite(F.since_touch_s), F.since_touch_s, 120), 0, 120)

    # --- shape
    cols["shape_diff"] = F.shape_diff
    cols["shape_diff_m2"] = _nanroll(F.shape_diff, s(2))
    directions = feat.attack_directions(tracking, spec)
    observed = tracking.player_observed
    for team, tname in ((0, "home"), (1, "away")):
        adv = np.full(n, np.nan)
        spread = np.full(n, np.nan)
        tcount = np.full(n, np.nan)
        cols_t = tracking.slot_team == team
        for half, first, last in spec.halves:
            sl = slice(first, last + 1)
            x = np.where(observed[sl][:, cols_t], p[sl][:, cols_t, 0], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cx = np.nanmean(x, axis=1)
                sx = np.nanstd(x, axis=1)
            adv[sl] = (cx - length / 2.0) * directions[(team, half)]
            spread[sl] = sx
            tcount[sl] = observed[sl][:, cols_t].sum(1)
        cols["adv_" + tname] = feat._smooth(adv, s(1))
        cols["spread_" + tname] = feat._smooth(spread, s(1))
        cols["tracked_" + tname] = tcount

    names = list(cols)
    X = np.column_stack([np.asarray(cols[k], dtype=np.float64) for k in names]).astype(np.float32)
    graded = spec.graded_mask()
    X[~graded] = np.nan
    return names, X


# Mirroring H <-> A: pairs swap columns, signed columns flip sign.
PAIRS = (("d_home", "d_away"), ("n_home_2", "n_away_2"), ("n_home_5", "n_away_5"),
         ("adv_home", "adv_away"), ("spread_home", "spread_away"),
         ("tracked_home", "tracked_away"))
SIGNED = ("margin", "margin_m05", "margin_m1", "margin_m3", "touch_signed", "shape_diff",
          "shape_diff_m2")


def swap_teams(names, X):
    """X with home and away exchanged (labels must swap H <-> A alongside)."""
    idx = {name: i for i, name in enumerate(names)}
    Y = X.copy()
    for a, b in PAIRS:
        Y[:, idx[a]], Y[:, idx[b]] = X[:, idx[b]], X[:, idx[a]]
    for name in SIGNED:
        Y[:, idx[name]] = -X[:, idx[name]]
    return Y
