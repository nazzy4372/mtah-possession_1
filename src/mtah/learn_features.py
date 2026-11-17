"""Feature matrix for the learned models. Per frame, no player identities, H/A swappable.

- player speed only enters relative to its own half (SkillCorner runs on a different speed
  scale from Metrica and the real match)
- every home/away pair can be swapped, so training mirrors H <-> A
- rolling windows of +-0.5/1/2/5 s, plus 2 s look-back/look-ahead (offline, so we can)

GROUP_GAP and GROUP_DEADBALL are newer feature groups that no shipped model uses.
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

    # --- ball gaps (GROUP_GAP). the real ball goes missing mostly when play is slow,
    # so only use these with the "speed" ball model in training
    known = F.ball_known
    frame = np.arange(n)
    last_known = np.maximum.accumulate(np.where(known, frame, -1))
    next_known = np.minimum.accumulate(np.where(known, frame, 10 ** 9)[::-1])[::-1]
    has_last, has_next = last_known >= 0, next_known < 10 ** 9
    elapsed = np.where(known, 0.0, np.where(has_last, (frame - last_known) / fps, 60.0))
    remaining = np.where(known, 0.0, np.where(has_next, (next_known - frame) / fps, 60.0))
    cols["gap_len"] = np.clip(np.where(known, 0.0, elapsed + remaining), 0, 120)
    cols["gap_elapsed"] = np.clip(elapsed, 0, 60)
    cols["gap_remaining"] = np.clip(remaining, 0, 60)
    with np.errstate(invalid="ignore"):
        line = np.minimum.reduce([bx, length - bx, by, width - by])      # negative = off the pitch
    cols["last_seen_line"] = np.clip(np.where(has_last, line[np.maximum(last_known, 0)], np.nan),
                                     -5, 34)                             # ball kicked out?
    cols["next_seen_line"] = np.clip(np.where(has_next, line[np.minimum(next_known, n - 1)],
                                              np.nan), -5, 34)           # restart from a line?
    cols["rank_at_last_seen"] = np.where(has_last, ps_rank[np.maximum(last_known, 0)], np.nan)
    cols["known_share_m5"] = _nanroll(known.astype(float), s(5))
    # a whistle: players slow down sharply from the 3 s before to the 3 s after
    cols["ps_drop3"] = _nanroll(ps_rank, s(3), "past") - _nanroll(ps_rank, s(3), "future")
    out_now = known & (np.nan_to_num(F.out_dist, nan=-1.0) > 0)
    last_out = np.maximum.accumulate(np.where(out_now, frame, -1))
    cols["since_out"] = np.clip(np.where(last_out >= 0, (frame - last_out) / fps, 60.0), 0, 60)

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

    # --- touch sequence (GROUP_TOUCH): who's on the ball just before / after
    dh = np.nan_to_num(F.d_home, nan=np.inf)
    da = np.nan_to_num(F.d_away, nan=np.inf)
    touch = {"home": ((dh < feat.TOUCH_M) & (dh + feat.TOUCH_MARGIN_M < da)).astype(float),
             "away": ((da < feat.TOUCH_M) & (da + feat.TOUCH_MARGIN_M < dh)).astype(float)}
    idx = np.arange(n)
    for tname, mask in touch.items():
        for w in (1, 3):
            cols["touch_{}_past{}".format(tname, w)] = _nanroll(mask, s(w), "past")
            cols["touch_{}_next{}".format(tname, w)] = _nanroll(mask, s(w), "future")
        last = np.maximum.accumulate(np.where(mask > 0, idx, -1))
        nxt = np.minimum.accumulate(np.where(mask > 0, idx, 10 ** 9)[::-1])[::-1]
        cols["since_" + tname] = np.clip(np.where(last >= 0, (idx - last) / fps, 15.0), 0, 15)
        cols["until_" + tname] = np.clip(np.where(nxt < 10 ** 9, (nxt - idx) / fps, 15.0), 0, 15)
    cols["touch_delta1"] = (cols["touch_home_next1"] - cols["touch_away_next1"]) \
        - (cols["touch_home_past1"] - cols["touch_away_past1"])
    cols["margin_past1"] = _nanroll(margin, s(1), "past")
    cols["margin_next1"] = _nanroll(margin, s(1), "future")
    cols["margin_delta1"] = cols["margin_next1"] - cols["margin_past1"]

    # ball kinematics: a kick or a first touch changes the ball's velocity vector
    k = max(1, s(0.2))
    v_before = np.full((n, 2), np.nan)
    v_after = np.full((n, 2), np.nan)
    v_before[k:] = (F.ball_xy[k:] - F.ball_xy[:-k]) * (fps / float(k))
    v_after[:-k] = (F.ball_xy[k:] - F.ball_xy[:-k]) * (fps / float(k))
    kick = np.hypot(v_after[:, 0] - v_before[:, 0], v_after[:, 1] - v_before[:, 1])
    cols["ball_kick"] = np.clip(kick, 0, 40)
    cols["ball_kick_max05"] = np.clip(_nanrollmax(kick, s(0.5)), 0, 40)
    home_dir = np.zeros(n)
    for half, first, last_f in spec.halves:
        home_dir[first:last_f + 1] = feat.attack_directions(tracking, spec)[(0, half)]
    vx = (v_before[:, 0] + v_after[:, 0]) / 2.0
    cols["ball_vx_home"] = _nanroll(np.clip(vx, -30, 30) * home_dir, s(0.5))
    cols["ball_adv_home"] = (F.ball_xy[:, 0] - length / 2.0) * home_dir

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

    # --- dead-ball evidence from a collective slowdown (deadball.py), for later models only
    from mtah import deadball
    cols.update(deadball.features(deadball.compute(tracking, spec)))

    names = list(cols)
    X = np.column_stack([np.asarray(cols[k], dtype=np.float64) for k in names]).astype(np.float32)
    graded = spec.graded_mask()
    X[~graded] = np.nan
    return names, X


# Mirroring H <-> A: pairs swap columns, signed columns flip sign.
PAIRS = (("d_home", "d_away"), ("n_home_2", "n_away_2"), ("n_home_5", "n_away_5"),
         ("adv_home", "adv_away"), ("spread_home", "spread_away"),
         ("tracked_home", "tracked_away"),
         ("touch_home_past1", "touch_away_past1"), ("touch_home_next1", "touch_away_next1"),
         ("touch_home_past3", "touch_away_past3"), ("touch_home_next3", "touch_away_next3"),
         ("since_home", "since_away"), ("until_home", "until_away"))
SIGNED = ("margin", "margin_m05", "margin_m1", "margin_m3", "touch_signed", "shape_diff",
          "shape_diff_m2", "touch_delta1", "margin_past1", "margin_next1", "margin_delta1",
          "ball_vx_home", "ball_adv_home")

# Ball-gap and stoppage cues, for models trained on the "speed" ball model only.
GROUP_GAP = ("gap_len", "gap_elapsed", "gap_remaining", "last_seen_line", "next_seen_line",
             "rank_at_last_seen", "known_share_m5", "ps_drop3", "since_out")

# slowdown detector signals - no shipped model uses them
GROUP_DEADBALL = ("db_ratio", "db_candidate", "db_box", "db_cue_out", "db_cue_still",
                  "db_cue_missing", "db_score")

# added after M5. separate group so the v2 feature set still rebuilds
GROUP_TOUCH = ("touch_home_past1", "touch_away_past1", "touch_home_next1", "touch_away_next1",
               "touch_home_past3", "touch_away_past3", "touch_home_next3", "touch_away_next3",
               "since_home", "since_away", "until_home", "until_away", "touch_delta1",
               "margin_past1", "margin_next1", "margin_delta1", "ball_kick", "ball_kick_max05",
               "ball_vx_home", "ball_adv_home")


def swap_teams(names, X):
    """X with home and away exchanged (labels must swap H <-> A alongside)."""
    idx = {name: i for i, name in enumerate(names)}
    Y = X.copy()
    for a, b in PAIRS:
        Y[:, idx[a]], Y[:, idx[b]] = X[:, idx[b]], X[:, idx[a]]
    for name in SIGNED:
        Y[:, idx[name]] = -X[:, idx[name]]
    return Y
