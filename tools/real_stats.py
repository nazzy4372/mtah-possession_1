"""Label-free aggregates of a match's tracking, used to calibrate the noise.

Only prints shares / counts / histograms / percentiles, never a frame or position, so the
output is safe to share.

    .venv\\Scripts\\python.exe tools\\real_stats.py --data-dir C:\\Hackathon\\mtah-data
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import io_atd  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

GAP_BINS = (1, 2, 3, 6, 13, 26, 51, 126, 251, 751)          # lower edges, frames
FRAGMENT_BINS = (1, 2, 6, 26, 126, 751, 2501, 7501)
CROWD_BINS = (0, 1, 2, 3)                                   # 3 means "3 or more"
SPEED_BINS = (0.0, 2.0, 8.0, 15.0)                          # m/s lower edges
PERCENTILES = (5, 25, 50, 75, 95, 99)


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def histogram(lengths, edges):
    lengths = np.asarray(lengths)
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        sel = (lengths >= lo) & ((lengths < hi) if hi is not None else True)
        label = "{}-{}".format(lo, hi - 1) if hi is not None and hi - 1 != lo else (
            str(lo) if hi is not None else "{}+".format(lo))
        out.append({"bin_frames": label, "count": int(sel.sum()),
                    "frames": int(lengths[sel].sum())})
    return out


def pct(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {"p{}".format(p): round(float(np.percentile(values, p)), 3) for p in PERCENTILES}


def collect(tracking, spec):
    blocks = [(first, last + 1) for _, first, last in spec.halves]
    graded = spec.graded_mask()
    stats = {"graded_frames": int(graded.sum()), "fps": spec.fps}

    # -- ball --------------------------------------------------------------------------------
    seen = tracking.ball_observed
    stats["ball_missing_share_graded"] = round(float(1 - seen[graded].mean()), 4)
    gaps, visible, gap_rows = [], [], []
    players = tracking.players_xy
    for lo, hi in blocks:
        s, e = runs(~seen[lo:hi])
        gaps += list(e - s)
        vs, ve = runs(seen[lo:hi])
        visible += list(ve - vs)
        for start, stop in zip(s + lo, e + lo):
            if start == lo:
                continue  # no last-seen ball inside the block
            t0 = start - 1
            d = np.hypot(players[t0, :, 0] - tracking.ball_xy[t0, 0],
                         players[t0, :, 1] - tracking.ball_xy[t0, 1])
            crowd = int(np.sum(d[np.isfinite(d)] < 3.0))
            gap_rows.append((stop - start, crowd, tracking.ball_speed[t0]))
    stats["ball_missing_gaps"] = {"n": len(gaps), "hist": histogram(gaps, GAP_BINS)}
    stats["ball_visible_runs"] = {"n": len(visible), "hist": histogram(visible, GAP_BINS)}

    rows = np.array(gap_rows, dtype=float).reshape(-1, 3)
    by_crowd = []
    for i, lo in enumerate(CROWD_BINS):
        sel = rows[:, 1] >= lo if i == len(CROWD_BINS) - 1 else rows[:, 1] == lo
        by_crowd.append({"players_within_3m": "{}+".format(lo) if i == len(CROWD_BINS) - 1
                         else str(lo), "n_gaps": int(sel.sum()),
                         "mean_len": round(float(rows[sel, 0].mean()), 2) if sel.any() else None,
                         "median_len": float(np.median(rows[sel, 0])) if sel.any() else None})
    stats["gap_length_by_crowding"] = by_crowd
    by_speed = []
    for i, lo in enumerate(SPEED_BINS):
        hi = SPEED_BINS[i + 1] if i + 1 < len(SPEED_BINS) else np.inf
        sel = (rows[:, 2] >= lo) & (rows[:, 2] < hi)
        by_speed.append({"ball_speed_ms": "{}-{}".format(lo, hi), "n_gaps": int(sel.sum()),
                         "mean_len": round(float(rows[sel, 0].mean()), 2) if sel.any() else None})
    stats["gap_length_by_ball_speed"] = by_speed

    # hazard: P(ball lost on the next frame | seen now), by crowding
    hazard = []
    crowd_all = np.zeros(tracking.n_frames, dtype=np.int16)
    chunk = 20000
    for lo in range(0, tracking.n_frames, chunk):
        hi = min(tracking.n_frames, lo + chunk)
        d = np.hypot(players[lo:hi, :, 0] - tracking.ball_xy[lo:hi, None, 0],
                     players[lo:hi, :, 1] - tracking.ball_xy[lo:hi, None, 1])
        crowd_all[lo:hi] = np.sum(np.nan_to_num(d, nan=np.inf) < 3.0, axis=1)
    now = np.zeros(tracking.n_frames, dtype=bool)
    for lo, hi in blocks:
        now[lo:hi - 1] = seen[lo:hi - 1]
    lost_next = np.zeros(tracking.n_frames, dtype=bool)
    lost_next[:-1] = ~seen[1:]
    for i, lo in enumerate(CROWD_BINS):
        sel = now & ((crowd_all >= lo) if i == len(CROWD_BINS) - 1 else (crowd_all == lo))
        hazard.append({"players_within_3m": "{}+".format(lo) if i == len(CROWD_BINS) - 1
                       else str(lo), "frames": int(sel.sum()),
                       "p_lost_next_frame": round(float(lost_next[sel].mean()), 5)
                       if sel.any() else None})
    stats["ball_loss_hazard_by_crowding"] = hazard

    length, width = tracking.pitch
    bx, by = tracking.ball_xy[:, 0], tracking.ball_xy[:, 1]
    vis = seen & graded
    with np.errstate(invalid="ignore"):
        outside = (bx < 0) | (bx > length) | (by < 0) | (by > width)
        outside_margin = (bx < -1) | (bx > length + 1) | (by < -1) | (by > width + 1)
    stats["ball_outside_pitch_share_of_visible"] = round(float(outside[vis].mean()), 4)
    stats["ball_outside_pitch_1m_share_of_visible"] = round(float(outside_margin[vis].mean()), 4)

    # -- players -----------------------------------------------------------------------------
    observed = tracking.player_observed
    count = observed.sum(axis=1)
    stats["tracked_players_per_frame"] = {
        "mean": round(float(count[graded].mean()), 3), **pct(count[graded]),
        "hist": {str(k): int(np.sum(count[graded] == k)) for k in range(0, 31)},
        "over_30": int(np.sum(count[graded] > 30))}
    for team, name in ((0, "home"), (1, "away")):
        cols = tracking.slot_team == team
        team_count = observed[:, cols].sum(axis=1)
        stats["tracked_{}_per_frame".format(name)] = {
            "mean": round(float(team_count[graded].mean()), 3), **pct(team_count[graded])}
        stats["slots_declared_{}".format(name)] = int(cols.sum())
        stats["slots_used_{}".format(name)] = int(np.sum(observed[graded][:, cols].any(axis=0)))

    fragments = []
    for lo, hi in blocks:
        for k in range(observed.shape[1]):
            s, e = runs(observed[lo:hi, k])
            fragments += list(e - s)
    fragments = np.array(fragments)
    minutes = graded.sum() / spec.fps / 60.0
    stats["fragments"] = {"n": int(len(fragments)), "per_minute": round(len(fragments) / minutes, 2),
                          "mean_len": round(float(fragments.mean()), 1) if len(fragments) else None,
                          "median_len": float(np.median(fragments)) if len(fragments) else None,
                          "hist": histogram(fragments, FRAGMENT_BINS)}

    dup_frames = np.zeros(tracking.n_frames, dtype=bool)
    for team in (0, 1):
        cols = np.flatnonzero(tracking.slot_team == team)
        for lo in range(0, tracking.n_frames, chunk):
            hi = min(tracking.n_frames, lo + chunk)
            xy = players[lo:hi][:, cols]
            d = np.hypot(xy[:, :, None, 0] - xy[:, None, :, 0], xy[:, :, None, 1] - xy[:, None, :, 1])
            iu = np.triu_indices(len(cols), 1)
            close = np.nan_to_num(d[:, iu[0], iu[1]], nan=np.inf) < 0.5
            dup_frames[lo:hi] |= close.any(axis=1)
    stats["duplicate_proxy_share_frames"] = round(float(dup_frames[graded].mean()), 4)

    # -- speeds ------------------------------------------------------------------------------
    ps = tracking.players_speed[graded]
    stats["player_speed_channel_ms"] = pct(ps[np.isfinite(ps)])
    stats["ball_speed_channel_ms"] = pct(tracking.ball_speed[graded & seen])
    step = np.hypot(np.diff(players[..., 0], axis=0), np.diff(players[..., 1], axis=0))
    stats["player_step_m_per_frame"] = pct(step[graded[1:] & graded[:-1]].ravel())
    bstep = np.hypot(np.diff(bx), np.diff(by))
    stats["ball_step_m_per_frame"] = pct(bstep[graded[1:] & graded[:-1]])
    stats["layout_check"] = tracking.meta.layout_check
    stats["n_slots"] = tracking.meta.n_slots
    return stats


def show(stats, indent=""):
    for key, value in stats.items():
        if isinstance(value, dict):
            print("{}{}:".format(indent, key))
            show(value, indent + "  ")
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print("{}{}:".format(indent, key))
            for row in value:
                print("{}  {}".format(indent, "  ".join("{}={}".format(k, v)
                                                        for k, v in row.items())))
        else:
            print("{}{}: {}".format(indent, key, value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    parser.add_argument("--json", help="also write the aggregates to this JSON file")
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("pass --data-dir or set MTAH_DATA_DIR")
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
    tracking = io_atd.load_game(args.data_dir, spec)
    stats = collect(tracking, spec)
    show(stats)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
