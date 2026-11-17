"""What does the ball do when the players are slow? Label-free.

Bins frames by player-speed rank within the half and shows, per bin, whether the ball is tracked,
at rest, near a player, off the pitch. Works on the real match and on external games.

    .venv\\Scripts\\python.exe tools\\slow_play_stats.py --data-dir C:\\Hackathon\\mtah-data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from mtah import features, io_atd, learn_features  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

BINS = ((0.0, 0.10, "slowest 10%"), (0.10, 0.25, "10-25%"), (0.25, 0.50, "25-50%"),
        (0.50, 1.01, "fastest 50%"))
NEAR_M = 1.5
LINE_M = 2.0
DROP = 0.35          # fall in speed rank between the 3 s before and the 3 s after


def describe(mask, F, line_dist):
    """Aggregates over the frames in mask."""
    n = int(mask.sum())
    if n == 0:
        return None
    known = F.ball_known[mask]
    k = mask & F.ball_known
    nearest = np.fmin(F.d_home, F.d_away)[k]
    speed = F.ball_speed[k]
    return {
        "frames": n,
        "ball known": float(known.mean()),
        "at rest": float(np.mean(np.nan_to_num(speed, nan=99.0) < 0.5)) if k.any() else np.nan,
        "ball speed p50": float(np.nanmedian(speed)) if k.any() else np.nan,
        "with a player": float(np.mean(nearest < NEAR_M)) if k.any() else np.nan,
        "moving + player": float(np.mean((nearest < NEAR_M) & (np.nan_to_num(speed, nan=0.0)
                                                             >= 0.5))) if k.any() else np.nan,
        "off pitch": float(np.mean(F.out_dist[k] > 0.5)) if k.any() else np.nan,
        "near a line": float(np.mean(line_dist[k] < LINE_M)) if k.any() else np.nan,
    }


def show(title, rows):
    keys = ["frames", "ball known", "at rest", "ball speed p50", "with a player",
            "moving + player", "off pitch", "near a line"]
    print("\n" + title)
    print("{:<16}".format("") + "".join("{:>16}".format(k) for k in keys))
    for name, r in rows:
        if r is None:
            print("{:<16}{:>16}".format(name, "no frames"))
            continue
        print("{:<16}".format(name) + "".join(
            "{:>16}".format(r[k]) if k == "frames" else "{:>16.3f}".format(r[k]) for k in keys))


def collect(tracking, spec, gt=None):
    F = features.compute(tracking, spec)
    graded = spec.graded_mask()
    rank, _ = learn_features._relative(F.player_speed, spec)
    length, width = tracking.pitch
    bx, by = F.ball_xy[:, 0], F.ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        line_dist = np.minimum.reduce([bx, length - bx, by, width - by])   # negative = outside

    show("ball behaviour by player-speed rank (graded frames; shares are of known-ball frames, "
         "except 'ball known')",
         [(name, describe(graded & (rank >= lo) & (rank < hi), F, line_dist))
          for lo, hi, name in BINS])

    # sudden slow-downs: speed rank falls sharply from the 3 s before to the 3 s after
    w = int(round(3 * spec.fps))
    past = learn_features._nanroll(rank, w, "past")
    future = learn_features._nanroll(rank, w, "future")
    drop = graded & (past - future > DROP)
    edges = np.diff(np.concatenate(([0], drop.astype(np.int8), [0])))
    n_events = int(np.sum(edges == 1))
    minutes = graded.sum() / spec.fps / 60.0
    print("\nsudden slow-downs (speed rank falls by more than {} from the 3 s before to the 3 s "
          "after)".format(DROP))
    print("  events {}  ({:.2f} per minute), frames inside {}".format(
        n_events, n_events / minutes, int(drop.sum())))
    show("ball behaviour inside sudden slow-downs", [("slow-down", describe(drop, F, line_dist))])

    # where was the ball last seen before each long gap?
    seen = tracking.ball_observed
    min_gap = spec.fps      # gaps of at least 1 s
    last_line, last_rank, n_gaps = [], [], 0
    for _, first, last in spec.halves:
        s = seen[first:last + 1]
        e = np.diff(np.concatenate(([0], (~s).astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if b - a < min_gap or a == 0:
                continue
            t = first + a - 1
            n_gaps += 1
            last_line.append(line_dist[t])
            last_rank.append(rank[t])
    last_line, last_rank = np.array(last_line), np.array(last_rank)
    print("\nball gaps of 1 s or more: {}".format(n_gaps))
    if n_gaps:
        print("  last seen within {} m of a line or outside: {:.3f}".format(
            LINE_M, float(np.mean(last_line < LINE_M))))
        print("  last seen already outside the pitch:        {:.3f}".format(
            float(np.mean(last_line < 0))))
        print("  players already slow (rank < 0.25) then:    {:.3f}".format(
            float(np.nanmean(last_rank < 0.25))))

    if gt is not None:
        from mtah.official import pm
        show("external game only, by TRUE state (how the augmented dead ball looks)",
             [("dead", describe(graded & (gt == pm.D), F, line_dist)),
              ("alive", describe(graded & ((gt == pm.H) | (gt == pm.A)), F, line_dist))])
        dead = graded & (gt == pm.D)
        print("  player-speed rank while dead: p25 {:.2f} p50 {:.2f} p75 {:.2f}".format(
            *np.nanpercentile(rank[dead], [25, 50, 75])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--external", help="external bundle name instead of --data-dir")
    parser.add_argument("--condition", default="full", help="clean | balldrop | full")
    args = parser.parse_args(argv)
    if args.external:
        from ext import evaluate as ev
        path, spec, gt = ev.load(args.external)
        tracking = ev.tracking_for(path, spec, args.condition, seed=1000)
        print("slow-play stats: {} ({})".format(args.external, args.condition))
        collect(tracking, spec, gt)
    else:
        if not args.data_dir:
            parser.error("pass --data-dir, set MTAH_DATA_DIR, or use --external")
        spec = GameSpec.from_upstream("BAR-ATH")
        print("slow-play stats: {}".format(spec.name))
        collect(io_atd.load_game(args.data_dir, spec), spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
