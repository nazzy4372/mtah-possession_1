"""Where does a submission disagree with the PUBLIC 5-minute ground truth? One row per segment.

Purpose: understanding what the organisers mean by "dead" and by "possession" (CHALLENGE.md: the
public ground truth "shows frame by frame ... how these cases are labelled. Use it to understand
what is expected"). It is NOT a tuning set: nothing may be fitted to these frames, and no rule may
refer to them. Any change it suggests must be general and must be validated on external games.
Every run is a look at the public window and is logged in docs/public_window_log.md.

Prints, for each ground-truth segment in the window: its label, video time and length, how our
labels split inside it, whether a possession was matched (best IoU with one of ours of the same
team), and for dead-ball segments how much we covered and how early or late we were. With
--data-dir it adds label-free context per segment (share of frames with a known ball, median
player speed, ball at rest, ball outside the pitch). Then the same from our side: each of our
segments with the ground-truth labels inside it. No positions and no per-frame values.

    .venv\\Scripts\\python.exe tools\\public_errors.py --data-dir C:\\Hackathon\\mtah-data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import features, io_atd  # noqa: E402
from mtah.official import pm  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

PUBLIC_GT = ROOT / "upstream" / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
NAMES = "HAD"


def clock(frame, fps):
    seconds = frame / float(fps)
    return "{:d}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def shares(codes, a, b):
    counts = np.bincount(codes[a:b], minlength=4)[:3]
    return counts / float(max(1, b - a))


def best_iou(seg, others):
    a, b, k = seg
    best = 0.0
    for c, d, j in others:
        if j != k or d <= a or b <= c:
            continue
        best = max(best, (min(b, d) - max(a, c)) / float(max(b, d) - min(a, c)))
    return best


def context(F, a, b):
    known = F.ball_known[a:b]
    speed = F.ball_speed[a:b]
    out = F.out_dist[a:b]
    with np.errstate(invalid="ignore"):
        still = np.nanmean(speed < 0.5) if known.any() else float("nan")
        outside = np.nanmean(out > 0.5) if known.any() else float("nan")
    return (float(known.mean()), float(np.nanmedian(F.player_speed[a:b])), float(still),
            float(outside))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(ROOT / "outputs" / "real" / "submission.csv"))
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"),
                        help="data bundle, for label-free context per segment (optional)")
    args = parser.parse_args(argv)
    spec = GameSpec.from_upstream("BAR-ATH")
    fps = spec.fps

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    table = pm.read_label_csv(args.file)
    reason = pm.validate_submission(table, spec.n_frames)
    if reason:
        print("INVALID: {}".format(reason))
        return 1
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pm.label_codes(table), runs)
    gt_segs, pred_segs = pm.segments(gt, runs), pm.segments(pred, runs)

    F = None
    if args.data_dir:
        F = features.compute(io_atd.load_game(args.data_dir, spec), spec)

    graded = gt != pm.X
    print("public window: {} graded frames, {} ground-truth segments, {} of ours".format(
        int(graded.sum()), len(gt_segs), len(pred_segs)))
    print("frames by error type")
    g, p = gt[graded], pred[graded]
    alive_g, alive_p = g != pm.D, p != pm.D
    for name, mask in (("correct", g == p),
                       ("dead called alive", ~alive_g & alive_p),
                       ("alive called dead", alive_g & ~alive_p),
                       ("wrong team", alive_g & alive_p & (g != p))):
        print("  {:<20} {:>5}  {:5.1f}%".format(name, int(mask.sum()), 100.0 * mask.mean()))

    head = "{:>3} {:>2} {:>8} {:>6}  {:>4} {:>4} {:>4}  {:<28}".format(
        "#", "GT", "video", "len s", "H%", "A%", "D%", "verdict")
    if F is not None:
        head += " {:>5} {:>6} {:>6} {:>5}".format("ball", "plSpd", "still", "out")
    print("\nground-truth segments (how our labels split inside each)")
    print(head)
    for i, (a, b, k) in enumerate(gt_segs, start=1):
        s = shares(pred, a, b)
        if k == pm.D:
            ours = [(c, d) for c, d, j in pred_segs if j == pm.D and d > a and c < b]
            if ours:
                verdict = "D found {:.0f}%, start {:+d}f end {:+d}f".format(
                    100 * s[2], ours[0][0] - a, ours[-1][1] - b)
            else:
                verdict = "D MISSED (played through)"
        else:
            iou = best_iou((a, b, k), pred_segs)
            verdict = "{} IoU {:.2f}".format("matched" if iou >= 0.5 else "MISSED ", iou)
        row = "{:>3} {:>2} {:>8} {:>6.1f}  {:>4.0f} {:>4.0f} {:>4.0f}  {:<28}".format(
            i, NAMES[k], clock(a, fps), (b - a) / float(fps), 100 * s[0], 100 * s[1], 100 * s[2],
            verdict)
        if F is not None:
            known, speed, still, outside = context(F, a, b)
            row += " {:>5.2f} {:>6.2f} {:>6.2f} {:>5.2f}".format(known, speed, still, outside)
        print(row)

    print("\nour segments (ground-truth labels inside each)")
    print("{:>3} {:>4} {:>8} {:>6}  {:>4} {:>4} {:>4}  {}".format(
        "#", "ours", "video", "len s", "H%", "A%", "D%", "verdict"))
    for i, (a, b, k) in enumerate(pred_segs, start=1):
        s = shares(gt, a, b)
        if k == pm.D:
            verdict = "true dead ball" if s[2] >= 0.5 else "SPURIOUS dead ball"
        else:
            iou = best_iou((a, b, k), gt_segs)
            verdict = "{} IoU {:.2f}".format("matched " if iou >= 0.5 else "SPURIOUS", iou)
        print("{:>3} {:>4} {:>8} {:>6.1f}  {:>4.0f} {:>4.0f} {:>4.0f}  {}".format(
            i, NAMES[k], clock(a, fps), (b - a) / float(fps), 100 * s[0], 100 * s[1], 100 * s[2],
            verdict))
    if F is not None:
        print("\ncontext columns: ball = share of frames with a known ball; plSpd = median player "
              "speed (m/s);\nstill = share of known-ball frames with the ball at rest; out = share "
              "with the ball outside the pitch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
