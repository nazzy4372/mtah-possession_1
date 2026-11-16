"""Aggregates of the slowdown dead-ball detector (src/mtah/deadball.py) on a match. No labels.

Prints, per half, how many dead-ball stretches the detector proposes, their median duration and
their share of the graded frames; then how often each cue fired, how the start of each stretch
was found and how many candidates were dropped by the penalty-area rule or for lack of a second
cue. Counts, shares and medians only: no frame number, no position. Safe to share.

The detector is evidence for the decoder, not a labeller, and it is not part of the submitted
model; this tool only shows how it behaves on a match.

    .venv\\Scripts\\python.exe tools\\deadball_stats.py --data-dir C:\\Hackathon\\mtah-data
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import deadball, io_atd, io_tagging  # noqa: E402
from mtah.deadball import DeadballParams  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402


def set_piece_starts(paths, fps):
    """Start frames of the tagged set pieces, or an empty array when there is no tagging file."""
    path = Path(paths["smart_tagging"])
    if not path.is_file():
        return np.array([], dtype=np.int64)
    return np.array([iv.start for iv in io_tagging.load(path, fps) if iv.is_set_piece],
                    dtype=np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    parser.add_argument("--no-box", action="store_true", help="without the penalty-area exclusion")
    parser.add_argument("--no-tags", action="store_true", help="ignore the Smart Tagging cue")
    args = parser.parse_args(argv)
    if not args.data_dir:
        print("no data directory: pass --data-dir or set MTAH_DATA_DIR")
        return 2
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
    tracking = io_atd.load_game(args.data_dir, spec)
    tags = np.array([], dtype=np.int64) if args.no_tags else set_piece_starts(
        spec.paths(args.data_dir), spec.fps)
    params = replace(DeadballParams(), box_exclusion=not args.no_box)
    ev = deadball.compute(tracking, spec, params, tags)

    print("dead-ball detector  {}  ({} frames; box exclusion {}; tagged set pieces used: {})".format(
        spec.name, spec.n_frames, "on" if params.box_exclusion else "OFF", len(tags)))
    print("{:<8}{:>12}{:>20}{:>18}".format("half", "intervals", "median duration s", "dead-ball share"))
    for row in deadball.summary(ev, spec):
        print("{:<8}{:>12}{:>20.1f}{:>18.3f}".format(row["half"], row["intervals"], row["median_s"],
                                                     row["share"]))
    s = ev.stats
    print("\nslowdown candidates {}: accepted {}, dropped by the penalty-area rule {}, dropped for "
          "lack of a second cue {}".format(s["candidates"], s["accepted"], s["excluded_box"],
                                           s["no_extra_cue"]))
    print("accepted at low confidence (few players tracked or ball position unknown): {}".format(
        s["low_confidence"]))
    print("cues on the accepted stretches (a stretch can have several):")
    for cue, text in (("out", "ball outside the pitch"), ("still", "ball at rest for 1 s"),
                      ("missing", "ball untracked for 1 s"), ("tag", "tagged set piece nearby")):
        print("  {:<26}{:>5}".format(text, s["cue_" + cue]))
    print("start of the accepted stretches found by:")
    for reason in deadball.START_REASONS:
        print("  {:<26}{:>5}".format(reason, s["start_" + reason.replace(" ", "_")]))
    graded = spec.graded_mask()
    ratio = ev.ratio[graded]
    ok = np.isfinite(ratio)
    print("\nspeed ratio (top speeds / their 10 s baseline) on graded frames: p10 {:.2f}  p50 {:.2f}"
          "  p90 {:.2f}; at or below the threshold on {:.3f} of frames".format(
              *np.percentile(ratio[ok], [10, 50, 90]),
              float(np.mean(ratio[ok] <= params.ratio_max))))
    print("frames with fewer than {} tracked players: {:.3f}".format(
        params.min_players, float(np.mean(ev.low_confidence[graded]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
