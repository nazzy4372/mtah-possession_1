"""Explain kickoff detection on a bundle.

For each half: which method fired, the offset from the published block start, and the check against
the published bounds and anchors. Then counts and run lengths of the conditions the detector uses,
in three windows relative to the published block start (the 10 s before, the first 10 s, and the
whole search window). These are aggregates only.

With --runs it also lists the formation runs in the search window (start, end and length in seconds
from the published block start, how each ended, and its share of UNKNOWN frames). Those are event
timings inside the kickoff window, of the same kind as the kickoff offset itself, and include no
positions. Leave the flag off if that is more detail than you want to share.

    .venv\\Scripts\\python.exe tools\\kickoff_diag.py --data-dir C:\\Hackathon\\mtah-data [--runs]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import halves, io_atd  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    parser.add_argument("--runs", action="store_true",
                        help="also list formation runs (timings in the kickoff window)")
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("pass --data-dir or set MTAH_DATA_DIR")
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
    tracking = io_atd.load_game(args.data_dir, spec)
    for kickoff in halves.detect_kickoffs(tracking, spec):
        print(kickoff.describe(spec.fps))
        diagnostics = halves.kickoff_diagnostics(tracking, kickoff.block_first, spec.fps)
        for window, values in diagnostics.items():
            print("  {:<9}".format(window))
            for key, value in values.items():
                print("    {:<32} {}".format(key, value))
        if args.runs:
            print("  formation runs >= 1 s (seconds from the published block start)")
            for row in halves.formation_run_table(tracking, kickoff.block_first,
                                                  kickoff.block_last, spec.fps):
                print("    " + "  ".join("{}={}".format(k, v) for k, v in row.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
