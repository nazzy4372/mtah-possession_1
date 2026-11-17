"""Why did kickoff detection pick what it picked? Aggregates per half.

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
