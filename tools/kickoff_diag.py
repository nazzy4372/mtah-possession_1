"""Explain kickoff detection on a bundle, with aggregates only.

For each half: which method fired and the offset from the published block start, then counts and
run lengths of the conditions the detector looks at, in three windows relative to the published
block start: the 10 s before, the first 10 s, and the whole search window. No positions and no
per-frame values are printed, so the output is safe to share.

    .venv\\Scripts\\python.exe tools\\kickoff_diag.py --data-dir C:\\Hackathon\\mtah-data
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
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("pass --data-dir or set MTAH_DATA_DIR")
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
    tracking = io_atd.load_game(args.data_dir, spec)
    for half, first, last in spec.halves:
        kickoff = halves.detect_kickoff(tracking, half, first, last, spec.fps)
        print(kickoff.describe(spec.fps))
        diagnostics = halves.kickoff_diagnostics(tracking, first, spec.fps)
        for window, values in diagnostics.items():
            print("  {:<9}".format(window))
            for key, value in values.items():
                print("    {:<32} {}".format(key, value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
