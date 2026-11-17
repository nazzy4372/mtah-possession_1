"""Compare two submission CSVs without ground truth: agreement, 3x3 table, possession counts.

    .venv\\Scripts\\python.exe tools\\compare_submissions.py OLD.csv NEW.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah.official import pm  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402


def compare(a, b, spec):
    graded = spec.graded_mask()
    out = {"graded_frames": int(graded.sum()),
           "agreement": float(np.mean(a[graded] == b[graded])),
           "table": [[int(np.sum((a[graded] == i) & (b[graded] == j))) for j in range(3)]
                     for i in range(3)],
           "x_mismatch": int(np.sum((a == pm.X) != (b == pm.X)))}
    runs = spec.runs()
    alive = graded & (a != pm.D) & (b != pm.D)
    out["team_agreement_both_alive"] = float(np.mean(a[alive] == b[alive])) if alive.any() else 0.0
    out["alive_dead_agreement"] = float(np.mean((a[graded] == pm.D) == (b[graded] == pm.D)))
    out["possessions"] = (len(pm.possession_spells(a, runs)), len(pm.possession_spells(b, runs)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    args = parser.parse_args(argv)
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
    codes = []
    for path in (args.a, args.b):
        table = pm.read_label_csv(path)
        reason = pm.validate_submission(table, spec.n_frames)
        if reason:
            print("INVALID {}: {}".format(Path(path).name, reason))
            return 1
        codes.append(pm.label_codes(table))
    r = compare(codes[0], codes[1], spec)
    print("compare  A = {}   B = {}".format(Path(args.a).name, Path(args.b).name))
    print("graded frames               {}".format(r["graded_frames"]))
    print("same label                  {:.4f}".format(r["agreement"]))
    print("same alive/dead call        {:.4f}".format(r["alive_dead_agreement"]))
    print("same team (both alive)      {:.4f}".format(r["team_agreement_both_alive"]))
    print("possessions                 A {}   B {}".format(*r["possessions"]))
    print("X placement differs on      {} frames".format(r["x_mismatch"]))
    print("label counts, rows = A, cols = B      H        A        D")
    for name, row in zip("HAD", r["table"]):
        print("                              {}  {:>8} {:>8} {:>8}".format(name, *row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
