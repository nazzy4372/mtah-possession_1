"""Aggregate sanity numbers for a submission CSV (no ground truth needed).

Prints counts and durations only, so its output on the real-match submission is safe to share.
Possessions are counted exactly as the official scorer counts them: X forward-filled within each
graded half (``pm.fill_abstentions``), then maximal H or A runs (``pm.possession_spells``).

    .venv\\Scripts\\python.exe tools\\report.py outputs\\real\\submission.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah.official import pm  # noqa: E402
from mtah.pipeline import sha256_of  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402


def summarise(codes, spec):
    rows = []
    for half, first, last in spec.halves:
        run = [(first, last + 1)]
        filled = pm.fill_abstentions(codes, run)
        window = codes[first:last + 1]
        spells = pm.possession_spells(filled, run)
        lengths = np.array([b - a for a, b, _ in spells], dtype=float) / spec.fps
        dead = [(a, b) for a, b, k in pm.segments(filled, run) if k == pm.D]
        dead_len = np.array([b - a for a, b in dead], dtype=float) / spec.fps
        n_h, n_a = int(np.sum(window == pm.H)), int(np.sum(window == pm.A))
        rows.append({
            "half": half,
            "frames": last + 1 - first,
            "H": n_h, "A": n_a, "D": int(np.sum(window == pm.D)), "X": int(np.sum(window == pm.X)),
            "possessions": len(spells),
            "home_possessions": sum(1 for s in spells if s[2] == pm.H),
            "mean_possession_s": float(lengths.mean()) if len(lengths) else 0.0,
            "median_possession_s": float(np.median(lengths)) if len(lengths) else 0.0,
            "possessions_under_1.5s": int(np.sum(lengths < 1.5)),
            "possessions_under_0.5s": int(np.sum(lengths < 0.5)),
            "dead_runs": len(dead),
            "mean_dead_run_s": float(dead_len.mean()) if len(dead_len) else 0.0,
            "D_share": float(np.mean(window == pm.D)),
            "home_share_of_alive": n_h / float(n_h + n_a) if n_h + n_a else 0.0,
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    args = parser.parse_args(argv)
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")

    table = pm.read_label_csv(args.file)
    reason = pm.validate_submission(table, spec.n_frames)
    if reason:
        print("INVALID: {}".format(reason))
        return 1
    codes = pm.label_codes(table)
    rows = summarise(codes, spec)

    print("report  {}  ({} frames)".format(Path(args.file).name, spec.n_frames))
    keys = [k for k in rows[0] if k != "half"]
    print("{:<26}".format("") + "".join("{:>12}".format("half {}".format(r["half"])) for r in rows))
    for key in keys:
        cells = []
        for r in rows:
            value = r[key]
            cells.append("{:>12.3f}".format(value) if isinstance(value, float)
                         else "{:>12}".format(value))
        print("{:<26}".format(key) + "".join(cells))
    total_spells = sum(r["possessions"] for r in rows)
    graded = sum(r["frames"] for r in rows)
    print("\nwhole match: {} possessions, X on graded frames: {}".format(
        total_spells, sum(r["X"] for r in rows)))
    print("graded frames: {}".format(graded))
    print("sha256: {}".format(sha256_of(args.file)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
