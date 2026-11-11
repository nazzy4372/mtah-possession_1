"""Correctness check against the PUBLIC 5-minute ground truth. Run once, after freezing.

The public window is a correctness check only (CLAUDE.md hard rule 3; upstream RULES.md): it shows
that frames are aligned, the file is well-formed and the labels mean what we think. It is never
used for training, tuning or choosing between variants. Every look is logged in
docs/public_window_log.md.

Runs the official scorer (upstream/evaluation/run_evaluation.py) on one submission, with every
output redirected to outputs/real/public_check/ (gitignored) so nothing is written into the
upstream submodule, and prints the ranked board only (aggregates over the 5-minute window).

    .venv\\Scripts\\python.exe tools\\public_check.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORER = ROOT / "upstream" / "evaluation" / "run_evaluation.py"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(ROOT / "outputs" / "real" / "submission.csv"))
    args = parser.parse_args(argv)
    work = ROOT / "outputs" / "real" / "public_check"
    results = work / "results"
    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True)
    shutil.copyfile(args.file, results / "ours.csv")
    return subprocess.call([sys.executable, str(SCORER), "--gt", "public",
                            "--results", str(results), "--out", str(work), "--no-detail"],
                           cwd=str(SCORER.parent))


if __name__ == "__main__":
    sys.exit(main())
