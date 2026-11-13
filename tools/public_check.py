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
    parser.add_argument("--detail", action="store_true",
                        help="also print the scorer's detail block: frame accuracy, confusion "
                             "matrix (GT x ours) and per-class precision/recall on the graded "
                             "frames of the public window")
    parser.add_argument("--force", action="store_true",
                        help="score a file other than the one already checked (logs a new look)")
    args = parser.parse_args(argv)

    # Every look is logged in docs/public_window_log.md. Scoring a new version needs --force, as
    # a deliberate step: the window may check a frozen file, never choose between versions.
    import hashlib
    digest = hashlib.sha256(Path(args.file).read_bytes()).hexdigest()
    checked = (ROOT / "config" / "submission_v2.sha256").read_text(encoding="utf-8").split()[0]
    if digest != checked and not args.force:
        print("REFUSED: {} is not the file the public window was checked on (v2, {}...).\n"
              "The public 5 minutes are a correctness check only and must not be used to compare "
              "versions.\nSee docs/public_window_log.md. To check alignment of a new version "
              "without ground truth, use tools/compare_submissions.py.".format(
                  Path(args.file).name, checked[:8]))
        return 2
    work = ROOT / "outputs" / "real" / "public_check"
    results = work / "results"
    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True)
    shutil.copyfile(args.file, results / "ours.csv")
    command = [sys.executable, str(SCORER), "--gt", "public", "--results", str(results),
               "--out", str(work)]
    if not args.detail:
        command.append("--no-detail")
    return subprocess.call(command, cwd=str(SCORER.parent))


if __name__ == "__main__":
    sys.exit(main())
