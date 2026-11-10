"""Smoke tests for the real-data tools, run on a synthetic bundle."""

import subprocess
import sys

from mtah import io_atd, pipeline
from mtah.official import REPO_ROOT

TOOLS = REPO_ROOT / "tools"


def run_tool(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_real_stats_aggregates(small_bundle):
    sys.path.insert(0, str(TOOLS))
    import real_stats

    tracking = io_atd.load_game(small_bundle.path, small_bundle.spec)
    stats = real_stats.collect(tracking, small_bundle.spec)
    assert 0.2 < stats["ball_missing_share_graded"] < 0.6
    assert 18 < stats["tracked_players_per_frame"]["mean"] <= 22
    assert stats["fragments"]["n"] > 0

    proc = run_tool(TOOLS / "real_stats.py", "--data-dir", small_bundle.path,
                    "--spec", small_bundle.path / "game_spec.json")
    assert proc.returncode == 0, proc.stderr
    assert "ball_missing_share_graded" in proc.stdout


def test_parity_tool(small_bundle):
    proc = run_tool(TOOLS / "parity_baseline.py", "--data-dir", small_bundle.path,
                    "--spec", small_bundle.path / "game_spec.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PARITY OK" in proc.stdout


def test_kickoff_diag_tool(small_bundle):
    proc = run_tool(TOOLS / "kickoff_diag.py", "--data-dir", small_bundle.path,
                    "--spec", small_bundle.path / "game_spec.json", "--runs")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("kickoff detected by formation+ball") == 2
    assert "longest_run_formation_ok" in proc.stdout and "ended=violated" in proc.stdout


def test_report_tool(small_bundle, tmp_path):
    result = pipeline.run(small_bundle.path, small_bundle.spec)
    path = tmp_path / "sub.csv"
    pipeline.write_submission(path, result.labels)
    proc = run_tool(TOOLS / "report.py", path, "--spec", small_bundle.path / "game_spec.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "possessions" in proc.stdout and pipeline.sha256_of(path) in proc.stdout
