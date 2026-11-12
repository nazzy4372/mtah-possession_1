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
    assert "sanity alarms" in proc.stdout and "warning(s)" in proc.stdout


def test_compare_submissions_tool(small_bundle, tmp_path):
    paths = []
    for method in ("v2", "baseline"):
        result = pipeline.run(small_bundle.path, small_bundle.spec, method=method)
        paths.append(tmp_path / "{}.csv".format(method))
        pipeline.write_submission(paths[-1], result.labels)
    proc = run_tool(TOOLS / "compare_submissions.py", paths[0], paths[1],
                    "--spec", small_bundle.path / "game_spec.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "same label" in proc.stdout and "X placement differs on      0 frames" in proc.stdout
    same = run_tool(TOOLS / "compare_submissions.py", paths[0], paths[0],
                    "--spec", small_bundle.path / "game_spec.json")
    assert "same label                  1.0000" in same.stdout


def test_report_alarm_ranges():
    sys.path.insert(0, str(TOOLS))
    import report

    good = {"half": 1, "possessions": 150, "mean_possession_s": 12.0, "D_share": 0.38,
            "home_share_of_alive": 0.6, "possessions_under_0.5s": 1, "X": 0}
    assert report.alarms([good])["n_warn"] == 0
    bad = dict(good, possessions=320, D_share=0.6, X=5)
    assert report.alarms([bad])["n_warn"] == 3
