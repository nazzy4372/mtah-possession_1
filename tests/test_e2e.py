"""End to end on synthetic matches: bundle -> pipeline -> CSV -> official validator and metric."""

import subprocess
import sys

import numpy as np
import pytest

from mtah import pipeline
from mtah.official import REPO_ROOT, VALIDATE_SCRIPT, pm

N_REAL_FRAMES = 199319


def score(gt_path, labels):
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    return pm.evaluate(gt, pm.label_codes(labels))


@pytest.mark.parametrize("method", pipeline.METHODS)
def test_small_match_end_to_end(small_bundle, tmp_path, method):
    spec = small_bundle.spec
    result = pipeline.run(small_bundle.path, spec, method=method)
    path = tmp_path / "small.csv"
    pipeline.write_submission(path, result.labels)
    assert pm.validate_submission(pm.read_label_csv(path), spec.n_frames) is None
    assert score(small_bundle.gt, result.labels)["spell_f1_mean"] > 0.0  # sanity, not a target


@pytest.mark.slow
def test_full_length_submission_is_valid_and_deterministic(full_bundle, tmp_path):
    spec = full_bundle.spec
    assert spec.n_frames == N_REAL_FRAMES and spec.name == "BAR-ATH"

    first = pipeline.run(full_bundle.path, spec)
    path_a = tmp_path / "a.csv"
    digest_a = pipeline.write_submission(path_a, first.labels)

    # the official validator, run exactly as a participant runs it
    proc = subprocess.run([sys.executable, str(VALIDATE_SCRIPT), str(path_a)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("OK")

    data = path_a.read_bytes()
    assert b"\r" not in data
    assert data.count(b"\n") == N_REAL_FRAMES + 1 and data.endswith(b"\n")

    labels = first.labels
    graded = spec.graded_mask()
    assert (labels[~graded] == "X").all() and not (labels[graded] == "X").any()
    for kickoff in first.kickoffs:
        assert abs(kickoff.frame - full_bundle.kickoffs[kickoff.half]) <= 5
        assert (labels[kickoff.block_first:kickoff.frame] == "D").all()

    # a second, independent run produces the same bytes
    second = pipeline.run(full_bundle.path, spec)
    path_b = tmp_path / "b.csv"
    assert pipeline.write_submission(path_b, second.labels) == digest_a
    assert path_b.read_bytes() == data

    assert score(full_bundle.gt, labels)["spell_f1_mean"] > 0.0


@pytest.mark.slow
def test_tasks_cli_submission_and_validate(full_bundle, tmp_path):
    out = tmp_path / "cli.csv"
    tasks = str(REPO_ROOT / "tasks.py")
    proc = subprocess.run([sys.executable, tasks, "submission", "--data-dir",
                           str(full_bundle.path), "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    digest = pipeline.sha256_of(out)
    assert digest in proc.stdout
    proc = subprocess.run([sys.executable, tasks, "validate", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LF only" in proc.stdout and digest in proc.stdout
