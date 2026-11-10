"""Data bundle -> per-frame labels -> submission CSV.

M1 wiring: parse ATD -> placeholder labeller (upstream baseline port) -> graded-frame policy with
algorithmic kickoff detection. M2 swaps the labeller for emissions + decoder; the contract around
it (row count, X on ungraded frames only, LF line endings, deterministic bytes) stays as is.
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from mtah import baseline_port, halves, io_atd
from mtah.spec import GameSpec

VALID_LABELS = ("H", "A", "D", "X")


@dataclass
class RunResult:
    labels: np.ndarray
    kickoffs: List[halves.Kickoff]
    timings: dict = field(default_factory=dict)


def run(data_dir, spec: GameSpec) -> RunResult:
    timings = {}
    t0 = time.perf_counter()
    tracking = io_atd.load_game(data_dir, spec)
    timings["parse_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw = baseline_port.placeholder_labels(tracking)
    kickoffs = halves.detect_kickoffs(tracking, spec)
    labels = halves.apply_policy(raw, spec, kickoffs)
    timings["label_s"] = time.perf_counter() - t1

    check_labels(labels, spec)
    return RunResult(labels=labels, kickoffs=kickoffs, timings=timings)


def check_labels(labels: np.ndarray, spec: GameSpec) -> None:
    """The submission contract, checked before anything is written."""
    if len(labels) != spec.n_frames:
        raise AssertionError("{} labels, expected {}".format(len(labels), spec.n_frames))
    bad = set(np.unique(labels).tolist()) - set(VALID_LABELS)
    if bad:
        raise AssertionError("labels outside {}: {}".format(VALID_LABELS, sorted(bad)))
    graded = spec.graded_mask()
    if np.any(labels[~graded] != "X"):
        raise AssertionError("non-X label on an ungraded frame")
    if np.any(labels[graded] == "X"):
        raise AssertionError("X on a graded frame (policy: never abstain on graded frames)")


def write_submission(path, labels) -> str:
    """Write frame,label with LF line endings on every OS; return the SHA-256 of the bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "frame,label\n" + "".join("{},{}\n".format(i, label) for i, label in enumerate(labels))
    data = body.encode("ascii")
    with open(path, "wb") as handle:  # binary: no newline translation, ever
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label_counts(labels: np.ndarray) -> dict:
    return {name: int(np.sum(labels == name)) for name in VALID_LABELS}
