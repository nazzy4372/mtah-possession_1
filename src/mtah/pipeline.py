"""Data bundle -> per-frame labels -> submission CSV.

    parse ATD -> kickoff detection (halves.py)
              -> features -> emissions (rules v1) -> Viterbi per half -> boundary refinement
              -> graded-frame policy (X outside the halves, D before each kickoff)

``method="baseline"`` swaps in the ported upstream baseline (nearest player, +/-50-frame vote) and is
the reference every external comparison is made against. The contract around either method (row
count, X on ungraded frames only, LF line endings, deterministic bytes) is checked before writing.
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from mtah import baseline_port, decode, emission, features, halves, io_atd, refine
from mtah.decode import DecoderParams
from mtah.emission import EmissionParams
from mtah.refine import RefineParams
from mtah.spec import GameSpec

VALID_LABELS = ("H", "A", "D", "X")


class KickoffInconsistent(RuntimeError):
    """A detected kickoff disagrees with the published boundaries or anchors: no file is written.
    The message carries the per-half summary and aggregate diagnostics only."""

    def __init__(self, kickoffs, fps):
        self.kickoffs = kickoffs
        lines = ["kickoff detection is inconsistent with the published bounds; no file written"]
        for kickoff in kickoffs:
            lines.append("  " + kickoff.describe(fps))
            for window, values in kickoff.diagnostics.items():
                lines.append("    diag {:<9} {}".format(window, "  ".join(
                    "{}={}".format(k, v) for k, v in values.items())))
        super().__init__("\n".join(lines))


@dataclass
class RunResult:
    labels: np.ndarray
    kickoffs: List[halves.Kickoff]
    timings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Params:
    emission: EmissionParams = EmissionParams()
    decoder: DecoderParams = DecoderParams()
    refine: RefineParams = RefineParams()
    use_refine: bool = True


METHODS = ("v1", "baseline")


def label_tracking(tracking, spec: GameSpec, method: str = "v1", params: Params = Params(),
                   kickoffs=None):
    """(labels, kickoffs) for an in-memory Tracking. The evaluation harness calls this directly so
    it can corrupt a Tracking (ball dropout) before labelling."""
    if kickoffs is None:
        kickoffs = halves.detect_kickoffs(tracking, spec)
        if not all(k.consistent for k in kickoffs):
            raise KickoffInconsistent(kickoffs, spec.fps)
    if method == "baseline":
        raw = baseline_port.placeholder_labels(tracking)
    elif method == "v1":
        F = features.compute(tracking, spec)
        logp = emission.rules_v1(F, params.emission)
        raw = decode.decode(logp, spec, kickoffs, params.decoder)
        if params.use_refine:
            raw = refine.refine(raw, F, spec, params.refine, kickoffs)
    else:
        raise ValueError("unknown method {!r}; have {}".format(method, METHODS))
    labels = halves.apply_policy(raw, spec, kickoffs)
    check_labels(labels, spec)
    return labels, kickoffs


def run(data_dir, spec: GameSpec, method: str = "v1", params: Params = Params()) -> RunResult:
    timings = {}
    t0 = time.perf_counter()
    tracking = io_atd.load_game(data_dir, spec)
    timings["parse_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    labels, kickoffs = label_tracking(tracking, spec, method, params)
    timings["label_s"] = time.perf_counter() - t1
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
