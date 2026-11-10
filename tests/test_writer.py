import numpy as np
import pytest

from mtah import pipeline
from mtah.official import pm
from mtah.spec import GameSpec


def test_submission_bytes_contract(tmp_path):
    labels = np.array(["X", "H", "A", "D", "X"])
    path = tmp_path / "sub.csv"
    digest = pipeline.write_submission(path, labels)
    data = path.read_bytes()
    assert b"\r" not in data
    assert data == b"frame,label\n0,X\n1,H\n2,A\n3,D\n4,X\n"
    assert digest == pipeline.sha256_of(path)
    assert pm.validate_submission(pm.read_label_csv(path), len(labels)) is None


def test_contract_checks_reject_bad_labels():
    spec = GameSpec(name="T", stem="T_tactical", n_frames=10, fps=25, pitch=(105.3, 68.0),
                    halves=((1, 2, 7),))
    good = np.array(list("XXHHADDHXX"))
    pipeline.check_labels(good, spec)
    with pytest.raises(AssertionError):
        pipeline.check_labels(np.array(list("XXHHAXDHXX")), spec)   # X on a graded frame
    with pytest.raises(AssertionError):
        pipeline.check_labels(np.array(list("HXHHADDHXX")), spec)   # label on an ungraded frame
    with pytest.raises(AssertionError):
        pipeline.check_labels(good[:-1], spec)                       # wrong length
