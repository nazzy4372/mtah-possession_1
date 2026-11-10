"""External-data plumbing: label derivation rules and augmentation (no download needed)."""

import numpy as np
import pytest

from ext import augment, metrica


def ev(team, type_, start, end=None, subtypes=(), period=1):
    return {"team": team, "type": type_, "subtypes": tuple(subtypes), "period": period,
            "start": start, "end": start if end is None else end}


def test_label_derivation_rules():
    frames = np.arange(1, 1001)
    periods = np.ones(1000, dtype=int)
    events = [
        ev(0, "SET PIECE", 101, subtypes=("KICK OFF",)),        # kickoff, home: alive H
        ev(0, "PASS", 101, 130),
        ev(1, "CHALLENGE", 200),                                 # contested: stays H
        ev(1, "RECOVERY", 300),                                  # away gains control
        ev(1, "BALL OUT", 350, 400),                             # dead from 400
        ev(0, "SET PIECE", 500, subtypes=("THROW IN",)),        # restart, home
        ev(0, "PASS", 500, 520),
        ev(0, "PASS", 700, 720),                                 # long gap, then a goal kick:
        ev(1, "PASS", 900, 930, subtypes=("GOAL KICK",)),       # unannotated stoppage 720..900
    ]
    labels = metrica.derive_labels(frames, periods, events)
    at = lambda f: labels[f - 1]  # noqa: E731
    assert at(50) == "D" and at(101) == "H" and at(250) == "H"
    assert at(300) == "A" and at(399) == "A" and at(400) == "D" and at(499) == "D"
    assert at(500) == "H" and at(719) == "H" and at(720) == "D" and at(899) == "D"
    assert at(900) == "A" and at(1000) == "A"


def test_subtype_splitting():
    assert metrica._split_subtypes("HEAD-ON TARGET-GOAL") == ("HEAD", "ON TARGET", "GOAL")
    assert metrica._split_subtypes(float("nan")) == ()
    assert metrica._is_restart(ev(0, "BALL LOST", 1, subtypes=("GOAL KICK", "INTERCEPTION")))


def test_ball_dropout_is_deterministic_and_long(small_bundle):
    from mtah import io_atd

    a = io_atd.load_game(small_bundle.path, small_bundle.spec)
    b = io_atd.load_game(small_bundle.path, small_bundle.spec)
    share_a = augment.ball_dropout(a, seed=7)
    share_b = augment.ball_dropout(b, seed=7)
    assert share_a == share_b and np.array_equal(np.isnan(a.ball_xy), np.isnan(b.ball_xy))


@pytest.mark.skipif(not (metrica.BUNDLES / "metrica_g3" / "gt.csv").is_file(),
                    reason="external bundles not prepared (python tasks.py external --prepare)")
@pytest.mark.slow
def test_v1_beats_baseline_on_held_out_external_game():
    from ext import evaluate

    rows = evaluate.evaluate(["metrica_g3"], bootstrap=0, verbose=False)
    by = {(r["condition"], r["method"]): r["spell_f1_mean"] for r in rows}
    for condition in evaluate.CONDITIONS:
        assert by[(condition, "v1")] > by[(condition, "baseline")]
