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


def test_full_augmentation_shape_and_determinism(small_bundle):
    from mtah import io_atd

    tr = io_atd.load_game(small_bundle.path, small_bundle.spec)
    a = augment.full(tr, seed=3)
    b = augment.full(tr, seed=3)
    assert a.players_xy.shape[1] == 46 and a.slot_team.tolist() == [0] * 23 + [1] * 23
    assert np.array_equal(np.isnan(a.players_xy), np.isnan(b.players_xy))
    assert np.allclose(np.nan_to_num(a.players_xy), np.nan_to_num(b.players_xy))
    graded = small_bundle.spec.graded_mask()
    before = tr.player_observed[graded].sum(axis=1).mean()
    after = a.player_observed[graded].sum(axis=1).mean()
    assert after < before                      # players lost to view loss and gaps
    used = a.player_observed[graded].any(axis=0).sum()
    assert used > 22                           # tracklets spread over more slots than players
    assert tr.players_xy.shape[1] == 46 and not np.isnan(tr.ball_xy).all()   # input untouched


def test_restore_dead_ball_fills_only_missing_frames(small_bundle):
    from mtah import io_atd

    tr = io_atd.load_game(small_bundle.path, small_bundle.spec)
    tr.ball_xy[1000:1100] = np.nan
    tr.ball_xy[1090] = (1.0, 2.0)
    augment.restore_dead_ball(tr, [(1000, 1100, 50.0, 30.0, "end")], 0.5, 0.0, seed=0)
    assert np.isnan(tr.ball_xy[1000:1050]).all()
    assert np.allclose(tr.ball_xy[1050:1090], (50.0, 30.0))
    assert tr.ball_xy[1090].tolist() == [1.0, 2.0]


def test_skillcorner_label_rules():
    from ext import skillcorner

    frames = np.arange(100, 130)
    groups = np.array(["H"] * 5 + [""] * 3 + ["A"] * 4 + [""] * 6 + [""] * 4 + ["H"] * 8)
    phases = [(100, 117, "H"), (122, 129, "H")]   # frames 118-121 are outside every phase
    labels = skillcorner.derive_labels(frames, groups, phases)
    assert "".join(labels[:8]) == "HHHHHHHH"           # loose ball inside a phase: stays H
    assert "".join(labels[8:18]) == "AAAAAAAAAA"       # A, then loose inside the phase: stays A
    assert "".join(labels[18:22]) == "DDDD"            # no group, no phase: dead
    assert "".join(labels[22:]) == "HHHHHHHH"


def test_skillcorner_resampling_10_to_25_fps():
    from ext import skillcorner

    src_t = np.array([0.0, 0.1, 0.2])
    values = np.array([[0.0, 0.0], [1.0, 2.0], [np.nan, np.nan]])
    out = skillcorner._resample(values, src_t, np.array([0.0, 0.04, 0.08, 0.1, 0.12]))
    assert np.allclose(out[:4], [[0, 0], [0.4, 0.8], [0.8, 1.6], [1.0, 2.0]])
    assert np.isnan(out[4]).all()                 # a missing neighbour stays missing


@pytest.mark.skipif(not (metrica.BUNDLES / "metrica_g3" / "gt.csv").is_file(),
                    reason="external bundles not prepared (python tasks.py external --prepare)")
@pytest.mark.slow
def test_v1_beats_baseline_on_held_out_external_game():
    from ext import evaluate

    rows = evaluate.evaluate(["metrica_g3"], bootstrap=0, verbose=False)
    by = {(r["condition"], r["method"]): r["spell_f1_mean"] for r in rows}
    for condition in evaluate.CONDITIONS:
        assert by[(condition, "v1")] > by[(condition, "baseline")]
