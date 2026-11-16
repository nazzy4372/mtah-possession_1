"""Dead-ball evidence from a collective slowdown, on hand-built sequences."""

import numpy as np
import pytest

from mtah import deadball, io_atd
from mtah.deadball import DeadballParams
from mtah.spec import GameSpec

FPS = 25
N = 45 * FPS                  # 45 s
PITCH = (105.0, 68.0)
STOP, RESUME = 15 * FPS, 30 * FPS


def make(player_speed, ball, n_players=22):
    """A Tracking in which every player moves at player_speed[t] m/s and the ball follows `ball`
    ((n, 2) metres, NaN where untracked)."""
    step = np.cumsum(player_speed / FPS)
    players = np.full((N, 46, 2), np.nan)
    for i in range(n_players):
        players[:, i, 0] = 20.0 + 3.0 * i + step * (1 if i % 2 else -1)
        players[:, i, 1] = 5.0 + 2.5 * i
    slot_team = np.array([0, 1] * 23, dtype=np.int8)
    tracking = io_atd.Tracking(players_xy=players, players_speed=np.full((N, 46), np.nan),
                               ball_xy=ball, ball_speed=np.full(N, np.nan), slot_team=slot_team,
                               pitch=PITCH, meta=None)
    spec = GameSpec(name="t", stem="t", n_frames=N, fps=FPS, pitch=PITCH, halves=((1, 0, N - 1),))
    return tracking, spec


def speeds(slow_from, slow_to, fast=4.0, slow=1.0):
    s = np.full(N, fast)
    s[slow_from:slow_to] = slow
    return s


def passing_ball(x_lo=30.0, x_hi=70.0, y=34.0, v=8.0):
    """A ball passed back and forth along x at v m/s: always moving, always inside the pitch."""
    t = np.arange(N) / FPS
    span = x_hi - x_lo
    phase = (t * v) % (2 * span)
    x = x_lo + np.where(phase < span, phase, 2 * span - phase)
    return np.column_stack([x, np.full(N, y)])


def test_ball_out_then_slowdown_then_restart_starts_at_the_crossing():
    ball = passing_ball()
    out_from = 14 * FPS
    ramp = np.linspace(34.0, 69.5, STOP + 5 - out_from)        # kicked over the touchline
    ball[out_from:STOP + 5, 1] = ramp
    ball[STOP + 5:RESUME] = ball[STOP + 4]                      # lies outside until the restart
    crossing = int(np.flatnonzero(ball[:, 1] > PITCH[1] + 0.5)[0])
    tracking, spec = make(speeds(STOP + 40, RESUME), ball)      # players slow 1.6 s later
    ev = deadball.compute(tracking, spec)
    assert len(ev.intervals) == 1
    iv = ev.intervals[0]
    assert iv.start == crossing and iv.reason == "line crossing"
    assert iv.start < iv.onset                                  # dead before the slowdown
    assert abs(iv.end - RESUME) <= FPS // 2                     # ends at the restart
    assert "out" in iv.cues and iv.confidence == 1.0
    assert ev.score[crossing:RESUME - FPS].min() == 1.0 and ev.score[:crossing].max() == 0.0
    assert ev.score[RESUME + FPS:].max() == 0.0
    assert ev.stats["accepted"] == 1 and ev.stats["cue_out"] == 1


def test_goalkeeper_holding_the_ball_in_the_box_is_not_dead():
    ball = passing_ball()
    ball[STOP:RESUME] = (8.0, 34.0)                             # held inside the penalty area
    tracking, spec = make(speeds(STOP + 12, RESUME), ball)
    ev = deadball.compute(tracking, spec)
    assert ev.intervals == [] and ev.score.max() == 0.0
    assert ev.stats["candidates"] >= 1 and ev.stats["excluded_box"] >= 1
    assert ev.in_box[STOP + 5:RESUME - 5].all()
    # without the exclusion the same sequence is called dead: the exclusion is what protects it
    off = deadball.compute(tracking, spec, DeadballParams(box_exclusion=False))
    assert len(off.intervals) == 1 and "still" in off.intervals[0].cues


def test_slow_build_up_without_another_cue_is_not_dead():
    tracking, spec = make(speeds(STOP, RESUME), passing_ball())
    ev = deadball.compute(tracking, spec)
    assert ev.candidate[STOP + 2 * FPS:STOP + 5 * FPS].all()    # the slowdown is seen ...
    assert ev.intervals == [] and ev.score.max() == 0.0         # ... but nothing confirms it
    assert ev.stats["no_extra_cue"] >= 1
    # a tagged set piece nearby is such a confirmation
    tagged = deadball.compute(tracking, spec, set_piece_starts=np.array([STOP + 4 * FPS]))
    assert len(tagged.intervals) == 1 and tagged.intervals[0].cues == ("tag",)
    assert tagged.stats["cue_tag"] == 1


def test_ball_missing_during_a_stoppage_is_still_detected():
    ball = passing_ball()
    ball[STOP:RESUME] = np.nan                                  # the tracker loses the ball
    tracking, spec = make(speeds(STOP + 25, RESUME), ball)
    ev = deadball.compute(tracking, spec)
    assert len(ev.intervals) == 1
    iv = ev.intervals[0]
    assert iv.start == STOP and iv.reason == "ball disappears" and "missing" in iv.cues
    assert abs(iv.end - RESUME) <= FPS // 2
    assert ev.cue_missing[STOP + FPS:RESUME].all() and not ev.cue_missing[:STOP].any()


def test_few_tracked_players_lower_the_confidence():
    ball = passing_ball()
    ball[STOP:RESUME] = np.nan
    tracking, spec = make(speeds(STOP + 25, RESUME), ball, n_players=8)
    ev = deadball.compute(tracking, spec)
    assert len(ev.intervals) == 1 and ev.intervals[0].confidence == 0.5
    assert ev.low_confidence.all() and ev.score.max() == 0.5


def test_penalty_zone_geometry():
    zone = lambda x, y: bool(deadball.in_penalty_zone(np.array([x]), np.array([y]), PITCH)[0])  # noqa: E731
    assert zone(8.0, 34.0) and zone(105.0 - 8.0, 34.0)          # both ends
    assert zone(16.4, 34.0 + 20.0) and not zone(16.4, 34.0 + 20.3)   # 40.32 m wide
    assert not zone(16.6, 34.0 + 15.0)                          # just outside the box
    assert zone(19.5, 34.0) and not zone(20.5, 34.0)            # the arc: 11 + 9.15 from the line
    assert not zone(52.5, 34.0) and not zone(-1.0, 34.0)        # midfield; behind the goal line


def test_evidence_only_touches_the_dead_ball_score_and_exposes_features():
    ball = passing_ball()
    ball[STOP:RESUME] = np.nan
    tracking, spec = make(speeds(STOP + 25, RESUME), ball)
    ev = deadball.compute(tracking, spec)
    logp = np.zeros((N, 3))
    out = deadball.add_evidence(logp, ev, 2.0)
    assert (out[:, :2] == 0).all() and (logp == 0).all()
    assert out[STOP + FPS, 2] == 2.0 and out[5, 2] == 0.0
    feats = deadball.features(ev)
    assert tuple(feats) == deadball.FEATURES
    assert all(len(v) == N for v in feats.values())
    assert np.nanmin(feats["db_ratio"][STOP + 2 * FPS:STOP + 5 * FPS]) == pytest.approx(0.25, abs=0.02)
    rows = deadball.summary(ev, spec)
    assert rows[0]["intervals"] == 1 and 13.0 < rows[0]["median_s"] < 16.5


def test_features_reach_the_learned_matrix_but_no_frozen_feature_set(small_bundle):
    import sys

    from mtah import learn_features
    from mtah.official import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT))
    from ext import train

    assert learn_features.GROUP_DEADBALL == deadball.FEATURES
    tr = io_atd.load_game(small_bundle.path, small_bundle.spec)
    names, X = learn_features.build(tr, small_bundle.spec)
    for name in deadball.FEATURES:
        assert name in names
    for variant in ("all", "touch", "gap", "rawspeed", "short", "noshape"):
        assert not set(train.VARIANTS[variant](names)) & set(deadball.FEATURES), variant
    assert set(deadball.FEATURES) <= set(train.VARIANTS["dead"](names))
    # home/away mirroring leaves the detector's signals alone
    i = [names.index(k) for k in deadball.FEATURES]
    swapped = learn_features.swap_teams(names, X)
    assert np.array_equal(np.nan_to_num(swapped[:, i]), np.nan_to_num(X[:, i]))


def test_v1d_runs_end_to_end_and_only_adds_dead_ball_evidence(small_bundle):
    from mtah import pipeline

    spec = small_bundle.spec
    tr = io_atd.load_game(small_bundle.path, spec)
    v1, kos = pipeline.label_tracking(tr, spec, "v1")
    v1d, _ = pipeline.label_tracking(tr, spec, "v1d", kickoffs=kos)
    assert np.sum(v1d == "D") >= np.sum(v1 == "D")
    zero = pipeline.Params(deadball=DeadballParams(weight=0.0))
    same, _ = pipeline.label_tracking(tr, spec, "v1d", zero, kickoffs=kos)
    assert np.array_equal(same, v1)


def test_deadball_stats_tool_prints_aggregates(small_bundle):
    from test_tools import TOOLS, run_tool

    proc = run_tool(TOOLS / "deadball_stats.py", "--data-dir", small_bundle.path,
                    "--spec", small_bundle.path / "game_spec.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "slowdown candidates" in proc.stdout and "median duration s" in proc.stdout
    assert "ball untracked for 1 s" in proc.stdout and "line crossing" in proc.stdout
