from dataclasses import replace

import numpy as np
import pytest

from mtah import halves, io_atd, pipeline


@pytest.fixture
def tracking(small_bundle):
    return io_atd.load_game(small_bundle.path, small_bundle.spec)


def detect(tracking, bundle, spec=None):
    return {k.half: k for k in halves.detect_kickoffs(tracking, spec or bundle.spec)}


def assert_near(kickoff, truth, tol_frames):
    assert kickoff.consistent
    assert abs(kickoff.frame - truth) <= tol_frames, (kickoff.method, kickoff.frame - truth)


def test_formation_break_refined_by_ball_rest(tracking, small_bundle):
    for half, kickoff in detect(tracking, small_bundle).items():
        assert kickoff.method == "formation+ball"
        assert_near(kickoff, small_bundle.kickoffs[half], 2)


def test_ball_jitter_at_rest_is_tolerated(tracking, small_bundle):
    rng = np.random.default_rng(1)
    tracking.ball_xy += rng.normal(0.0, 0.1, size=tracking.ball_xy.shape)
    for half, kickoff in detect(tracking, small_bundle).items():
        assert kickoff.method == "formation+ball"
        assert_near(kickoff, small_bundle.kickoffs[half], 3)


def test_ball_invisible_while_resting(tracking, small_bundle):
    """The real-match pattern: the ball is only picked up once it moves off the spot."""
    fps = small_bundle.spec.fps
    for half, k in small_bundle.kickoffs.items():
        tracking.ball_xy[k - 10 * fps:k] = np.nan
    for half, kickoff in detect(tracking, small_bundle).items():
        assert kickoff.method == "formation+ball"
        assert_near(kickoff, small_bundle.kickoffs[half], 2)


def test_no_ball_at_all_uses_formation_break(tracking, small_bundle):
    tracking.ball_xy[:] = np.nan
    fps = small_bundle.spec.fps
    for half, kickoff in detect(tracking, small_bundle).items():
        truth = small_bundle.kickoffs[half]
        assert kickoff.method == "formation" and kickoff.consistent
        assert truth <= kickoff.frame <= truth + 3 * fps


def test_poorly_tracked_team_does_not_end_the_lineup(tracking, small_bundle):
    """Around the real second-half restart one team had ~4.6 tracked players: UNKNOWN, not over."""
    half, first, last = small_bundle.spec.halves[1]
    truth = small_bundle.kickoffs[half]
    fps = small_bundle.spec.fps
    away = np.flatnonzero(tracking.slot_team == 1)
    tracking.players_xy[truth - 8 * fps:truth - 2 * fps, away[:18]] = np.nan
    tracking.ball_xy[truth - 10 * fps:truth] = np.nan
    assert_near(detect(tracking, small_bundle)[half], truth, 2)


def test_stationary_ball_near_centre_in_open_play_is_not_a_kickoff(tracking, small_bundle):
    """Real half 1 was first read as a kickoff 229 s in: a ball resting near the centre in play."""
    spec = small_bundle.spec
    length, width = spec.pitch
    for half, k in small_bundle.kickoffs.items():
        tracking.ball_xy[k + 40 * spec.fps:k + 45 * spec.fps] = (length / 2.0, width / 2.0)
    for half, kickoff in detect(tracking, small_bundle).items():
        assert_near(kickoff, small_bundle.kickoffs[half], 2)


def test_tracking_resuming_before_lineup_is_not_a_kickoff(tracking, small_bundle):
    half, first, last = small_bundle.spec.halves[1]
    truth = small_bundle.kickoffs[half]
    tracking.ball_xy[:] = np.nan
    walk = slice(first, first + 4 * small_bundle.spec.fps)   # players scattered, not lined up
    tracking.players_xy[walk, :, 0] = small_bundle.spec.pitch[0] / 2.0 + np.linspace(
        -30, 30, tracking.players_xy.shape[1])
    kickoff = detect(tracking, small_bundle)[half]
    assert kickoff.method == "formation"
    assert truth <= kickoff.frame <= truth + 3 * small_bundle.spec.fps


def test_data_start_when_tracking_begins_at_the_block(tracking, small_bundle):
    half, first, last = small_bundle.spec.halves[0]
    tracking.players_xy[:first] = np.nan
    tracking.ball_xy[:first + 3] = np.nan
    kickoff = detect(tracking, small_bundle)[half]
    assert kickoff.method == "data-start" and kickoff.frame == first


def test_nothing_found_falls_back_with_diagnostics(tracking, small_bundle):
    tracking.ball_xy[:] = np.nan
    tracking.players_xy[:] = np.nan
    for kickoff in detect(tracking, small_bundle).values():
        assert kickoff.method == "fallback" and not kickoff.detected
        assert kickoff.frame == kickoff.block_first
        assert kickoff.diagnostics["search"]["frames_any_player"] == 0
        assert "NOT DETECTED" in kickoff.describe(25)


def test_published_anchor_check(tracking, small_bundle):
    spec = small_bundle.spec
    k1, k2 = small_bundle.kickoffs[1], small_bundle.kickoffs[2]
    good = replace(spec, kickoff_anchors=((1, k1 + 1), (2, k2 - 1)))
    assert all(k.consistent for k in detect(tracking, small_bundle, good).values())
    bad = replace(spec, kickoff_anchors=((1, k1), (2, k2 + 8 * spec.fps)))
    kickoffs = detect(tracking, small_bundle, bad)
    assert kickoffs[1].consistent and not kickoffs[2].consistent
    assert "consistent with published bounds: NO" in kickoffs[2].describe(spec.fps)


def test_pipeline_refuses_inconsistent_kickoff(small_bundle, tmp_path):
    spec = small_bundle.spec
    bad = replace(spec, kickoff_anchors=((2, small_bundle.kickoffs[2] + 8 * spec.fps),))
    with pytest.raises(pipeline.KickoffInconsistent) as info:
        pipeline.run(small_bundle.path, bad)
    assert "no file written" in str(info.value)


def test_policy_x_outside_blocks_and_d_before_kickoff(tracking, small_bundle):
    spec = small_bundle.spec
    kickoffs = halves.detect_kickoffs(tracking, spec)
    labels = halves.apply_policy(np.full(spec.n_frames, "H"), spec, kickoffs)
    graded = spec.graded_mask()
    assert (labels[~graded] == "X").all()
    assert not (labels[graded] == "X").any()
    for kickoff in kickoffs:
        assert (labels[kickoff.block_first:kickoff.frame] == "D").all()
        assert labels[kickoff.frame] == "H"
