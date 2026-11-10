import numpy as np
import pytest

from mtah import halves, io_atd


@pytest.fixture
def tracking(small_bundle):
    return io_atd.load_game(small_bundle.path, small_bundle.spec)


def detect(tracking, bundle):
    return {k.half: k for k in halves.detect_kickoffs(tracking, bundle.spec)}


def test_kickoffs_detected_from_ball_on_spot(tracking, small_bundle):
    for half, kickoff in detect(tracking, small_bundle).items():
        assert kickoff.method == "ball+shape"
        assert abs(kickoff.frame - small_bundle.kickoffs[half]) <= 2


def test_ball_jitter_at_rest_is_tolerated(tracking, small_bundle):
    """The real ball track jitters by ~0.1 m a frame; a resting ball must still count as resting."""
    rng = np.random.default_rng(1)
    tracking.ball_xy += rng.normal(0.0, 0.1, size=tracking.ball_xy.shape)
    for half, kickoff in detect(tracking, small_bundle).items():
        assert kickoff.method == "ball+shape"
        assert abs(kickoff.frame - small_bundle.kickoffs[half]) <= 3


def test_shape_fallback_when_ball_is_never_seen(tracking, small_bundle):
    """HYPOTHESIS check of the fallback: without a ball, the end of the kickoff formation."""
    tracking.ball_xy[:] = np.nan
    for half, kickoff in detect(tracking, small_bundle).items():
        truth = small_bundle.kickoffs[half]
        if half == 1:
            continue  # half 1 has tracking before its block, so it is the shape case too
        assert kickoff.method == "shape"
        assert truth <= kickoff.frame <= truth + 5 * small_bundle.spec.fps


def test_data_start_when_tracking_begins_at_the_block(tracking, small_bundle):
    """A first half graded from its kickoff, with no tracked rest before it."""
    (half, first, last) = small_bundle.spec.halves[0]
    tracking.players_xy[:first] = np.nan
    tracking.ball_xy[:first + 3] = np.nan
    kickoff = detect(tracking, small_bundle)[half]
    assert kickoff.method == "data-start" and kickoff.frame == first


def test_tracking_resuming_before_lineup_is_not_a_kickoff(tracking, small_bundle):
    """Second-half risk: tracking resumes at the block start while players walk out, no ball.
    That must not be read as a kickoff at the block start."""
    half, first, last = small_bundle.spec.halves[1]
    truth = small_bundle.kickoffs[half]
    tracking.ball_xy[:] = np.nan
    walk = slice(first, first + 4 * small_bundle.spec.fps)   # players scattered, not lined up
    tracking.players_xy[walk, :, 0] = small_bundle.spec.pitch[0] / 2.0 + np.linspace(
        -30, 30, tracking.players_xy.shape[1])
    kickoff = detect(tracking, small_bundle)[half]
    assert kickoff.method != "data-start"
    assert truth <= kickoff.frame <= truth + 5 * small_bundle.spec.fps


def test_nothing_found_falls_back_with_diagnostics(tracking, small_bundle):
    tracking.ball_xy[:] = np.nan
    tracking.players_xy[:] = np.nan
    for kickoff in detect(tracking, small_bundle).values():
        assert kickoff.method == "fallback" and not kickoff.detected
        assert kickoff.frame == kickoff.block_first
        assert kickoff.diagnostics["search"]["frames_any_player"] == 0
        assert "NOT DETECTED" in kickoff.describe(25)


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
