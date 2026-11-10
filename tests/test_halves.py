import numpy as np
import pytest

from mtah import halves, io_atd


def test_kickoffs_detected_from_ball_shape(small_bundle):
    tracking = io_atd.load_game(small_bundle.path, small_bundle.spec)
    kickoffs = halves.detect_kickoffs(tracking, small_bundle.spec)
    assert [k.half for k in kickoffs] == [1, 2]
    for kickoff in kickoffs:
        assert kickoff.method == "ball"
        assert abs(kickoff.frame - small_bundle.kickoffs[kickoff.half]) <= 5


def test_shape_fallback_when_ball_is_never_seen(small_bundle):
    """HYPOTHESIS check of the fallback: without a ball, the end of the kickoff formation."""
    tracking = io_atd.load_game(small_bundle.path, small_bundle.spec)
    tracking.ball_xy[:] = np.nan
    for kickoff in halves.detect_kickoffs(tracking, small_bundle.spec):
        assert kickoff.method == "shape"
        truth = small_bundle.kickoffs[kickoff.half]
        assert truth <= kickoff.frame <= truth + 2 * small_bundle.spec.fps


def test_no_kickoff_shape_raises(small_bundle):
    tracking = io_atd.load_game(small_bundle.path, small_bundle.spec)
    tracking.ball_xy[:] = np.nan
    tracking.players_xy[:] = np.nan
    with pytest.raises(halves.KickoffNotFound):
        halves.detect_kickoffs(tracking, small_bundle.spec)


def test_policy_x_outside_blocks_and_d_before_kickoff(small_bundle):
    spec = small_bundle.spec
    tracking = io_atd.load_game(small_bundle.path, spec)
    kickoffs = halves.detect_kickoffs(tracking, spec)
    labels = halves.apply_policy(np.full(spec.n_frames, "H"), spec, kickoffs)
    graded = spec.graded_mask()
    assert (labels[~graded] == "X").all()
    assert not (labels[graded] == "X").any()
    for kickoff in kickoffs:
        assert (labels[kickoff.block_first:kickoff.frame] == "D").all()
        assert labels[kickoff.frame] == "H"
