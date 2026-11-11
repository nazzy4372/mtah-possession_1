import numpy as np
import pytest

from mtah import emission_v2, io_atd, learn_features


@pytest.fixture(scope="module")
def built(small_bundle):
    tr = io_atd.load_game(small_bundle.path, small_bundle.spec)
    names, X = learn_features.build(tr, small_bundle.spec)
    return small_bundle.spec, names, X


def test_matrix_shape_and_graded_only(built):
    spec, names, X = built
    assert X.shape == (spec.n_frames, len(names)) and X.dtype == np.float32
    assert len(set(names)) == len(names)
    graded = spec.graded_mask()
    assert np.isnan(X[~graded]).all()
    rank = X[graded, names.index("ps_rank")]
    assert np.nanmin(rank) >= 0.0 and np.nanmax(rank) <= 1.0


def test_no_feature_encodes_ball_missingness_directly(built):
    _, names, _ = built
    assert not any("missing" in n or "known" in n or "observed" in n for n in names)


def test_team_swap_is_an_involution_and_mirrors(built):
    _, names, X = built
    Y = learn_features.swap_teams(names, X)
    i = {n: k for k, n in enumerate(names)}
    assert np.array_equal(np.nan_to_num(learn_features.swap_teams(names, Y)), np.nan_to_num(X))
    assert np.array_equal(np.nan_to_num(Y[:, i["d_home"]]), np.nan_to_num(X[:, i["d_away"]]))
    assert np.array_equal(np.nan_to_num(Y[:, i["margin"]]), np.nan_to_num(-X[:, i["margin"]]))
    assert np.array_equal(np.nan_to_num(Y[:, i["ps_rank"]]), np.nan_to_num(X[:, i["ps_rank"]]))
    for a, b in learn_features.PAIRS:
        assert a in i and b in i
    for name in learn_features.SIGNED:
        assert name in i


def test_relative_speed_is_scale_free(small_bundle):
    """Rank features must not change when every speed is scaled (the cross-source gap)."""
    x = np.random.default_rng(0).gamma(2.0, 0.7, size=small_bundle.spec.n_frames)
    r1, _ = learn_features._relative(x, small_bundle.spec)
    r2, _ = learn_features._relative(x * 1.3, small_bundle.spec)
    assert np.array_equal(np.nan_to_num(r1), np.nan_to_num(r2))


@pytest.mark.skipif(not emission_v2.MODEL_TXT.is_file(), reason="model not trained yet")
def test_v2_emissions_deterministic(small_bundle):
    from mtah import features
    tr = io_atd.load_game(small_bundle.path, small_bundle.spec)
    F = features.compute(tr, small_bundle.spec)
    a = emission_v2.emissions(tr, small_bundle.spec, F)
    b = emission_v2.emissions(tr, small_bundle.spec, F)
    assert np.array_equal(a, b) and np.isfinite(a).all()
