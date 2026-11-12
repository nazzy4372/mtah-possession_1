import numpy as np
import pytest
import yaml

from mtah import decode, tune


def test_batched_viterbi_matches_single_decoder():
    rng = np.random.default_rng(0)
    seqs, costs = [], []
    for i in range(9):
        n = int(rng.integers(1, 1500))
        x = rng.normal(0, 1, size=(n, 3))
        if i % 3 == 0:
            x = np.round(x)                 # ties
        if i % 2 == 0 and n > 6:
            x[: n // 3, :2] = -np.inf       # forced dead prefix
        seqs.append(x)
        costs.append(decode.transition_costs(decode.DecoderParams(
            *rng.uniform(0, 20, size=3).tolist())))
    batch = decode.viterbi_batch(seqs, np.array(costs))
    for s, c, b in zip(seqs, costs, batch):
        assert np.array_equal(decode.viterbi(s, c), b)


def test_dynamic_viterbi_with_zero_weight_equals_static():
    rng = np.random.default_rng(1)
    seqs = [rng.normal(0, 1, size=(int(rng.integers(2, 800)), 3)) for _ in range(6)]
    params = [decode.DecoderParams(*rng.uniform(1, 15, size=3).tolist()) for _ in seqs]
    static = decode.viterbi_batch(seqs, np.array([decode.transition_costs(p) for p in params]))
    base = np.array([(p.c_switch, p.c_die, p.c_restart) for p in params])
    ratios = [rng.normal(0, 2, size=(len(s), 3)).astype(np.float32) for s in seqs]
    dyn = decode.viterbi_batch_dyn(seqs, base, np.zeros(len(seqs)), ratios)
    for a, b in zip(static, dyn):
        assert np.array_equal(a, b)


def test_dynamic_viterbi_switches_where_a_change_is_likely():
    n = 200
    logp = np.zeros((n, 3))
    logp[:, 2] = -50.0                        # never dead
    logp[:100, 0], logp[100:, 1] = 0.05, 0.05  # weak evidence: H then A
    base = np.array([[20.0, 20.0, 20.0]])
    flat = np.zeros((n, 3), dtype=np.float32)
    assert len(set(decode.viterbi_batch_dyn([logp], base, np.array([1.0]), [flat])[0])) == 1
    cue = flat.copy()
    cue[100, 0] = 25.0                        # the change model is sure of a switch at frame 100
    path = decode.viterbi_batch_dyn([logp], base, np.array([1.0]), [cue])[0]
    assert (path[:100] == decode.H).all() and (path[100:] == decode.A).all()


class _Trial:
    def __init__(self, value, params):
        self.value, self.params = value, params


class _Study:
    def __init__(self, trials):
        self.trials = trials


def test_median_of_top_ten_and_majority_vote():
    trials = [_Trial(1.0 - k / 100.0, {"c_switch": float(k), "c_die": 10.0, "c_restart": 5.0,
                                       "temperature": 1.0, "dead_bias": 0.0,
                                       "refine": k % 3 == 0}) for k in range(20)]
    params, scores = tune.median_of_top(_Study(trials))
    assert params["c_switch"] == 4.5              # median of 0..9
    assert params["refine"] is False              # 4 of the top 10 refine
    assert scores == sorted(scores, reverse=True) and len(scores) == 10


def test_config_round_trip(tmp_path):
    params = {"c_switch": 12.3, "c_die": 30.0, "c_restart": 20.5, "temperature": 0.9,
              "dead_bias": -0.25, "refine": False}
    path = tmp_path / "final.yaml"
    tune.write_config(params, "all", [0.41, 0.40], 0.39, path=path)
    data = path.read_bytes()
    assert b"\r" not in data
    cfg = yaml.safe_load(data)
    assert cfg["decoder"]["c_switch"] == pytest.approx(12.3)
    assert cfg["emission"]["dead_bias"] == pytest.approx(-0.25)
    assert cfg["refine"] is False and cfg["model_variant"] == "all"


def test_folds_mix_both_sources():
    games = ["metrica_g1", "metrica_g2", "metrica_g3"] + ["sc_{}".format(i) for i in range(10)]
    a, b = tune.folds(games)
    assert sorted(a + b) == sorted(games) and not set(a) & set(b)
    for fold in (a, b):
        assert any(g.startswith("metrica") for g in fold) and any(g.startswith("sc_") for g in fold)
