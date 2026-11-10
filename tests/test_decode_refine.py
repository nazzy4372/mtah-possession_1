import numpy as np

from mtah import decode, features, refine
from mtah.halves import Kickoff
from mtah.spec import GameSpec

SPEC = GameSpec(name="T", stem="T_tactical", n_frames=400, fps=25, pitch=(105.0, 68.0),
                halves=((1, 50, 349),))


def emissions(states, strength=1.0, n=400):
    """(n, 3) log-emissions favouring the given per-frame state string by `strength` nats."""
    logp = np.full((n, 3), -strength)
    for i, s in enumerate(states):
        logp[i, "HAD".index(s)] = 0.0
    return logp


def test_viterbi_ignores_short_flicker_and_follows_sustained_change():
    states = "H" * 100 + "A" * 5 + "H" * 100 + "A" * 95 + "D" * 100
    labels = decode.decode(emissions(states), SPEC, [], decode.DecoderParams(15, 25, 25))
    graded = labels[50:350]
    assert set(labels[:50]) == {"X"} and set(labels[350:]) == {"X"}
    assert (graded[:155] == "H").all()             # the 5-frame A blip costs more than it earns
    assert (labels[205:300] == "A").all()          # 95 frames of A are worth a switch
    assert (labels[300:350] == "D").all()


def test_pre_kickoff_frames_are_forced_dead():
    kick = Kickoff(half=1, block_first=50, block_last=349, frame=120, method="formation")
    labels = decode.decode(emissions("H" * 400, strength=5.0), SPEC, [kick])
    assert (labels[50:120] == "D").all() and (labels[120:350] == "H").all()


def _features(n, d_home, d_away, out_dist=None, ball_speed=None):
    nan = np.full(n, np.nan)
    return features.Features(
        ball_xy=np.zeros((n, 2)), ball_observed=np.ones(n, bool), ball_known=np.ones(n, bool),
        ball_speed=ball_speed if ball_speed is not None else nan.copy(),
        ball_missing_s=np.zeros(n), d_home=d_home, d_away=d_away,
        out_dist=out_dist if out_dist is not None else np.full(n, -5.0),
        player_speed=nan.copy(), last_touch=np.zeros(n, int), since_touch_s=nan.copy(),
        shape_diff=nan.copy())


def test_refine_snaps_team_switch_to_first_touch_by_new_team():
    n = 400
    labels = np.array(["X"] * 50 + ["H"] * 150 + ["A"] * 150 + ["X"] * 50)
    d_home, d_away = np.full(n, 1.0), np.full(n, 8.0)
    d_away[185:] = 0.8                      # an away player reaches the ball 15 frames earlier
    d_home[185:] = 3.0
    out = refine.refine(labels, _features(n, d_home, d_away), SPEC)
    assert (out[50:185] == "H").all() and (out[185:350] == "A").all()


def test_refine_snaps_ball_death_to_out_of_bounds_and_respects_neighbours():
    n = 400
    labels = np.array(["X"] * 50 + ["H"] * 150 + ["D"] * 20 + ["A"] * 130 + ["X"] * 50)
    out_dist = np.full(n, -5.0)
    out_dist[180:] = 1.0                    # ball crossed the line 20 frames before the call
    F = _features(n, np.full(n, 1.0), np.full(n, 9.0), out_dist=out_dist)
    out = refine.refine(labels, F, SPEC)
    assert (out[180:200] == "D").all() and out[179] == "H"


def test_ball_interpolation_only_bridges_short_gaps():
    ball = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [3.0, 0.0],
                     [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                     [8.0, 0.0]])
    seen = ~np.isnan(ball[:, 0])
    out, interp = features.interpolate_ball(ball, seen, max_gap=2)
    assert out[1].tolist() == [1.0, 0.0] and out[2].tolist() == [2.0, 0.0]
    assert interp.tolist() == [False, True, True, False, False, False, False, False, False]
    assert np.isnan(out[4:8]).all()
