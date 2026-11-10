"""Emission model v1: hand-written rules -> per-frame log p(H), log p(A), log p(D).

Factorised as  p(D) = sigmoid(z_dead)  and  p(H | alive) = sigmoid(z_team), so team evidence never
argues about alive vs dead and vice versa.

z_team (positive = home):
  * proximity: who is nearest the (observed or interpolated) ball, weighted by how close the
    nearest player is: margin d_away - d_home through tanh;
  * last touch: the team that last had a player clearly on the ball, fading with time. It is used
    when nobody is near the ball, which covers the long real-match ball gaps;
  * shape: the team in possession pushes its centroid towards the goal it attacks (weak).

z_dead:
  * players slow: the median player speed below SPEED_PIVOT means a stoppage (strongest cue);
  * ball outside the pitch;
  * ball known and stationary.
  Deliberately NOT used: "the ball is missing". In the Metrica open data the ball is blanked while
  the ball is dead, but on the real match it goes missing mostly during live play. A rule built on
  it would fit the external games and fail on the real one.

Weights are hand-set from feature distributions on external games 1-2 (game 3 held out). They are
tuned properly in M5.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmissionParams:
    k_ball: float = 2.5
    near_m: float = 2.5          # proximity counts fully when the nearest player is within this
    margin_scale_m: float = 2.0
    k_touch: float = 1.2
    touch_fade_s: float = 8.0
    k_shape: float = 0.4
    shape_scale_m: float = 15.0
    dead_bias: float = -0.5
    k_speed: float = 0.7
    speed_pivot: float = 1.35    # m/s; median player speed below this argues for a stoppage
    speed_scale: float = 0.3
    k_out: float = 3.0
    k_still: float = 0.7


def _logsig(z):
    return -np.logaddexp(0.0, -z)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def team_logit(F, p: EmissionParams) -> np.ndarray:
    nearest = np.fmin(F.d_home, F.d_away)
    margin = np.nan_to_num(F.d_away - F.d_home, nan=0.0)
    w_near = np.where(np.isfinite(nearest), _sig((p.near_m - np.nan_to_num(nearest, nan=99.0))
                                                 / 0.5), 0.0)
    z = p.k_ball * w_near * np.tanh(margin / p.margin_scale_m)
    fade = np.exp(-np.where(np.isfinite(F.since_touch_s), F.since_touch_s, 1e9) / p.touch_fade_s)
    z += p.k_touch * F.last_touch * fade * (1.0 - w_near)
    z += p.k_shape * np.tanh(np.nan_to_num(F.shape_diff, nan=0.0) / p.shape_scale_m)
    return z


def dead_logit(F, p: EmissionParams) -> np.ndarray:
    z = np.full(len(F.ball_known), p.dead_bias)
    speed = np.clip((p.speed_pivot - F.player_speed) / p.speed_scale, -3.0, 3.0)
    z += p.k_speed * np.nan_to_num(speed, nan=0.0)
    z += p.k_out * np.where(np.isfinite(F.out_dist), _sig((F.out_dist - 0.5) / 0.25), 0.0)
    still = F.ball_known & (np.nan_to_num(F.ball_speed, nan=99.0) < 0.5)
    z += p.k_still * still
    return z


def rules_v1(F, params: EmissionParams = EmissionParams()) -> np.ndarray:
    """(n, 3) log-probabilities for states (H, A, D)."""
    zt = team_logit(F, params)
    zd = dead_logit(F, params)
    log_alive = _logsig(-zd)
    out = np.empty((len(zt), 3))
    out[:, 0] = log_alive + _logsig(zt)
    out[:, 1] = log_alive + _logsig(-zt)
    out[:, 2] = _logsig(zd)
    return out
