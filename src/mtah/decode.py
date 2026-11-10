"""Viterbi decoding over {H, A, D}, run separately per graded half.

The metric ranks whole possessions, so the decoder's job is to stop flicker without merging.
Transitions are priced in log units: switching team (H <-> A) costs ``c_switch``; the ball dying
(alive -> D) costs ``c_die``; a restart (D -> alive) costs ``c_restart``. A change is made only when
the evidence for it, summed over the frames it covers, outweighs the cost. That is the
"controlled possession" rule: a touch or a deflection does not flip possession, a sustained change
does.

Pre-kickoff frames inside a half are forced to D (the kickoff is detected in halves.py).
"""

from dataclasses import dataclass

import numpy as np

H, A, D = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"])


@dataclass(frozen=True)
class DecoderParams:
    # M2 defaults from a 27-point grid on external dev games 1-2 (game 3 held out), picked from
    # the flat top of the grid for a possession count closest to the truth. Re-tuned in M5.
    c_switch: float = 15.0
    c_die: float = 25.0
    c_restart: float = 25.0


def transition_costs(p: DecoderParams) -> np.ndarray:
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = p.c_switch
    c[H, D] = c[A, D] = p.c_die
    c[D, H] = c[D, A] = p.c_restart
    return c


def viterbi(logp: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Most likely state path for (n, 3) log-emissions under additive transition costs."""
    n = len(logp)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logp[0].copy()
    neg_cost = -costs
    for t in range(1, n):
        cand = score[:, None] + neg_cost          # (from, to)
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, (0, 1, 2)] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(logp: np.ndarray, spec, kickoffs, params: DecoderParams = DecoderParams()) -> np.ndarray:
    """Labels (H/A/D/X) for the whole video: decoded per half, X outside the halves."""
    labels = np.full(len(logp), "X", dtype="<U1")
    costs = transition_costs(params)
    kick = {k.half: k.frame for k in kickoffs}
    for half, first, last in spec.halves:
        segment = logp[first:last + 1].copy()
        k = kick.get(half, first)
        if k > first:
            segment[:min(k, last + 1) - first, :2] = -np.inf   # dead until the kickoff
            segment[:min(k, last + 1) - first, 2] = 0.0
        labels[first:last + 1] = STATE_LABELS[viterbi(segment, costs)]
    return labels
