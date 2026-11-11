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


def viterbi_batch(logps, costs) -> list:
    """Viterbi for many sequences at once: identical paths to ``viterbi`` for each item.

    logps: list of (n_i, 3) arrays; costs: (B, 3, 3). One pass over time serves every item (the
    per-frame numpy overhead is paid once, not B times), which is what makes tuning affordable.
    Tie-breaking matches ``viterbi`` exactly: the same additions, the first maximum wins.
    """
    B = len(logps)
    if B == 0:
        return []
    lens = np.array([len(x) for x in logps])
    N = int(lens.max())
    E = np.zeros((N, B, 3))
    for i, x in enumerate(logps):
        E[:len(x), i] = x
    neg_cost = -np.asarray(costs, dtype=float)                # (B, from, to)
    rows = np.arange(B)
    back = np.zeros((N, B, 3), dtype=np.int8)
    score = E[0].copy()
    final = np.where(lens[:, None] == 1, score, 0.0)
    for t in range(1, N):
        cand = score[:, :, None] + neg_cost                   # (B, from, to)
        best = cand.argmax(axis=1)                            # (B, to)
        back[t] = best
        score = np.take_along_axis(cand, best[:, None, :], axis=1)[:, 0, :] + E[t]
        ends = lens - 1 == t
        if ends.any():
            final[ends] = score[ends]
    start = final.argmax(axis=1)
    cur = np.zeros(B, dtype=np.int64)
    path = np.zeros((B, N), dtype=np.int8)
    for t in range(N - 1, -1, -1):
        begin = lens - 1 == t
        cur[begin] = start[begin]
        path[:, t] = cur
        if t > 0:
            active = t <= lens - 1
            cur[active] = back[t, rows[active], cur[active]]
    return [path[i, :lens[i]] for i in range(B)]


def half_segments(logp: np.ndarray, spec, kickoffs):
    """[(first, last, segment)] per half, with frames before the kickoff forced to D."""
    kick = {k.half: k.frame for k in kickoffs}
    out = []
    for half, first, last in spec.halves:
        segment = logp[first:last + 1].copy()
        k = kick.get(half, first)
        if k > first:
            segment[:min(k, last + 1) - first, :2] = -np.inf   # dead until the kickoff
            segment[:min(k, last + 1) - first, 2] = 0.0
        out.append((first, last, segment))
    return out


def decode_many(jobs):
    """jobs: [(logp, spec, kickoffs, DecoderParams)] -> [labels], decoded in one batch."""
    items, costs, where = [], [], []
    for j, (logp, spec, kickoffs, params) in enumerate(jobs):
        for first, last, segment in half_segments(logp, spec, kickoffs):
            items.append(segment)
            costs.append(transition_costs(params))
            where.append((j, first, last))
    paths = viterbi_batch(items, np.array(costs))
    out = [np.full(len(job[0]), "X", dtype="<U1") for job in jobs]
    for (j, first, last), path in zip(where, paths):
        out[j][first:last + 1] = STATE_LABELS[path]
    return out


def decode(logp: np.ndarray, spec, kickoffs, params: DecoderParams = DecoderParams()) -> np.ndarray:
    """Labels (H/A/D/X) for the whole video: decoded per half, X outside the halves."""
    return decode_many([(logp, spec, kickoffs, params)])[0]
