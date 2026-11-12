# Ablations on external games (M4)

Every score is the official `spellF1` (imported from `upstream/evaluation/possession_metrics.py`),
averaged over 13 external games (3 Metrica, 10 SkillCorner). Each game is scored under the `full`
noise model (`ext/augment.py`), which is calibrated to the real match's label-free aggregates.
Learned variants are scored **leave-one-game-out**: the scored game is never in training, and its
corrupted copy uses a different noise seed from the training copies.

Reproduce with `python -m ext.train cache`, `python -m ext.train logo --variant <v> [--grid]`,
then `--reuse` for decoding-only changes. `python -m ext.evaluate --conditions full` scores v1 and
the baseline.

| variant | spellF1, 13 games | Metrica (3) | SkillCorner (10) |
|---|---|---|---|
| **v2: LightGBM, all features, no refinement (shipped)** | **0.395** | **0.473** | **0.371** |
| v2, with boundary refinement | 0.388 | 0.469 | 0.364 |
| v2 without shape features (centroid advance, spread) | 0.368 | 0.447 | 0.344 |
| v2 with absolute instead of game-relative player speed | 0.389 | 0.474 | 0.364 |
| v1: hand-written rules + the same decoder (with refinement) | 0.301 (0.280–0.323) | 0.381 | 0.277 |
| ported upstream baseline (nearest player, ±50-frame vote) | 0.189 (0.176–0.200) | 0.229 | 0.176 |

Notes:
- v2 beats v1 on **all 13 games**, by 0.06 to 0.15 each. Per-game bootstrap intervals are in
  `outputs/external/logo_all.json`.
- **Refinement** (boundary snapping) helps the rules but slightly hurts v2: 10 of 13 games are
  better without it and 2 are ties. v2's boundaries are already sharp, so v2 ships with refinement
  off (`"refine": false` in `models/emission_v2.json`).
- **Shape features** are worth about 0.027. They carry the team call while the ball is unknown,
  which is 41% of the real match.
- **Relative speed** (rank and robust z within the half) scores about the same as absolute speed
  on these games (+0.006). It is kept because it is invariant to a source's speed scale. Real and
  Metrica share a scale; SkillCorner does not (`tools/feature_stats.py`).
- **Smart Tagging: not evaluable.** No external source has an independent automatic tagging stream.
  SkillCorner's phases of play are the nearest equivalent, but the SkillCorner labels are derived
  from them, so any test on them would be circular. v2 therefore does not use tagging. Revisit
  only with an evaluation that does not leak.
- **Decoder costs**: a 12-point grid over (switch, dead) on the out-of-fold predictions was flat
  (0.375–0.388). (15, 25) was the best point and equals v1's existing defaults. It is chosen on
  the same 13 games it is scored on, so the v2 figures are slightly optimistic. M5 tunes
  properly.
- **Calibration**: a single temperature fitted on the out-of-fold predictions came out at 1.0, so
  LightGBM's probabilities were already calibrated.
- **Determinism**: retraining produces a byte-identical `models/emission_v2.txt`
  (SHA-256 `d33c42ac…3c85`). Inference is single-threaded.

# Tuning and freeze (M5)

`python tasks.py tune` (`src/mtah/tune.py`): Optuna, seeded TPE, 60 trials, over the three
decoder costs, the emission temperature, a dead-ball bias and refinement on/off. The objective is
the mean official spellF1 on the leave-one-game-out predictions under `full` noise. Trials are
decoded in batches with `decode.viterbi_batch`, which gives identical paths to the single decoder.

| | spellF1, 13 external games |
|---|---|
| M4 settings (costs 15 / 25 / 25, T 1.0, no refinement) | 0.3947 (chosen on the same games) |
| best single Optuna trial | 0.3955 |
| **nested 2-fold estimate** (tune on one fold, score the other, swap; folds mix both sources) | **0.3932** |
| frozen median of the top 10 trials, re-scored on all 13 | 0.3940 (optimistic) |

Frozen in `config/final.yaml`: c_switch 7.46, c_die 11.21, c_restart 5.57, temperature 1.77,
dead-ball bias +0.24, refinement off.

Findings:
- **Tuning gains nothing measurable.** The landscape is flat, and the M4 settings were already at
  the optimum. A temperature of 1.77 roughly halves the emission scale, so the frozen costs
  correspond to about 13 / 20 / 10 at temperature 1, close to M4's.
- **Under-segmentation is a model limit, not a decoder one.** Across all tuning the predicted
  possession count stays at 0.59–0.74 of the truth per game. Lower switch costs add spurious
  switches faster than they recover real short possessions, so spellF1 does not improve. The fix
  would have to come from sharper team evidence at changes of control.
- **Wide windows are not the cause.** Retraining without the ±5 s windows (variant `short`)
  scores the same (0.3941 vs 0.3947 at the M4 settings), so the shipped `all` model is kept.

# Model v3: change-of-possession evidence (after M5)

An error analysis of v2 on the external out-of-fold predictions (no data from the provided match):

| | true | predicted | matched within ±1 s |
|---|---|---|---|
| team changes (H ↔ A) | 3,069 | 1,567 | 15% of the true ones |
| ball dies (alive → D) | 1,333 | 1,359 | 43% |
| restarts (D → alive) | 1,331 | 1,356 | 53% |

- There is **no systematic lag** (median 0 to −1 frames), so a constant boundary shift cannot help.
- The loss is in team changes. Possessions shorter than 3 s are a third of all true possessions
  and v2 matches 6% of them; longer ones are matched 46% of the time.

**Touch features** (`learn_features.GROUP_TOUCH`, 20 features, variant `touch`): for each team, the
share of frames on the ball in the 1 s and 3 s before and after the frame, the time since its last
touch and until its next one, the before/after difference of touch and proximity margin, the size
of the ball's velocity change (a kick or first touch), and the ball's speed and position along the
home team's attacking direction. All are mirrored when home and away are swapped.

| variant (leave-one-game-out, `full` noise) | nested 2-fold spellF1 | note |
|---|---|---|
| v2 (39 features), frozen in M5 | 0.3932 | |
| **v3: v2 + touch features (shipped)** | **0.4168** | better on 11 of 13 games at equal decoder settings (0.4195 vs 0.3947); the two losses are −0.002 and −0.004 |
| v3 + change-point model, weight searched in 0–3 | 0.4226 | weight at the upper edge |
| v3 + change-point model, weight searched in 0–8 | 0.4289 | weight at the upper edge |
| v3 + change-point model, weight searched in 0–25 | 0.4125 | below touch-only |

**Change-point model: tried, not shipped.** A second LightGBM model predicts whether a switch, a
stoppage or a restart happens within ±5 frames, and the decoder's transition costs follow it
(`decode.viterbi_batch_dyn`). Out of fold it flags 80–87% of true switch frames. But its effect on
spellF1 depends on the search range for its weight (+0.006, +0.012, then −0.004 against
touch-only), which is not a robust gain. The acceptance rule, set before the test, was a clear win
on most games on the nested estimate. It failed, so v3 ships without it. The code is kept and tested.

v3's frozen decoding settings are in `config/final_v3.yaml` (switch 5.01, die 12.86, restart 6.04,
temperature 1.87, dead-ball bias +0.38, no refinement). The predicted possession count is still
0.56–0.78 of the truth per game: the touch features place boundaries better, but do not recover
most of the missed short possessions.
