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

**Emphasising frames near changes in training: tried, not shipped.** spellF1 is decided at the
changes of state, which are a small share of frames. Two training variants on the v3 features
(`ext/train.py`, `TRAINING`):

| variant | spellF1 at equal decoder settings | nested 2-fold | against v3, per game (nested) |
|---|---|---|---|
| v3 (every 5th frame, equal weights) | 0.4195 | 0.4168 | |
| `touch_w3`: frames within ±1 s of a change weighted ×3 | 0.4246 | not run | better on 9 of 13 at equal settings |
| `touch_dense`: every frame within ±1 s of a change kept | 0.4282 | 0.4234 | better on 9 of 13, mean +0.007, paired t = 1.2 |

The predicted possession count moves toward the truth (0.61 → 0.67 of it), and all three Metrica
games improve (+0.024 to +0.037 nested). But SkillCorner is unchanged on average, with four
games down by 0.016–0.022, and the overall gain is within noise. By the rule fixed beforehand (a
clear win on most games on the nested estimate) it is not adopted. The Metrica-only gain is noted
as a lead, not as evidence: choosing by a subset of games after seeing the results would be
selection, not validation.

v3's frozen decoding settings are in `config/final_v3.yaml` (switch 5.01, die 12.86, restart 6.04,
temperature 1.87, dead-ball bias +0.38, no refinement). The predicted possession count is still
0.56–0.78 of the truth per game: the touch features place boundaries better, but do not recover
most of the missed short possessions.

## Model v4: the v3 features, trained on corrected ball-visibility noise

**What was wrong.** Models v2 and v3 were trained on noise condition `full`, in which the ball
drops out independently of the state of play. A label-free measurement on the real match
(`tools/slow_play_stats.py`, aggregates only) showed that the real ball is tracked on 8.6% of the
slowest tenth of frames (by mean player speed) and on 82.7% of the faster half: it goes untracked
mostly during stoppages. Condition `full2` reproduces this (`docs/label_derivation.md`).

**Test.** All variants are scored on the same `full2` evaluation copies of the 13 external games,
leave-one-game-out, with the nested 2-fold tuning of the decoder (60 trials per fold). The
acceptance rule is the one fixed before v3: a clear win on most games on the nested estimate.

| variant | trained on | nested 2-fold spellF1 on `full2` | Metrica (3) | SkillCorner (10) |
|---|---|---|---|---|
| `touch_x2`: the v3 recipe (59 features) | `full` | 0.4189 | 0.556 | 0.378 |
| **`touch_f2`: the same 59 features (model v4)** | `full2` | **0.4385** | 0.604 | 0.389 |
| `gap_f2`: + 9 ball-gap and stoppage features | `full2` | 0.4367 | 0.606 | 0.386 |
| the frozen v3 file's model and settings, in sample | `full` | 0.4394 | 0.569 | 0.401 |

- **`touch_f2` against `touch_x2`: accepted.** Better on 10 of 13 games, mean +0.020, paired
  t = 3.1; better on 7 of 10 SkillCorner games (mean +0.011) and on all three Metrica games
  (+0.04 to +0.07). The three losses are −0.006 to −0.009. Training on noise that matches the
  real ball visibility helps when the test data has that visibility.
- **`gap_f2` against `touch_f2`: rejected.** The gap features (`learn_features.GROUP_GAP`: gap
  length, time into and until the end of the gap, whether the ball was last or next seen near a
  line, player-speed rank when it was lost, share of the last 5 s with a known ball, a sudden
  drop in player speed, time since the ball was last outside the pitch) are better on only 6 of
  13 games, mean −0.002. The existing features already carry the signal once the training data
  has the right visibility. The features stay in the code, unused by any shipped model.
- **The in-sample row is not a fair reference** and is shown only for transparency: that model
  was trained on every game it is scored on. On the old data the same in-sample figure is 0.434
  against an honest nested 0.417, an optimism of about 0.017. `touch_x2` is the honest version of
  that row.

What the test can and cannot show: it shows that v4 is better than v3 **if** the real ball
behaves as under `full2`. That premise rests on a label-free measurement of the real match, not
on its labels. No label, ground truth or public-window score of the provided match was used to
build `full2`, to choose v4, or to tune it.

**A degenerate decoder regime, found and excluded.** The first search for v4's decoder settings
returned `c_switch` 31.9, `c_die` 4.5, `c_restart` 4.6: the coordinate-wise median of a top 10
that mixed two regimes. A direct change of team then costs more than the ball dying and
restarting, so the decoder never switches directly and bridges every change with a short dead
ball. Measured on the 13 external games (means per game):

| | dead-ball runs | of them under 1 s | direct H↔A changes | spellF1 |
|---|---|---|---|---|
| external truth | 103 | 3.3 | 236 | |
| first search (bridging) | 222 | 105.8 | 0 | 0.434 |
| **frozen v4 settings** | 102 | 0.0 | 116 | 0.439 |

spellF1 is blind to the difference, because possessions are runs of H or of A either way; the
labels are plainly wrong, and macroF1 and any reader of the file would see it. The rule
`c_switch ≤ c_die + c_restart` (`tune.no_bridge`) is now part of the search for the `full2`
variants: invalid trials are not decoded and are scored 0, so the whole budget goes to valid
settings. All three variants in the table above were re-tuned under the rule. The v2 and v3
settings, frozen earlier, satisfy it. `tools/report.py` gained a matching alarm (at most 10% of
dead-ball runs under 1 s), fixed before v4 was run on the real match.

v4's frozen decoding settings are in `config/final_v4.yaml` (switch 6.34, die 24.23, restart 4.43,
temperature 1.77, dead-ball bias +0.53, no refinement).

## Model v5: v4 plus a team prior from Smart Tagging

**Why it was left out before, and why that was too cautious.** Smart Tagging exists only for the
provided match, so we first judged that its weight could not be validated. But SkillCorner's open
data carries its own automatic phases of play with a team (build-up, create, finish, transition,
quick break), which is the same kind of stream. Our SkillCorner dead-ball labels were derived
from those phases, so they cannot test anything about alive against dead. They can test a prior
on **which team** has the ball, which is where the tags have something to add: a label-free check
on the real match (`tools/tagging_stats.py`) showed that the dead-ball side is already consistent
(a dead ball of ours before 55 of 57 tagged set pieces), while inside the tagged possession
phases our team call differs from the tags on 8–15% of alive frames.

**The prior.** On frames inside a BUILD UP, PROGRESSION, CREATING CHANCES or ATTACKING TRANSITION
phase of one team, the other team's log-score is lowered by `weight` nats per frame before
decoding (`io_tagging.team_evidence`, `add_team_prior`). Frames where phases of both teams
overlap are skipped. The dead-ball score is never touched, and v4's model and decoder settings
are unchanged.

**The test** (`python -m ext.tags experiment`): the frozen v4 settings on the leave-one-game-out
predictions of the 10 SkillCorner games, with and without the prior. SkillCorner's phases come
from the same possession model as its labels, so they are cleaner than an independent tagger.
They are therefore degraded: boundaries moved by about 1 s, 22% of phases dropped to match the
real coverage, and a share of phases given to the wrong team.

| tag quality | tags name the true team on | we agree with the tags on | spellF1 gain at weight 0.1 | **0.2** | 0.4 | 0.8 |
|---|---|---|---|---|---|---|
| clean | 87.1% | 91.9% | +0.006 | **+0.009** (10 of 10 games) | +0.014 | +0.020 |
| real-like (jitter, 22% dropped) | 87.0% | 91.7% | +0.004 | **+0.007** (8 of 10) | +0.010 | +0.013 |
| + 10% of phases to the wrong team | 80.4% | 84.4% | +0.001 | **+0.003** (6 of 10) | +0.003 | +0.001 |
| + 20% wrong team | 72.1% | 75.4% | +0.004 | **+0.005** (8 of 10) | −0.001 | −0.011 |
| + 30% wrong team | 64.0% | 65.5% | +0.002 | **+0.001** (6 of 10) | −0.001 | −0.010 |

(Baseline without tags: 0.3881 on these 10 games.)

**The weight rule** (`ext.tags.choose_weight`, fixed before the real run): no mean loss in any
scenario, including the implausibly bad ones; then the best worst case over the plausible ones.
It gives **0.2**, frozen in `config/final_v5.yaml`. Larger weights earn more when the tags are
good and lose when they are bad.

**What to expect.** A small gain: about +0.003 to +0.009 spellF1 on the stand-in games, and it
does not recover missed possessions (the predicted possession count does not move). On the real
match our agreement with the tagging evidence is close to the "real-like" and "10% wrong team"
rows, far from the 20% and 30% rows. The stand-in is imperfect (a different vendor's phases),
which is why the rule buys safety before gain.

## Where the score is lost, and possession-balanced training (tried, not shipped)

**Loss budget** on the 13 external games (v4, leave-one-game-out, frozen settings; mean spellF1
0.439):

| true possession length | share of all | found (IoU ≥ 0.5) | missed entirely | overlapped but merged or misplaced | frames where the model favours the true team |
|---|---|---|---|---|---|
| under 1.5 s | 18% | 2% | 73% | 25% | 37% |
| 1.5–3 s | 15% | 14% | 54% | 32% | 51% |
| 3–8 s | 28% | 32% | 31% | 36% | 67% |
| 8–20 s | 25% | 61% | 7% | 33% | 83% |
| over 20 s | 15% | 73% | 1% | 26% | 91% |

- **Boundary timing is not the problem.** Of the true changes that have one of ours within 2 s,
  the median error is 0.44 s with no bias (median signed error 0.00 s). Moving every such change
  onto the true frame would give 0.471 (+0.03).
- **Short possessions are.** Writing every true possession under 3 s into our output would give
  0.695. A missed short possession costs three matches: itself, and the two neighbours of the
  other team that get merged into one long spell (our overlapping spell is 2–3 times too long).
- **The cause is the per-frame model, not the decoder.** The decoder finds what the summed
  evidence supports; for possessions under 3 s the evidence favours the wrong team as often as
  the right one.

**Possession-balanced training.** spellF1 counts every possession once; per-frame training
counts a 20 s possession twenty times more than a 1 s one. Two variants on the v4 data
(`ext/train.py`, `BALANCE`) keep every frame of a possession shorter than 8 s and weight it so
that the possession carries the mass of an 8 s one:

| variant | nested 2-fold spellF1 | against v4 (0.4385), per game | predicted / true possessions |
|---|---|---|---|
| v4 | 0.4385 | | 0.63 |
| `touch_f2_bal`: every possession under 8 s | 0.4403 | better on 8 of 13, mean +0.002, t = 0.3 | 0.69 |
| `touch_f2_bal15`: only 1.5–8 s (shorter ones left alone) | 0.4356 | better on 3 of 13, mean −0.003 | 0.67 |

Neither passes the rule (a clear win on most games), so v4's model stays. The balanced models
do predict more possessions, but the extra ones match the truth no better than chance of a
match, and the tuned decoder answers with a higher switch cost. Together with the earlier
near-change weighting (+0.007, rejected), this is the third reweighting that fails: the features
do not separate a short controlled possession from a touch in a duel, and reweighting cannot add
information that is not there. The ball is also untracked on about a third of those frames.

## Dead-ball evidence from a collective slowdown (tried, not shipped)

`src/mtah/deadball.py` turns a sudden, collective slowdown of the players into evidence for the
dead-ball state. It never writes a label: its score is added to the dead-ball emission and the
decoder weighs it.

- **Signal.** Mean of the 7 highest player speeds, smoothed with a centred 0.5 s rolling median,
  divided by its rolling median over the previous 10 s (leaving out the last second). A
  slowdown candidate is a ratio at or below 0.60 for at least 1 s.
- **Penalty-area exclusion.** A candidate whose ball (or its position seen within the last 2 s)
  is inside either penalty area or its arc is dropped: a goalkeeper holding the ball slows
  everyone down while the ball is alive.
- **Timing.** The dead stretch starts up to 3 s before the slowdown, at the ball crossing a line,
  else a sharp ball stop, else the ball disappearing. It ends when the ball moves off after
  resting or the top speeds recover to 0.8 of the baseline.
- **Guard.** A candidate counts only with a second cue: ball outside the pitch, ball at rest for
  1 s, ball untracked for 1 s, or a tagged set piece starting within 3 s.

All numbers are parameters (`DeadballParams`) and were left at their first values: none was
tuned. Ablation on the 13 external games under `full2` noise (`python -m ext.deadball_ablation`;
means; the external tagging stand-in has no set pieces, so the tag cue never fires here):

| variant | dead-ball precision | dead-ball recall | alive/dead F1 | spellF1 | games better |
|---|---|---|---|---|---|
| rules v1, no detector | 0.818 | 0.900 | 0.882 | 0.312 | |
| rules v1 + detector | 0.761 | 0.926 | 0.849 | 0.296 | 0 of 13 |
| rules v1 + detector, no penalty-area exclusion | 0.738 | 0.935 | 0.832 | 0.289 | 0 of 13 |
| rules v1 + detector, no second-cue guard | 0.744 | 0.927 | 0.838 | 0.284 | 0 of 13 |
| the detector alone (score > 0 as a label) | 0.747 | 0.597 | | | |
| v4 (learned, leave-one-game-out), no detector | 0.899 | 0.944 | 0.937 | 0.439 | |
| v4 + detector, weight 0.5 | 0.883 | 0.954 | 0.932 | 0.435 | 5 of 13 |
| v4 + detector, weight 1 | 0.865 | 0.959 | 0.924 | 0.431 | 4 of 13 |
| v4 + detector, weight 2 | 0.827 | 0.964 | 0.903 | 0.414 | 2 of 13 |
| v4 + detector, weight 1, no penalty-area exclusion | 0.854 | 0.962 | 0.918 | 0.428 | 3 of 13 |

Over the 13 games the detector saw 2,012 slowdown candidates: 323 were dropped by the
penalty-area rule, 275 for lack of a second cue, 1,414 accepted (534 at low confidence). Cues on
the accepted ones: ball untracked 1,356, ball outside 428, ball at rest 150. Starts: line
crossing 240, ball stop 50, ball disappearing 190, slowdown onset 934.

**Reading.** The two safeguards work as designed: removing the penalty-area exclusion or the
second-cue guard makes every number worse. But the detector as a whole is not adopted. It finds
a little more dead ball (recall +0.01 to +0.03) at a larger cost in false dead ball, because its
own precision (0.75) is below that of the model it is added to (0.82 for the rules, 0.90 for
v4), and spellF1 falls in every configuration. Two reasons. First, the learned model already
uses the same information (relative player speed with past and future windows, ball outside,
ball at rest), so the detector repeats evidence instead of adding it. Second, under realistic
ball visibility the ball is untracked through most slow play, so "ball untracked for 1 s" fires
for 96% of accepted candidates and hardly filters anything. The code, the tests on hand-built
sequences and the aggregate tool (`tools/deadball_stats.py`) are kept; the signals are exposed
to the learned-model feature matrix (`learn_features.GROUP_DEADBALL`) but belong to no shipped
feature set. A learned variant on those features (`VARIANTS["dead"]`) was not trained.
