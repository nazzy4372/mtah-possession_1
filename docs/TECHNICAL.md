# Ball possession from automatic tracking: technical document

**Team Nuttertools.** More than a Hack 2027, Phase 1. Match: FC Barcelona (H) vs Athletic Club (A).
Code: this repository; every command is listed in section 8. Submission file: `Nuttertools.csv`,
produced by model v5 (SHA-256 `1dcde4f6111998025c7a75f554f754e9d699dbb5c9e0330146943de759300fc6`).

## 1. Approach, and why

The task is scored on **spellF1**: whole possessions matched by temporal overlap. Two failure modes
dominate that metric. Per-frame decisions flicker and invent possessions; heavy smoothing merges
real ones. The method is therefore framed as **per-frame evidence followed by a decoder that
prices every change of state**:

1. **Parse** the tactical ATD with a fast numpy reader (bit-exact with the upstream baseline's own
   reader on all 199,319 frames).
2. **Detect each kickoff** from the tracking (section 4). Frames before it inside a graded half are
   dead ball; ungraded frames are `X`.
3. **Per-frame evidence.** 59 collective, identity-free features: relative player speed, ball
   speed and position, proximity of each team to the ball, team shape, and the **touch sequence
   around the frame**: which team is on the ball in the seconds before and after, and whether
   the ball's velocity has just changed. Most have rolling windows. A small LightGBM model maps
   them to p(H, A, D).
4. **Smart Tagging as a team prior.** Inside a tagged phase of settled possession for one team,
   the other team's score is lowered slightly on every frame (section 4). The dead-ball score is
   left to the tracking.
5. **Decode** each half with a Viterbi decoder over {H, A, D}. Switching team costs `c_switch`,
   the ball dying costs `c_die`, a restart costs `c_restart`. A change of state happens only when
   the evidence for it, summed over its frames, outweighs its cost. This is our reading of
   "controlled possession": a touch, deflection or aerial duel does not move possession; a
   sustained change does.
6. **Write** the CSV: LF line endings, exact row contract, byte-identical on every run.

Why this design rather than an end-to-end sequence model: we have no labelled data from this match
and must not create any. All learning therefore happens on other games, and **the domain gap is
the main risk, not model capacity**. A small model on robust, collective features, with the
temporal structure written explicitly into the decoder, transfers better than a large one, and
every part can be inspected.

## 2. Assets used and dropped

| asset | used? | why |
|---|---|---|
| Tactical ATD (players + ball) | **yes** | the main input: every alive/dead decision and almost every team call comes from it |
| Tactical ATD metadata XML | **yes** | team of each slot; home = first `<Team>` |
| Published half blocks and kickoff anchors (`games.py`) | **checks only** | blocks decide X vs graded; anchors only *assert* that the detected kickoff is consistent (within 2 s), never set a label |
| Smart Tagging | **yes, as a weak team prior** | the phases of settled possession (Build Up, Progression, Creating Chances, Attacking Transition) lower the other team's score inside them. The weight was set on other games, with SkillCorner's automatic phases of play standing in for the tagging (sections 5 and 6). Set pieces are not an input: a label-free check showed our dead-ball runs already precede 55 of the 57 tagged ones. The ball-location codes, Long Ball and the `N/A` team are not used |
| Homography | **no** | the ATD is already in pitch coordinates; the homography only maps to pixels, which we do not use |
| Video (tactical, broadcast) | **no** | not needed for a tracking-based method |
| Broadcast ATD | **no** | not synchronised to the tactical timeline; the tactical ATD covers the whole match |

Facts about the real tracking were measured with label-free aggregate tools (`tools/real_stats.py`,
`tools/feature_stats.py`) and drove the design:
- the ball is missing on 41% of graded frames, in about 150 long gaps (median ≈ 11 s);
- 20.5 tracked players per frame on average, with fragmented tracklets over 46 slots;
- the speed channel `s` is in m/s.

## 3. Training data (all external; nothing from the provided game)

| source | licence / terms | labelled by | size | labels (`docs/label_derivation.md`) |
|---|---|---|---|---|
| Metrica Sports open sample data, games 1–3 (github.com/metrica-sports/sample-data, pinned commit, SHA-256 checked) | no formal licence; the README asks for responsible use and acknowledgement. Acknowledged here | human event annotation by Metrica | 3 matches, about 430,000 frames at 25 fps | possession = team of each controlled action (pass, carry, shot, recovery, ball lost); challenges never change possession; dead from ball out / shot out or goal / foul / offside until the next restart (set piece or goal kick) |
| SkillCorner open data, 10 A-League 2024/25 matches (github.com/SkillCorner/opendata, pinned, LFS objects SHA-256 checked) | MIT; the README asks to credit SkillCorner. Credited here | SkillCorner's automatic pipeline | 10 matches, about 1.5 M frames after 10 → 25 fps resampling | per-frame `possession.group`; a frame without possession *outside* every phase of play is dead ball; *inside* a phase it is a loose ball and stays with the last team (definition checked before use) |

Both sources are converted into **our exact ATD text and XML format** at 25 fps, laid out like the
real file, with untracked padding and a half-time gap. The same pipeline therefore runs end to end
on them. No hand labels were made on any game; **no data, label or annotation of the provided game
was used to train, tune or select anything.** The provided match's own Smart Tagging is an input
at inference only (section 4); its weight was set on the SkillCorner games, whose phases of play
(`<id>_phases_of_play.csv`, same source and licence) were written out as a tagging file in the
real layout (`ext/tags.py`).

**Noise model** (`ext/augment.py`). The external tracking is corrupted to match the real match's
label-free aggregates:
- **ball:** long dropouts (frequency, length histogram and crowding dependence as measured);
- **players:** loss of view (so some frames have few players), individual player gaps;
- **slots:** tracklet fragmentation onto 46 slots, so a slot is not a person; duplicate tracks.

Calibration (real vs Metrica):

| | real | Metrica, corrupted |
|---|---|---|
| ball missing | 0.41 | 0.39–0.48 |
| tracked players (mean) | 20.5 | 20.7–20.9 |
| tracklets per minute | 11.5 | 11.4–12.6 |
| player step p50 (m per frame) | 0.054 | 0.054–0.060 |

**When the ball goes missing: an assumption we corrected.** Metrica blanks the ball whenever it is
dead, so "no ball" would look like a perfect dead-ball cue that exists only in that source. For
models v2 and v3 (noise condition `full`) we assumed that the real ball goes missing mostly in
live play, restored a resting ball on most dead stretches, and dropped the ball independently of
the state of play. A later label-free measurement of the whole match
(`tools/slow_play_stats.py`) showed the opposite. Ranking graded frames by mean player speed:

| ball tracked on | slowest 10% of frames | 10–25% | 25–50% | fastest 50% |
|---|---|---|---|---|
| real match | 8.6% | 29.8% | 50.7% | 82.7% |
| external games under `full` (v2, v3) | about 69% | 66% | 60% | 58% |
| external games under `full2` (v4) | 11–24% | 29–31% | 46–51% | 73–82% |

The real tracker loses the ball mostly while play is slow, which is when the ball is dead. Noise
condition `full2` reproduces this: ball gaps start with a probability calibrated per player-speed
bin, and a dead ball is restored only a quarter of the time, carried by the player taking the
restart. Model v4 is the v3 feature set retrained on `full2`. The measurement uses player speed
and ball visibility only: no label, ground truth or public-window score.

## 4. Possession rules implemented

- **Alive vs dead.** Dead ball is detected from the play itself: players slowing down, relative to
  their own speed distribution in that half (so it does not depend on a source's speed scale); the
  ball outside the pitch; the ball at rest. A dead run ends at the restart.
- **Restarts and restart team.** Whoever the evidence favours when play resumes, typically the
  team nearest the ball as it starts to move. There is no special case for throw-ins tagged `N/A`;
  none is needed because set-piece tags are not an input.
- **Smart Tagging team prior.** On frames inside a Build Up, Progression, Creating Chances or
  Attacking Transition phase of one team, the other team's log-score is lowered by 0.2 per frame
  before decoding. Frames where phases of both teams overlap are skipped, as are `N/A` teams. The
  prior is weak on purpose: a phase can span two possessions, so it only tips frames on which
  the tracking evidence is close. The dead-ball score is never changed, because the phases run
  through stoppages (we call 20% of their frames dead).
- **Loose and contested balls** stay with the last team. Contested phases carry weak, mixed
  evidence, which never repays `c_switch`.
- **Touches and deflections without control** do not change possession: a few frames of opposing
  evidence cost more than they earn.
- **Kickoffs.** Frames inside a graded half before its kickoff are `D`. The kickoff is found
  algorithmically: the end of the first kickoff line-up (each team in its own half, at least 2 s,
  tolerant of tracking dropouts), sharpened by the ball leaving the centre spot. It is asserted
  against the published half blocks and kickoff anchors. On BAR-ATH: half 1 at +0.08 s from its
  block start, half 2 at +117.04 s, 0.04 s and 0.28 s from the anchors.
- **Abstention.** `X` only on ungraded frames; never on graded ones.

## 5. Validation protocol

- **Leave-one-game-out** on the 13 external games. Each game is scored by a model trained on the
  other 12. Its corrupted copy uses a different noise seed from any training copy.
- **Metric:** the official `possession_metrics.evaluate`, imported from the upstream submodule and
  never reimplemented.
- **Tuning** (Optuna, seeded TPE, 60 trials) of the decoder costs, emission temperature, dead-ball
  bias and boundary refinement, on the out-of-fold predictions only. The frozen parameters are the
  median of the top 10 trials (`config/final_v4.yaml`). The honest estimate is a **nested 2-fold**
  (tune on half the games, score the other half, swap).
- **Four versions were frozen.** v2 (39 features) came first. An error analysis of v2 on the
  external out-of-fold predictions then showed that the loss was in changes of possession, which
  led to the touch-sequence features of v3. v4 keeps v3's features and changes only the training
  noise (section 3). v5 is v4 plus the Smart Tagging team prior. Each step was accepted on the
  external games alone, by a rule fixed beforehand. v2, v3 and v4 stay reproducible with
  `--method v2`, `--method v3` and `--method v4`.
- **Validating the tagging prior without a second tagged game.** Smart Tagging exists only for
  the provided match. SkillCorner's automatic phases of play (build-up, create, finish,
  transition, each with a team) are the same kind of stream, so they stand in for it on the 10
  SkillCorner games. They share a possession model with our SkillCorner labels, which makes them
  cleaner than an independent tagger, so they are degraded first: boundaries moved by about 1 s,
  22% of phases dropped, and up to 30% of phases given to the wrong team. The weight follows a
  rule fixed before the real run: no mean loss in any of those scenarios, then the best worst
  case over the plausible ones (`ext.tags.choose_weight`). The stand-in is only used for the team
  call: our SkillCorner dead-ball labels were derived from the same phases, so a test of alive
  against dead would be circular, and the prior leaves that decision alone.
- **A structural check on the decoder settings**, added with v4. The first v4 search returned
  settings in which a direct change of team cost more than the ball dying and restarting. Such a
  decoder never switches team directly; it inserts a sub-second dead ball at every change (106
  dead runs under 1 s per external game against 3 in the truth). spellF1 cannot see this, so the
  search could not either. Settings of that kind are now excluded from the search
  (`tune.no_bridge`), and `tools/report.py` raises an alarm if more than 10% of dead-ball runs
  are under 1 s. The frozen v2 and v3 settings already satisfied the rule.
- **The provided match was never used to choose anything.** Only label-free aggregates of its
  tracking were used to calibrate the noise model, plus sanity alarms on our output (possessions
  per half, durations, dead-ball share, Barcelona's share), fixed in advance and never tuned on.
- **Public 5-minute window:** used as a correctness check and, as CHALLENGE.md invites, to
  understand what is expected. It was looked at **seven times**: twice on the frozen v2 file
  (the scoreboard, then its frame-level breakdown), once on the frozen v3 file, once per segment
  on the v3 file, once on the v4 file (scoreboard and per segment), once by hand on one stoppage
  of the v5 file, and once on the submitted v5 file as a final check. Each scored file had been
  chosen on the external games, frozen, reproduced and, for v4 and v5, packaged for upload
  before its look, and no look changed which file was submitted. One earlier run scored only the
  upstream baselines, to test the tool. See `docs/public_window_log.md`.
- **What the window did and did not influence.** No parameter, threshold or model choice was ever
  fitted to it or selected with it: v3 was chosen over v2, and v4 over v3, on the external games
  alone. The fourth look did prompt a question. Most of our dead-ball misses in the window came
  from one stoppage, which made us doubt our assumption about when the real ball goes untracked.
  We then answered that question without the window and without labels, by measuring ball
  visibility against player speed over the whole match (section 3). That measurement, not the
  window, defines the corrected noise model, and the external games decided whether to adopt it.
  The fifth look reported that 7.6% of the window's frames carried the wrong team. The decision
  to add the tagging prior did not rest on that number: it followed from a label-free comparison
  of our labels with the tags over the whole match, and from the SkillCorner test.

## 6. Results

External games, official spellF1, mean over 13 games. Learned variants are leave-one-game-out.
First, on the corrected noise `full2`, the comparison that decided v4:

| method, scored on `full2` | all 13 (nested estimate) | Metrica (3) | SkillCorner (10) |
|---|---|---|---|
| **v4: v3's features, trained on `full2`** | **0.439** | 0.604 | 0.389 |
| v3's recipe, trained on `full` | 0.419 | 0.556 | 0.378 |
| v4 + 9 ball-gap and slow-down features | 0.437 | 0.606 | 0.386 |

v4 is better than v3's recipe on 10 of 13 games (mean +0.020, paired t = 3.1; the three losses
are under 0.01). The extra features, which encode a ball gap and a sudden drop in player speed
directly, add nothing once the training data has the right ball visibility (better on 6 of 13),
so they are not shipped.

**The Smart Tagging team prior (v5)**, on the 10 SkillCorner games with their phases of play as
the stand-in stream, v4's settings unchanged (0.388 without tags):

| quality of the stand-in tags | they name the true team on | we agree with them on | spellF1 gain at the frozen weight 0.2 | at 0.4 | at 0.8 |
|---|---|---|---|---|---|
| as delivered | 87.1% | 91.9% | **+0.009** (10 of 10 games) | +0.014 | +0.020 |
| boundaries moved by 1 s, 22% of phases dropped | 87.0% | 91.7% | **+0.007** (8 of 10) | +0.010 | +0.013 |
| and 10% of phases given to the wrong team | 80.4% | 84.4% | **+0.003** (6 of 10) | +0.003 | +0.001 |
| and 20% | 72.1% | 75.4% | **+0.005** (8 of 10) | −0.001 | −0.011 |
| and 30% | 64.0% | 65.5% | **+0.001** (6 of 10) | −0.001 | −0.010 |

The weight 0.2 is the largest that never loses on average, whatever the tag quality. On the real
match the tagging evidence covers 54.5% of the frames we call alive and we agree with it on 88.3%
of those (label-free, `tools/tagging_stats.py`), which places the real stream between the second
and third rows. The gain is real but small, and the prior does not recover missed possessions:
the predicted possession count does not rise. The stand-in also differs from the real stream
(its phases are about half as long), so these numbers are a guide, not a forecast.

Then the earlier results, on the older noise `full`:

| method, scored on `full` | all 13 | Metrica (3) | SkillCorner (10) |
|---|---|---|---|
| v3 = v2 + touch-sequence features, tuned decoder (nested estimate) | 0.417 | 0.52–0.56 per game | 0.34–0.44 per game |
| v3 + change-point model driving the decoder's costs (three searches) | 0.413–0.429 | | |
| v2 (39 features), tuned decoder (nested estimate) | 0.393 | 0.45–0.49 per game | 0.33–0.43 per game |
| v2, M4 decoder settings | 0.395 | 0.473 | 0.371 |
| v2 with boundary refinement | 0.388 | 0.469 | 0.364 |
| v2 without team-shape features | 0.368 | 0.447 | 0.344 |
| v2 with absolute instead of relative speed | 0.389 | 0.474 | 0.364 |
| hand-written rules + the same decoder | 0.301 | 0.381 | 0.277 |
| upstream baseline, ported (nearest player, ±2 s vote) | 0.189 | 0.229 | 0.176 |

The learned model beats the rules on all 13 games. Team-shape features carry the team call while
the ball is unknown. The touch-sequence features of v3 improve on v2 on 11 of 13 games at equal
decoder settings. A change-point model that drives the decoder's transition costs was also tried;
its gain was not stable across searches, so it is not shipped (`docs/ablation.md`).
Under the realistic noise, the ported baseline scores 0.18–0.23, close to
its published full-match score on the real game (0.243). The noise model was never tuned to that
number, so the closeness is independent evidence that the noise model is realistic.

**Real match: sanity alarms only** (no labels; ranges fixed before the check). Submitted file,
v5, with the earlier v4 and v3 files for comparison (half 1 / half 2):

| alarm | v5 (submitted) | v4 | v3 | range | |
|---|---|---|---|---|---|
| possessions | 134 / 123 | 134 / 127 | 153 / 144 | 100–200 | pass |
| mean possession | 14.0 s / 13.9 s | 14.1 s / 13.5 s | 13.2 s / 13.4 s | 5–25 s | pass |
| dead-ball share | 41.1% / 46.0% | 40.8% / 45.9% | 36.4% / 39.4% | 25–50% | pass |
| Barcelona share of ball-in-play | 62.4% / 59.7% | 62.5% / 59.0% | 61.3% / 58.8% | 45–75% | pass |
| possessions under 0.5 s; X on graded frames | 0; 0 | 0; 0 | 0; 0 | ≤5%; 0 | pass |
| dead-ball runs under 1 s (alarm added with v4) | 1 of 57 / 0 of 60 | 1 of 55 / 0 of 60 | | ≤10% | pass |

**v5 against v4 on the real match, without ground truth:** the same label on 99.1% of the 158,857
graded frames, the same alive/dead call on 99.8%, the same team on 98.8% of the frames both call
alive, `X` on the same frames, 257 possessions against 261. The prior moves 1,113 frames from
one team to the other and 349 frames from alive to dead; 5 go the other way. A small, targeted
change, as intended for a weak prior.

**v4 against v3 on the real match, without ground truth** (`tools/compare_submissions.py`): the two
files give the same label on 92.1% of the 158,857 graded frames, the same alive/dead call on 93.8%,
and the same team on 97.0% of the frames both call alive. `X` sits on exactly the same frames and
both kickoffs are detected at the same frames. The difference is almost entirely dead ball: v4
calls 9,219 frames dead that v3 called alive, and only 584 the other way. That is the intended
effect of the corrected noise model: stoppages in which the ball is untracked or carried are now
recognised. With more dead ball, v4 reports 261 possessions against v3's 297. (v3 against v2
agreed on 91.8% of graded frames.)

**Smart Tagging set pieces as an independent check of the dead ball** (`tools/tagging_stats.py`,
run on the tag-free v4 file; no ground truth). Every tagged set piece is a restart, so a dead
ball must precede it. Of the 57 tagged set pieces on graded frames (throw-ins, corners, goal
kicks, free kicks, kickoffs after a goal), the file has a dead ball before 55, and its own
restart falls inside the tagged interval for 51. The two without a dead ball are throw-ins. The
tags cover about half of our stoppages (58 of 112 dead-ball runs end at a tagged set piece);
offsides and most fouls are not tagged. Two automatic sources that share no code agree this
closely on alive against dead, so the set-piece tags were not made an input: forcing the last two
would move a handful of frames. Inside the tagged possession phases the same check showed less
agreement on the team (Build Up 92%, Progression 90%, Creating Chances 85%, Attacking Transition
68% of the frames we call alive), which is where the v5 prior acts.

**Public window (correctness check, each file checked after it was frozen):**

| on the public 5 minutes (7,500 frames, 17 possessions) | macroF1 | spellF1 | frame accuracy | alive/dead F1 | team accuracy (both alive) | possessions |
|---|---|---|---|---|---|---|
| **submitted file, v5** | **0.793** | **0.588** | 0.823 | 0.945 | 0.897 | 17 of mean 15.8 s |
| earlier file, v4 | 0.798 | 0.587 | 0.830 | 0.945 | 0.905 | 13 of mean 20.7 s |
| earlier file, v3 | 0.793 | 0.563 | 0.826 | 0.942 | 0.905 | 15 of mean 17.8 s |
| earlier file, v2 | 0.783 | 0.513 | 0.814 | 0.941 | 0.892 | 15 of mean 17.8 s |
| ground truth | | | | | | 17 of mean 14.2 s |
| best upstream baseline on this window | 0.621 | 0.424 | | | | |

The check shows that frames are aligned, the file is well-formed and the labels mean what we
intend. With only 17 possessions it cannot rank or tune anything: one possession more or less
moves spellF1 by about 0.06, more than the difference between any two of our rows. No setting was
changed after any look, and each version had been chosen on the external games and frozen before
its check. The v5 and v4 rows show exactly that: v5 is a hair higher on spellF1 and a hair lower
on frame accuracy and team accuracy, and neither difference means anything on 17 possessions.
v5 was kept because it had been chosen beforehand, not because of this row. Its dead-ball calls
in the window are identical to v4's (recall 0.52, precision 1.00); the tagging prior changes
team calls only. It reports 17 possessions, the true count, of which 11 match at IoU 0.5.

What the v4 file gets wrong on this window, by frames (the per-segment study was not repeated
for v5): 83.0% correct, 9.3% dead ball called
alive, 7.6% wrong team, and 3 frames alive called dead.
- **Short possessions.** 10 of the 17 possessions are matched. Five of the seven missed ones last
  under 4 s (0.8 s to 3.6 s). This is the under-segmentation measured on external games
  (section 7).
- **Dead ball** is the weakest class for both v3 and v4 (recall 0.53 and 0.52; precision 1.00).
  Of the four stoppages, two are found (83% and 92% of their frames). A 5 s one is played
  through. In a 20 s one the ball is tracked and moving throughout, among slow players, and we
  keep a team in possession for most of it. The extra dead ball that v4 finds on the whole match
  (comparison with v3 above) lies outside this window.
- **Boundary timing.** Only 5 of the 20 changes of state are placed within 1 s.

## 7. Limitations and next steps

- **Under-segmentation.** On external games the model reports 0.6–0.7× the true number of
  possessions. It misses quick turnovers: possessions under 3 s are a third of the true ones and
  are rarely matched. Tuning the decoder cannot fix this: cheaper switches add false switches
  faster than they recover real ones. The touch-sequence features of v3 place boundaries better
  but do not recover most short possessions. About 40% of frames have no ball, and a change of
  control inside such a gap cannot be located from the tracking.
- **Proxy labels.** External labels approximate, not reproduce, the competition's definition (which
  comes from corrected tracking). SkillCorner's labels are automatic and its tracking is broadcast
  (extrapolated off camera). External scores are a relative guide, not a forecast.
- **Smart Tagging is used cautiously.** The team prior was validated on another vendor's phases
  of play, not on Metrica's own tags, so its weight is deliberately small. With Smart Tagging on
  other games it could carry more weight and also inform restarts.
- **Dead ball with a tracked, moving ball** (a ball carried or kicked back during a stoppage) is
  the weakest dead-ball case. A detector of sudden collective slowdowns with a penalty-area
  exclusion (`src/mtah/deadball.py`) was built and tested on the external games; it lowered
  spellF1 (0.439 → 0.435 at its mildest setting) and is not shipped. Possession-balanced
  training, aimed at the short possessions, did not help either. Details in `docs/ablation.md`.
- **Kickoff timing** is a few frames off (half 2: about 7 frames early), and boundaries are not
  refined for the learned model because snapping hurt it on external games.
- **Noise model gaps.** It does not reproduce the real 2.4% of frames with 23 or more tracks, and
  its tracklets are about twice as long as the real ones.

## 8. Reproduction

Windows Command Prompt, Python 3.11, with the official data bundle unpacked anywhere:

```
git clone --recurse-submodules <repository URL> mtah-possession
cd mtah-possession
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tasks.py submission --data-dir C:\path\to\mtah-data --expect-sha256 recorded
.venv\Scripts\python.exe tasks.py validate --expect-sha256 recorded
```

The output is `outputs\real\submission.csv`, in about 20–50 s. `--expect-sha256 recorded` fails
unless the file equals the frozen one (`config/submission.sha256`). Byte-exact reproduction from
a fresh clone with a fresh virtual environment was verified for v2 on the real data, and for every
version on a full-length synthetic match. The v3, v4 and v5 files were each regenerated
byte-identically on the real data in a second run. The inputs are the bundle's tactical ATD
text, its metadata XML and its Smart Tagging XML. Everything else that determines the bytes is
pinned:
- all dependencies (`requirements.txt`, full closure);
- the upstream submodule commit;
- the model file, stored byte-exact;
- the frozen config;
- single-threaded LightGBM inference.

The earlier frozen versions are reproduced with `tasks.py submission --method v4 --expect-sha256
recorded-v4`, and likewise `v3` / `recorded-v3` and `v2` / `recorded-v2` (add `--out` to keep the
files apart).

Rebuilding the model from external data only (each step is deterministic):

```
.venv\Scripts\python.exe tasks.py external --prepare
.venv\Scripts\python.exe -m ext.train cache --condition full2
.venv\Scripts\python.exe -m ext.train logo --variant touch_f2
.venv\Scripts\python.exe -m ext.train final-v4
.venv\Scripts\python.exe tasks.py tune --variant touch_f2 --freeze
.venv\Scripts\python.exe -m ext.tags write
.venv\Scripts\python.exe -m ext.tags experiment
.venv\Scripts\python.exe -m ext.tags freeze
```

The last three write the stand-in tagging files, run the tagging test and freeze the weight in
`config/final_v5.yaml`.
(v3: `cache`, `logo --variant touch`, `final-v3 --variant touch`, `tune --variant touch --freeze`.)

`python tasks.py test` runs the unit and end-to-end tests (83) on synthetic matches in the exact
ATD format.

**Development note.** The code was written with an AI coding assistant. In line with the terms,
the assistant never had access to the match data, video or any extract of them. A team member ran
every command on the real data locally and shared only aggregate numbers (counts, shares,
percentiles), as recorded in this document.
