# Ball possession from automatic tracking: technical document

**Team Nuttertools.** More than a Hack 2027, Phase 1. Match: FC Barcelona (H) vs Athletic Club (A).
Code: this repository; every command is listed in section 8. Submission file: `Nuttertools.csv`
(SHA-256 `6ad9d4addca4cc8da84292254678682215e433888e534f7eb2f016d7f46d92e1`).

## 1. Approach, and why

The task is scored on **spellF1**: whole possessions matched by temporal overlap. Two failure modes
dominate that metric. Per-frame decisions flicker and invent possessions; heavy smoothing merges
real ones. The method is therefore framed as **per-frame evidence followed by a decoder that
prices every change of state**:

1. **Parse** the tactical ATD with a fast numpy reader (bit-exact with the upstream baseline's own
   reader on all 199,319 frames).
2. **Detect each kickoff** from the tracking (section 4). Frames before it inside a graded half are
   dead ball; ungraded frames are `X`.
3. **Per-frame evidence.** 39 collective, identity-free features: relative player speed, ball
   speed and position, proximity of each team to the ball, last touch, team shape, all with rolling
   windows. A small LightGBM model maps them to p(H, A, D).
4. **Decode** each half with a Viterbi decoder over {H, A, D}. Switching team costs `c_switch`,
   the ball dying costs `c_die`, a restart costs `c_restart`. A change of state happens only when
   the evidence for it, summed over its frames, outweighs its cost. This is our reading of
   "controlled possession": a touch, deflection or aerial duel does not move possession; a
   sustained change does.
5. **Write** the CSV: LF line endings, exact row contract, byte-identical on every run.

Why this design rather than an end-to-end sequence model: we have no labelled data from this match
and must not create any. All learning therefore happens on other games, and **the domain gap is
the main risk, not model capacity**. A small model on robust, collective features, with the
temporal structure written explicitly into the decoder, transfers better than a large one, and
every part can be inspected.

## 2. Assets used and dropped

| asset | used? | why |
|---|---|---|
| Tactical ATD (players + ball) | **yes** | the only input to the labels |
| Tactical ATD metadata XML | **yes** | team of each slot; home = first `<Team>` |
| Published half blocks and kickoff anchors (`games.py`) | **checks only** | blocks decide X vs graded; anchors only *assert* that the detected kickoff is consistent (within 2 s), never set a label |
| Smart Tagging | **no** | no external dataset has an independent automatic tagging stream to train or validate its weight on. SkillCorner's phases of play are the closest, but our SkillCorner labels derive from them, so any test would be circular. An unvalidated prior was judged riskier than none |
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
was used to train, tune or select anything.**

**Noise model** (`ext/augment.py`, condition `full`). The external tracking is corrupted to match the
real match's label-free aggregates:
- **ball:** long dropouts (frequency, length histogram and crowding dependence as measured);
- **players:** loss of view (so some frames have few players), individual player gaps;
- **slots:** tracklet fragmentation onto 46 slots, so a slot is not a person; duplicate tracks.

Calibration (real vs Metrica under `full`):

| | real | Metrica with `full` |
|---|---|---|
| ball missing | 0.41 | 0.39–0.48 |
| tracked players (mean) | 20.5 | 20.7–20.9 |
| tracklets per minute | 11.5 | 11.4–12.6 |
| player step p50 (m per frame) | 0.054 | 0.054–0.060 |

One artefact needed care. **Metrica blanks the ball whenever it is dead**, so "no ball" would
look like a perfect dead-ball cue. On the real match the ball goes missing mostly in live play.
The converter therefore restores a resting ball at the restart spot from the event coordinates,
and no feature encodes missingness directly. After this, P(missing | dead) is 0.34 and P(missing |
alive) is 0.43, against an estimated 0.34 / 0.45 on the real match.

## 4. Possession rules implemented

- **Alive vs dead.** Dead ball is detected from the play itself: players slowing down, relative to
  their own speed distribution in that half (so it does not depend on a source's speed scale); the
  ball outside the pitch; the ball at rest. A dead run ends at the restart.
- **Restarts and restart team.** Whoever the evidence favours when play resumes, typically the
  team nearest the ball as it starts to move. There is no special case for throw-ins tagged `N/A`;
  none is needed because Smart Tagging is not used.
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
  median of the top 10 trials (`config/final.yaml`). The honest estimate is a **nested 2-fold**
  (tune on half the games, score the other half, swap).
- **The provided match was never used to choose anything.** Only label-free aggregates of its
  tracking were used to calibrate the noise model, plus sanity alarms on our output (possessions
  per half, durations, dead-ball share, Barcelona's share), fixed in advance and never tuned on.
- **Public 5-minute window:** used as a correctness check only, after everything was frozen, and
  looked at **once** (one earlier run scored only the upstream baselines, to test the tool). See
  section 6 and `docs/public_window_log.md`.

## 6. Results

External games, official spellF1, mean over 13 games under `full` noise. Learned variants are
leave-one-game-out.

| method | all 13 | Metrica (3) | SkillCorner (10) |
|---|---|---|---|
| **shipped: LightGBM v2 + tuned decoder (nested estimate)** | **0.393** | 0.47–0.49 per game | 0.33–0.43 per game |
| v2, M4 decoder settings | 0.395 | 0.473 | 0.371 |
| v2 with boundary refinement | 0.388 | 0.469 | 0.364 |
| v2 without team-shape features | 0.368 | 0.447 | 0.344 |
| v2 with absolute instead of relative speed | 0.389 | 0.474 | 0.364 |
| hand-written rules + the same decoder | 0.301 | 0.381 | 0.277 |
| upstream baseline, ported (nearest player, ±2 s vote) | 0.189 | 0.229 | 0.176 |

The learned model beats the rules on all 13 games. Team-shape features carry the team call while
the ball is unknown. Under the realistic noise, the ported baseline scores 0.18–0.23, close to
its published full-match score on the real game (0.243). The noise model was never tuned to that
number, so the closeness is independent evidence that the noise model is realistic.

**Real match: sanity alarms only** (no labels; ranges fixed before any check):

| alarm | half 1 | half 2 | range | |
|---|---|---|---|---|
| possessions | 152 | 151 | 100–200 | pass |
| mean possession | 13.8 s | 12.8 s | 5–25 s | pass |
| dead-ball share | 33.8% | 39.1% | 25–50% | pass |
| Barcelona share of ball-in-play | 59.9% | 58.8% | 45–75% | pass |
| possessions under 0.5 s; X on graded frames | 0; 0 | 0; 0 | ≤5%; 0 | pass |

**Public window (correctness check, looked at once, after freezing):** the frozen submission scores
macroF1 **0.783** and spellF1 **0.513** on the public 5 minutes. It reports 15 possessions of mean
17.8 s, against the truth's 17 of mean 14.2 s. For reference, the upstream baselines score
0.58–0.62 macroF1 and up to 0.424 spellF1 on the same window. The check shows that frames are
aligned, the file is well-formed and the labels mean what we intend. With only 17 possessions, it
cannot rank or tune anything, and nothing was changed after it. The slightly low possession count
is the under-segmentation already measured on external games (section 7).

## 7. Limitations and next steps

- **Under-segmentation.** On external games the model reports 0.6–0.7× the true number of
  possessions. It misses quick turnovers. Tuning the decoder cannot fix this: cheaper switches add
  false switches faster than they recover real ones. The next step is sharper evidence at changes
  of control: touch and reception events, ball direction changes near a player.
- **Proxy labels.** External labels approximate, not reproduce, the competition's definition (which
  comes from corrected tracking). SkillCorner's labels are automatic and its tracking is broadcast
  (extrapolated off camera). External scores are a relative guide, not a forecast.
- **Smart Tagging unused**, for lack of a fair external validation. With an independent tagging
  source on other games it could become a prior on restarts and team.
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

The output is `outputs\real\submission.csv`, in about 15–40 s. `--expect-sha256 recorded` fails
unless the file equals the frozen one (`config/submission.sha256`, `6ad9d4ad…`). This was verified
on a fresh clone with a fresh virtual environment. Everything that determines the bytes is pinned:
- all dependencies (`requirements.txt`, full closure);
- the upstream submodule commit;
- the model file, stored byte-exact;
- the frozen config;
- single-threaded LightGBM inference.

Rebuilding the model from external data only (each step is deterministic):

```
.venv\Scripts\python.exe tasks.py external --prepare
.venv\Scripts\python.exe -m ext.train cache
.venv\Scripts\python.exe -m ext.train logo --variant all
.venv\Scripts\python.exe -m ext.train final --variant all --c-switch 15 --c-dead 25 --temperature 1.0
.venv\Scripts\python.exe tasks.py tune
```

`python tasks.py test` runs 57 unit and end-to-end tests on synthetic matches in the exact ATD
format.

**Development note.** The code was written with an AI coding assistant. In line with the terms,
the assistant never had access to the match data, video or any extract of them. A team member ran
every command on the real data locally and shared only aggregate numbers (counts, shares,
percentiles), as recorded in this document.
