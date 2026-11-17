# Ball possession from tracking: Nuttertools

More than a Hack 2027, Phase 1. FC Barcelona (H) vs Athletic Club (A).

Submitted file: `Nuttertools.csv`, made by our model v5. SHA-256
`1dcde4f6111998025c7a75f554f754e9d699dbb5c9e0330146943de759300fc6`. Section 8 has the commands
to rebuild it.

## 1. What we did and why

spellF1 rewards getting whole possessions right, start to end. Two things kill it: labels that
flicker frame to frame (lots of fake little possessions), and smoothing so heavy that real
possessions get merged. So we split the problem in two. First we work out, frame by frame, how
likely H, A and dead ball are. Then a decoder picks the most likely sequence, and every change of
state has a price.

The pipeline:

1. Read the tactical ATD (our reader matches the upstream baseline's on all 199,319 frames).
2. Find each kickoff from the tracking. Graded frames before it are dead ball; ungraded frames
   are `X`.
3. Compute 59 features per frame. All are collective (no player identities, since the 46 slots
   are tracklets, not people): player speeds relative to the rest of the half, where the ball is
   and how fast it moves, which team is closest to it, team shape, and which team was on the ball
   in the seconds before and after. A small LightGBM model turns them into p(H, A, D).
4. Nudge the team call with the Smart Tagging: inside a tagged possession phase for one team, the
   other team gets a small penalty on every frame.
5. Decode each half with Viterbi over {H, A, D}. Switching team costs `c_switch`, the ball going
   dead costs `c_die`, a restart costs `c_restart`. A change only happens when the evidence over
   its frames pays for it. That is basically our version of "controlled possession": one touch
   or a header in a duel doesn't move possession, a sustained change does.
6. Write the CSV with LF endings. Same bytes every run.

Why not a big sequence model? We have no labels for this match and aren't allowed to make any, so
everything is learned on other games. The real risk is the gap between those games and this one,
not lack of model capacity. Small model, sturdy features, and the time structure written into the
decoder where we can see it.

## 2. What we used from the data bundle

| asset | used? | notes |
|---|---|---|
| Tactical ATD (players + ball) | yes | main input, almost every label comes from it |
| ATD metadata XML | yes | which team each slot belongs to; first `<Team>` is home |
| Half blocks and kickoff anchors (`games.py`) | as checks | blocks decide X vs graded. Anchors only confirm our detected kickoff is within 2 s |
| Smart Tagging | yes, lightly | possession phases (Build Up, Progression, Creating Chances, Attacking Transition) give a small team prior. Set pieces, location codes, Long Ball and `N/A` teams aren't used |
| Homography | no | pitch to pixels; we never touch pixels |
| Video | no | |
| Broadcast ATD | no | different timeline, and the tactical ATD already covers everything |

A few facts about the real tracking that shaped things (measured with label-free tools,
`tools/real_stats.py` and `tools/feature_stats.py`): the ball is missing on 41% of graded frames,
mostly in long gaps (median around 11 s); about 20.5 players are tracked per frame, spread over 46
fragmented slots; and the speed channel is in m/s, not km/h.

## 3. Training data

Everything we trained or tuned on is from other matches.

| source | licence | labelled by | size | how we got H/A/D |
|---|---|---|---|---|
| Metrica open sample data, games 1–3 (github.com/metrica-sports/sample-data, pinned) | no formal licence, asks for acknowledgement (done here) | Metrica, human event data | 3 matches, ~430k frames at 25 fps | team of each controlled action; challenges don't change possession; dead from out / goal / foul / offside until the restart |
| SkillCorner open data, 10 A-League 2024/25 matches (github.com/SkillCorner/opendata, pinned) | MIT, asks for credit (done here) | SkillCorner's automatic pipeline | 10 matches, ~1.5M frames after 10→25 fps | their per-frame `possession.group`; no possession outside any phase of play = dead; inside a phase = loose ball, stays with the last team |

Details in `docs/label_derivation.md`. Both sources get converted into exactly the same ATD text +
XML format as the real file, so the whole pipeline runs on them unchanged. Nobody hand-labelled
anything, and nothing from the provided match (data, labels, tags) went into training or tuning.
The match's own Smart Tagging is only read at inference time; its weight was set on the
SkillCorner games (section 5).

**Noise.** Clean external tracking looks nothing like the real ATD, so we corrupt it
(`ext/augment.py`) to match the real match's label-free numbers: long ball dropouts, players
dropping out of view, tracklets chopped and shuffled across 46 slots, duplicate tracks.

| | real | Metrica after noise |
|---|---|---|
| ball missing | 0.41 | 0.39–0.48 |
| tracked players (mean) | 20.5 | 20.7–20.9 |
| tracklets per minute | 11.5 | 11.4–12.6 |
| player step, median (m/frame) | 0.054 | 0.054–0.060 |

**One assumption we got wrong.** Metrica blanks the ball whenever it's dead, which would make "no
ball" a perfect (and fake) dead-ball signal. For v2 and v3 we assumed the real ball mostly goes
missing during live play, so we dropped it at random. Later we measured the whole real match
(`tools/slow_play_stats.py`, player speed and ball visibility only, no labels) and it's the
other way round:

| ball tracked on | slowest 10% of frames | 10–25% | 25–50% | fastest 50% |
|---|---|---|---|---|
| real match | 8.6% | 29.8% | 50.7% | 82.7% |
| external, old noise (v2, v3) | ~69% | 66% | 60% | 58% |
| external, new noise (v4, v5) | 11–24% | 29–31% | 46–51% | 73–82% |

So the real tracker loses the ball mostly when play is slow, i.e. stoppages. The new noise
(`full2`) copies that, and v4 is the same features retrained on it.

## 4. Possession rules

- **Alive vs dead** comes from the play: players slowing down (relative to their own speeds in
  that half, so it works across data sources), ball out of the pitch, ball at rest. A dead
  stretch ends at the restart.
- **Restart team**: whoever the evidence points to when play starts again, usually the team
  nearest the ball. No special case for throw-ins tagged `N/A`.
- **Smart Tagging prior**: inside a possession phase of one team, the other team's log-score
  drops by 0.2 per frame. Overlapping phases of both teams are skipped. It's deliberately weak,
  since a phase can cover two possessions; it only tips frames where the tracking is unsure. It
  never touches the dead-ball score (phases run straight through stoppages).
- **Loose and contested balls** stay with the last team. Mixed evidence in a duel never pays for a
  switch.
- **Touches and deflections** without control don't change possession, for the same reason.
- **Kickoffs**: found as the end of the kickoff line-up (both teams in their own half for at least
  2 s), sharpened by the ball leaving the centre spot, then checked against the published anchors.
  Half 1: +0.08 s after its block start. Half 2: +117.04 s. Within 0.04 s and 0.28 s of the
  anchors.
- **X** only on ungraded frames. Never on graded ones.

## 5. How we validated

- Leave-one-game-out over the 13 external games: each game is scored by a model trained on the
  other 12, on a noise copy it never saw.
- Scoring always uses the official `possession_metrics.evaluate` from the upstream submodule. We
  never reimplemented it.
- Decoder costs, temperature and dead-ball bias were tuned with Optuna (seeded TPE, 60 trials) on
  the out-of-fold predictions, and we froze the median of the top 10 trials. The number we trust
  is a nested 2-fold estimate (tune on half the games, score the other half, swap).
- Every new version had to win clearly on most external games, by a rule we fixed before looking.
  v2 → v3 added touch-sequence features (the error analysis said we were losing at changes of
  possession). v3 → v4 only changed the training noise. v4 → v5 added the tagging prior. Older
  versions still rebuild with `--method v2/v3/v4`.
- **Testing the tagging prior with no second tagged game.** Only this match has Smart Tagging.
  SkillCorner's phases of play are the same kind of thing (build-up, create, finish, transition,
  each with a team), so they stood in for it. They're cleaner than a real independent tagger, so
  we made them worse first: boundaries shifted ~1 s, 22% of phases dropped, up to 30% given to
  the wrong team. We picked the biggest weight that never lost on average in any of those
  scenarios. We only used this for the team call; for alive/dead it would be circular, because
  our SkillCorner dead-ball labels come from the same phases.
- **A trap we caught.** The first v4 search found settings where a direct team switch cost more
  than dying + restarting. The decoder then never switched directly and slipped a tiny dead ball
  into every change (106 sub-second dead runs per game vs 3 in the truth). spellF1 can't see
  this, so the search couldn't either. Those settings are now excluded (`tune.no_bridge`), and
  `tools/report.py` flags it if more than 10% of dead runs are under 1 s.
- **The real match never picked anything.** We only used label-free aggregates of its tracking
  (to calibrate the noise) and sanity alarms on our output, with ranges fixed in advance.
- **Public 5-minute window.** We looked at it seven times, all logged in
  `docs/public_window_log.md`: twice for v2, twice for v3 (scoreboard, then per segment), once
  for v4, once by hand on one stoppage in v5, and once on the final v5 file. Each file was frozen
  before its look and no look changed what we submitted. One honest caveat: the fourth look made
  us suspicious about when the real ball goes missing. We answered that with the whole-match
  measurement in section 3, not with the window, and the external games decided whether to adopt
  it.

## 6. Results

All external numbers are mean spellF1 over the 13 games, leave-one-game-out.

On the corrected noise, which is what decided v4:

| | all 13 (nested) | Metrica (3) | SkillCorner (10) |
|---|---|---|---|
| **v4** (v3 features, new noise) | **0.439** | 0.604 | 0.389 |
| v3 recipe (old noise) | 0.419 | 0.556 | 0.378 |
| v4 + 9 extra ball-gap / slowdown features | 0.437 | 0.606 | 0.386 |

v4 beats v3's recipe on 10 of 13 games (+0.020, paired t = 3.1). The extra features didn't add
anything once the noise was right, so we left them out.

Tagging prior (v5), on the 10 SkillCorner games, v4 settings unchanged, 0.388 without tags:

| stand-in tag quality | tags name the true team | gain at our weight 0.2 | at 0.4 | at 0.8 |
|---|---|---|---|---|
| as delivered | 87.1% | +0.009 (10/10 games) | +0.014 | +0.020 |
| shifted 1 s, 22% dropped | 87.0% | +0.007 (8/10) | +0.010 | +0.013 |
| + 10% wrong team | 80.4% | +0.003 (6/10) | +0.003 | +0.001 |
| + 20% wrong team | 72.1% | +0.005 (8/10) | −0.001 | −0.011 |
| + 30% wrong team | 64.0% | +0.001 (6/10) | −0.001 | −0.010 |

Bigger weights win more when the tags are good and lose when they're bad; 0.2 never lost. On the
real match the tags cover 54.5% of the frames we call alive and agree with us on 88.3% of them,
which puts the real stream somewhere around the second or third row. So: a small gain, and it
doesn't find the possessions we miss.

Older results, on the old noise:

| | all 13 | Metrica | SkillCorner |
|---|---|---|---|
| v3 (nested) | 0.417 | 0.52–0.56 | 0.34–0.44 |
| v3 + change-point model (three searches) | 0.413–0.429 | | |
| v2 (nested) | 0.393 | 0.45–0.49 | 0.33–0.43 |
| v2 without team shape | 0.368 | 0.447 | 0.344 |
| hand-written rules + same decoder | 0.301 | 0.381 | 0.277 |
| upstream baseline, ported | 0.189 | 0.229 | 0.176 |

The change-point model wasn't stable across searches, so it's out. Side note: under our noise the
ported baseline scores 0.18–0.23, close to its published 0.243 on the real game. We never tuned
for that, so it's a decent sign the noise is realistic.

**Real match, sanity alarms only** (no labels, ranges set before we looked), half 1 / half 2:

| | v5 (submitted) | v4 | v3 | range |
|---|---|---|---|---|
| possessions | 134 / 123 | 134 / 127 | 153 / 144 | 100–200 |
| mean possession | 14.0 / 13.9 s | 14.1 / 13.5 s | 13.2 / 13.4 s | 5–25 s |
| dead-ball share | 41.1 / 46.0% | 40.8 / 45.9% | 36.4 / 39.4% | 25–50% |
| Barcelona share of ball in play | 62.4 / 59.7% | 62.5 / 59.0% | 61.3 / 58.8% | 45–75% |
| possessions < 0.5 s, X on graded | 0, 0 | 0, 0 | 0, 0 | ≤5%, 0 |
| dead runs < 1 s | 1 of 57 / 0 of 60 | 1 of 55 / 0 of 60 | | ≤10% |

All pass. v5 vs v4 on the real match: same label on 99.1% of graded frames, same alive/dead on
99.8%. v4 vs v3: same label on 92.1%, and nearly all the difference is extra dead ball in v4
(9,219 frames), which is what the noise fix was supposed to do.

**Smart Tagging as a cross-check.** Every tagged set piece is a restart, so there should be a dead
ball just before it. Our file has one before 55 of the 57 tagged set pieces (the two misses are
throw-ins). The tags only cover about half our stoppages, since offsides and most fouls aren't
tagged. Two automatic systems that share no code agreeing that well was enough for us to leave
set pieces out of the model.

**Public window** (correctness check only, each file already frozen):

| 7,500 frames, 17 possessions | macroF1 | spellF1 | frame acc. | alive/dead F1 | team acc. | possessions |
|---|---|---|---|---|---|---|
| **v5 (submitted)** | **0.793** | **0.588** | 0.823 | 0.945 | 0.897 | 17, mean 15.8 s |
| v4 | 0.798 | 0.587 | 0.830 | 0.945 | 0.905 | 13, mean 20.7 s |
| v3 | 0.793 | 0.563 | 0.826 | 0.942 | 0.905 | 15, mean 17.8 s |
| v2 | 0.783 | 0.513 | 0.814 | 0.941 | 0.892 | 15, mean 17.8 s |
| ground truth | | | | | | 17, mean 14.2 s |
| best upstream baseline | 0.621 | 0.424 | | | | |

It tells us the frames line up and the labels mean what we think. It can't rank anything: with 17
possessions, one more or less moves spellF1 by ~0.06. v5 and v4 are level here; we kept v5
because it was picked on the external games beforehand.

What goes wrong in the window (per-segment study of v4; v5 has the same dead-ball calls): 83% of
frames right, 9.3% dead ball called alive, 7.6% wrong team. Five of the seven missed possessions
are under 4 s. Dead ball is the weak class (recall 0.52, precision 1.00): one 5 s stoppage is
played through, and in a 20 s one the ball stays tracked and moving among slow players, so we
keep giving it to a team. Only 5 of the 20 changes land within 1 s.

## 7. Limitations

- **Short possessions.** This is the big one. We predict 0.6–0.7× the true number of possessions
  on external games. Possessions under 3 s are a third of all possessions and we find very few of
  them; the per-frame model favours the right team on only 37% of frames in possessions under
  1.5 s. If we could find every one of them, the external score would go from 0.439 to about
  0.69. Perfect boundary timing, by contrast, would only add ~0.03. We tried three ways of
  re-weighting training towards short possessions and none helped: the features just can't tell a
  short controlled possession from a touch in a duel, and 40% of the time there's no ball.
- **Stoppages with a visible, moving ball** (ball carried or kicked back to the restart spot).
  We built a detector for sudden collective slowdowns, with a penalty-area exclusion
  (`src/mtah/deadball.py`). On external games it lowered spellF1 (0.439 → 0.435 at its mildest
  setting), so it's not in the submission. The code and its tests are kept.
- **Proxy labels.** External labels approximate the competition's definition, they don't copy
  it. SkillCorner's labels are automatic and its tracking is broadcast. Treat external scores as a
  relative guide.
- **Tagging prior is small on purpose.** It was validated on another vendor's phases, not
  Metrica's.
- **Kickoff timing** is a few frames off (half 2 ~7 frames early). We don't snap boundaries for
  the learned model; it made things worse externally.
- **Noise gaps.** We don't reproduce the 2.4% of real frames with 23+ tracks, and our tracklets
  are about twice as long as the real ones.

More detail on all the experiments, including the ones that failed, is in `docs/ablation.md`.

## 8. Reproduction

Windows Command Prompt, Python 3.11, data bundle unpacked anywhere:

```
git clone --recurse-submodules <repository URL> mtah-possession
cd mtah-possession
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tasks.py submission --data-dir C:\path\to\mtah-data --expect-sha256 recorded
.venv\Scripts\python.exe tasks.py validate --expect-sha256 recorded
```

That writes `outputs\real\submission.csv` in 20–50 s. `--expect-sha256 recorded` fails unless the
bytes match `config/submission.sha256`. Inputs are the tactical ATD text, its metadata XML and
the Smart Tagging XML. Everything else is pinned: `requirements.txt` (full closure), the upstream
submodule commit, the model file, the frozen configs, and single-threaded LightGBM at inference.
We regenerated v3, v4 and v5 a second time on the real data and got identical bytes, and every
version reproduces on a full-length synthetic match. Older files: `--method v4 --expect-sha256
recorded-v4` (same for v3, v2; add `--out` so they don't overwrite each other).

Rebuilding the model from external data:

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

The last three redo the tagging test and freeze its weight in `config/final_v5.yaml`.
`python tasks.py test` runs the tests (83) on synthetic matches in the exact ATD format.

**How this was built.** We wrote the code with an AI coding assistant. As the terms require, the
assistant never saw the match data, the video or any extract of them. One of us ran everything on
the real data locally and only shared aggregate numbers (counts, shares, percentiles) back, which
are the numbers quoted here.
