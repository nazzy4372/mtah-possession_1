# CLAUDE.md: More than a Hack 2027, Phase 1 (Ball Possession)

You are building a competition entry for a scored football-analytics challenge. Read this whole
file before writing code. It is the standing brief for every session in this repo.

Upstream repo (public, MIT, read it freely): https://github.com/metrica-sports/more-than-a-hack-2027
It is checked out as a git submodule at `upstream/`. Never modify it; import from it.
Environment: Windows, Python 3.11 venv at `.venv`. There is no `make`, so every command goes through
`python tasks.py <command>` and all code must be cross-platform (use `pathlib`, no shell-specific
syntax).
Read, in order: README.md, CHALLENGE.md, DATA.md, EVALUATION.md, SUBMISSION.md, RULES.md, FAQ.md,
`evaluation/possession_metrics.py`, `starter/baseline.py`, `starter/load_atd.py`,
`starter/load_smart_tagging.py`, `games.py`. If this brief and those files disagree, they win.

---

## 0. The task in five lines

- Output `frame,label` CSV, exactly 199,319 rows, frames 0..199318, labels `H` (Barcelona has ball,
  alive), `A` (Athletic has ball, alive), `D` (dead ball), `X` (abstain).
- Graded frames: 1st half 11185–90651, 2nd half 110808–190197. The 2nd half is graded from 110808
  but kicks off at 113742, so those frames are all `D`. Ungraded frames: write `X`.
- **Ranked on `spellF1`**: possessions (maximal runs of H or of A) matched 1-to-1 by temporal IoU,
  F1 averaged over IoU 0.3–0.7. `macroF1` is the tie-break. A 1-frame lag is free, 8 frames costs
  about 0.1, 1 s costs about 0.35. Too many possessions (flicker) or too few (merging) both kill it.
- `X` on graded frames is forward-filled with our previous label before spellF1 runs, and counts as
  a miss for macroF1. **Default: never emit `X` on graded frames.**
- Baseline to beat: spellF1 0.2433 (full match). Our target is above 0.45 (hypothesis, not a promise).

---

## 1. HARD RULES. Breaking any of these disqualifies us.

1. **You never read, open, print, grep, head, sample or otherwise access the provided match data.**
   It lives outside this repo at `%MTAH_DATA_DIR%` = `C:\Hackathon\mtah-data`. The terms forbid
   giving any cloud LLM the videos, datasets or extracts of them. This includes command output,
   so never run any command whose output contains frame-level values from that data. The human
   runs every command that touches the real match and pastes back **aggregate numbers only**.
   If you need something from the real data, write a script that prints aggregates and ask the
   human to run it.
2. **No hand labels on the provided game, for any purpose.** No frame-specific constants or rules
   ("if frame between X and Y then ..."). Every label must come out of an algorithm.
3. **The public 5-minute ground truth (`evaluation/gt/...public.csv`) is a correctness check only.**
   Never train on it, tune on it, or pick between model variants with it. All tuning happens on
   external games (Section 3).
4. **Deterministic, exact reproduction.** `python tasks.py submission` on the data bundle must regenerate the
   byte-identical CSV. Pin every dependency, fix all seeds, set single-threaded or deterministic
   modes (LightGBM `deterministic=True`, `num_threads=1` at inference, `force_row_wise=True`).
   Ship trained model files in the repo (they come from external data, so they may be public).
5. **No match data in git**, and no derived per-frame exports of it. `.gitignore` must cover
   `data/`, `*.mp4`, `*_atd.*`, `*homography*.csv`, `*smart_tagging*.xml`, `outputs/real/`.
6. External data must be about **other games** and openly accessible to the organisers. Record its
   source and licence in the technical doc.
7. Write `X` on ungraded frames, and detect the 2nd-half kickoff algorithmically (kickoff shape:
   ball at the centre spot, each team in its own half). Assert that the detected frame is consistent
   with the published half boundaries rather than hard-coding it.

---

## 2. Data facts you must design around (all from the public docs)

- ATD raw line: `<frame_id>:<p1>;<p2>;...;<p46>:<ball>`, each group `x,y,speed`, `NaN` when missing.
  `video_frame = frame_id - 1`. Coordinates are normalised 0–1; convert to metres with
  105.3 × 68.0 (`games.py`).
- 46 track slots for 22 players. Slots are tracklets, not people: no identities, frequent ID
  switches and fragments. The team of each slot comes from the metadata XML. The first `<Team>` is
  home. **All features must be collective and identity-free.**
- The ball is missing on 41% of graded frames, disproportionately in duels, crowded boxes and aerial
  balls. Design for it from day one.
- **Schema confirmed (checked locally, names only):** the ATD XML declares exactly three channels,
  `x` (normalised), `y` (normalised) and `s` (**speed in m/s**, as upstream DATA.md says; an earlier
  note here said km/h, but real_stats showed position-derived speed equals raw `s` at every
  percentile. The ball's `s` is unreliable, so derive ball speed from positions). The ball
  and all 46 player slots each carry exactly those three fields. There is no ball-state or ownership
  channel, so alive/dead and team in possession must be inferred entirely. `tools/inspect_schema.py`
  stays in the repo as documentation.
- The setup has been verified: `check_alignment_raw()` agrees, and the upstream baseline rebuilds
  byte-identically on this machine. The ball is located on 56.9% of all frames.
- **Where the points are (from reading the baseline reports, not from tuning):** all four baselines
  find almost none of the dead ball (recall 0.08–0.34), and their team calls while the ball is alive
  are already 73–89% correct. Unsmoothed, the frame-level calls flicker into about 8× too many
  possessions. Priorities in order: (1) dead-ball detection, which splits possessions correctly;
  (2) a decoder that stops flicker without merging; (3) boundary timing.
- Smart Tagging: 638 automatic intervals, `frame = round(seconds * 25)`. Codes are upper case; tags
  are title case (`Team`, `Type`). Match case-insensitively. Team is `N/A` on 36 intervals, including
  every throw-in. It is noisy evidence, not truth, and its intervals are phases, not possessions.
- Possession definition (CHALLENGE.md): one team always has the ball while it is alive. Loose and
  contested balls stay with the last controlling team. A touch, deflection or aerial duel without
  control is not a change of possession. The ground truth is derived mechanically from corrected
  tracking.
- Public ground truth (what is *expected*, not a tuning set): 17 possessions in 5 minutes, mean 14 s,
  but some last under 1.5 s. Do not impose a hard minimum possession length above about 0.5 s.
  Use soft switch penalties instead.

---

## 3. External training and tuning data (other games, allowed)

Build `ext/` loaders that convert each source into **our exact ATD text and XML format at 25 fps**,
plus a `frame,label` ground-truth CSV, so the same pipeline runs end to end on external and real data.

| priority | source | why | labels |
|---|---|---|---|
| 1 | Metrica open sample data (games 1–3; `kloppy.metrica.load_open_data`, GitHub `metrica-sports/sample-data`) | same vendor, 25 fps; game 3 is EPTS | derive H/A/D from human event data: possession by the team of each controlled event, D from ball-out, foul, set-piece and goal events until the restart event |
| 2 | SkillCorner open data (GitHub `SkillCorner/opendata`, 10 A-League 2024/25 matches) | broadcast tracking with natural noise; has a per-frame `possession` group | use `possession.group` for H/A; derive D from null possession and ball-out logic. Verify the definition before trusting it. Resample 10 to 25 fps |
| 3 (optional) | PFF FC World Cup 2022 open data, if obtainable | volume | event-derived |

Write a short `docs/label_derivation.md` for each source, describing exactly how H/A/D were derived.

**Noise augmentation (essential).** Corrupt the external tracking to look like the real ATD:
- Ball dropout in correlated bursts, target about 41% missing, higher when 3 or more players are
  within 3 m of the ball or the ball is fast.
- Track fragmentation: split each player into tracklets with random cuts, then reassign slot indices.
- Occasional duplicate tracks, position jitter, and about 20.5 tracked players on average.

The human supplies the real match's *label-free* aggregates (ball-missing gap-length histogram,
track fragment lengths, tracked-player count) from `tools/real_stats.py` to calibrate this.

---

## 4. Architecture (build in this order)

```
src/mtah/
  io_atd.py        fast numpy parser for raw ATD + metadata (no kloppy at inference; keep it fast)
  io_tagging.py    Smart Tagging -> per-frame arrays
  features.py      per-frame, identity-free features (+ rolling-window versions)
  deadball.py      alive/dead evidence
  emission.py      p(H, A, D | features) model: rules v1, LightGBM v2
  decode.py        HSMM / Viterbi with switch costs, per half
  refine.py        boundary snapping
  pipeline.py      data dir -> labels array -> CSV
  tune.py          Optuna on external games, objective = official spellF1
tools/  inspect_schema.py, real_stats.py, report.py (aggregates only)
ext/    loaders + converters + augmentation for external games
tests/  unit tests + an end-to-end test on a synthetic 3-minute match
```

**Features per frame** (metres, NaN-safe, each with an "observed" flag):
- Ball: position (interpolated over gaps of 1 s or less, with a flag), speed, acceleration, how long
  it has been missing, inside or outside the pitch (with margin), distance to the nearest touchline
  or goal line, stationary flag.
- Proximity: nearest home and away distance to the ball, margin (away minus home), home and away
  counts within 2 m and 5 m, and which team was last within 1.5 m (with time since).
- Collective shape (works while the ball is invisible): per-team centroid, spread, convex-hull
  area, depth, defensive-line height, centroid velocity toward goal, team speeds, count of players
  sprinting. The team in possession usually stretches out and pushes forward; the defending team
  compacts.
- Stoppage signals: whole-pitch mean player speed (players slowing down), drop in sprinters, ball
  stationary on a line, corner, penalty spot or centre spot.
- Smart Tagging priors: active phase code and team per frame (BUILD UP, PROGRESSION, CREATING
  CHANCES or ATTACKING TRANSITION for team X is evidence for X; DEFENDING ... for team X is evidence
  for the other team), and set-piece start frames as restart evidence. Treat `N/A` as missing.
- Rolling versions of the key features over ±0.5 s, ±1 s, ±2 s and ±5 s windows (centred and causal).

**Emission model**
- v1 (rules, day 2): hand-designed log-scores from proximity margin, last-touch team, shape signals
  and dead-ball signals. Purpose: have a valid submission that beats baseline early.
- v2 (learned, day 4): LightGBM multiclass on augmented external games with leave-one-game-out cross-
  validation, then probability calibration. Keep the model small; the domain gap is the risk, not
  capacity.

**Decoder.** Run the hidden semi-Markov / Viterbi decode separately per half, over states {H, A, D}:
- Costs: switch H↔A (`c_switch`), alive→D (`c_die`), D→alive (`c_restart`). Optional duration
  priors: D at least about 1 s; possession duration with a soft log-normal prior.
- After a D run, the restart team is the team nearest the ball when it starts moving. No special-
  casing of `N/A` throw-ins is needed beyond that.
- Contested phases: the cost structure keeps possession with the last team unless the evidence for a
  sustained change outweighs `c_switch`. That is the "controlled possession" rule.

**Boundary refinement.** Move each H↔A boundary to the first frame at which a player of the new team
is within r metres of the (possibly interpolated) ball, inside a ±1.5 s search window. Move each
alive→D boundary to the out-of-bounds crossing frame when one exists. Tune r and the window on
external games only.

**Tuning.** Optuna (seeded TPE) over the decoder costs, duration priors, refinement r and window, and
the rolling-window choices. The objective is the mean spellF1 across held-out external games,
computed by **importing the official `evaluation/possession_metrics.py`**; never reimplement the
metric. Choose the median parameter set of the top 10 trials rather than the single best (robustness
over overfitting). Report spellF1 ± bootstrap.

---

## 5. Milestones and acceptance criteria

Work milestone by milestone. At the end of each one, stop and report back.

- **M1, scaffolding (day 1).**
  - Repo, pinned `requirements.txt` (Python 3.11), `tasks.py` (`python tasks.py test|external|tune|submission|validate|report`), `.gitignore`, and
    `.claude/settings.json` deny rules (already present, do not weaken them).
  - Vendor or depend on the upstream evaluation code.
  - Synthetic-match generator in exact ATD format, and an end-to-end test that produces a 199,319-row
    file which passes `validate_submission.py`.
  - `tools/inspect_schema.py` and `tools/real_stats.py`.
- **M2, rules v1 (day 2).** `python tasks.py submission` produces a valid file using v1 emissions + decoder.
  On external games it beats the reproduced baseline on spellF1.
- **M3, external data (days 2–3).** Metrica and SkillCorner converted, labels derived and
  documented, augmentation in place. Baseline + v1 scored on every external game.
- **M4, learned emissions (days 3–4).** LightGBM leave-one-game-out spellF1 clearly above v1 on
  external games. Run an ablation table (with and without shape features, tagging, refinement).
- **M5, tuning and freeze (day 5).** Optuna done, parameters frozen in `config/final.yaml`. The human
  runs `python tasks.py submission` and `python tasks.py report` on the real match **once** and shares the aggregates.
- **M6, robustness (day 6).** Sanity alarms, which are checks rather than tuning targets:
  - possessions per half within football norms (roughly 100–200 per half, hypothesis);
  - mean possession duration plausible;
  - D share plausible;
  - Barcelona's share of possession plausible.
  - Then: two consecutive runs produce identical bytes; a clean clone + data bundle reproduces the file.
- **M7, deliverables (day 7, submit day 7, not day 8).** `docs/TECHNICAL.md`, README reproduction
  steps, and `<team_name>.csv` validated with the official validator immediately before upload.

---

## 6. Technical document outline (`docs/TECHNICAL.md`, 3–5 pages)

1. Approach and why: per-frame evidence → semi-Markov decoding optimised for spellF1.
2. Assets used and dropped: ATD, Smart Tagging, homography (state whether it was used), video
   (not used), and why.
3. Training data: every external dataset, its source, licence, who labelled it, size, and how H/A/D
   were derived. State explicitly that nothing comes from the provided game.
4. Possession rules implemented: loose balls, contested balls, deflections, restarts, throw-ins with
   `N/A` team.
5. Validation protocol: leave-one-game-out on external games; the public window used only as a
   correctness check, and how many times it was looked at.
6. Results: ablation table on external games; public-window check.
7. Limitations and next steps.
8. Reproduction: exact commands.

---

## 7. How to report back to the human

End every milestone with:
- What was built, and which tests pass.
- External-game scores (spellF1 mean ± bootstrap, macroF1, possession counts).
- **The exact commands the human must run on the real data**, and which aggregate numbers to paste
  back.
- Open questions or assumptions, labelled **HYPOTHESIS** where unverified.

Never declare a milestone done without running its tests. Keep code readable; the organisers inspect
the repo.
