# External data: sources and label derivation

Every external dataset used for development, tuning or training is described here: source,
terms, size, who labelled it, and exactly how per-frame `H` / `A` / `D` labels were derived.
**Nothing here comes from the provided game (BAR-ATH).** Code: `ext/`.

## Metrica Sports open sample data (games 1–3)

| | |
|---|---|
| Source | https://github.com/metrica-sports/sample-data, pinned to commit `e706dd5`; every file is SHA-256 checked against `ext/manifests/metrica.json` |
| Terms | No formal licence. The repository README asks users to "be responsible with the use of this data" and to acknowledge the source for public use. Acknowledged here and in `docs/TECHNICAL.md`. |
| Content | 3 anonymised professional matches at 25 fps, 105 × 68 m. Games 1–2: CSV tracking (home and away files) and CSV events. Game 3: FIFA EPTS tracking and metadata, JSON events. |
| Who labelled it | The event data is human-annotated by Metrica Sports. The tracking is Metrica's own. We added no hand labels. |
| Size | About 182 MB raw. After conversion: games 1–3 cover 145,006 / 141,156 / 143,761 tracked frames over two periods each. |
| Use | Development and evaluation of the rules (M2): games 1–2 are the dev set, game 3 is held out. Training and LOGO cross-validation from M4. |

### Conversion to our ATD format (`ext/metrica.py`)
- Each game is written as a bundle in exactly the real layout: `tactical/<stem>_atd.txt` + `_atd.xml`
  (EPTS, readable by kloppy), plus `gt.csv`, `game_spec.json` and `truth.json`. The same pipeline
  then runs on it unchanged.
- Timeline: 30 s of untracked frames, period 1, 60 s untracked "half-time", period 2, 10 s
  untracked. This reproduces the real file's structure. The graded halves are the two periods.
- Slots: one slot per Metrica player column (home first). Coordinates stay normalised.
  The speed channel `s` is computed from positions in m/s, the real unit. Game 3 is read with kloppy
  (`coordinates="metrica"`); games 1–2 are read directly from the CSVs.

### Labels from events
Events are processed in time order within each period. A label holds until the next change point.

| rule | event(s) | effect, from frame |
|---|---|---|
| restart | any `SET PIECE`; any event with subtype `GOAL KICK` (Metrica annotates goal kicks as `PASS` or `BALL LOST`) | alive, restarting team, from the start frame |
| control | `PASS`, `CARRY`, `SHOT`, `RECOVERY`, `BALL LOST` by team T | T in possession from the start frame, if T did not already have it |
| no change | `CHALLENGE` (any outcome), `CARD` | none: loose and contested balls stay with the last team |
| ball out | `BALL OUT` | dead from the end frame (the ball crossing the line) |
| shot ends play | `SHOT` with subtype `OUT` or `GOAL` | dead from the end frame |
| foul | `FAULT RECEIVED`; subtype `HAND BALL` | dead from the start frame |
| offside, end of half | subtype `OFFSIDE`, `END HALF` | dead from the end frame |
| unannotated stoppage | a restart while play is still alive, more than 1 s after the previous event ended | dead from that previous event's end frame |
| before kickoff | — | dead until the first restart of the period |

Resulting ground truth (whole match): possessions 381 / 359 / 405, mean 7–10 s; dead-ball share
44% / 41% / 32%. That is within football norms; a typical match has the ball out of play for
35–45% of the time.

**Known differences from the competition ground truth (HYPOTHESIS until checked against more
data):** the BAR-ATH truth comes from corrected tracking, not from events. Event frames are
accurate to a few frames. A possession change is placed on the new team's first controlled
action, which can come slightly after the moment control was gained.

### Resting dead ball (augmentation input, `rests.json`)
Because Metrica blanks the ball while it is dead (next section), the converter also records, from
the **event data only**, where a dead ball rests: just beyond the nearest line at the point it went
out (`BALL OUT` / `SHOT`-out end coordinates, pushed 1.5 m out, for up to 2 s), and on the restart
spot before the restart (the restart event's start coordinates; the centre spot for a kickoff).
`ext/augment.py` puts a jittered ball back over 70% of each such stretch, but only where the
source blanked it. The `clean` condition never uses this.

### Domain artefact to be aware of
In the Metrica sample data the **ball is blanked while it is dead**: 86–97% of dead frames have no
ball against about 1% of alive frames. On the real match the ball goes missing mostly during live
play (the published baselines, which call every ball-less frame dead, recover only 8–34% of the dead
frames). So (a) the v1 rules never use "ball missing" as dead-ball evidence, and (b) every
evaluation is also run under real-calibrated ball dropout (`ext/augment.py`), which inserts long
gaps independently of the true state, with the frequency, length distribution and crowding
dependence measured on the real match by `tools/real_stats.py`.

## SkillCorner open data (10 A-League 2024/25 matches)

| | |
|---|---|
| Source | https://github.com/SkillCorner/opendata, pinned in `ext/manifests/skillcorner.json`. The repository holds 20 matches; we use the first 10 match ids in sorted order. Tracking files are Git LFS objects, and each one's SHA-256 is checked against its LFS pointer. |
| Licence | MIT. The README asks users to credit SkillCorner. Credited here and in `docs/TECHNICAL.md`. |
| Content | Broadcast tracking at 10 fps (players and ball, metres, origin at the centre), a per-frame `possession` group, phases of play and dynamic events. Pitch size per match from `<id>_match.json`. |
| Who labelled it | SkillCorner's automatic pipeline (computer vision plus their possession model). Automatic, not human, so treat it as a second opinion on the labels, not as gold. |
| Size | About 0.9 GB raw for 10 matches; about 60,000 frames at 10 fps each. |

### Conversion (`ext/skillcorner.py`)
- 10 fps to 25 fps: positions are linearly interpolated between neighbouring samples (missing if
  either neighbour is missing); labels take the most recent sample.
- Coordinates: metres with the centre at the origin and y up, normalised to the ATD convention
  (0–1, top-left) with the match's own pitch size, which the GameSpec carries.
- Slots: one per player in `match.json`, home team first. SkillCorner's extrapolated
  (off-camera) positions are kept as a stand-in for the tactical camera's whole-pitch view. The
  ATD noise model is applied on top.
- Same timeline layout as Metrica: untracked padding, period 1, 60 s untracked half-time,
  period 2.

### Labels
Checked on match 1874553 before trusting the definition:
- `possession.group` is `home team`, `away team` or None, and it persists while no player has the
  ball (the team-possession definition we need).
- None frames **outside** every phase of play are dead ball. They make up 95% of None frames,
  come in long runs (median 17 s), and the ball is off the pitch on 16% of them against 0.4% of
  possession frames.
- None frames **inside** a phase are loose balls in short runs (median 0.8 s). They stay with the
  last team, or the phase's team if there is none yet.
- The group and the phase's team agree on 86.5% of frames.

Result for 1874553: 166 / 177 possessions per half, mean about 10 s, dead-ball share 40% / 47%
(broadcast replays during stoppages may lengthen dead stretches; **HYPOTHESIS**).

## Noise model (`ext/augment.py`, condition `full`)
Calibrated against label-free aggregates of the real match (`tools/real_stats.py`, run locally by
a team member; only aggregates were shared with the development assistant, no frame-level data).
Achieved on Metrica games 1 and 3:

| aggregate | real match | Metrica g1 / g3 with `full` |
|---|---|---|
| ball missing (graded) | 0.407 | 0.438 / 0.478 |
| ball missing while alive / dead | ≈0.45 / ≈0.34 (estimated from the published baselines) | 0.38 / 0.52, 0.46 / 0.53 |
| tracked players: mean, p5 | 20.5, 14 | 20.9, 17 / 20.7, 14 |
| frames with ≤ 13 players | 5.1% | 2.9% / 4.3% |
| tracklets per minute | 11.5 | 11.4 / 12.6 |
| tracklet median length (frames) | 888 | 1,789 / 1,685 |
| player step p50 / p95 (m per frame) | 0.054 / 0.187 | 0.054 / 0.179, 0.060 / 0.182 |
| same-team pair within 0.5 m | 3.5% | 7.1% / 3.0% (mostly natural) |
| frames with ≥ 23 tracks | 2.4% | 0.3% / 0.3% (not reproduced) |
