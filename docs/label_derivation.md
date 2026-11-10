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

### Domain artefact to be aware of
In the Metrica sample data the **ball is blanked while it is dead**: 86–97% of dead frames have no
ball against about 1% of alive frames. On the real match the ball goes missing mostly during live
play (the published baselines, which call every ball-less frame dead, recover only 8–34% of the dead
frames). So (a) the v1 rules never use "ball missing" as dead-ball evidence, and (b) every
evaluation is also run under real-calibrated ball dropout (`ext/augment.py`), which inserts long
gaps independently of the true state, with the frequency, length distribution and crowding
dependence measured on the real match by `tools/real_stats.py`.
