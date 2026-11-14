# Public-window log

The public 5-minute ground truth (`upstream/evaluation/gt/BAR-ATH_possession_gt_public.csv`) is a
**correctness check only**: never training data, never a tuning set, and never used to choose
between variants (CLAUDE.md hard rule 3; upstream RULES.md). Every look at it is recorded here, so
the technical document can state how many times it was used and why.

| # | date | milestone | who ran it | purpose | what was looked at |
|---|------|-----------|------------|---------|--------------------|
| 0 | 2026-11-11 | M7 | assistant | check that `tools/public_check.py` redirects the scorer's outputs out of `upstream/` | the four **upstream baselines** only, whose public-window scores are already published in the upstream README; reproduced exactly. Not our submission. |
| 1 | 2026-11-11 | M7 | team member (`tools/public_check.py`) | correctness check of the **frozen** submission (`6ad9d4ad…`), after every parameter was frozen | scoreboard row only: macroF1 0.7833, spellF1 0.5125, 15 possessions of mean 17.8 s (truth: 17 of 14.2 s), transF1 0.3077, latency +4.7 frames. Passed: frames aligned, labels as intended. |
| 2 | 2026-11-12 | M7 | team member (`tools/public_check.py --detail`) | the same frozen file, at the team's request: the frame-level breakdown of the same 7,500 frames | scorer detail block: accuracy 0.814, per-class F1 H 0.851 / A 0.825 / D 0.674, D recall 0.530, alive/dead F1 0.941, team accuracy (both alive) 0.892, possessions 15 vs 17 |
| 2b | 2026-11-12 | after M7 | team member | the look #2 command was shown once more on the **same v2 file** (before the v3 file was generated) | identical output to look #2; no new information |
| 3 | 2026-11-14 | after v3 was frozen | team member (`tools/public_check.py --detail --force`) | correctness check of the **frozen v3** file (`f6fb5445…`), the file to be uploaded, at the team's request | scorer detail block: macroF1 0.7932, spellF1 0.5625, accuracy 0.826, per-class F1 H 0.863 / A 0.839 / D 0.677, D recall 0.527, alive/dead F1 0.942, team accuracy (both alive) 0.905, 15 possessions of mean 17.8 s (truth: 17 of 14.2 s), latency −0.8 frames |
| 4 | 2026-11-14 | after v3 was frozen | team member (`tools/public_errors.py`) | understanding the organisers' definitions (CHALLENGE.md: "use it to understand what is expected"): which ground-truth segments the frozen v3 file gets wrong | one row per ground-truth segment (21) and per segment of ours (20), with label shares, match verdicts and label-free context. Findings: most "dead called alive" frames come from one 20 s stoppage in which the ball is tracked and moving; 5 of the 7 missed possessions last under 4 s. |
| 5 | 2026-11-14 | after v4 was frozen, reproduced and packaged | team member (`tools/public_check.py --detail --force`, `tools/public_errors.py`) | correctness check of the **submitted v4** file (`0432a1ed…`), at the team's request, with the per-segment breakdown | scorer detail block: macroF1 0.7979, spellF1 0.5867, accuracy 0.830, per-class F1 H 0.866 / A 0.841 / D 0.686, D recall 0.524 (precision 0.996), alive/dead F1 0.945, team accuracy (both alive) 0.905, 13 possessions of mean 20.7 s (truth: 17 of 14.2 s), latency −7.0 frames. Per segment: 10 of 17 possessions matched; 5 of the 7 missed last under 4 s; of the 4 stoppages two are found (83%, 92%), one of 5 s is played through, one of 20 s with a tracked, moving ball is found for 11%. |

No parameter, threshold or model choice was fitted to the window or selected with it. v3 was
chosen over v2 on the external games alone, and frozen with its SHA-256 recorded, **before** look
#3. Look #3 was agreed in advance to be a correctness check that could not change which version is
submitted.

**What look #4 led to (stated plainly).** Look #4 showed that most of the dead-ball frames v3
missed in the window came from one stoppage. That made us question an assumption in our training
noise: when does the real ball go untracked? We did not answer it with the window. We measured it
on the whole match without labels (`tools/slow_play_stats.py`: ball visibility by player-speed
bin), found the assumption wrong, rebuilt the noise model to match that label-free measurement
(`full2`), and accepted the retrained model v4 on the 13 external games by the rule fixed before
v3. So the window prompted a question; a label-free measurement and the external games answered
it.

**The v4 file (`0432a1ed…`, the submitted file at the time) was scored on the window once, in
look #5.** By then v4 had been chosen on the external games, frozen with its SHA-256 recorded,
reproduced byte-for-byte in a second run and packaged for upload. It was agreed before the look
that its result could not change the choice between v3 and v4, in either direction, and it did
not. The v4 and v3 rows differ by less than one possession's worth of spellF1, so the window
says nothing about which is better; that was decided on the external games.

**What came after look #5 (stated plainly).** The next day the team asked why the provided Smart
Tagging was not an input, since the entry is judged on this match alone. That led to model v5 =
v4 plus a weak team prior from the tagged possession phases. Look #5 had reported 7.6% of the
window's frames with the wrong team, and that number was in our minds. It decided nothing: the
case for the prior was made with a label-free comparison of our labels and the tags over the
whole match (`tools/tagging_stats.py`), and its weight was set on 10 SkillCorner games with
their phases of play as a stand-in stream (`ext/tags.py`), by a rule fixed before v5 was run on
the real match.

**The submitted v5 file (`1dcde4f6…`, frozen 2026-11-15) has not been scored on the window.** It
keeps 99.1% of the v4 file's labels and `X` on the same frames, so the alignment check of look
#5 carries over. v5 was not, and will not be, compared with v4 on the window.
