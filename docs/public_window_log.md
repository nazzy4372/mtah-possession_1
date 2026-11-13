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

Nothing was changed as a result of any look. v3 was chosen over v2 on the external games alone,
and frozen with its SHA-256 recorded, **before** look #3. Look #3 was agreed in advance to be a
correctness check that could not change which version is submitted.
