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

Nothing was changed as a result of any look. The v3 file has **not** been scored on the public
window. v3 was accepted on the external games alone, and checked against v2 frame by frame without
ground truth (`tools/compare_submissions.py`), so that the window could not become a way of
choosing between versions.
