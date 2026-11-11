# Public-window log

The public 5-minute ground truth (`upstream/evaluation/gt/BAR-ATH_possession_gt_public.csv`) is a
**correctness check only**: never training data, never a tuning set, and never used to choose
between variants (CLAUDE.md hard rule 3; upstream RULES.md). Every look at it is recorded here, so
the technical document can state how many times it was used and why.

| # | date | milestone | who ran it | purpose | what was looked at |
|---|------|-----------|------------|---------|--------------------|
| 0 | 2026-11-11 | M7 | assistant | check that `tools/public_check.py` redirects the scorer's outputs out of `upstream/` | the four **upstream baselines** only, whose public-window scores are already published in the upstream README; reproduced exactly. Not our submission. |
| 1 | pending | M7 | team member | correctness check of the **frozen** submission (`6ad9d4ad…`), after every parameter was frozen | scoreboard row (macroF1, spellF1, possession count and duration) |

Nothing was changed as a result of any look.
