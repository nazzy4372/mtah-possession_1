# More than a Hack 2027, Phase 1: ball possession

Per-frame possession labels (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, built from
the provided automatic tracking. The official challenge, scorer and starter code live in the
`upstream/` submodule ([metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)),
imported and never modified.

Status: **M2, rules v1.** The pipeline runs identity-free features through hand-written
emission rules, a per-half Viterbi decoder with switch costs, and boundary refinement. On external
games (Metrica open data) it beats the ported upstream baseline on spellF1. See
`docs/label_derivation.md` for the external data. The learned emissions come in M4 and tuning in M5.

## Setup (Windows, Python 3.11)

```
git clone --recurse-submodules <this repo>
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The match data is not in this repository and never will be. Unpack the bundle anywhere and either
set `MTAH_DATA_DIR` or pass `--data-dir`.

## Commands

All commands go through `tasks.py`:

```
.venv\Scripts\python.exe tasks.py test                 all tests (add --fast to skip the full-length match)
.venv\Scripts\python.exe tasks.py submission --data-dir C:\path\to\mtah-data
.venv\Scripts\python.exe tasks.py validate             official validator + LF check + SHA-256
.venv\Scripts\python.exe tasks.py report               aggregate sanity numbers
.venv\Scripts\python.exe tasks.py synthetic --preset full
.venv\Scripts\python.exe tasks.py external --prepare     download (pinned) + convert + score external games
```

`submission` writes `outputs\real\submission.csv` with LF line endings and prints its SHA-256. A
second run on the same bundle reproduces the same bytes.

## Layout

```
src/mtah/      io_atd (fast ATD parser), halves (kickoff detection, graded-frame policy),
               pipeline, synthetic (test matches in the exact ATD format), official (upstream imports)
tools/         check_setup, inspect_schema, real_stats, report, parity_baseline (aggregates only)
tests/         unit + end-to-end tests on synthetic matches
docs/          technical notes; public_window_log.md records every look at the public ground truth
```
