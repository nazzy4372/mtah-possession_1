# More than a Hack 2027, Phase 1: ball possession

Team Nuttertools. Per-frame possession labels (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic
Club, from the provided automatic tracking. The official challenge, scorer and starter code sit in
the `upstream/` submodule ([metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)).
We import from it and never change it.

**How it works, in short.** Collective, identity-free features per frame go into a small LightGBM
model that gives p(H, A, D). A Viterbi decoder then labels each half, and every change of state
costs something, so labels don't flicker but real possessions don't get merged either. The
model only ever saw other games (Metrica and SkillCorner open data), with noise added to look
like the real tracking. The submitted version, v5, is model v4 plus a small team nudge from the
match's Smart Tagging.

External games, leave-one-game-out spellF1: 0.439 for v4, against 0.419 for our earlier v3
recipe, 0.301 for our hand-written rules and 0.189 for the ported upstream baseline. The tagging
nudge added another +0.003 to +0.009 in our tests. Full write-up in `docs/TECHNICAL.md`, all the
experiments (including the ones that didn't work) in `docs/ablation.md`, and how we labelled the
external data in `docs/label_derivation.md`.

## Reproduce the submission (Windows, Python 3.11)

The match data isn't in this repo and won't be. You need the official data bundle (the
`tactical/` folder, see upstream DATA.md) somewhere on disk.

```
git clone --recurse-submodules <this repo> mtah-possession
cd mtah-possession
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tasks.py submission --data-dir C:\path\to\mtah-data --expect-sha256 recorded
.venv\Scripts\python.exe tasks.py validate --expect-sha256 recorded
```

You get `outputs\real\submission.csv` (LF line endings) in 20–50 s. With `--expect-sha256
recorded` it fails unless the bytes match `config/submission.sha256`. Older versions: `--method
v4 --expect-sha256 recorded-v4`, same idea for v3 and v2 (use `--out` so they don't overwrite each
other).

It reads the tactical ATD text, its metadata XML and the Smart Tagging XML. Dependencies are
pinned (`requirements.txt`, full closure), the submodule commit is pinned, the model files are
stored byte-exact and LightGBM runs single-threaded at inference, so the output is the same bytes
every time.

## Commands

Everything goes through `tasks.py`:

```
.venv\Scripts\python.exe tasks.py test                 all tests (--fast skips the full-length match)
.venv\Scripts\python.exe tasks.py submission --data-dir C:\path\to\mtah-data
.venv\Scripts\python.exe tasks.py validate             official validator + LF check + SHA-256
.venv\Scripts\python.exe tasks.py report               aggregates + sanity alarms
.venv\Scripts\python.exe tasks.py synthetic --preset full
.venv\Scripts\python.exe tasks.py external --prepare   download (pinned), convert, score external games
.venv\Scripts\python.exe tasks.py tune                 Optuna on out-of-fold predictions
.venv\Scripts\python.exe tools\feature_stats.py        label-free feature distributions
```

Retraining from scratch (external data only):

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

`final-v4` + `tune` rebuild `models/emission_v4.txt` byte for byte and rewrite
`config/final_v4.yaml`. The `ext.tags` steps redo the tagging test and rewrite
`config/final_v5.yaml`. (v3: `cache`, `logo --variant touch`, `final-v3 --variant touch`, `tune
--variant touch --freeze`.)

## Layout

```
src/mtah/   io_atd (ATD reader), halves (kickoffs, X/D policy), features + learn_features,
            emission (rules v1), emission_v2..v5 (LightGBM + tagging prior), io_tagging,
            deadball (slowdown detector, not used in the submission), decode (Viterbi),
            refine, pipeline, tune, synthetic, official (upstream imports)
ext/        external data: download, Metrica + SkillCorner converters, noise, evaluation,
            training, tags (stand-in tagging stream)
models/     emission_v4.txt (the one we ship) + .json; v3 and v2 kept for reproduction
config/     final_v4.yaml + final_v5.yaml, submission.sha256; older configs and hashes too
tools/      label-free helpers (real_stats, slow_play_stats, tagging_stats, deadball_stats,
            report, compare_submissions, public_check, ...); all print aggregates only
tests/      unit + end-to-end tests on synthetic matches
docs/       TECHNICAL.md, ablation.md, label_derivation.md, public_window_log.md
```
