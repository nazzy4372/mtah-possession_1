# More than a Hack 2027, Phase 1: ball possession

Per-frame possession labels (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, built from
the provided automatic tracking. The official challenge, scorer and starter code live in the
`upstream/` submodule ([metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)),
imported and never modified.

**Approach.** Identity-free per-frame features feed a small LightGBM model that outputs p(H, A, D).
Each half is then decoded with a Viterbi decoder that charges for every switch, which stops
flicker without merging possessions. The model is trained only on noise-augmented external games
(Metrica and SkillCorner open data) and shipped in `models/`. The decoding settings were tuned with
Optuna on held-out external predictions and frozen in `config/final.yaml`. Leave-one-game-out
spellF1 on 13 external games is 0.393 (nested estimate), against 0.301 for our rules and 0.189 for
the ported upstream baseline (`docs/ablation.md`). External data and labels:
`docs/label_derivation.md`.

## Reproduce the submission (Windows Command Prompt, Python 3.11)

The match data is not in this repository and never will be. You need the official data bundle
(the `tactical/` folder, see upstream DATA.md) somewhere on disk.

```
git clone --recurse-submodules <this repo> mtah-possession
cd mtah-possession
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tasks.py submission --data-dir C:\path\to\mtah-data --expect-sha256 recorded
.venv\Scripts\python.exe tasks.py validate --expect-sha256 recorded
```

`submission` writes `outputs\real\submission.csv` (LF line endings) in about 15 s. With
`--expect-sha256 recorded` it fails unless the bytes equal the frozen file, whose SHA-256 is in
`config/submission.sha256`. Everything that determines the output is pinned:
- `requirements.txt`, with the full dependency closure;
- the upstream submodule commit;
- the model file (stored byte-exact);
- the frozen config.

LightGBM runs single-threaded at inference.

## Commands

All commands go through `tasks.py`:

```
.venv\Scripts\python.exe tasks.py test                 all tests (add --fast to skip the full-length match)
.venv\Scripts\python.exe tasks.py submission --data-dir C:\path\to\mtah-data
.venv\Scripts\python.exe tasks.py validate             official validator + LF check + SHA-256
.venv\Scripts\python.exe tasks.py report               aggregate numbers + sanity alarms
.venv\Scripts\python.exe tasks.py synthetic --preset full
.venv\Scripts\python.exe tasks.py external --prepare   download (pinned) + convert + score external games
.venv\Scripts\python.exe tasks.py tune                 Optuna on external out-of-fold predictions
.venv\Scripts\python.exe tools\feature_stats.py        label-free feature distributions (domain gap)
```

Retraining the model from scratch, all from external data:

```
.venv\Scripts\python.exe tasks.py external --prepare
.venv\Scripts\python.exe -m ext.train cache
.venv\Scripts\python.exe -m ext.train logo --variant all
.venv\Scripts\python.exe -m ext.train final --variant all --c-switch 15 --c-dead 25 --temperature 1.0
.venv\Scripts\python.exe tasks.py tune
```

The last two steps reproduce `models/emission_v2.txt` byte for byte and rewrite `config/final.yaml`.

## Layout

```
src/mtah/      io_atd (fast ATD parser), halves (kickoff detection, graded-frame policy),
               features + learn_features, emission (rules v1) + emission_v2 (LightGBM),
               decode (Viterbi), refine, pipeline, tune, synthetic, official (upstream imports)
ext/           external data: pinned download, Metrica + SkillCorner converters, noise model,
               evaluation harness, training
models/        emission_v2.txt (LightGBM) + emission_v2.json (features, provenance)
config/        final.yaml (frozen decoding settings), submission.sha256 (frozen output hash)
tools/         check_setup, inspect_schema, real_stats, feature_stats, kickoff_diag, report,
               parity_baseline (all print aggregates only)
tests/         unit + end-to-end tests on synthetic matches
docs/          label_derivation.md, ablation.md, public_window_log.md
```
