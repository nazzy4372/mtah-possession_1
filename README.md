# More than a Hack 2027, Phase 1: ball possession

Per-frame possession labels (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, built from
the provided automatic tracking. The official challenge, scorer and starter code live in the
`upstream/` submodule ([metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)),
imported and never modified.

**Approach.** Identity-free per-frame features feed a small LightGBM model that outputs p(H, A, D).
Each half is then decoded with a Viterbi decoder that charges for every switch, which stops
flicker without merging possessions. The model is trained only on noise-augmented external games
(Metrica and SkillCorner open data) and shipped in `models/`. The decoding settings were tuned with
Optuna on held-out external predictions and frozen in `config/final_v4.yaml`. Leave-one-game-out
spellF1 on 13 external games (nested estimate) is 0.439 for the shipped model v4 against 0.419
for the v3 recipe, on noise that matches how the real ball goes untracked in slow play. On the
older noise v3 scored 0.417, the earlier v2 0.393, our rules 0.301 and the ported upstream
baseline 0.189 (`docs/ablation.md`). The submitted model, v5, is v4 plus a weak team prior from
the match's Smart Tagging (`config/final_v5.yaml`); its weight was set on the SkillCorner games,
using their phases of play as a stand-in tagging stream (+0.003 to +0.009 spellF1 there).
External data and labels: `docs/label_derivation.md`. Full write-up: `docs/TECHNICAL.md`.

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

`submission` writes `outputs\real\submission.csv` (LF line endings) in about 20–50 s. With
`--expect-sha256 recorded` it fails unless the bytes equal the frozen file, whose SHA-256 is in
`config/submission.sha256`. The default method is the submitted model (v5); the earlier files are
reproduced with `--method v4 --expect-sha256 recorded-v4`, and likewise `v3` / `recorded-v3` and
`v2` / `recorded-v2` (add `--out` to keep them apart). The inputs are the bundle's tactical ATD
text, its metadata XML and its Smart Tagging XML. Everything else that determines the output
is pinned:
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
.venv\Scripts\python.exe -m ext.train cache --condition full2
.venv\Scripts\python.exe -m ext.train logo --variant touch_f2
.venv\Scripts\python.exe -m ext.train final-v4
.venv\Scripts\python.exe tasks.py tune --variant touch_f2 --freeze
.venv\Scripts\python.exe -m ext.tags write
.venv\Scripts\python.exe -m ext.tags experiment
.venv\Scripts\python.exe -m ext.tags freeze
```

`final-v4` and `tune` reproduce `models/emission_v4.txt` byte for byte and rewrite
`config/final_v4.yaml`; the `ext.tags` steps rerun the tagging test and rewrite
`config/final_v5.yaml`. The earlier v3 is rebuilt with `cache`, `logo --variant touch`,
`final-v3 --variant touch` and `tune --variant touch --freeze`.

## Layout

```
src/mtah/      io_atd (fast ATD parser), halves (kickoff detection, graded-frame policy),
               features + learn_features, emission (rules v1) + emission_v2/v3/v4 (LightGBM),
               io_tagging + emission_v5 (Smart Tagging team prior),
               decode (Viterbi), refine, pipeline, tune, synthetic, official (upstream imports)
ext/           external data: pinned download, Metrica + SkillCorner converters, noise model,
               evaluation harness, training, tags (stand-in tagging stream and its test)
models/        emission_v4.txt (shipped LightGBM model) + .json (features, provenance);
               emission_v3.*, emission_v2.* (the earlier frozen models)
config/        final_v4.yaml + final_v5.yaml (frozen settings), submission.sha256 (frozen output
               hash); final_v3.yaml, final.yaml and submission_v4/_v3/_v2.sha256 for the earlier
               versions
tools/         check_setup, inspect_schema, real_stats, feature_stats, slow_play_stats,
               tagging_stats, kickoff_diag, report, parity_baseline, compare_submissions,
               public_check, public_errors (all print aggregates only)
tests/         unit + end-to-end tests on synthetic matches
docs/          label_derivation.md, ablation.md, public_window_log.md
```
