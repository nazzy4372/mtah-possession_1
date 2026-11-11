"""Tuning (M5): Optuna over the decoding parameters, on external games only.

    python tasks.py tune [--variant all] [--trials 80]

What is tuned, and on what:
  * search space: the decoder's three transition costs, the emission temperature, a dead-ball
    bias added to log p(D), and whether to refine boundaries;
  * data: the LEAVE-ONE-GAME-OUT predictions of the learned model (ext/train.py logo), so every
    game is scored by a model that never saw it, under the real-calibrated `full` noise;
  * objective: mean spellF1 over the games, from upstream possession_metrics.evaluate (imported,
    never reimplemented);
  * sampler: seeded TPE. Trials are asked in rounds of ROUND and decoded together in one batched
    Viterbi pass (decode.viterbi_batch, identical paths to the single decoder);
  * the frozen parameters are the median of the top 10 trials (majority vote for the yes/no
    refine setting), not the single best: robustness over overfitting.

Honest estimate: a nested 2-fold split (each fold mixes Metrica and SkillCorner games). Tune on one
fold, take the median of its top 10, score the other fold; then swap. The nested score is the
number to quote; the final tuning on all 13 games is what gets frozen.

The provided match is never touched; the public 5-minute window is never used.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from mtah import decode, features, halves, refine
from mtah.official import pm

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "final.yaml"
OUT = ROOT / "outputs" / "external"
ROUND = 6
TOP = 10

SPACE = {
    "c_switch": ("log", 2.0, 60.0),
    "c_die": ("log", 3.0, 80.0),
    "c_restart": ("log", 3.0, 80.0),
    "temperature": ("lin", 0.5, 2.0),
    "dead_bias": ("lin", -1.5, 1.5),
    "refine": ("cat", (False, True)),
}


def _train_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ext import train
    return train


def load_oof(variant):
    """games, cached eval sets, out-of-fold raw scores, class prior."""
    train = _train_module()
    path = OUT / "oof_{}.npz".format(variant)
    if not path.is_file():
        raise FileNotFoundError("{} missing: run python -m ext.train logo --variant {}".format(
            path, variant))
    saved = np.load(path)
    games = train.game_list()
    eval_sets = [train.prepare(g, "full", train.EVAL_SEED + i) for i, g in enumerate(games)]
    return games, eval_sets, {g: saved[g] for g in games}, saved["prior"]


def emissions(raw, prior, temperature, dead_bias):
    z = raw / temperature
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True)) - np.log(prior)[None, :]
    logp[:, 2] += dead_bias
    return logp


def evaluate_params(param_list, games, eval_sets, oof, prior):
    """spellF1 per game for each parameter set, decoded in one batch."""
    train = _train_module()
    jobs, meta = [], []
    for k, prm in enumerate(param_list):
        dp = decode.DecoderParams(prm["c_switch"], prm["c_die"], prm["c_restart"])
        for g, d in zip(games, eval_sets):
            spec = train._Spec(d)
            logp = np.zeros((spec.n_frames, 3))
            logp[spec.graded_mask()] = emissions(oof[g], prior, prm["temperature"],
                                                 prm["dead_bias"])
            jobs.append((logp, spec, train.kickoffs_of(d), dp))
            meta.append((k, g, d, spec))
    labels = decode.decode_many(jobs)
    scores = [dict() for _ in param_list]
    for (k, g, d, spec), lab in zip(meta, labels):
        kos = train.kickoffs_of(d)
        if param_list[k]["refine"]:
            F = features.Features(ball_xy=None, ball_observed=None, ball_known=None,
                                  ball_speed=d["ball_speed"], ball_missing_s=None,
                                  d_home=d["d_home"], d_away=d["d_away"],
                                  out_dist=d["out_dist"], player_speed=None, last_touch=None,
                                  since_touch_s=None, shape_diff=None)
            lab = refine.refine(lab, F, spec, refine.RefineParams(), kos)
        lab = halves.apply_policy(lab, spec, kos)
        r = pm.evaluate(d["gt"].astype(np.int8), pm.label_codes(lab))
        scores[k][g] = {"spell_f1_mean": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
                        "poss_ratio": r["n_pred_spells"] / float(r["n_gt_spells"])}
    return scores


def _suggest(trial):
    out = {}
    for name, spec in SPACE.items():
        if spec[0] == "log":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif spec[0] == "lin":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        else:
            out[name] = trial.suggest_categorical(name, list(spec[1]))
    return out


def optimise(games, eval_sets, oof, prior, n_trials, seed=0, label=""):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = 0
    while done < n_trials:
        trials = [study.ask() for _ in range(min(ROUND, n_trials - done))]
        params = [_suggest(t) for t in trials]
        started = time.perf_counter()
        scores = evaluate_params(params, games, eval_sets, oof, prior)
        for t, s in zip(trials, scores):
            study.tell(t, float(np.mean([v["spell_f1_mean"] for v in s.values()])))
        done += len(trials)
        print("  {} trials {:>3}/{}  best {:.4f}  ({:.0f} s/round)".format(
            label, done, n_trials, study.best_value, time.perf_counter() - started), flush=True)
    return study


def median_of_top(study, top=TOP):
    ranked = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:top]
    out = {}
    for name, spec in SPACE.items():
        values = [t.params[name] for t in ranked]
        if spec[0] == "cat":
            out[name] = bool(sum(bool(v) for v in values) * 2 > len(values))
        else:
            out[name] = float(np.median(values))
    return out, [t.value for t in ranked]


def folds(games):
    metrica = [g for g in games if g.startswith("metrica")]
    sc = [g for g in games if g.startswith("sc_")]
    a = metrica[:2] + sc[:len(sc) // 2]
    b = metrica[2:] + sc[len(sc) // 2:]
    return a, b


def _subset(names, games, eval_sets):
    idx = [games.index(n) for n in names]
    return names, [eval_sets[i] for i in idx]


def nested(variant, n_trials):
    games, eval_sets, oof, prior = load_oof(variant)
    fold_a, fold_b = folds(games)
    per_game = {}
    chosen = []
    for tune_on, score_on, name in ((fold_a, fold_b, "A->B"), (fold_b, fold_a, "B->A")):
        g_t, s_t = _subset(tune_on, games, eval_sets)
        study = optimise(g_t, s_t, oof, prior, n_trials, label=name)
        params, _ = median_of_top(study)
        chosen.append(params)
        g_s, s_s = _subset(score_on, games, eval_sets)
        per_game.update(evaluate_params([params], g_s, s_s, oof, prior)[0])
    mean = float(np.mean([v["spell_f1_mean"] for v in per_game.values()]))
    return mean, per_game, chosen


def write_config(params, variant, scores, nested_score, path=CONFIG):
    """config/final.yaml, written by hand-formatting so the bytes are stable."""
    path.parent.mkdir(exist_ok=True)
    lines = [
        "# Frozen M5 parameters. Written by `python tasks.py tune`; read by the pipeline (v2).",
        "# Tuned with Optuna (seeded TPE) on leave-one-game-out predictions of 13 external",
        "# games under the real-calibrated noise model; median of the top {} trials.".format(TOP),
        "# Nothing here was chosen with the provided match or the public 5-minute window.",
        "model_variant: {}".format(variant),
        "decoder:",
        "  c_switch: {:.4f}".format(params["c_switch"]),
        "  c_die: {:.4f}".format(params["c_die"]),
        "  c_restart: {:.4f}".format(params["c_restart"]),
        "emission:",
        "  temperature: {:.4f}".format(params["temperature"]),
        "  dead_bias: {:.4f}".format(params["dead_bias"]),
        "refine: {}".format("true" if params["refine"] else "false"),
        "validation:",
        "  nested_2fold_spell_f1: {:.4f}".format(nested_score),
        "  top_trials_spell_f1: [{}]".format(", ".join("{:.4f}".format(s) for s in scores)),
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="all")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--nested-only", action="store_true")
    args = parser.parse_args(argv)

    print("nested 2-fold estimate, variant {}".format(args.variant), flush=True)
    nested_score, per_game, chosen = nested(args.variant, args.trials)
    print("nested spellF1 {:.4f}".format(nested_score))
    for g, v in per_game.items():
        print("  {:<11} {:.4f}  poss/gt {:.2f}".format(g, v["spell_f1_mean"], v["poss_ratio"]))
    result = {"variant": args.variant, "nested_spell_f1": nested_score, "per_game": per_game,
              "fold_params": chosen}
    if not args.nested_only:
        games, eval_sets, oof, prior = load_oof(args.variant)
        study = optimise(games, eval_sets, oof, prior, args.trials, label="all")
        params, top_scores = median_of_top(study)
        frozen = evaluate_params([params], games, eval_sets, oof, prior)[0]
        print("frozen (median of top {}): {}".format(TOP, json.dumps(params)))
        print("frozen params on all 13 (tuned on them, optimistic): {:.4f}".format(
            np.mean([v["spell_f1_mean"] for v in frozen.values()])))
        write_config(params, args.variant, top_scores, nested_score)
        result.update({"frozen": params, "frozen_per_game": frozen, "top_scores": top_scores})
        print("wrote {}".format(CONFIG))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tune_{}.json".format(args.variant), "w", encoding="utf-8",
              newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
