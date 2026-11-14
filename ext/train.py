"""Learned emissions (v2): LightGBM on noise-augmented external games, leave-one-game-out.

    python -m ext.train cache            # feature matrices for every external game (cached)
    python -m ext.train logo [--variant all|noshape|rawspeed] [--grid]
    python -m ext.train final            # train on every game, write models/emission_v2.txt

Protocol:
  * training rows: each game under the FULL noise model with seed 2000 + i, every SUBSAMPLE-th
    graded frame, plus a home/away-mirrored copy (labels swapped);
  * evaluation: the held-out game under the full noise model with a different seed (1000 + i,
    the same corrupted copy v1 and the baseline are scored on in ext/evaluate.py), every frame;
  * the model outputs p(H, A, D | x). It is calibrated with one temperature fitted on the
    out-of-fold predictions, divided by the class prior to give emission likelihoods, then
    decoded with the same Viterbi and refinement as v1;
  * LightGBM is deterministic (deterministic=True, force_row_wise=True, fixed seeds and thread
    count); inference runs single-threaded.
Nothing here touches the provided match.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from ext import evaluate as ev  # noqa: E402
from mtah import decode, features, halves, learn_features, refine  # noqa: E402
from mtah.official import pm  # noqa: E402

CACHE = ROOT / "ext_cache" / "features"
MODELS = ROOT / "models"
OUT = ROOT / "outputs" / "external"
SUBSAMPLE = 5
TRAIN_SEED, EVAL_SEED = 2000, 1000

LGB_PARAMS = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 400, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "deterministic": True,
    "force_row_wise": True, "num_threads": 4, "verbosity": -1,
}
NUM_ROUNDS = 250

def _touch(names):
    """The frozen v3 feature set: everything except raw speed and the later gap group."""
    return [n for n in names if n not in learn_features.GROUP_SPEED_RAW
            and n not in learn_features.GROUP_GAP]


def _m4(names):
    """The frozen M4 (v2) feature set: the v3 set without the touch group."""
    return [n for n in _touch(names) if n not in learn_features.GROUP_TOUCH]


VARIANTS = {
    "all": _m4,
    "noshape": lambda names: [n for n in _m4(names) if n not in learn_features.GROUP_SHAPE],
    "rawspeed": lambda names: [n for n in names if not n.startswith("ps_rank") and n != "ps_z"
                               and n not in learn_features.GROUP_TOUCH
                               and n not in learn_features.GROUP_GAP],
    # M5: without the widest (+/-5 s) windows
    "short": lambda names: [n for n in _m4(names) if not n.endswith("_m5")],
    # post-M5: plus touch-sequence and ball-kinematics features around the frame
    "touch": _touch,
    # plus ball-gap and stoppage cues; only for training data with the "speed" ball model
    "gap": lambda names: [n for n in names if n not in learn_features.GROUP_SPEED_RAW],
}
FEATURE_VERSION = 3   # bump when learn_features.build changes
# Caches of the older condition keep their version: the columns the older variants use have not
# changed, and their models (v2, v3) were trained from those caches.
CACHE_VERSION = {"full": 2}


# -- feature cache -------------------------------------------------------------------------------

def cache_path(game, condition, seed):
    return CACHE / "{}_{}_{}_v{}.npz".format(game, condition, seed,
                                             CACHE_VERSION.get(condition, FEATURE_VERSION))


def prepare(game, condition, seed):
    """Features + everything needed to decode and score one (game, condition, seed)."""
    path = cache_path(game, condition, seed)
    if path.is_file():
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    bundle, spec, gt = ev.load(game)
    tracking = ev.tracking_for(bundle, spec, condition, seed)
    kickoffs = halves.detect_kickoffs(tracking, spec)
    F = features.compute(tracking, spec)
    names, X = learn_features.build(tracking, spec, F)
    data = {"X": X, "names": np.array(names), "gt": gt.astype(np.int8),
            "graded": spec.graded_mask(),
            "kickoffs": np.array([[k.half, k.block_first, k.block_last, k.frame]
                                  for k in kickoffs], dtype=np.int64),
            "d_home": F.d_home, "d_away": F.d_away, "out_dist": F.out_dist,
            "ball_speed": F.ball_speed}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def game_list():
    return ev.available_games()


# -- model ---------------------------------------------------------------------------------------

# Training-set variants that emphasise frames near a true change of state (within CHANGE_NEAR
# frames). spellF1 is decided at the changes, but they are a small share of all frames.
#   name -> (feature variant, weight on near-change frames, keep every near-change frame,
#            noise condition of the training and evaluation data)
TRAINING = {
    "touch_w3": ("touch", 3.0, False, "full"),     # every 5th frame, near-change frames x3
    "touch_dense": ("touch", 1.0, True, "full"),   # every near-change frame, every 5th elsewhere
    # Noise condition full2: ball visibility follows player speed as on the real match.
    "touch_f2": ("touch", 1.0, False, "full2"),    # the v3 features on the new data
    "gap_f2": ("gap", 1.0, False, "full2"),        # plus ball-gap and stoppage cues
    # The v3 recipe (trained on full), held-out games scored on full2: the honest reference for
    # the two variants above. A fifth entry is the training condition when it differs.
    "touch_x2": ("touch", 1.0, False, "full2", "full"),
}


def resolve(variant):
    """(feature variant, near-change weight, dense near changes) for a training variant name."""
    return TRAINING.get(variant, (variant, 1.0, False, "full"))[:3]


def condition_of(variant):
    """Noise condition a variant is evaluated under (and trained under, unless stated)."""
    return TRAINING.get(variant, (variant, 1.0, False, "full"))[3]


def train_condition_of(variant):
    """Noise condition of a variant's training copies."""
    entry = TRAINING.get(variant, (variant, 1.0, False, "full"))
    return entry[4] if len(entry) > 4 else entry[3]


def all_variants():
    return sorted(VARIANTS) + sorted(TRAINING)


def training_rows(datasets, keep, near_weight=1.0, dense=False):
    """(X, y, weights or None): subsampled graded rows of several games plus the H <-> A mirrored
    copy. With near_weight > 1 or dense, frames near a true change count for more."""
    xs, ys, ws = [], [], []
    emphasise = near_weight != 1.0 or dense
    for d in datasets:
        names = list(d["names"])
        cols = [names.index(k) for k in keep]
        usable = d["graded"] & (d["gt"] != pm.X)
        if emphasise:
            _, near = change_targets(d["gt"])
            idx = np.arange(len(usable))
            pick = usable & ((idx % SUBSAMPLE == 0) | near) if dense else None
            rows = np.flatnonzero(pick) if dense else np.flatnonzero(usable)[::SUBSAMPLE]
            w = np.where(near[rows], near_weight, 1.0)
            ws += [w, w]
        else:
            rows = np.flatnonzero(usable)[::SUBSAMPLE]
        X = d["X"][rows]
        y = d["gt"][rows].astype(np.int64)
        X_sw = learn_features.swap_teams(names, X)
        y_sw = np.where(y == pm.H, pm.A, np.where(y == pm.A, pm.H, y))
        xs += [X[:, cols], X_sw[:, cols]]
        ys += [y, y_sw]
    return np.vstack(xs), np.concatenate(ys), (np.concatenate(ws) if emphasise else None)


def fit(X, y, rounds=NUM_ROUNDS, num_class=3, weight=None):
    import lightgbm as lgb
    data = lgb.Dataset(X, label=y, weight=weight, params={"verbosity": -1}, free_raw_data=True)
    return lgb.train(dict(LGB_PARAMS, num_class=num_class), data, num_boost_round=rounds)


# -- change-point model ---------------------------------------------------------------------------
# A second model answers "does the state change here, and how?" It never decides a label. Its
# output only makes the decoder's transitions cheaper or dearer frame by frame
# (decode.viterbi_batch_dyn).
CHANGE_HALF_WIDTH = 5     # frames either side of a true change that count as "change here"
CHANGE_NEAR = 25          # every frame this close to a change is kept for training
CHANGE_CLASSES = ("none", "switch", "die", "restart")


def change_targets(gt, half_width=CHANGE_HALF_WIDTH, near_width=CHANGE_NEAR):
    """(y, near): y per frame is 0 none, 1 switch (H<->A), 2 die (alive->D), 3 restart
    (D->alive) within +/- half_width frames of a true change; near marks +/- near_width."""
    n = len(gt)
    y = np.zeros(n, dtype=np.int64)
    near = np.zeros(n, dtype=bool)
    bounds = np.flatnonzero((gt[1:] != gt[:-1]) & (gt[1:] != pm.X) & (gt[:-1] != pm.X)) + 1
    for b in bounds:
        kind = 2 if gt[b] == pm.D else 3 if gt[b - 1] == pm.D else 1
        y[max(0, b - half_width):b + half_width + 1] = kind
        near[max(0, b - near_width):b + near_width + 1] = True
    return y, near


def change_rows(datasets, keep):
    """Training rows for the change model: every frame near a change, every SUBSAMPLE-th
    elsewhere, plus the home/away mirror (the target does not depend on which team is home)."""
    xs, ys = [], []
    for d in datasets:
        names = list(d["names"])
        cols = [names.index(k) for k in keep]
        y_all, near = change_targets(d["gt"])
        idx = np.arange(len(y_all))
        rows = np.flatnonzero(d["graded"] & (d["gt"] != pm.X) & (near | (idx % SUBSAMPLE == 0)))
        X = d["X"][rows]
        xs += [X[:, cols], learn_features.swap_teams(names, X)[:, cols]]
        ys += [y_all[rows], y_all[rows]]
    return np.vstack(xs), np.concatenate(ys)


def change_logratio(raw, prior):
    """(n, 3) log(p_k / prior_k) for switch, die, restart."""
    return (log_softmax(raw)[:, 1:] - np.log(prior[1:])[None, :]).astype(np.float32)


def logo_change(variant="touch", verbose=True):
    """Out-of-fold change log-ratios for every game -> outputs/external/oofc_<variant>.npz."""
    games = game_list()
    train_sets = [prepare(g, "full", TRAIN_SEED + i) for i, g in enumerate(games)]
    eval_sets = [prepare(g, "full", EVAL_SEED + i) for i, g in enumerate(games)]
    keep = VARIANTS[variant](list(train_sets[0]["names"]))
    out = {}
    for i, g in enumerate(games):
        started = time.perf_counter()
        X, y = change_rows([d for j, d in enumerate(train_sets) if j != i], keep)
        booster = fit(X, y, num_class=4)
        prior = np.bincount(y, minlength=4).astype(float) / len(y)
        d = eval_sets[i]
        names = list(d["names"])
        Xe = d["X"][d["graded"]][:, [names.index(k) for k in keep]]
        out[g] = change_logratio(raw_scores(booster, Xe), prior)
        if verbose:
            y_true, _ = change_targets(d["gt"])
            y_true = y_true[d["graded"]]
            top = [float(np.mean(out[g][y_true == k, k - 1] > 0)) for k in (1, 2, 3)]
            print("  fold {:<11} {} rows, {:.0f} s; share of true change frames with ratio > 1: "
                  "switch {:.2f} die {:.2f} restart {:.2f}".format(
                      g, len(y), time.perf_counter() - started, *top), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "oofc_{}.npz".format(variant), **out)
    return out


def raw_scores(booster, X):
    return booster.predict(X, raw_score=True, num_threads=1)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fit_temperature(raws, labels):
    """One temperature minimising the out-of-fold negative log-likelihood (grid search)."""
    grid = np.round(np.arange(0.5, 4.01, 0.1), 2)
    best = min(grid, key=lambda t: -np.mean(
        log_softmax(raws / t)[np.arange(len(labels)), labels]))
    return float(best)


def emissions(raw, temperature, prior):
    """log p(x | s) up to a constant: calibrated log posterior minus log prior."""
    return log_softmax(raw / temperature) - np.log(prior)[None, :]


# -- decoding + scoring --------------------------------------------------------------------------

class _Spec:
    """Just enough of a GameSpec for decode/refine/policy, rebuilt from the cache."""
    def __init__(self, d, fps=25):
        self.fps = fps
        self.n_frames = len(d["gt"])
        self.halves = tuple((int(h), int(a), int(b)) for h, a, b, _ in d["kickoffs"])
        self._graded = d["graded"]

    def graded_mask(self):
        return self._graded


def kickoffs_of(d):
    return [halves.Kickoff(int(h), int(a), int(b), int(f), "cached")
            for h, a, b, f in d["kickoffs"]]


def decode_and_score(d, logp_graded, dec_params, use_refine=True, bootstrap=0):
    spec = _Spec(d)
    kos = kickoffs_of(d)
    logp = np.zeros((spec.n_frames, 3))
    logp[spec.graded_mask()] = logp_graded
    labels = decode.decode(logp, spec, kos, dec_params)
    if use_refine:
        F = features.Features(ball_xy=None, ball_observed=None, ball_known=None,
                              ball_speed=d["ball_speed"], ball_missing_s=None,
                              d_home=d["d_home"], d_away=d["d_away"], out_dist=d["out_dist"],
                              player_speed=None, last_touch=None, since_touch_s=None,
                              shape_diff=None)
        labels = refine.refine(labels, F, spec, refine.RefineParams(), kos)
    labels = halves.apply_policy(labels, spec, kos)
    r = pm.evaluate(d["gt"].astype(np.int8), pm.label_codes(labels), bootstrap_n=bootstrap)
    return {"spell_f1_mean": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
            "spell_bootstrap_low": r["spell_bootstrap_low"],
            "spell_bootstrap_high": r["spell_bootstrap_high"],
            "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"],
            "alive_dead_f1": r["alive_dead_f1"], "team_acc_both_alive": r["team_acc_both_alive"]}


# -- leave-one-game-out --------------------------------------------------------------------------

def logo(variant="all", verbose=True):
    """Out-of-fold raw scores for every game (held-out copy, eval seed)."""
    games = game_list()
    train_sets = [prepare(g, train_condition_of(variant), TRAIN_SEED + i)
                  for i, g in enumerate(games)]
    eval_sets = [prepare(g, condition_of(variant), EVAL_SEED + i) for i, g in enumerate(games)]
    feature_variant, near_weight, dense = resolve(variant)
    keep = VARIANTS[feature_variant](list(train_sets[0]["names"]))
    oof = {}
    for i, g in enumerate(games):
        started = time.perf_counter()
        X, y, w = training_rows([d for j, d in enumerate(train_sets) if j != i], keep,
                                near_weight, dense)
        booster = fit(X, y, weight=w)
        d = eval_sets[i]
        names = list(d["names"])
        Xe = d["X"][d["graded"]][:, [names.index(k) for k in keep]]
        oof[g] = raw_scores(booster, Xe)
        if verbose:
            print("  fold {:<11} trained on {} rows in {:.0f} s".format(
                g, len(y), time.perf_counter() - started), flush=True)
    prior = np.bincount(np.concatenate([d["gt"][d["graded"] & (d["gt"] != pm.X)]
                                        for d in train_sets]), minlength=3)[:3].astype(float)
    prior = (prior + prior[[1, 0, 2]]) / 2.0          # the mirrored copy makes H and A equal
    prior /= prior.sum()
    return games, eval_sets, oof, prior, keep


def calibrate(games, eval_sets, oof):
    raws = np.vstack([oof[g] for g in games])
    labels = np.concatenate([d["gt"][d["graded"]].astype(np.int64) for d in eval_sets])
    keep = labels < 3
    return fit_temperature(raws[keep], labels[keep])


def score_all(games, eval_sets, oof, temperature, prior, dec_params, use_refine=True,
              bootstrap=0):
    rows = []
    for g, d in zip(games, eval_sets):
        r = decode_and_score(d, emissions(oof[g], temperature, prior), dec_params, use_refine,
                             bootstrap)
        rows.append({"game": g, **r})
    return rows


def mean_spell(rows):
    return float(np.mean([r["spell_f1_mean"] for r in rows]))


def final(variant, c_switch, c_dead, temperature):
    """Train on every external game; write models/emission_v2.txt + .json (with provenance)."""
    import lightgbm as lgb

    games = game_list()
    train_sets = [prepare(g, "full", TRAIN_SEED + i) for i, g in enumerate(games)]
    keep = VARIANTS[variant](list(train_sets[0]["names"]))
    X, y, _ = training_rows(train_sets, keep)
    booster = fit(X, y)
    prior = np.bincount(y, minlength=3)[:3].astype(float)
    prior /= prior.sum()
    MODELS.mkdir(exist_ok=True)
    booster.save_model(str(MODELS / "emission_v2.txt"))
    manifests = {name: json.loads((ROOT / "ext" / "manifests" / (name + ".json")).read_text(
        encoding="utf-8"))["ref"] for name in ("metrica", "skillcorner")}
    meta = {"features": keep, "temperature": temperature, "prior": prior.tolist(),
            "decoder": {"c_switch": c_switch, "c_die": c_dead, "c_restart": c_dead},
            # LOGO: boundary snapping lowers v2's spellF1 on 10 of 13 games (0.388 vs 0.395)
            "refine": False,
            "variant": variant, "train_games": games, "rows": int(len(y)),
            "subsample": SUBSAMPLE, "augment_seed_base": TRAIN_SEED,
            "lightgbm": {"version": lgb.__version__, "params": LGB_PARAMS,
                         "num_boost_round": NUM_ROUNDS},
            "sources": manifests,
            "note": "trained only on external games (Metrica open data, SkillCorner open data); "
                    "nothing from the provided match"}
    with open(MODELS / "emission_v2.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote models/emission_v2.txt ({} features, {} rows)".format(len(keep), len(y)))
    return 0


def _provenance(games, n_rows):
    import lightgbm as lgb
    manifests = {name: json.loads((ROOT / "ext" / "manifests" / (name + ".json")).read_text(
        encoding="utf-8"))["ref"] for name in ("metrica", "skillcorner")}
    return {"train_games": games, "rows": int(n_rows), "subsample": SUBSAMPLE,
            "augment_seed_base": TRAIN_SEED, "feature_version": FEATURE_VERSION,
            "lightgbm": {"version": lgb.__version__, "params": LGB_PARAMS,
                         "num_boost_round": NUM_ROUNDS},
            "sources": manifests,
            "note": "trained only on external games (Metrica open data, SkillCorner open data); "
                    "nothing from the provided match"}


def _write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def final_model(variant="touch", with_change=False, version="v3"):
    """Train a shipped model on every external game: models/emission_<version>.* (and
    change_<version>.* on request). v3 is variant `touch` (noise `full`); v4 is `touch_f2` (the
    same features, noise `full2`). Decoding settings are not stored here; they are frozen in
    config/final_<version>.yaml."""
    games = game_list()
    train_sets = [prepare(g, train_condition_of(variant), TRAIN_SEED + i)
                  for i, g in enumerate(games)]
    feature_variant, near_weight, dense = resolve(variant)
    keep = VARIANTS[feature_variant](list(train_sets[0]["names"]))
    MODELS.mkdir(exist_ok=True)

    X, y, w = training_rows(train_sets, keep, near_weight, dense)
    fit(X, y, weight=w).save_model(str(MODELS / "emission_{}.txt".format(version)))
    if w is None:
        prior = np.bincount(y, minlength=3)[:3].astype(float) / len(y)
    else:
        # emphasised sampling distorts the class shares; use the true shares of all graded
        # frames (made symmetric in H and A, as the mirrored training copy is)
        prior = np.bincount(np.concatenate([d["gt"][d["graded"] & (d["gt"] != pm.X)]
                                            for d in train_sets]), minlength=3)[:3].astype(float)
        prior = (prior + prior[[1, 0, 2]]) / 2.0
        prior /= prior.sum()
    _write_json(MODELS / "emission_{}.json".format(version), {
        "features": keep, "prior": prior.tolist(), "variant": variant,
        "near_change_weight": near_weight, "dense_near_changes": dense,
        # recorded from v4 on, so that emission_v3.json still retrains byte-identically
        **({} if version == "v3" else {"noise_condition": train_condition_of(variant)}),
        **_provenance(games, len(y))})
    print("wrote models/emission_{}.txt ({} features, {} rows)".format(
        version, len(keep), len(y)))
    del X, y
    if with_change:
        X, y = change_rows(train_sets, keep)
        fit(X, y, num_class=4).save_model(str(MODELS / "change_{}.txt".format(version)))
        prior = np.bincount(y, minlength=4).astype(float) / len(y)
        _write_json(MODELS / "change_{}.json".format(version), {
            "features": keep, "prior": prior.tolist(), "variant": variant,
            "classes": list(CHANGE_CLASSES), "half_width": CHANGE_HALF_WIDTH,
            "near_width": CHANGE_NEAR, **_provenance(games, len(y))})
        print("wrote models/change_{}.txt ({} rows)".format(version, len(y)))
    return 0


def score_shipped(condition="full2", verbose=True):
    """The frozen v3 model (models/emission_v3.txt, config/final_v3.yaml), scored on the external
    evaluation copies of another noise condition. v3 was trained on every game, so this is not
    leave-one-game-out: it flatters v3, which makes it a conservative reference for a new model."""
    import lightgbm as lgb
    import yaml

    from mtah import emission_v3, tune

    booster = lgb.Booster(model_file=str(emission_v3.EMISSION_TXT))
    meta = json.loads(emission_v3.EMISSION_META.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(emission_v3.CONFIG.read_text(encoding="utf-8"))
    dec = decode.DecoderParams(cfg["decoder"]["c_switch"], cfg["decoder"]["c_die"],
                               cfg["decoder"]["c_restart"])
    rows = []
    for i, g in enumerate(game_list()):
        d = prepare(g, condition, EVAL_SEED + i)
        names = list(d["names"])
        X = d["X"][d["graded"]][:, [names.index(k) for k in meta["features"]]]
        logp = tune.emissions(raw_scores(booster, X), np.asarray(meta["prior"]),
                              cfg["emission"]["temperature"], cfg["emission"]["dead_bias"])
        r = decode_and_score(d, logp, dec, use_refine=bool(cfg["refine"]))
        rows.append({"game": g, **r})
        if verbose:
            print("  {:<11} {:.4f}  poss/gt {:.2f}".format(
                g, r["spell_f1_mean"], r["n_pred_spells"] / r["n_gt_spells"]), flush=True)
    print("frozen v3 on {}: mean spellF1 {:.4f} (Metrica {:.4f}, SkillCorner {:.4f})".format(
        condition, mean_spell(rows),
        mean_spell([r for r in rows if r["game"].startswith("metrica")]),
        mean_spell([r for r in rows if r["game"].startswith("sc_")])))
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "shipped_v3_{}.json".format(condition), rows)
    return rows


COST_GRID = [(cs, cd) for cs in (4.0, 8.0, 15.0, 25.0) for cd in (8.0, 15.0, 25.0)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--condition", default="full", choices=("full", "full2"))
    p = sub.add_parser("score-shipped")
    p.add_argument("--condition", default="full2", choices=("full", "full2"))
    p = sub.add_parser("logo-change")
    p.add_argument("--variant", default="touch", choices=all_variants())
    p = sub.add_parser("final-v3")
    p.add_argument("--variant", default="touch", choices=all_variants())
    p.add_argument("--with-change", action="store_true",
                   help="also train the change-point model (an experiment that is not shipped: "
                        "its gain was not robust, see docs/ablation.md)")
    p = sub.add_parser("final-v4")
    p.add_argument("--variant", default="touch_f2", choices=all_variants())
    p = sub.add_parser("final")
    p.add_argument("--variant", default="all", choices=all_variants())
    p.add_argument("--c-switch", type=float, required=True)
    p.add_argument("--c-dead", type=float, required=True)
    p.add_argument("--temperature", type=float, required=True,
                   help="from the out-of-fold calibration of the same variant (logo_<v>.json)")
    p = sub.add_parser("logo")
    p.add_argument("--variant", default="all", choices=all_variants())
    p.add_argument("--grid", action="store_true", help="also grid the decoder costs on OOF")
    p.add_argument("--c-switch", type=float)
    p.add_argument("--c-dead", type=float)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--reuse", action="store_true",
                   help="reuse saved out-of-fold predictions (decoding-only changes)")
    args = parser.parse_args(argv)

    if args.cmd == "score-shipped":
        score_shipped(args.condition)
        return 0

    if args.cmd == "final-v3":
        return final_model(args.variant, args.with_change, "v3")

    if args.cmd == "final-v4":
        return final_model(args.variant, False, "v4")

    if args.cmd == "logo-change":
        logo_change(args.variant)
        return 0

    if args.cmd == "final":
        return final(args.variant, args.c_switch, args.c_dead, args.temperature)

    if args.cmd == "cache":
        for i, g in enumerate(game_list()):
            t = time.perf_counter()
            prepare(g, args.condition, TRAIN_SEED + i)
            prepare(g, args.condition, EVAL_SEED + i)
            print("cached {} ({}) in {:.0f} s".format(g, args.condition, time.perf_counter() - t),
                  flush=True)
        return 0

    oof_file = OUT / "oof_{}.npz".format(args.variant)
    if args.reuse and oof_file.is_file():
        games = game_list()
        eval_sets = [prepare(g, condition_of(args.variant), EVAL_SEED + i)
                     for i, g in enumerate(games)]
        saved = np.load(oof_file)
        oof = {g: saved[g] for g in games}
        prior, temperature = saved["prior"], float(saved["temperature"])
        keep = VARIANTS[resolve(args.variant)[0]](list(eval_sets[0]["names"]))
        print("variant {}: reusing out-of-fold predictions".format(args.variant), flush=True)
    else:
        games, eval_sets, oof, prior, keep = logo(args.variant)
        temperature = calibrate(games, eval_sets, oof)
        print("variant {}: {} features, prior {}, temperature {}".format(
            args.variant, len(keep), np.round(prior, 3).tolist(), temperature), flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof_file, prior=prior, temperature=temperature,
                            **{g: oof[g] for g in games})
    costs = [(args.c_switch, args.c_dead)] if args.c_switch else \
        (COST_GRID if args.grid else [(decode.DecoderParams().c_switch,
                                       decode.DecoderParams().c_die)])
    results = []
    for cs, cd in costs:
        dp = decode.DecoderParams(c_switch=cs, c_die=cd, c_restart=cd)
        rows = score_all(games, eval_sets, oof, temperature, prior, dp, not args.no_refine,
                         args.bootstrap)
        results.append({"c_switch": cs, "c_dead": cd, "mean_spell_f1": mean_spell(rows),
                        "rows": rows})
        print("  c_switch {:>5} c_dead {:>5}  mean spellF1 {:.4f}  (Metrica {:.4f}, "
              "SkillCorner {:.4f})".format(
                  cs, cd, mean_spell(rows),
                  mean_spell([r for r in rows if r["game"].startswith("metrica")]),
                  mean_spell([r for r in rows if r["game"].startswith("sc_")])), flush=True)
    with open(OUT / "logo_{}{}.json".format(args.variant, "_norefine" if args.no_refine else ""),
              "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"variant": args.variant, "features": keep, "temperature": temperature,
                   "prior": prior.tolist(), "results": results}, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
