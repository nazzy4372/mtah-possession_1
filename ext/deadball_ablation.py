"""Does the slowdown detector (src/mtah/deadball.py) help? Ablation on the 13 external games.

    python -m ext.deadball_ablation [--games N]

Short answer: no (docs/ablation.md).
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
from ext import tags, train  # noqa: E402
from mtah import deadball, decode, emission, features, halves, refine, tune  # noqa: E402
from mtah.deadball import DeadballParams  # noqa: E402
from mtah.official import pm  # noqa: E402
from mtah.pipeline import Params  # noqa: E402

OUT = ROOT / "outputs" / "external"
V4_WEIGHTS = (0.5, 1.0, 2.0, 4.0)


def metrics(gt, codes, graded):
    r = pm.evaluate(gt, codes)
    pred_d, true_d = (codes == pm.D) & graded, (gt == pm.D) & graded
    both = float(np.sum(pred_d & true_d))
    return {"precision": both / max(1.0, float(pred_d.sum())),
            "recall": both / max(1.0, float(true_d.sum())),
            "alive_dead_f1": r["alive_dead_f1"], "spell": r["spell_f1_mean"]}


def run(n_games=None, verbose=True):
    games, eval_sets, oof, prior = tune.load_oof("touch_f2")
    s4 = tags.v4_settings()
    dp4 = decode.DecoderParams(s4["c_switch"], s4["c_die"], s4["c_restart"])
    base = Params()
    variants = {"v1 + detector": DeadballParams(),
                "v1 + detector, no box rule": DeadballParams(box_exclusion=False),
                "v1 + detector, no cue guard": DeadballParams(require_cue=False)}
    rows, stats = {}, {}

    def add(name, value):
        rows.setdefault(name, []).append(value)

    for i, (g, d) in enumerate(list(zip(games, eval_sets))[:n_games]):
        started = time.perf_counter()
        bundle, spec, gt = ev.load(g)
        gt = gt.astype(np.int8)
        graded = spec.graded_mask()
        kos = train.kickoffs_of(d)
        tracking = ev.tracking_for(bundle, spec, "full2", train.EVAL_SEED + games.index(g))
        F = features.compute(tracking, spec)
        rules = emission.rules_v1(F, base.emission)

        def label_v1(logp):
            raw = decode.decode(logp, spec, kos, base.decoder)
            raw = refine.refine(raw, F, spec, base.refine, kos)
            return np.asarray(pm.label_codes(halves.apply_policy(raw, spec, kos)))

        add("rules v1", metrics(gt, label_v1(rules), graded))
        evidences = {}
        for name, prm in variants.items():
            evidences[name] = deadball.compute(tracking, spec, prm)
            add(name, metrics(gt, label_v1(deadball.add_evidence(rules, evidences[name],
                                                                 prm.weight)), graded))
        e = evidences["v1 + detector"]
        for k, v in e.stats.items():
            stats[k] = stats.get(k, 0) + v
        fired, true_d = (e.score > 0) & graded, (gt == pm.D) & graded
        add("detector alone", {"precision": float(np.sum(fired & true_d)) / max(1, fired.sum()),
                               "recall": float(np.sum(fired & true_d)) / max(1, true_d.sum()),
                               "alive_dead_f1": float("nan"), "spell": float("nan")})

        logp4 = np.zeros((spec.n_frames, 3))
        logp4[graded] = tune.emissions(oof[g], prior, s4["temperature"], s4["dead_bias"])

        def label_v4(logp):
            raw = decode.decode(logp, spec, kos, dp4)
            return np.asarray(pm.label_codes(halves.apply_policy(raw, spec, kos)))

        add("v4", metrics(gt, label_v4(logp4), graded))
        for w in V4_WEIGHTS:
            add("v4 + detector (weight {:g})".format(w),
                metrics(gt, label_v4(deadball.add_evidence(logp4, e, w)), graded))
        no_box = evidences["v1 + detector, no box rule"]
        add("v4 + detector, no box rule (weight 1)",
            metrics(gt, label_v4(deadball.add_evidence(logp4, no_box, 1.0)), graded))
        if verbose:
            print("{} done in {:.0f} s".format(g, time.perf_counter() - started), flush=True)

    n = len(next(iter(rows.values())))
    print("\nexternal games: {} (noise full2), means".format(n))
    print("{:<40} {:>10} {:>8} {:>14} {:>9} {:>14}".format(
        "variant", "D precision", "D recall", "alive/dead F1", "spellF1", "better than base"))
    table = {}
    for name, vals in rows.items():
        mean = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        ref = rows["v4" if name.startswith("v4") else "rules v1"]
        better = int(np.sum(np.array([v["spell"] for v in vals])
                            > np.array([v["spell"] for v in ref])))
        table[name] = dict(mean, better=better, per_game=[v["spell"] for v in vals])
        print("{:<40} {:>10.3f} {:>8.3f} {:>14.3f} {:>9.4f} {:>14}".format(
            name, mean["precision"], mean["recall"], mean["alive_dead_f1"], mean["spell"],
            "-" if name in ("rules v1", "v4", "detector alone") else "{}/{}".format(better, n)))
    print("\ndetector on all games: " + "  ".join("{} {}".format(k, v) for k, v in stats.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "deadball_ablation.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"games": games[:n_games], "table": table, "stats": stats}, handle, indent=2,
                  sort_keys=True)
        handle.write("\n")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, help="only the first N games (quick check)")
    args = parser.parse_args(argv)
    run(args.games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
