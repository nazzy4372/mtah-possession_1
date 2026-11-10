"""Score labelling methods on external games with the official metric.

    python tasks.py external [--games 1 2 3] [--bootstrap 200]

For each converted external game (ext_cache/bundles/<name>/) and each condition:
  clean      the external tracking as converted;
  balldrop   plus real-calibrated ball dropout (ext/augment.py, fixed seed per game);
it runs every method (the ported upstream baseline and our v1), scores against the event-derived
ground truth with upstream possession_metrics.evaluate (imported, never reimplemented) and prints
spellF1 with a bootstrap interval, macroF1 and possession counts. Results go to
outputs/external/results.json as well.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ext import augment  # noqa: E402
from mtah import halves, io_atd, pipeline  # noqa: E402
from mtah.official import pm  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

BUNDLES = ROOT / "ext_cache" / "bundles"
OUT = ROOT / "outputs" / "external"
CONDITIONS = ("clean", "balldrop")


def load(name):
    path = BUNDLES / name
    spec = GameSpec.load(path / "game_spec.json")
    gt = pm.label_codes(pm.read_label_csv(path / "gt.csv"))
    return path, spec, gt


def tracking_for(path, spec, condition, seed):
    tracking = io_atd.load_game(path, spec)
    if condition == "balldrop":
        augment.ball_dropout(tracking, seed=seed)
    return tracking


def score(gt, labels, bootstrap):
    r = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=bootstrap)
    keys = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1",
            "n_pred_spells", "n_gt_spells", "mean_spell_duration_s", "gt_mean_spell_duration_s",
            "alive_dead_f1", "team_acc_both_alive")
    out = {k: r[k] for k in keys}
    out["recall_D"] = r["per_class"]["D"]["recall"]
    return out


def evaluate(games=("metrica_g1", "metrica_g2", "metrica_g3"), methods=("baseline", "v1"),
             conditions=CONDITIONS, params=pipeline.Params(), bootstrap=200, verbose=True):
    rows = []
    for gi, name in enumerate(games):
        path, spec, gt = load(name)
        for condition in conditions:
            tracking = tracking_for(path, spec, condition, seed=1000 + gi)
            kickoffs = halves.detect_kickoffs(tracking, spec)
            for method in methods:
                started = time.perf_counter()
                labels, _ = pipeline.label_tracking(tracking, spec, method, params, kickoffs)
                row = {"game": name, "condition": condition, "method": method,
                       **score(gt, labels, bootstrap),
                       "seconds": round(time.perf_counter() - started, 1)}
                rows.append(row)
                if verbose:
                    print_row(row)
    return rows


HEADER = "{:<11} {:<9} {:<9} {:>7} {:>15} {:>7} {:>9} {:>7} {:>7} {:>6}".format(
    "game", "cond", "method", "spellF1", "bootstrap 5-95", "macroF1", "poss/gt", "aliveF1",
    "teamAcc", "D rec")


def print_row(r):
    if not getattr(print_row, "header_done", False):
        print(HEADER)
        print_row.header_done = True
    ci = "{:.3f}-{:.3f}".format(r["spell_bootstrap_low"], r["spell_bootstrap_high"]) \
        if r["spell_bootstrap_low"] is not None else "--"
    print("{:<11} {:<9} {:<9} {:>7.4f} {:>15} {:>7.4f} {:>9} {:>7.3f} {:>7.3f} {:>6.3f}".format(
        r["game"], r["condition"], r["method"], r["spell_f1_mean"], ci, r["macro_f1"],
        "{}/{}".format(r["n_pred_spells"], r["n_gt_spells"]), r["alive_dead_f1"],
        r["team_acc_both_alive"], r["recall_D"]))


def summarise(rows):
    print("\nmean over games")
    for condition in sorted({r["condition"] for r in rows}):
        for method in sorted({r["method"] for r in rows}):
            sel = [r for r in rows if r["condition"] == condition and r["method"] == method]
            if sel:
                print("  {:<9} {:<9} spellF1 {:.4f}  macroF1 {:.4f}  poss/gt {:.2f}".format(
                    condition, method, np.mean([r["spell_f1_mean"] for r in sel]),
                    np.mean([r["macro_f1"] for r in sel]),
                    np.mean([r["n_pred_spells"] / r["n_gt_spells"] for r in sel])))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args(argv)
    rows = evaluate(["metrica_g{}".format(g) for g in args.games], conditions=args.conditions,
                    bootstrap=args.bootstrap)
    summarise(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
