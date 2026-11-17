"""SkillCorner phases of play written out as a Smart-Tagging-like file, to test the v5 tag prior.

    python -m ext.tags write | experiment | freeze

build_up -> BUILD UP, create -> PROGRESSION, finish -> CREATING CHANCES, transition/quick_break
-> ATTACKING TRANSITION. These phases are cleaner than a real tagger, so we make them worse
first (shifted boundaries, dropped phases, wrong team) and pick a weight that never loses.
Only the team call is tested; dead ball would be circular here.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from ext import evaluate as ev  # noqa: E402
from ext import skillcorner as sc  # noqa: E402
from ext import train  # noqa: E402
from mtah import decode, halves, io_atd, io_tagging, tune  # noqa: E402
from mtah.official import pm  # noqa: E402

CODE_OF = {"build_up": "BUILD UP", "create": "PROGRESSION", "finish": "CREATING CHANCES",
           "transition": "ATTACKING TRANSITION", "quick_break": "ATTACKING TRANSITION",
           "direct": "LONG BALL"}
OUT = ROOT / "outputs" / "external"

# tag quality: (boundary jitter s, share of phases dropped, share given to wrong team).
# real-like ~ what tools/tagging_stats.py shows on the real match
SCENARIOS = (("clean", 0.0, 0.0, 0.0), ("real-like", 1.0, 0.22, 0.0),
             ("10% wrong team", 1.0, 0.22, 0.10), ("20% wrong team", 1.0, 0.22, 0.20),
             ("30% wrong team", 1.0, 0.22, 0.30))
WEIGHTS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
SEED = 7000


def games():
    return [g for g in train.game_list() if g.startswith("sc_")]


def tagging_path(game):
    bundle, spec, _ = ev.load(game)
    return spec.paths(bundle)["smart_tagging"]


def write_tagging(game):
    """SportsCode XML of the game's SkillCorner phases, on the bundle's video clock."""
    match_id = game.split("_", 1)[1]
    bundle, spec, _ = ev.load(game)
    folder = sc.CACHE / match_id
    meta = json.loads((folder / "{}_match.json".format(match_id)).read_text(encoding="utf-8"))
    home_id = meta["home_team"]["id"]
    first = {int(p["period"]): int(p["start_frame"]) for p in meta["match_periods"]}
    block = {int(h): int(a) for h, a, _ in spec.halves}
    names = io_atd.read_meta(spec.paths(bundle)["atd_meta"])
    rows = []
    with open(folder / "{}_phases_of_play.csv".format(match_id), encoding="utf-8") as fh:
        for p in csv.DictReader(fh):
            code = CODE_OF.get(p["team_in_possession_phase_type"])
            period = int(p["period"])
            if code is None or period not in block:
                continue
            start, end = (block[period] + (int(p[k]) - first[period]) * sc.FPS / float(sc.SRC_FPS)
                          for k in ("frame_start", "frame_end"))
            team = names.home_name if int(p["team_in_possession_id"]) == home_id else names.away_name
            rows.append((start / sc.FPS, (end + sc.FPS / float(sc.SRC_FPS)) / sc.FPS, code, team,
                         "1st Half" if period == 1 else "2nd Half"))
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "<ALL_INSTANCES>"]
    for i, (start, end, code, team, half) in enumerate(rows):
        lines.append("  <instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                     "<label><group>Team</group><text>{}</text></label>"
                     "<label><group>Half</group><text>{}</text></label></instance>".format(
                         i + 1, start, end, code, team, half))
    lines += ["</ALL_INSTANCES>", "</file>", ""]
    path = tagging_path(game)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return path, len(rows)


def load_intervals(game):
    """(in-possession intervals, {team name: side}, spec) for a SkillCorner game."""
    bundle, spec, _ = ev.load(game)
    intervals = io_tagging.load(tagging_path(game), spec.fps)
    names = io_atd.read_meta(spec.paths(bundle)["atd_meta"])
    sides = io_tagging.team_sides(intervals, names.home_name, names.away_name)
    if not sides:
        raise ValueError("{}: team names of the tagging do not map to home/away".format(game))
    return intervals, sides, spec


def degrade(intervals, sides, fps, jitter_s, drop, flip, seed):
    """A worse copy of the stream: boundaries moved, phases dropped, teams swapped."""
    rng = np.random.default_rng(seed)
    other = {a: b for a, b in zip(sides, list(sides)[::-1])}
    out = []
    for iv in intervals:
        if rng.random() < drop:
            continue
        start = iv.start + int(round(rng.normal(0.0, jitter_s) * fps))
        end = iv.end + int(round(rng.normal(0.0, jitter_s) * fps))
        if end <= start:
            continue
        tags = dict(iv.tags)
        if rng.random() < flip and tags.get("team") in other:
            tags["team"] = other[tags["team"]]
        out.append(io_tagging.Interval(iv.code, start, end, tags))
    return out


def evidence_for(game, scenario, seed):
    intervals, sides, spec = load_intervals(game)
    _, jitter, drop, flip = scenario
    if jitter or drop or flip:
        intervals = degrade(intervals, sides, spec.fps, jitter, drop, flip, seed)
    return io_tagging.team_evidence(intervals, sides, spec.n_frames)


def v4_settings():
    from mtah import emission_v4
    return emission_v4.settings()


def decode_games(names, eval_sets, oof, prior, s, evidences, weight):
    """Labels of each game: the frozen v4 emissions and decoder, plus the tagging prior."""
    dp = decode.DecoderParams(s["c_switch"], s["c_die"], s["c_restart"])
    jobs = []
    for g, d, evidence in zip(names, eval_sets, evidences):
        spec = train._Spec(d)
        logp = np.zeros((spec.n_frames, 3))
        logp[spec.graded_mask()] = tune.emissions(oof[g], prior, s["temperature"], s["dead_bias"])
        if weight:
            logp = io_tagging.add_team_prior(logp, evidence, weight)
        jobs.append((logp, spec, train.kickoffs_of(d), dp))
    labels = decode.decode_many(jobs)
    return [np.asarray(pm.label_codes(halves.apply_policy(lab, train._Spec(d), train.kickoffs_of(d))))
            for lab, d in zip(labels, eval_sets)]


def stream_stats(codes, evidence, gt, graded):
    """Label-free: share of our alive frames the tags cover, and our agreement with them there.
    With the truth: how often the tags name the right team on the true alive frames they cover."""
    alive = graded & (codes != pm.D)
    covered = alive & (evidence != io_tagging.NONE)
    true_cov = graded & (gt != pm.D) & (gt != pm.X) & (evidence != io_tagging.NONE)
    return (covered.sum() / max(1, alive.sum()),
            float(np.mean(codes[covered] == evidence[covered])) if covered.any() else 0.0,
            float(np.mean(gt[true_cov] == evidence[true_cov])) if true_cov.any() else 0.0)


def experiment(verbose=True):
    names = games()
    all_games, all_sets, oof, prior = tune.load_oof("touch_f2")
    eval_sets = [all_sets[all_games.index(g)] for g in names]
    s = v4_settings()
    result = {"weights": list(WEIGHTS), "scenarios": {}, "games": names}

    def score(codes):
        scores = [pm.evaluate(d["gt"].astype(np.int8), c) for d, c in zip(eval_sets, codes)]
        return {"spell": [r["spell_f1_mean"] for r in scores],
                "macro": [r["macro_f1"] for r in scores],
                "ratio": [r["n_pred_spells"] / float(r["n_gt_spells"]) for r in scores]}

    base_codes = decode_games(names, eval_sets, oof, prior, s, [None] * len(names), 0.0)
    base = score(base_codes)                       # the frozen v4, no tags
    for k, scenario in enumerate(SCENARIOS):
        evidences = [evidence_for(g, scenario, SEED + 100 * k + i) for i, g in enumerate(names)]
        rows = [base if w == 0.0 else
                score(decode_games(names, eval_sets, oof, prior, s, evidences, w))
                for w in WEIGHTS]
        stats = np.array([stream_stats(c, e, d["gt"].astype(np.int8), d["graded"])
                          for c, e, d in zip(base_codes, evidences, eval_sets)])
        result["scenarios"][scenario[0]] = {
            "coverage_of_our_alive": float(stats[:, 0].mean()),
            "our_agreement_inside": float(stats[:, 1].mean()),
            "tag_team_accuracy": float(stats[:, 2].mean()), "rows": rows}
        if verbose:
            print()
            print("{}: tags cover {:.0%} of our alive frames, we agree with them on {:.1%}, "
                  "they name the true team on {:.1%}".format(scenario[0], *stats.mean(axis=0)))
            print("  {:>7} {:>8} {:>8} {:>9} {:>9}".format(
                "weight", "spellF1", "macroF1", "poss/gt", "better"))
            for w, row in zip(WEIGHTS, rows):
                better = int(np.sum(np.array(row["spell"]) > np.array(base["spell"])))
                print("  {:>7.2f} {:>8.4f} {:>8.4f} {:>9.2f} {:>9}".format(
                    w, np.mean(row["spell"]), np.mean(row["macro"]), np.mean(row["ratio"]),
                    "{}/{}".format(better, len(names)) if w else "-"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tag_experiment.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


# plausible = about where our real-match agreement with the tags sits
PLAUSIBLE = ("clean", "real-like", "10% wrong team")


def choose_weight(result):
    """The weight rule, fixed before the real run: (1) the mean spellF1 gain must not be negative
    in ANY scenario, including the implausibly bad ones; (2) among those weights, take the one
    whose smallest mean gain over the plausible scenarios is largest. Returns (weight, table)."""
    weights = result["weights"]
    gain = {name: [float(np.mean(row["spell"]) - np.mean(sc["rows"][0]["spell"]))
                   for row in sc["rows"]] for name, sc in result["scenarios"].items()}
    best, best_floor = 0.0, 0.0
    for k, w in enumerate(weights):
        if w == 0.0 or any(gain[name][k] < 0.0 for name in gain):
            continue
        floor = min(gain[name][k] for name in PLAUSIBLE)
        if floor > best_floor:
            best, best_floor = w, floor
    return best, gain


def freeze():
    path = OUT / "tag_experiment.json"
    if not path.is_file():
        raise FileNotFoundError("{} missing: run python -m ext.tags experiment".format(path))
    result = json.loads(path.read_text(encoding="utf-8"))
    weight, gain = choose_weight(result)
    k = result["weights"].index(weight)
    lines = [
        "# Model v5 = model v4 (models/emission_v4.txt, config/final_v4.yaml) plus a team prior",
        "# from the match's Smart Tagging. Written by `python -m ext.tags freeze`.",
        "# The weight was chosen on 10 SkillCorner games, whose automatic phases of play stand in",
        "# for the tagging, degraded to be no better than the real stream (ext/tags.py).",
        "# Rule: no mean loss in any tag-quality scenario; then the best worst case over the",
        "# plausible ones. Nothing here was chosen with the provided match's labels or the",
        "# public 5-minute window.",
        "base_model: v4",
        "tagging:",
        "  weight: {:.4f}".format(weight),
        "validation:",
        "  games: {}".format(len(result["games"])),
        "  mean_spell_f1_gain_by_scenario:",
    ] + ["    {}: {:+.4f}".format(name.replace(" ", "_").replace("%", "pct"), gain[name][k])
         for name in result["scenarios"]] + [""]
    target = ROOT / "config" / "final_v5.yaml"
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print("weight {:.2f}; wrote {}".format(weight, target))
    return weight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("write", "experiment", "freeze"))
    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        freeze()
        return 0
    if args.cmd == "write":
        for g in games():
            path, n = write_tagging(g)
            print("{}: {} phases -> {}".format(g, n, path.name))
        return 0
    experiment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
