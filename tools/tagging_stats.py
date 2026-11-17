"""Does a submission line up with the Smart Tagging? Label-free.

Checks there's a dead ball before each tagged set piece, and how our team calls sit inside the
tagged phases. Aggregates only.

    .venv\\Scripts\\python.exe tools\\tagging_stats.py --data-dir C:\\Hackathon\\mtah-data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import io_atd, io_tagging  # noqa: E402
from mtah.official import pm  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

WINDOWS_S = (0.0, 2.0, 5.0)      # tolerance around the tagged interval when looking for a restart
LOOKBACK_S = 10.0                # "no dead ball at all" looks this far before the tagged start


def restarts_and_deaths(codes, spec):
    """Frames at which our labels go dead -> alive (restart) and alive -> dead, per graded half."""
    restarts, deaths = [], []
    for _, first, last in spec.halves:
        c = codes[first:last + 1]
        dead = c == pm.D
        change = np.flatnonzero(dead[1:] != dead[:-1]) + 1
        for k in change:
            (restarts if dead[k - 1] else deaths).append(first + int(k))
    return np.array(restarts, dtype=np.int64), np.array(deaths, dtype=np.int64)


def gap_to_interval(frames, start, end):
    """Distance in frames from each of `frames` to the interval [start, end] (0 inside)."""
    return np.maximum(0, np.maximum(start - frames, frames - end))


def collect(codes, intervals, spec, sides):
    fps = spec.fps
    graded = spec.graded_mask()
    restarts, _ = restarts_and_deaths(codes, spec)
    inside = [iv for iv in intervals if 0 <= iv.start < spec.n_frames and graded[iv.start]]
    pieces = [iv for iv in inside if iv.is_set_piece and iv.kind != io_tagging.HALF_START_TYPE]

    rows, offsets = [], []
    for iv in pieces:
        gap = gap_to_interval(restarts, iv.start, iv.end) if len(restarts) else np.array([10 ** 9])
        nearest = restarts[int(np.argmin(gap))] if len(restarts) else None
        before = codes[max(0, iv.start - int(3 * fps)):iv.start]
        span = codes[max(0, iv.start - int(LOOKBACK_S * fps)):min(spec.n_frames, iv.end)]
        row = {"kind": iv.kind or "(no type)",
               "near": [bool(gap.min() <= w * fps) for w in WINDOWS_S],
               "dead_before_3s": float(np.mean(before == pm.D)) if len(before) else 0.0,
               "no_dead_at_all": not bool(np.any(span == pm.D)),
               "alive_frames_2s_before": int(np.sum(codes[max(0, iv.start - 2 * fps):iv.start]
                                                    != pm.D))}
        rows.append(row)
        if nearest is not None and gap.min() <= LOOKBACK_S * fps:
            offsets.append((nearest - iv.start) / float(fps))

    # our dead-ball runs of at least 1 s: is their restart tagged?
    ours = []
    for _, first, last in spec.halves:
        for a, b, k in pm.segments(codes, [(first, last + 1)]):
            if k == pm.D and b - a >= fps and b <= last:
                ours.append(b)
    ours = np.array(ours, dtype=np.int64)
    tagged = []
    for w in WINDOWS_S:
        hit = np.zeros(len(ours), dtype=bool)
        for iv in pieces:
            hit |= gap_to_interval(ours, iv.start, iv.end) <= w * fps
        tagged.append(int(hit.sum()))

    # phases: how our labels sit inside each code that carries a team
    phases = {}
    for iv in inside:
        side = io_tagging.side_of(iv, sides)
        if iv.is_set_piece or side is None:
            continue
        c = codes[iv.start:min(iv.end, spec.n_frames)]
        c = c[graded[iv.start:min(iv.end, spec.n_frames)]]
        p = phases.setdefault(iv.code, {"n": 0, "frames": 0, "dead": 0, "alive": 0, "same": 0})
        p["n"] += 1
        p["frames"] += len(c)
        p["dead"] += int(np.sum(c == pm.D))
        p["alive"] += int(np.sum(c != pm.D))
        p["same"] += int(np.sum(c == (pm.H if side == io_tagging.HOME else pm.A)))
    # the team prior a model could take from the tags (io_tagging.team_evidence)
    evidence = io_tagging.team_evidence(intervals, sides, spec.n_frames) if sides else None
    prior = None
    if evidence is not None:
        alive = graded & (codes != pm.D)
        covered = alive & (evidence != io_tagging.NONE)
        on = graded & (evidence != io_tagging.NONE)
        runs = int(np.sum(on[1:] & ~on[:-1]) + int(on[0]))
        prior = {"coverage": covered.sum() / max(1, alive.sum()),
                 "agreement": float(np.mean(codes[covered] == evidence[covered]))
                 if covered.any() else 0.0,
                 "runs": runs, "mean_run_s": on.sum() / max(1, runs) / float(fps),
                 "dead_inside": float(np.mean(codes[on] == pm.D)) if on.any() else 0.0}
    return {"prior": prior,
            "n_instances": len(intervals), "n_inside": len(inside), "pieces": rows,
            "offsets_s": offsets, "n_restarts": int(len(restarts)), "our_dead_runs": int(len(ours)),
            "our_dead_runs_tagged": tagged, "phases": phases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(ROOT / "outputs" / "real" / "submission.csv"))
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    args = parser.parse_args(argv)
    if not args.data_dir:
        print("no data directory: pass --data-dir or set MTAH_DATA_DIR")
        return 2
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
    table = pm.read_label_csv(args.file)
    reason = pm.validate_submission(table, spec.n_frames)
    if reason:
        print("INVALID: {}".format(reason))
        return 1
    codes = np.asarray(pm.label_codes(table))
    paths = spec.paths(args.data_dir)
    intervals = io_tagging.load(paths["smart_tagging"], spec.fps)
    meta = io_atd.read_meta(paths["atd_meta"])
    sides = io_tagging.team_sides(intervals, meta.home_name, meta.away_name)
    r = collect(codes, intervals, spec, sides)

    print("tagging stats  {}  ({} tagged intervals, {} starting on graded frames)".format(
        Path(args.file).name, r["n_instances"], r["n_inside"]))
    print("team names mapped to home/away: {}".format("yes" if sides else "NO (phase table skipped)"))

    rows = r["pieces"]
    print("\nset pieces = tagged restarts (half-start kickoffs excluded): {}".format(len(rows)))
    print("our labels: {} restarts (dead -> alive), {} dead-ball runs of 1 s or more".format(
        r["n_restarts"], r["our_dead_runs"]))
    if rows:
        head = "  {:<24}{:>4}".format("type", "n") + "".join(
            "{:>12}".format("rst +/-{:g}s".format(w)) for w in WINDOWS_S) + "{:>12}{:>12}".format(
                "dead 3s bef", "no dead")
        print(head)
        kinds = sorted({row["kind"] for row in rows}) + ["ALL"]
        for kind in kinds:
            sel = [row for row in rows if kind == "ALL" or row["kind"] == kind]
            cells = "".join("{:>12}".format("{}/{}".format(
                sum(row["near"][i] for row in sel), len(sel))) for i in range(len(WINDOWS_S)))
            print("  {:<24}{:>4}{}{:>12.2f}{:>12}".format(
                kind, len(sel), cells, float(np.mean([row["dead_before_3s"] for row in sel])),
                sum(row["no_dead_at_all"] for row in sel)))
        print("  rst +/-Ws = set pieces with one of our restarts inside the tagged interval widened "
              "by W s")
        print("  dead 3s bef = mean share of our labels that are dead in the 3 s before the tagged "
              "start")
        print("  no dead = set pieces with no dead frame of ours from {:g} s before the tagged start "
              "to its end".format(LOOKBACK_S))
        print("  frames we call alive in the 2 s before a tagged start: {}".format(
            sum(row["alive_frames_2s_before"] for row in rows)))
    if r["offsets_s"]:
        q = np.percentile(r["offsets_s"], [10, 25, 50, 75, 90])
        print("\nour nearest restart minus the tagged start, seconds ({} set pieces with a restart "
              "within {:g} s)".format(len(r["offsets_s"]), LOOKBACK_S))
        print("  p10 {:+.1f}  p25 {:+.1f}  p50 {:+.1f}  p75 {:+.1f}  p90 {:+.1f}".format(*q))
    if r["our_dead_runs"]:
        print("\nour dead-ball runs whose restart is tagged as a set piece: " + "  ".join(
            "+/-{:g}s {}/{}".format(w, n, r["our_dead_runs"])
            for w, n in zip(WINDOWS_S, r["our_dead_runs_tagged"])))

    if sides and r["phases"]:
        print("\ntagged phases with a team (our labels inside them)")
        print("  {:<32}{:>5}{:>9}{:>9}{:>16}".format("code", "n", "frames", "dead", "tagged team has"))
        for code in sorted(r["phases"], key=lambda c: -r["phases"][c]["frames"]):
            p = r["phases"][code]
            print("  {:<32}{:>5}{:>9}{:>9.3f}{:>16.3f}".format(
                code.upper(), p["n"], p["frames"], p["dead"] / max(1, p["frames"]),
                p["same"] / max(1, p["alive"])))
        print("  dead = share of the phase's frames we call dead; tagged team has = share of the "
              "frames we\n  call alive on which we give the ball to the phase's tagged team")
    if r["prior"]:
        q = r["prior"]
        print()
        print("team evidence from the in-possession phases ({}), conflicts removed".format(
            ", ".join(c.upper() for c in io_tagging.IN_POSSESSION)))
        print("  covers {:.3f} of the frames we call alive; we agree with it on {:.3f} of those"
              .format(q["coverage"], q["agreement"]))
        print("  {} stretches of mean {:.1f} s; we call {:.3f} of their frames dead".format(
            q["runs"], q["mean_run_s"], q["dead_inside"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
