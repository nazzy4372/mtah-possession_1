"""Check our reader + nearest-player codes against the upstream baseline. Prints one line."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import baseline_port, io_atd  # noqa: E402
from mtah.official import baseline  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402


def compare(data_dir, spec):
    paths = spec.paths(data_dir)
    started = time.perf_counter()
    upstream_teams = baseline.team_of_each_track(str(paths["atd_meta"]))
    upstream = baseline.nearest_player_labels(str(paths["atd_raw"]), upstream_teams,
                                              spec.n_frames, spec.pitch)
    upstream_s = time.perf_counter() - started

    started = time.perf_counter()
    tracking = io_atd.load_game(data_dir, spec)
    ours = baseline_port.nearest_player_codes(tracking)
    ours_s = time.perf_counter() - started
    return {
        "slot_teams_identical": bool(np.array_equal(upstream_teams, tracking.slot_team)),
        "n_slots": int(len(upstream_teams)),
        "identical_frames": int(np.sum(upstream == ours)),
        "n_frames": int(spec.n_frames),
        "layout_check": tracking.meta.layout_check,
        "upstream_s": upstream_s,
        "ours_s": ours_s,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("pass --data-dir or set MTAH_DATA_DIR")
    spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
    r = compare(args.data_dir, spec)
    print("slot teams identical to upstream: {} ({} slots)".format(
        r["slot_teams_identical"], r["n_slots"]))
    print("nearest-player codes identical on {} / {} frames".format(
        r["identical_frames"], r["n_frames"]))
    print("metadata channel layout: {}".format(r["layout_check"]))
    print("time: upstream reader {:.1f} s, ours {:.1f} s".format(r["upstream_s"], r["ours_s"]))
    ok = r["slot_teams_identical"] and r["identical_frames"] == r["n_frames"]
    print("PARITY OK" if ok else "PARITY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
