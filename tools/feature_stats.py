"""Label-free distributions of the model's own features, for checking the domain gap.

Prints percentiles and shares over graded frames, never per-frame values. The same script runs on
the real match (by the human) and on external games (clean or augmented), so the two can be
compared directly. It needs no labels.

    .venv\\Scripts\\python.exe tools\\feature_stats.py --data-dir C:\\Hackathon\\mtah-data
    python tools/feature_stats.py --external metrica_g1 --condition full

Reported: median player speed (the main dead-ball cue), ball speed, nearest-player distance to the
ball, proximity margin, ball outside the pitch, shape difference, last-touch coverage, and the v1
emission outputs: the share of frames the rules call dead, and how decisive the team call is.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from mtah import emission, features, io_atd  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

PCTS = (5, 25, 50, 75, 95)


def pct(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "--"
    return " ".join("p{}={:.2f}".format(p, np.percentile(values, p)) for p in PCTS)


def collect(tracking, spec):
    F = features.compute(tracking, spec)
    g = spec.graded_mask()
    nearest = np.fmin(F.d_home, F.d_away)
    zd = emission.dead_logit(F, emission.EmissionParams())
    zt = emission.team_logit(F, emission.EmissionParams())
    return {
        "player_speed_ms": pct(F.player_speed[g]),
        "ball_known_share": round(float(F.ball_known[g].mean()), 3),
        "ball_speed_ms (known)": pct(F.ball_speed[g & F.ball_known]),
        "ball_still_share (known, <0.5 m/s)": round(float(np.mean(
            np.nan_to_num(F.ball_speed[g & F.ball_known], nan=99) < 0.5)), 3),
        "nearest_player_m (known)": pct(nearest[g]),
        "abs_margin_m (known)": pct(np.abs(F.d_away - F.d_home)[g]),
        "ball_outside_share (known)": round(float(np.mean(F.out_dist[g & F.ball_known] > 0.5)), 4),
        "abs_shape_diff_m": pct(np.abs(F.shape_diff[g])),
        "last_touch_within_5s_share": round(float(np.mean(F.since_touch_s[g] < 5)), 3),
        "v1_dead_share (z_dead>0)": round(float(np.mean(zd[g] > 0)), 3),
        "v1_abs_team_logit": pct(np.abs(zt[g])),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--spec", help="game_spec.json (default: BAR-ATH from upstream games.py)")
    parser.add_argument("--external", help="external bundle name instead of --data-dir")
    parser.add_argument("--condition", default="clean", help="clean | balldrop | full")
    args = parser.parse_args(argv)
    if args.external:
        from ext import evaluate as ev
        path, spec, _ = ev.load(args.external)
        tracking = ev.tracking_for(path, spec, args.condition, seed=1000)
        name = "{} ({})".format(args.external, args.condition)
    else:
        if not args.data_dir:
            parser.error("pass --data-dir, set MTAH_DATA_DIR, or use --external")
        spec = GameSpec.load(args.spec) if args.spec else GameSpec.from_upstream("BAR-ATH")
        tracking = io_atd.load_game(args.data_dir, spec)
        name = spec.name
    print("feature stats: {}".format(name))
    for key, value in collect(tracking, spec).items():
        print("  {:<36} {}".format(key, value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
