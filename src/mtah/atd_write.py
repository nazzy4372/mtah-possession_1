"""Write tracking arrays back out in the exact ATD text + XML format (for external games)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtah import synthetic
from mtah.spec import GameSpec


def speed_ms(norm_xy: np.ndarray, pitch, fps: int) -> np.ndarray:
    """Per-frame speed in m/s from normalised positions (NaN at gaps and on the first frame)."""
    metres = norm_xy * np.asarray(pitch, dtype=float)
    step = metres[1:] - metres[:-1]
    out = np.full(norm_xy.shape[:-1], np.nan)
    out[1:] = np.hypot(step[..., 0], step[..., 1]) * fps
    return out


def write_raw(path: Path, players: np.ndarray, ball: np.ndarray, pitch, fps: int) -> None:
    """players (n, k, 2), ball (n, 2), normalised; one ATD line per frame, frame ids 1..n."""
    n, k = players.shape[:2]
    p_speed = speed_ms(players, pitch, fps)
    b_speed = speed_ms(ball, pitch, fps)
    cols = {"frame": np.arange(1, n + 1)}
    for j in range(k):
        cols["x{}".format(j)] = players[:, j, 0]
        cols["y{}".format(j)] = players[:, j, 1]
        cols["s{}".format(j)] = np.where(np.isnan(players[:, j, 0]), np.nan, p_speed[:, j])
    cols["bx"], cols["by"] = ball[:, 0], ball[:, 1]
    cols["bs"] = np.where(np.isnan(ball[:, 0]), np.nan, b_speed)
    text = pd.DataFrame(cols).to_csv(header=False, index=False, na_rep="NaN",
                                     float_format="%.5f", lineterminator="\n")
    out_lines = []
    for line in text.split("\n"):
        if not line:
            continue
        fields = line.split(",")
        groups = [",".join(fields[1 + 3 * j:4 + 3 * j]) for j in range(k)]
        out_lines.append("{}:{}:{}".format(fields[0], ";".join(groups), ",".join(fields[-3:])))
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(out_lines) + "\n")


def write_bundle(out_dir, spec: GameSpec, players: np.ndarray, ball: np.ndarray,
                 slot_team: np.ndarray, gt: np.ndarray, truth: dict,
                 names=("Home", "Away")) -> None:
    out = Path(out_dir)
    (out / "tactical").mkdir(parents=True, exist_ok=True)
    paths = spec.paths(out)
    write_raw(paths["atd_raw"], players, ball, spec.pitch, spec.fps)
    meta = synthetic.metadata_xml(slot_team, spec.n_frames,
                                  [(first, last) for _, first, last in spec.halves],
                                  home=names[0], away=names[1], pitch=spec.pitch)
    with open(paths["atd_meta"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(meta)
    with open(out / "gt.csv", "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,label\n" + "".join("{},{}\n".format(i, v) for i, v in enumerate(gt)))
    spec.save(out / "game_spec.json")
    with open(out / "truth.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(truth, handle, indent=2, sort_keys=True)
        handle.write("\n")
