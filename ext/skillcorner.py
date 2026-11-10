"""SkillCorner open data (A-League 2024/25 broadcast tracking) -> our ATD bundle format + H/A/D.

    python -m ext.skillcorner              # every downloaded match -> ext_cache/bundles/sc_<id>/

Source and terms: see ext/download.py (MIT licence; credit SkillCorner). Label derivation:
docs/label_derivation.md. Summary:

Labels (checked on match 1874553 before trusting them, see the doc):
  * possession.group (home / away / None) is SkillCorner's per-frame team possession. It persists
    while no player has the ball, as our definition requires;
  * a None frame OUTSIDE every phase of play (phases_of_play.csv) is dead ball. Those frames come in
    long runs (median 17 s), with the ball off the pitch 16% of the time, against 0.4% when a
    team has it;
  * a None frame INSIDE a phase is a loose ball (runs with a median of 0.8 s). It stays with the
    last team, or the phase's team if there is none yet.

Resampling: 10 fps -> 25 fps. Positions are linearly interpolated between neighbouring samples
where both exist; labels take the most recent sample. Coordinates (metres, origin at the centre, y
up) are normalised to the ATD convention (0-1, origin top-left) with the match's own pitch size.
Positions SkillCorner extrapolated (is_detected = False, mostly off-camera) are kept: they stand in
for the whole-pitch view of the tactical camera. The ATD noise model is applied on top
(ext/augment.py).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import atd_write  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

CACHE = ROOT / "ext_cache" / "skillcorner"
BUNDLES = ROOT / "ext_cache" / "bundles"
SRC_FPS, FPS = 10, 25
PRE_PAD, HALFTIME, POST_PAD = 30 * FPS, 60 * FPS, 10 * FPS
GROUPS = {"home team": "H", "away team": "A"}


def load_match(match_id: str):
    folder = CACHE / match_id
    meta = json.loads((folder / "{}_match.json".format(match_id)).read_text(encoding="utf-8"))
    home_id = meta["home_team"]["id"]
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    players = sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    column = {p["id"]: j for j, p in enumerate(players)}
    slot_team = np.array([0 if p["team_id"] == home_id else 1 for p in players], dtype=np.int8)

    frames, periods, groups = [], [], []
    rows = []
    with open(folder / "{}_tracking_extrapolated.jsonl".format(match_id), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("period") not in (1, 2):
                continue
            rows.append(r)
    n = len(rows)
    pxy = np.full((n, len(players), 2), np.nan)
    bxy = np.full((n, 2), np.nan)
    for i, r in enumerate(rows):
        frames.append(int(r["frame"]))
        periods.append(int(r["period"]))
        groups.append(GROUPS.get(r["possession"].get("group"), ""))
        b = r["ball_data"]
        if b.get("x") is not None:
            bxy[i] = (b["x"], b["y"])
        for p in r["player_data"]:
            j = column.get(p["player_id"])
            if j is not None:
                pxy[i, j] = (p["x"], p["y"])
    # metres, centre origin, y up  ->  normalised, top-left origin
    def norm(xy):
        out = np.empty_like(xy)
        out[..., 0] = (xy[..., 0] + length / 2.0) / length
        out[..., 1] = (width / 2.0 - xy[..., 1]) / width
        return out

    phases = []
    with open(folder / "{}_phases_of_play.csv".format(match_id), encoding="utf-8") as fh:
        for p in csv.DictReader(fh):
            phases.append((int(p["frame_start"]), int(p["frame_end"]),
                           "H" if int(p["team_in_possession_id"]) == home_id else "A"))
    return (np.array(frames), np.array(periods), np.array(groups), norm(pxy), norm(bxy),
            slot_team, phases, (length, width))


def derive_labels(frames, groups, phases):
    """H/A/D per 10 fps sample (see module docstring)."""
    phase_team = np.full(len(frames), "", dtype="<U1")
    for a, b, team in phases:
        lo = np.searchsorted(frames, a)
        hi = np.searchsorted(frames, b, side="right")
        phase_team[lo:hi] = team
    labels = np.full(len(frames), "D", dtype="<U1")
    last = ""
    for i in range(len(frames)):
        g = groups[i]
        if g:
            labels[i] = last = g
        elif phase_team[i]:
            labels[i] = last if last else phase_team[i]
            last = labels[i]
        else:
            labels[i] = "D"
            last = ""
    return labels


def _resample(values, src_t, dst_t):
    """Linear interpolation over the first axis where both neighbours exist, else NaN."""
    idx = np.searchsorted(src_t, dst_t, side="right") - 1
    idx = np.clip(idx, 0, len(src_t) - 2)
    w = (dst_t - src_t[idx]) / (src_t[idx + 1] - src_t[idx])
    w = np.clip(w, 0.0, 1.0)
    shape = (-1,) + (1,) * (values.ndim - 1)
    a, b = values[idx], values[idx + 1]
    out = a + (b - a) * w.reshape(shape)
    exact = w == 0.0
    out[exact] = a[exact]
    return out


def convert(match_id: str, out_root: Path = BUNDLES) -> Path:
    frames, periods, groups, pxy, bxy, slot_team, phases, pitch = load_match(match_id)
    labels = derive_labels(frames, groups, phases)

    blocks, vid_players, vid_ball, gt, kickoffs = [], [], [], [], {}
    cursor = PRE_PAD
    for half in (1, 2):
        sel = np.flatnonzero(periods == half)
        if not np.array_equal(np.diff(frames[sel]), np.ones(len(sel) - 1)):
            raise ValueError("{}: non-contiguous frames in period {}".format(match_id, half))
        src_t = (frames[sel] - frames[sel][0]) / float(SRC_FPS)
        n_out = int(np.floor(src_t[-1] * FPS)) + 1
        dst_t = np.arange(n_out) / float(FPS)
        vid_players.append(_resample(pxy[sel], src_t, dst_t))
        vid_ball.append(_resample(bxy[sel], src_t, dst_t))
        lab_idx = np.clip(np.searchsorted(src_t, dst_t, side="right") - 1, 0, len(sel) - 1)
        gt.append(labels[sel][lab_idx])
        blocks.append((half, cursor, cursor + n_out - 1))
        alive = np.flatnonzero(gt[-1] != "D")
        if len(alive):
            kickoffs[str(half)] = int(cursor + alive[0])
        cursor += n_out + (HALFTIME if half == 1 else POST_PAD)
    n = cursor
    players = np.full((n, len(slot_team), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    truth_gt = np.full(n, "X", dtype="<U1")
    for (half, first, last), p, b, g in zip(blocks, vid_players, vid_ball, gt):
        players[first:last + 1], ball[first:last + 1], truth_gt[first:last + 1] = p, b, g

    spec = GameSpec(name="SC-{}".format(match_id), stem="SC-{}_tactical".format(match_id),
                    n_frames=int(n), fps=FPS, pitch=tuple(float(v) for v in pitch),
                    halves=tuple(blocks))
    truth = {"source": "SkillCorner/opendata match {}".format(match_id),
             "event_kickoffs_video_frame": kickoffs, "n_slots": int(len(slot_team)),
             "label_counts": {k: int(np.sum(truth_gt == k)) for k in "HADX"}}
    out = out_root / "sc_{}".format(match_id)
    atd_write.write_bundle(out, spec, players, ball, slot_team, truth_gt, truth)
    return out


def downloaded_matches():
    manifest = ROOT / "ext" / "manifests" / "skillcorner.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))["matches"]
    return sorted(p.name for p in CACHE.iterdir() if p.is_dir()) if CACHE.is_dir() else []


if __name__ == "__main__":
    ids = sys.argv[1:] or downloaded_matches()
    for match_id in ids:
        path = convert(match_id)
        truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
        print("{} -> {}  {}".format(match_id, path.name, truth["label_counts"]), flush=True)
