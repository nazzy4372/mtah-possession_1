"""Metrica Sports open sample games 1-3 -> our ATD bundle format + H/A/D ground truth.

    python -m ext.metrica            # converts every game into ext_cache/bundles/metrica_g<N>/

Source and terms: see ext/download.py. Label derivation: docs/label_derivation.md (summary below).

Timeline. The Metrica frames of each period are placed on a synthetic "video" timeline laid out like
the real match: PRE_PAD untracked frames, period 1, HALFTIME untracked frames, period 2, POST_PAD.
The graded halves are the two periods, so the pipeline sees the same structure as on BAR-ATH.
Games 1-2 start tracking at the kickoff (like the real first half); game 3 tracks before its kickoff
(like the real second half).

Labels from the human-annotated event data (who has the ball, when the ball is dead):
  * a restart (any SET PIECE, or a PASS whose subtype is GOAL KICK) makes the ball alive, with the
    restarting team in possession, from its start frame;
  * a controlling event (PASS, CARRY, SHOT, RECOVERY, BALL LOST) by team T at its start frame gives
    T the ball if it did not have it (possession changes on the new team's first controlled
    action; a CHALLENGE never changes possession: contested balls stay with the last team);
  * a stoppage makes the ball dead: BALL OUT (from its end frame, when the ball crosses the line);
    a SHOT whose subtype contains OUT or GOAL (from its end frame); FAULT RECEIVED and HAND BALL
    (from the start frame); OFFSIDE and END HALF (from the end frame). Dead lasts until the next
    restart, or until the next controlling event if the restart was not annotated;
  * before the first event of a period (the kickoff) the ball is dead.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mtah import atd_write  # noqa: E402
from mtah.spec import GameSpec  # noqa: E402

CACHE = ROOT / "ext_cache" / "metrica"
BUNDLES = ROOT / "ext_cache" / "bundles"
FPS = 25
PITCH = (105.0, 68.0)          # Metrica sample-data README: 105 x 68 m for all games
PRE_PAD, HALFTIME, POST_PAD = 30 * FPS, 60 * FPS, 10 * FPS

CONTROL = {"PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST"}


# -- reading -------------------------------------------------------------------------------------

def _read_csv_team(path: Path):
    """(period (n,), frame (n,), players (n, k, 2), ball (n, 2)) from a Metrica CSV tracking file."""
    with open(path, encoding="utf-8") as handle:
        header = [handle.readline() for _ in range(3)][2].rstrip("\n").split(",")
    data = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=float)
    n_players = sum(1 for name in header[3:] if name.startswith("Player"))
    players = data[:, 3:3 + 2 * n_players].reshape(len(data), n_players, 2)
    ball = data[:, 3 + 2 * n_players:5 + 2 * n_players]
    return data[:, 0].astype(int), data[:, 1].astype(int), players, ball


def load_csv_game(game: int):
    folder = CACHE / "Sample_Game_{}".format(game)
    stem = "Sample_Game_{}_RawTrackingData_{}_Team.csv"
    period, frame, home, ball = _read_csv_team(folder / stem.format(game, "Home"))
    period_a, frame_a, away, _ = _read_csv_team(folder / stem.format(game, "Away"))
    if not (np.array_equal(frame, frame_a) and np.array_equal(period, period_a)):
        raise ValueError("game {}: home and away tracking frames differ".format(game))
    players = np.concatenate([home, away], axis=1)
    slot_team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)

    ev = pd.read_csv(folder / "Sample_Game_{}_RawEventsData.csv".format(game))
    events = [{"team": 0 if str(r["Team"]).strip().lower() == "home" else 1,
               "type": str(r["Type"]).strip().upper(),
               "subtypes": _split_subtypes(r["Subtype"]),
               "period": int(r["Period"]), "start": int(r["Start Frame"]),
               "end": int(r["End Frame"])} for _, r in ev.iterrows()]
    return period, frame, players, ball, slot_team, events


def _split_subtypes(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(s.strip().upper() for s in re.split(r"[-]", str(value)) if s.strip())


def load_epts_game(game: int = 3):
    from kloppy import metrica

    folder = CACHE / "Sample_Game_{}".format(game)
    dataset = metrica.load_tracking_epts(
        meta_data=str(folder / "Sample_Game_{}_metadata.xml".format(game)),
        raw_data=str(folder / "Sample_Game_{}_tracking.txt".format(game)), coordinates="metrica")
    df = dataset.to_df()
    home, away = dataset.metadata.teams
    ids = [p.player_id for p in home.players] + [p.player_id for p in away.players]
    slot_team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    players = np.full((len(df), len(ids), 2), np.nan)
    for j, pid in enumerate(ids):
        if "{}_x".format(pid) in df:
            players[:, j, 0] = df["{}_x".format(pid)].to_numpy(dtype=float)
            players[:, j, 1] = df["{}_y".format(pid)].to_numpy(dtype=float)
    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=float)

    raw = json.loads((folder / "Sample_Game_{}_events.json".format(game)).read_text("utf-8"))
    events = []
    for e in raw["data"]:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        events.append({"team": 0 if e["team"]["id"] == home.team_id else 1,
                       "type": e["type"]["name"].strip().upper(),
                       "subtypes": tuple(s["name"].strip().upper() for s in (sub or [])),
                       "period": int(e["period"]), "start": int(e["start"]["frame"]),
                       "end": int(e["end"]["frame"])})
    return (df["period_id"].to_numpy(dtype=int), df["frame_id"].to_numpy(dtype=int), players,
            ball, slot_team, events)


# -- labels --------------------------------------------------------------------------------------

def _is_restart(event):
    """A set piece, or a goal kick (Metrica annotates those as PASS or BALL LOST / GOAL KICK)."""
    return event["type"] == "SET PIECE" or "GOAL KICK" in event["subtypes"]


def _stop_frame(event):
    """Frame from which this event makes the ball dead, or None."""
    t, sub = event["type"], event["subtypes"]
    if t == "BALL OUT":
        return event["end"]
    if t == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return event["end"]
    if t == "FAULT RECEIVED" or "HAND BALL" in sub:
        return event["start"]
    if "OFFSIDE" in sub or "END HALF" in sub:
        return max(event["start"], event["end"])
    return None


def derive_labels(frames: np.ndarray, periods: np.ndarray, events) -> np.ndarray:
    """H/A/D per tracking row from the event stream (see module docstring)."""
    labels = np.full(len(frames), "D", dtype="<U1")
    team_char = ("H", "A")
    for period in np.unique(periods):
        rows = np.flatnonzero(periods == period)
        period_frames = frames[rows]
        f0 = int(period_frames[0])
        evs = sorted((e for e in events if e["period"] == period), key=lambda e: e["start"])

        # change points (frame, label), applied in time order; a label holds until the next one
        points, state, last_end = [], "D", f0
        for e in evs:
            start = max(e["start"], f0)
            team = team_char[e["team"]]
            if _is_restart(e):
                if state != "D" and start - last_end > FPS:
                    points.append((last_end, "D"))  # unannotated stoppage before a restart
                state = team
                points.append((start, state))
            elif e["type"] in CONTROL and state != team:
                state = team
                points.append((start, state))
            stop = _stop_frame(e)
            if stop is not None:
                state = "D"
                points.append((max(stop, f0), "D"))
            last_end = max(last_end, e["start"], e["end"])

        out = np.full(len(rows), "D", dtype="<U1")
        for frame, label in sorted(points, key=lambda p: p[0]):
            out[int(np.searchsorted(period_frames, frame)):] = label
        labels[rows] = out
    return labels


# -- conversion ----------------------------------------------------------------------------------

def convert(game: int, out_root: Path = BUNDLES) -> Path:
    if game in (1, 2):
        periods, frames, players, ball, slot_team, events = load_csv_game(game)
    else:
        periods, frames, players, ball, slot_team, events = load_epts_game(game)
    keep = np.isin(periods, (1, 2))
    periods, frames, players, ball = periods[keep], frames[keep], players[keep], ball[keep]
    labels = derive_labels(frames, periods, events)

    # place the two periods on a video-like timeline with untracked padding and a halftime gap
    p1, p2 = np.flatnonzero(periods == 1), np.flatnonzero(periods == 2)
    for rows in (p1, p2):
        if not np.array_equal(np.diff(frames[rows]), np.ones(len(rows) - 1)):
            raise ValueError("game {}: non-contiguous frames inside a period".format(game))
    first1 = PRE_PAD
    first2 = first1 + len(p1) + HALFTIME
    n = first2 + len(p2) + POST_PAD
    video_rows = np.concatenate([first1 + np.arange(len(p1)), first2 + np.arange(len(p2))])
    rows = np.concatenate([p1, p2])

    vid_players = np.full((n, players.shape[1], 2), np.nan)
    vid_ball = np.full((n, 2), np.nan)
    gt = np.full(n, "X", dtype="<U1")
    vid_players[video_rows] = players[rows]
    vid_ball[video_rows] = ball[rows]
    gt[video_rows] = labels[rows]

    spec = GameSpec(name="METRICA-G{}".format(game), stem="METRICA-G{}_tactical".format(game),
                    n_frames=int(n), fps=FPS, pitch=PITCH,
                    halves=((1, first1, first1 + len(p1) - 1), (2, first2, first2 + len(p2) - 1)))
    kickoffs = {}
    for half, (offset, prow) in enumerate(((first1, p1), (first2, p2)), start=1):
        ko = [e for e in events if e["period"] == half and e["type"] == "SET PIECE"
              and "KICK OFF" in e["subtypes"]]
        if ko:
            kickoffs[str(half)] = int(offset + np.searchsorted(frames[prow], min(e["start"]
                                                                                  for e in ko)))
    truth = {"source": "metrica-sports/sample-data Sample_Game_{}".format(game),
             "event_kickoffs_video_frame": kickoffs, "n_slots": int(players.shape[1]),
             "label_counts": {k: int(np.sum(gt == k)) for k in "HADX"}}
    out = out_root / "metrica_g{}".format(game)
    atd_write.write_bundle(out, spec, vid_players, vid_ball, slot_team, gt, truth)
    return out


if __name__ == "__main__":
    games = [int(g) for g in sys.argv[1:]] or [1, 2, 3]
    for g in games:
        path = convert(g)
        truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
        print("game {} -> {}  {}".format(g, path, truth["label_counts"]))
