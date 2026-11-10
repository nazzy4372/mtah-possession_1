"""Fast parser for the raw ATD (Metrica FIFA/EPTS) tracking file and its metadata XML.

Raw line:  ``<frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<x,y,s>`` (players, then the ball), with
``NaN`` for a missing observation. x and y are normalised to 0-1; s is speed in **km/h**.
``video_frame = frame_id - 1``.

Speed: replacing ``:`` and ``;`` with ``,`` turns every line into ``1 + 3*n_slots + 3`` plain
numbers, which pandas' C reader parses in seconds. ``float_precision="round_trip"`` makes each
value bit-identical to Python's ``float()``, which is what the upstream baseline uses, so distances
computed here match it exactly (tests/test_parity.py).

Privacy: this module runs on the real match only on the human's machine. Error messages report
counts and line numbers, never values from the file.

No kloppy here: it is far slower, and inference must stay fast. kloppy is used only in the tests,
as an independent check that our synthetic files follow the real format.
"""

import io
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1
KMH_TO_MS = 1.0 / 3.6
FIELDS_PER_OBJECT = 3  # x, y, s


class ATDFormatError(ValueError):
    """The file does not have the expected shape. Messages never quote data values."""


def _local(tag: str) -> str:
    return tag.split("}")[-1]


@dataclass(frozen=True)
class ATDMeta:
    home_name: str
    away_name: str
    slot_team: np.ndarray          # (n_slots,) int8, HOME or AWAY, in <Player> order
    channel_ids: Tuple[str, ...]   # declared channel ids, file order
    layout_check: str              # "verified", "unverified", with a reason when unverified

    @property
    def n_slots(self) -> int:
        return int(len(self.slot_team))


def read_meta(path) -> ATDMeta:
    """Teams, per-slot team and channels from the ATD metadata XML.

    Home is the first ``<Team>`` (FIFA/EPTS convention, followed by kloppy and by the upstream
    baseline). Slot i is the i-th ``<Player>`` element, again as the upstream baseline reads it.
    When the data-format specification is present we also check that the player channels in the
    raw line follow that same slot order, three per slot. Contradiction raises, absence is noted.
    """
    root = ET.parse(str(path)).getroot()
    teams = [el for el in root.iter() if _local(el.tag) == "Team"]
    if len(teams) != 2:
        raise ATDFormatError("{}: expected 2 <Team> elements, found {}".format(
            Path(path).name, len(teams)))
    home_id = teams[0].get("id")
    names = []
    for team in teams:
        name = next((c.text for c in team if _local(c.tag) == "Name"), None)
        names.append((name or team.get("id") or "").strip())

    players = [el for el in root.iter() if _local(el.tag) == "Player"]
    if not players:
        raise ATDFormatError("{}: no <Player> elements".format(Path(path).name))
    slot_team = np.array([HOME if p.get("teamId") == home_id else AWAY for p in players],
                         dtype=np.int8)
    channel_ids = tuple(el.get("id") for el in root.iter() if _local(el.tag) == "Channel")

    layout_check = _check_layout(root, [p.get("id") for p in players])
    return ATDMeta(home_name=names[0], away_name=names[1], slot_team=slot_team,
                   channel_ids=channel_ids, layout_check=layout_check)


def _check_layout(root, player_ids: List[Optional[str]]) -> str:
    """Do the PlayerChannelRefs in the data-format spec run slot by slot, 3 per slot, in
    <Player> order? Returns "verified" or "unverified: <reason>"; raises on a contradiction."""
    channel_owner = {el.get("id"): el.get("playerId") for el in root.iter()
                     if _local(el.tag) == "PlayerChannel"}
    refs = [el.get("playerChannelId") for el in root.iter() if _local(el.tag) == "PlayerChannelRef"]
    if not refs or not channel_owner:
        return "unverified: no PlayerChannel/PlayerChannelRef elements"
    owners = [channel_owner.get(ref) for ref in refs]
    if any(owner is None for owner in owners) or any(pid is None for pid in player_ids):
        return "unverified: channel references could not be resolved to players"
    if len(owners) != FIELDS_PER_OBJECT * len(player_ids):
        raise ATDFormatError("data format lists {} player channels for {} players; expected {} "
                             "per player".format(len(owners), len(player_ids), FIELDS_PER_OBJECT))
    expected = [pid for pid in player_ids for _ in range(FIELDS_PER_OBJECT)]
    if owners != expected:
        raise ATDFormatError("player channels in the raw line do not follow <Player> order")
    return "verified"


@dataclass
class Tracking:
    """One match, indexed by video frame (row i = video frame i = ATD frame_id i + 1).

    Positions are metres (normalised value times pitch length / width, the same arithmetic as the
    upstream baseline). Speeds are m/s. Missing is NaN.
    """
    players_xy: np.ndarray     # (n, n_slots, 2) float64, metres
    players_speed: np.ndarray  # (n, n_slots) float64, m/s
    ball_xy: np.ndarray        # (n, 2) float64, metres
    ball_speed: np.ndarray     # (n,) float64, m/s
    slot_team: np.ndarray      # (n_slots,) int8
    pitch: Tuple[float, float]
    meta: ATDMeta

    @property
    def n_frames(self) -> int:
        return int(self.ball_xy.shape[0])

    @property
    def ball_observed(self) -> np.ndarray:
        return ~np.isnan(self.ball_xy[:, 0]) & ~np.isnan(self.ball_xy[:, 1])

    @property
    def player_observed(self) -> np.ndarray:
        return ~np.isnan(self.players_xy[..., 0]) & ~np.isnan(self.players_xy[..., 1])


def _check_fields_per_line(blob: bytes, expected: int, name: str, lines_per_chunk=20000) -> None:
    """Every non-blank line must have ``expected`` fields, or ``expected + 1`` on every line
    (a trailing separator). pandas would silently pad a short line with NaN, so this is counted
    explicitly, chunk by chunk to keep memory flat."""
    buf = np.frombuffer(blob, dtype=np.uint8)
    newlines = np.flatnonzero(buf == 10)
    ends = newlines if len(buf) and buf[-1] == 10 else np.append(newlines, len(buf))
    starts = np.concatenate(([0], newlines + 1))[:len(ends)]
    counts = np.empty(len(ends), dtype=np.int64)
    for lo in range(0, len(ends), lines_per_chunk):
        hi = min(len(ends), lo + lines_per_chunk)
        a, b = starts[lo], ends[hi - 1]
        commas = np.flatnonzero(buf[a:b] == 44) + a
        counts[lo:hi] = (np.searchsorted(commas, ends[lo:hi])
                         - np.searchsorted(commas, starts[lo:hi])) + 1
    blank = (ends - starts) <= 1  # empty, or a lone "\r"
    counts = counts[~blank]
    if not len(counts):
        raise ATDFormatError("{}: no data lines".format(name))
    if (counts == expected).all() or (counts == expected + 1).all():
        return
    bad = int(np.flatnonzero(counts != counts[0] if counts[0] in (expected, expected + 1)
                             else np.ones_like(counts, dtype=bool))[0]) + 1
    raise ATDFormatError("{}: {} fields per line on data line {}, expected {} (1 frame id + slots "
                         "x 3 + ball x 3)".format(name, int(counts[bad - 1]), bad, expected))


def read_raw(path, n_slots: int, n_frames: Optional[int] = None) -> np.ndarray:
    """(n, 3*n_slots + 3) float64 array of the raw numbers, row i = video frame i.

    Strict: the frame ids must be exactly 1..n with nothing missing, and every line must carry
    ``1 + 3*n_slots + 3`` fields. A single trailing separator (an empty last field on every line)
    is tolerated, as kloppy's reader tolerates it.
    """
    path = Path(path)
    blob = path.read_bytes().translate(bytes.maketrans(b":;", b",,"))
    expected_cols = 1 + FIELDS_PER_OBJECT * n_slots + FIELDS_PER_OBJECT
    _check_fields_per_line(blob, expected_cols, path.name)
    try:
        with warnings.catch_warnings():
            # Deprecated in pandas 3 but still honoured, and pandas is pinned. round_trip is the
            # one mode documented to match Python's float() bit for bit (see module docstring).
            warnings.filterwarnings("ignore", message=".*float_precision.*")
            table = pd.read_csv(io.BytesIO(blob), header=None, engine="c", dtype=np.float64,
                                float_precision="round_trip", skip_blank_lines=True)
    except (ValueError, pd.errors.ParserError) as error:
        # pandas' message can quote a token from the file; report only the kind of failure.
        raise ATDFormatError("{}: not a parseable ATD raw file ({})".format(
            path.name, type(error).__name__)) from None
    del blob
    values = table.to_numpy(dtype=np.float64)
    del table

    if values.shape[1] == expected_cols + 1 and np.isnan(values[:, -1]).all():
        values = values[:, :-1]
    if values.shape[1] != expected_cols:
        raise ATDFormatError("{}: {} fields per line, expected {} (1 frame id + {} slots x 3 + "
                             "ball x 3)".format(path.name, values.shape[1], expected_cols, n_slots))

    frame_ids = values[:, 0]
    n = len(frame_ids)
    if not np.array_equal(frame_ids, np.arange(1, n + 1, dtype=np.float64)):
        bad = int(np.flatnonzero(frame_ids != np.arange(1, n + 1))[0]) + 1
        raise ATDFormatError("{}: frame ids are not 1..{} in order (first mismatch on line {})"
                             .format(path.name, n, bad))
    if n_frames is not None and n != n_frames:
        raise ATDFormatError("{}: {} records, expected {}".format(path.name, n, n_frames))
    return values


def load(meta_path, raw_path, pitch: Tuple[float, float],
         n_frames: Optional[int] = None) -> Tracking:
    meta = read_meta(meta_path)
    values = read_raw(raw_path, meta.n_slots, n_frames)
    length, width = pitch
    n, k = values.shape[0], meta.n_slots

    players = values[:, 1:1 + FIELDS_PER_OBJECT * k].reshape(n, k, FIELDS_PER_OBJECT)
    players_xy = np.empty((n, k, 2), dtype=np.float64)
    players_xy[..., 0] = players[..., 0] * length
    players_xy[..., 1] = players[..., 1] * width
    players_speed = players[..., 2] * KMH_TO_MS

    ball = values[:, 1 + FIELDS_PER_OBJECT * k:]
    ball_xy = np.empty((n, 2), dtype=np.float64)
    ball_xy[:, 0] = ball[:, 0] * length
    ball_xy[:, 1] = ball[:, 1] * width
    ball_speed = ball[:, 2] * KMH_TO_MS

    return Tracking(players_xy=players_xy, players_speed=np.ascontiguousarray(players_speed),
                    ball_xy=ball_xy, ball_speed=np.ascontiguousarray(ball_speed),
                    slot_team=meta.slot_team, pitch=(float(length), float(width)), meta=meta)


def load_game(data_dir, spec) -> Tracking:
    """Tracking for a GameSpec from a data bundle directory."""
    paths = spec.paths(data_dir)
    for key in ("atd_meta", "atd_raw"):
        if not paths[key].is_file():
            raise FileNotFoundError("missing {} for {} under {}".format(key, spec.name, data_dir))
    return load(paths["atd_meta"], paths["atd_raw"], spec.pitch, spec.n_frames)
