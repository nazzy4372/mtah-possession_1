"""GameSpec: everything the pipeline needs to know about a match besides its data files.

The real match (BAR-ATH) takes every value from upstream ``games.py``, the published contract.
Synthetic and external games carry the same fields in a ``game_spec.json``, so one pipeline runs on
all of them unchanged.

``halves`` are the graded blocks as inclusive video-frame ranges ``(half, first, last)``. For the
real match they are published boundaries, not answers (DATA.md), and they are the only frame
numbers this project uses.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

from mtah.official import games

ASSET_SUFFIXES = {
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}


@dataclass(frozen=True)
class GameSpec:
    name: str
    stem: str                       # file stem inside <data_dir>/tactical/
    n_frames: int                   # rows in the submission
    fps: int
    pitch: Tuple[float, float]      # (length, width) in metres
    halves: Tuple[Tuple[int, int, int], ...]   # ((half, first_video_frame, last_video_frame), ...)
    # Published kickoff anchors ((half, video_frame), ...), used ONLY to check the detected
    # kickoff (CLAUDE.md rule 7), never to set a label. Empty when a game publishes none.
    kickoff_anchors: Tuple[Tuple[int, int], ...] = ()

    @classmethod
    def from_upstream(cls, game: str = "BAR-ATH") -> "GameSpec":
        entry = games.GAMES[game]
        anchors = []
        for half, key in ((1, "kickoff"), (2, "kickoff_2nd")):
            if key in entry:
                anchors.append((half, int(games.anchor_video_frame(entry[key]))))
        return cls(
            name=game,
            stem=games.SOURCES["tactical"]["stem"],
            n_frames=int(entry["n_video_frames"]),
            fps=int(entry["fps"]),
            pitch=tuple(float(v) for v in entry["pitch"]),
            halves=tuple((int(h), int(a), int(b)) for h, a, b in games.half_video_blocks(game)),
            kickoff_anchors=tuple(anchors),
        )

    @classmethod
    def load(cls, path) -> "GameSpec":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(name=data["name"], stem=data["stem"], n_frames=int(data["n_frames"]),
                   fps=int(data["fps"]), pitch=tuple(float(v) for v in data["pitch"]),
                   halves=tuple(tuple(int(v) for v in h) for h in data["halves"]),
                   kickoff_anchors=tuple(tuple(int(v) for v in a)
                                         for a in data.get("kickoff_anchors", ())))

    def save(self, path) -> None:
        text = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def paths(self, data_dir) -> Dict[str, Path]:
        """Asset paths. The real match goes through upstream games.asset_paths, so nothing here
        spells its file names out; other games use the same layout with their own stem."""
        if self.name in games.GAMES and self.stem == games.SOURCES["tactical"]["stem"]:
            return {k: Path(v) for k, v in games.asset_paths(self.name, str(data_dir)).items()}
        folder = Path(data_dir) / "tactical"
        return {k: folder / (self.stem + suffix) for k, suffix in ASSET_SUFFIXES.items()}

    def graded_mask(self):
        import numpy as np
        mask = np.zeros(self.n_frames, dtype=bool)
        for _, first, last in self.halves:
            mask[first:last + 1] = True
        return mask

    def runs(self):
        """Half-open [start, stop) graded runs, the convention of possession_metrics."""
        return [(first, last + 1) for _, first, last in self.halves]
