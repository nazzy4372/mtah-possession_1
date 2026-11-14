"""Smart Tagging (SportsCode XML) -> intervals in video frames.

Metrica's automatic tactical tags: phases of play and set pieces, each an interval with a code and
tags (Team, Half, Type, ...). `<start>` and `<end>` are seconds from the first video frame, so
`round(seconds * fps)` is a video frame (upstream DATA.md, verified there on both kickoffs).

Codes are upper case and tag values title case in the delivered file; everything here is matched
case-insensitively. Team is `N/A` on some intervals (every throw-in) and is then treated as
missing. The tags are noisy evidence, not truth, and their intervals are phases, not possessions.

Not used by any shipped model (docs/TECHNICAL.md, section 2). Used by tools/tagging_stats.py.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

SET_PIECE_CODE = "set pieces"
HALF_START_TYPE = "kick off (start)"
HOME, AWAY = 0, 1


@dataclass(frozen=True)
class Interval:
    code: str                 # lower case
    start: int                # first video frame
    end: int                  # first video frame after the interval
    tags: Dict[str, str]      # group (lower case) -> text (as written)

    def tag(self, group: str) -> str:
        return (self.tags.get(group.lower()) or "").strip()

    @property
    def is_set_piece(self) -> bool:
        return self.code == SET_PIECE_CODE

    @property
    def kind(self) -> str:
        """Set-piece type in lower case ('throw-in', 'corner kick', ...), '' for phases."""
        return self.tag("type").lower() if self.is_set_piece else ""


def load(path, fps: int) -> List[Interval]:
    """Every instance of the file, in file order."""
    out = []
    for instance in ET.parse(str(path)).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {(label.findtext("group") or "").strip().lower(): (label.findtext("text") or "")
                for label in instance.findall("label")}
        out.append(Interval(code=(instance.findtext("code") or "").strip().lower(),
                            start=int(round(start_s * fps)), end=int(round(end_s * fps)),
                            tags=tags))
    return out


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def team_sides(intervals: List[Interval], home_name: str, away_name: str) -> Dict[str, int]:
    """Map each club name used in the Team tag to HOME or AWAY by its similarity to the names in
    the ATD metadata (the two files do not spell the clubs the same way). Returns {} unless the
    file names exactly two clubs and they map to different sides."""
    names = sorted({iv.tag("team") for iv in intervals} - {"", "N/A", "n/a"})
    if len(names) != 2:
        return {}
    sides = {}
    for name in names:
        score = [SequenceMatcher(None, _norm(name), _norm(ref)).ratio()
                 for ref in (home_name, away_name)]
        sides[name] = HOME if score[0] >= score[1] else AWAY
    return sides if set(sides.values()) == {HOME, AWAY} else {}


def side_of(interval: Interval, sides: Dict[str, int]) -> Optional[int]:
    return sides.get(interval.tag("team"))
