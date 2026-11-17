"""Model v5 = v4 + a small team prior from the match's Smart Tagging. This is the submission.

Inside a BUILD UP / PROGRESSION / CREATING CHANCES / ATTACKING TRANSITION phase of one team,
the other team's score drops by `weight` per frame. Dead ball isn't touched.
Weight is in config/final_v5.yaml, set on SkillCorner games (see ext/tags.py).
"""

from functools import lru_cache
from pathlib import Path

from mtah import emission_v4, io_atd, io_tagging

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "final_v5.yaml"


def available() -> bool:
    return emission_v4.available() and CONFIG.is_file()


@lru_cache(maxsize=1)
def settings() -> dict:
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if cfg["base_model"] != "v4":
        raise ValueError("config/final_v5.yaml expects base model {!r}".format(cfg["base_model"]))
    out = dict(emission_v4.settings())
    out.update({"w_tagging": float(cfg["tagging"]["weight"]),
                "source": "{} + {}".format(out["source"], CONFIG.name)})
    return out


def tag_evidence(data_dir, spec):
    """Per-frame team evidence (io_tagging.HOME / AWAY / NONE) from the bundle's Smart Tagging."""
    paths = spec.paths(data_dir)
    if not Path(paths["smart_tagging"]).is_file():
        raise FileNotFoundError("model v5 needs the Smart Tagging file of the bundle "
                                "(<stem>_smart_tagging.xml in the tactical folder)")
    intervals = io_tagging.load(paths["smart_tagging"], spec.fps)
    meta = io_atd.read_meta(paths["atd_meta"])
    sides = io_tagging.team_sides(intervals, meta.home_name, meta.away_name)
    usable = [iv for iv in intervals
              if iv.code in io_tagging.IN_POSSESSION and iv.tag("team") not in ("", "N/A", "n/a")]
    if usable and not sides:
        raise ValueError("the Team tags of the Smart Tagging do not map onto the two teams of "
                         "the tracking metadata")
    return io_tagging.team_evidence(intervals, sides, spec.n_frames)
