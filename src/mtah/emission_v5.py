"""Model v5: model v4 plus a team prior from the match's Smart Tagging.

v5 uses the v4 LightGBM model and v4's frozen decoder settings unchanged (emission_v4.py,
config/final_v4.yaml). The only addition: on frames that the tagging places inside a phase of
settled possession for one team (BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION),
the other team's log-score is lowered by `weight` nats per frame before decoding
(io_tagging.team_evidence, io_tagging.add_team_prior). The dead-ball score is never touched.

The weight is frozen in config/final_v5.yaml by `python -m ext.tags freeze`. It was chosen on
SkillCorner games only, whose automatic phases of play stand in for the tagging, degraded to be
no better than the real stream (ext/tags.py, docs/ablation.md). Nothing from the provided match's
labels or public ground truth was used.
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
