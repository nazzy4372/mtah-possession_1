"""Emission model v2: LightGBM trained on noise-augmented external games (ext/train.py).

The model files are shipped in the repository (models/), trained only on other games' data:
  models/emission_v2.txt   LightGBM text model (stored byte-exact, see .gitattributes)
  models/emission_v2.json  feature list, class prior, training provenance
Decoding settings are frozen in config/final.yaml (written by `python tasks.py tune`): decoder
costs, emission temperature, dead-ball bias, refinement on/off. Without that file the model's own
defaults from emission_v2.json are used.

Inference is single-threaded and deterministic. The output is log p(x | state) up to a constant:
the temperature-scaled log posterior minus the log class prior, plus the dead-ball bias on D.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from mtah import learn_features
from mtah.decode import DecoderParams

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_TXT = MODEL_DIR / "emission_v2.txt"
MODEL_META = MODEL_DIR / "emission_v2.json"
CONFIG = ROOT / "config" / "final.yaml"


@lru_cache(maxsize=2)
def load(model_txt: str = str(MODEL_TXT), meta_json: str = str(MODEL_META)):
    import lightgbm as lgb
    if not Path(model_txt).is_file():
        raise FileNotFoundError("{} not found; train it with: python -m ext.train final"
                                .format(model_txt))
    booster = lgb.Booster(model_file=model_txt)
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    return booster, meta


@lru_cache(maxsize=2)
def settings(config_path: str = str(CONFIG)) -> dict:
    """Decoding settings: config/final.yaml if present, else the model's defaults."""
    meta = load()[1]
    out = {"c_switch": meta["decoder"]["c_switch"], "c_die": meta["decoder"]["c_die"],
           "c_restart": meta["decoder"]["c_restart"], "temperature": meta["temperature"],
           "dead_bias": 0.0, "refine": bool(meta.get("refine", False)), "source": "model"}
    path = Path(config_path)
    if path.is_file():
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg.get("model_variant") != meta.get("variant"):
            raise ValueError("config/final.yaml was tuned for variant {!r} but the shipped model "
                             "is {!r}".format(cfg.get("model_variant"), meta.get("variant")))
        out.update({"c_switch": float(cfg["decoder"]["c_switch"]),
                    "c_die": float(cfg["decoder"]["c_die"]),
                    "c_restart": float(cfg["decoder"]["c_restart"]),
                    "temperature": float(cfg["emission"]["temperature"]),
                    "dead_bias": float(cfg["emission"]["dead_bias"]),
                    "refine": bool(cfg["refine"]), "source": str(path.name)})
    return out


def decoder_params() -> DecoderParams:
    s = settings()
    return DecoderParams(c_switch=s["c_switch"], c_die=s["c_die"], c_restart=s["c_restart"])


def _log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def emissions(tracking, spec, F) -> np.ndarray:
    """(n, 3) log-emissions for H, A, D; zeros outside the graded halves."""
    booster, meta = load()
    s = settings()
    names, X = learn_features.build(tracking, spec, F)
    cols = [names.index(k) for k in meta["features"]]
    graded = spec.graded_mask()
    raw = booster.predict(X[graded][:, cols], raw_score=True, num_threads=1)
    logp = _log_softmax(raw / s["temperature"]) - np.log(np.asarray(meta["prior"]))
    logp[:, 2] += s["dead_bias"]
    out = np.zeros((tracking.n_frames, 3))
    out[graded] = logp
    return out
