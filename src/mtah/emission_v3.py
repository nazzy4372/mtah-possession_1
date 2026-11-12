"""Model v3: v2 plus change-of-possession evidence (post-M5 improvement).

A LightGBM model trained only on noise-augmented external games (ext/train.py):
  models/emission_v3.txt  p(H, A, D | x) on the v2 features plus touch-sequence and ball-kinematics
                          features around the frame (learn_features.GROUP_TOUCH)
Decoding settings are frozen in config/final_v3.yaml (written by `python tasks.py tune --freeze`).

Not shipped: a change-point model (models/change_v3.txt, p(no change, switch, die, restart | x))
that modulates the decoder's transition costs (decode.viterbi_batch_dyn). Its gain on external
games was not robust (docs/ablation.md), so the frozen config has no `change` block and the
model is neither loaded nor present. The code path is kept and tested as a documented experiment.

Inference is single-threaded and deterministic. v2 (emission_v2.py, config/final.yaml) is left
untouched so the earlier frozen submission stays reproducible with `--method v2`.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from mtah import learn_features
from mtah.decode import DecoderParams

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
EMISSION_TXT = MODEL_DIR / "emission_v3.txt"
EMISSION_META = MODEL_DIR / "emission_v3.json"
CHANGE_TXT = MODEL_DIR / "change_v3.txt"
CHANGE_META = MODEL_DIR / "change_v3.json"
CONFIG = ROOT / "config" / "final_v3.yaml"


def available() -> bool:
    return EMISSION_TXT.is_file() and CONFIG.is_file()


@lru_cache(maxsize=1)
def load():
    import lightgbm as lgb
    emission = lgb.Booster(model_file=str(EMISSION_TXT))
    meta = json.loads(EMISSION_META.read_text(encoding="utf-8"))
    change, change_meta = None, None
    if settings()["w_change"] > 0.0:
        change = lgb.Booster(model_file=str(CHANGE_TXT))
        change_meta = json.loads(CHANGE_META.read_text(encoding="utf-8"))
    return emission, meta, change, change_meta


@lru_cache(maxsize=1)
def settings() -> dict:
    import yaml
    if not CONFIG.is_file():
        raise FileNotFoundError("{} not found; freeze it with: python tasks.py tune --variant "
                                "touch --change touch --freeze".format(CONFIG))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return {"c_switch": float(cfg["decoder"]["c_switch"]),
            "c_die": float(cfg["decoder"]["c_die"]),
            "c_restart": float(cfg["decoder"]["c_restart"]),
            "temperature": float(cfg["emission"]["temperature"]),
            "dead_bias": float(cfg["emission"]["dead_bias"]),
            "refine": bool(cfg["refine"]),
            "w_change": float(cfg.get("change", {}).get("weight", 0.0)),
            "model_variant": cfg["model_variant"], "source": CONFIG.name}


def decoder_params() -> DecoderParams:
    s = settings()
    return DecoderParams(c_switch=s["c_switch"], c_die=s["c_die"], c_restart=s["c_restart"])


def _log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def evidence(tracking, spec, F):
    """(logp (n, 3), change log-ratio (n, 3) float32 or None); zeros outside the graded halves."""
    emission, meta, change, change_meta = load()
    s = settings()
    if meta["variant"] != s["model_variant"]:
        raise ValueError("config/final_v3.yaml was tuned for variant {!r} but the shipped model "
                         "is {!r}".format(s["model_variant"], meta["variant"]))
    names, X = learn_features.build(tracking, spec, F)
    graded = spec.graded_mask()
    Xg = X[graded]

    cols = [names.index(k) for k in meta["features"]]
    raw = emission.predict(Xg[:, cols], raw_score=True, num_threads=1)
    logp_g = _log_softmax(raw / s["temperature"]) - np.log(np.asarray(meta["prior"]))
    logp_g[:, 2] += s["dead_bias"]
    logp = np.zeros((tracking.n_frames, 3))
    logp[graded] = logp_g

    ratio = None
    if change is not None:
        ccols = [names.index(k) for k in change_meta["features"]]
        craw = change.predict(Xg[:, ccols], raw_score=True, num_threads=1)
        prior = np.asarray(change_meta["prior"])
        ratio = np.zeros((tracking.n_frames, 3), dtype=np.float32)
        ratio[graded] = (_log_softmax(craw)[:, 1:] - np.log(prior[1:])[None, :]).astype(np.float32)
    return logp, ratio
