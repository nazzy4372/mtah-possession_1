"""Model v4: the v3 features, trained on external games with the corrected ball-visibility noise.

Same 59 features and the same LightGBM settings as v3 (emission_v3.py). The only difference is the
training data: noise condition `full2` (ext/augment.py, ball model "speed"), in which the ball goes
untracked mostly while the players are slow, as measured on the real match without labels
(tools/slow_play_stats.py). Under the older condition `full` the dropout ignored the state of play.

  models/emission_v4.txt   p(H, A, D | x), trained by `python -m ext.train final-v4`
  config/final_v4.yaml     decoding settings, frozen by `python tasks.py tune --variant touch_f2
                           --freeze`

Inference is single-threaded and deterministic. v2 and v3 are left untouched, so the earlier frozen
files stay reproducible with `--method v2` and `--method v3`.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from mtah import learn_features
from mtah.decode import DecoderParams

ROOT = Path(__file__).resolve().parents[2]
EMISSION_TXT = ROOT / "models" / "emission_v4.txt"
EMISSION_META = ROOT / "models" / "emission_v4.json"
CONFIG = ROOT / "config" / "final_v4.yaml"


def available() -> bool:
    return EMISSION_TXT.is_file() and EMISSION_META.is_file() and CONFIG.is_file()


@lru_cache(maxsize=1)
def load():
    import lightgbm as lgb
    return (lgb.Booster(model_file=str(EMISSION_TXT)),
            json.loads(EMISSION_META.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def settings() -> dict:
    import yaml
    if not CONFIG.is_file():
        raise FileNotFoundError("{} not found; freeze it with: python tasks.py tune --variant "
                                "touch_f2 --freeze".format(CONFIG))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return {"c_switch": float(cfg["decoder"]["c_switch"]),
            "c_die": float(cfg["decoder"]["c_die"]),
            "c_restart": float(cfg["decoder"]["c_restart"]),
            "temperature": float(cfg["emission"]["temperature"]),
            "dead_bias": float(cfg["emission"]["dead_bias"]),
            "refine": bool(cfg["refine"]),
            "w_change": 0.0,
            "model_variant": cfg["model_variant"], "source": CONFIG.name}


def decoder_params() -> DecoderParams:
    s = settings()
    return DecoderParams(c_switch=s["c_switch"], c_die=s["c_die"], c_restart=s["c_restart"])


def evidence(tracking, spec, F):
    """(logp (n, 3), None); zeros outside the graded halves. The second value keeps the signature
    of emission_v3.evidence (v4 has no change-point model)."""
    booster, meta = load()
    s = settings()
    if meta["variant"] != s["model_variant"]:
        raise ValueError("config/final_v4.yaml was tuned for variant {!r} but the shipped model "
                         "is {!r}".format(s["model_variant"], meta["variant"]))
    names, X = learn_features.build(tracking, spec, F)
    graded = spec.graded_mask()
    cols = [names.index(k) for k in meta["features"]]
    raw = booster.predict(X[graded][:, cols], raw_score=True, num_threads=1)
    z = raw / s["temperature"]
    z = z - z.max(axis=1, keepdims=True)
    logp_g = z - np.log(np.exp(z).sum(axis=1, keepdims=True)) - np.log(np.asarray(meta["prior"]))
    logp_g[:, 2] += s["dead_bias"]
    logp = np.zeros((tracking.n_frames, 3))
    logp[graded] = logp_g
    return logp, None
