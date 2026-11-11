"""Emission model v2: LightGBM trained on noise-augmented external games (ext/train.py).

The model files are shipped in the repository (models/), trained only on other games' data:
  models/emission_v2.txt   LightGBM text model
  models/emission_v2.json  feature list, temperature, class prior, decoder costs, provenance

Inference is single-threaded and deterministic. The output is log p(x | state) up to a constant:
the temperature-calibrated log posterior minus the log class prior, ready for the Viterbi decoder.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from mtah import learn_features
from mtah.decode import DecoderParams

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_TXT = MODEL_DIR / "emission_v2.txt"
MODEL_META = MODEL_DIR / "emission_v2.json"


@lru_cache(maxsize=2)
def load(model_txt: str = str(MODEL_TXT), meta_json: str = str(MODEL_META)):
    import lightgbm as lgb
    if not Path(model_txt).is_file():
        raise FileNotFoundError("{} not found; train it with: python -m ext.train final"
                                .format(model_txt))
    booster = lgb.Booster(model_file=model_txt)
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    return booster, meta


def decoder_params(meta=None) -> DecoderParams:
    meta = meta or load()[1]
    d = meta["decoder"]
    return DecoderParams(c_switch=d["c_switch"], c_die=d["c_die"], c_restart=d["c_restart"])


def _log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def emissions(tracking, spec, F) -> np.ndarray:
    """(n, 3) log-emissions for H, A, D; zeros outside the graded halves."""
    booster, meta = load()
    names, X = learn_features.build(tracking, spec, F)
    cols = [names.index(k) for k in meta["features"]]
    graded = spec.graded_mask()
    raw = booster.predict(X[graded][:, cols], raw_score=True, num_threads=1)
    out = np.zeros((tracking.n_frames, 3))
    out[graded] = _log_softmax(raw / meta["temperature"]) - np.log(np.asarray(meta["prior"]))
    return out
