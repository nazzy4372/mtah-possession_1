"""The official upstream code, imported in one place.

The upstream repository is a git submodule at ``upstream/`` and is never modified. Its modules are
written as scripts that expect their own folders on ``sys.path``, so this is the only module in the
project that touches ``sys.path``. Everything else imports from here:

    from mtah.official import pm, games, baseline

``pm`` is ``evaluation/possession_metrics.py``: the metric is imported, never reimplemented.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "upstream"
EVALUATION_DIR = UPSTREAM / "evaluation"
STARTER_DIR = UPSTREAM / "starter"
VALIDATE_SCRIPT = EVALUATION_DIR / "validate_submission.py"

if not (EVALUATION_DIR / "possession_metrics.py").is_file():
    raise ImportError(
        "upstream submodule not found at {}. Run: git submodule update --init".format(UPSTREAM))

for _path in (UPSTREAM, EVALUATION_DIR, STARTER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import games  # noqa: E402  upstream/games.py: per-match constants and published half blocks
import possession_metrics as pm  # noqa: E402  upstream/evaluation/possession_metrics.py
import baseline  # noqa: E402  upstream/starter/baseline.py: reference algorithm (parity tests)

__all__ = ["pm", "games", "baseline", "REPO_ROOT", "UPSTREAM", "VALIDATE_SCRIPT"]
