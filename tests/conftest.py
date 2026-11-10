"""Shared fixtures: synthetic bundles in the exact ATD format, written once per test session.

Synthetic bundles live in pytest's temporary directory, never in the repo and never next to the
real match data.
"""

import json

import pytest

from mtah import synthetic
from mtah.spec import GameSpec


class Bundle:
    def __init__(self, path):
        self.path = path
        self.spec = GameSpec.load(path / "game_spec.json")
        self.truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
        self.gt = path / "gt.csv"

    @property
    def kickoffs(self):
        return {int(k): v for k, v in self.truth["kickoffs"].items()}


@pytest.fixture(scope="session")
def small_bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("syn_small")
    synthetic.generate(path, "small", seed=0)
    return Bundle(path)


@pytest.fixture(scope="session")
def full_bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("syn_full")
    synthetic.generate(path, "full", seed=0)
    return Bundle(path)
