"""Our parser + vectorised nearest player must equal upstream starter/baseline.py exactly."""

import numpy as np

from mtah import baseline_port, io_atd
from mtah.official import baseline


def test_nearest_player_codes_match_upstream_exactly(small_bundle):
    spec = small_bundle.spec
    paths = spec.paths(small_bundle.path)
    teams = baseline.team_of_each_track(str(paths["atd_meta"]))
    upstream = baseline.nearest_player_labels(str(paths["atd_raw"]), teams, spec.n_frames,
                                              spec.pitch)
    tracking = io_atd.load_game(small_bundle.path, spec)
    ours = baseline_port.nearest_player_codes(tracking)
    assert np.array_equal(teams, tracking.slot_team)
    assert np.array_equal(upstream, ours)
    assert (ours != baseline_port.NOBODY).sum() > 1000  # the check is not vacuous


def test_placeholder_is_upstream_smooth50(small_bundle):
    tracking = io_atd.load_game(small_bundle.path, small_bundle.spec)
    expected = baseline.smooth(baseline_port.nearest_player_codes(tracking), 50)
    assert np.array_equal(baseline_port.placeholder_labels(tracking), expected)
