"""Smart Tagging reader and the label-free set-piece diagnostic."""

import sys

import numpy as np

from mtah import io_tagging, pipeline
from mtah.official import REPO_ROOT, pm
from mtah.spec import GameSpec

TOOLS = REPO_ROOT / "tools"

XML = """<?xml version="1.0" encoding="utf-8"?>
<file><ALL_INSTANCES>
<instance><ID>1</ID><start>4.00</start><end>9.00</end><code>SET PIECES</code>
<label><group>Team</group><text>N/A</text></label>
<label><group>Type</group><text>Throw-in</text></label></instance>
<instance><ID>2</ID><start>10.00</start><end>20.00</end><code>BUILD UP</code>
<label><group>Team</group><text>Barcelona</text></label></instance>
<instance><ID>3</ID><start>20.00</start><end>24.00</end><code>DEFENDING IN MIDDLE THIRD</code>
<label><group>Team</group><text>Atletic Club</text></label></instance>
</ALL_INSTANCES></file>
"""


def test_reader_frames_case_and_team_mapping(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text(XML, encoding="utf-8")
    ivs = io_tagging.load(path, fps=25)
    assert [(iv.start, iv.end) for iv in ivs] == [(100, 225), (250, 500), (500, 600)]
    assert ivs[0].is_set_piece and ivs[0].kind == "throw-in" and not ivs[1].is_set_piece
    sides = io_tagging.team_sides(ivs, "FC Barcelona", "Athletic Club")
    assert sides == {"Barcelona": io_tagging.HOME, "Atletic Club": io_tagging.AWAY}
    assert io_tagging.side_of(ivs[0], sides) is None            # N/A is missing, not a team
    assert io_tagging.team_sides(ivs, "Team One", "Team One") == {}


def test_set_piece_diagnostic_counts():
    sys.path.insert(0, str(TOOLS))
    import tagging_stats

    spec = GameSpec(name="t", stem="t", n_frames=1000, fps=25, pitch=(105.0, 68.0),
                    halves=((1, 0, 999),))
    codes = np.full(1000, pm.H, dtype=np.int8)
    codes[50:110] = pm.D                     # dead run, restart at frame 110
    codes[600:700] = pm.A
    iv = io_tagging.Interval
    pieces = [iv("set pieces", 100, 225, {"type": "Throw-in", "team": "N/A"}),      # respected
              iv("set pieces", 800, 900, {"type": "Corner Kick", "team": "Home"})]  # no dead ball
    phase = [iv("build up", 600, 700, {"team": "Away"}), iv("build up", 300, 400, {"team": "Away"})]
    r = tagging_stats.collect(codes, pieces + phase, spec, {"Home": 0, "Away": 1})
    rows = {row["kind"]: row for row in r["pieces"]}
    assert rows["throw-in"]["near"] == [True, True, True]
    assert rows["throw-in"]["dead_before_3s"] > 0.6 and not rows["throw-in"]["no_dead_at_all"]
    assert rows["corner kick"]["near"] == [False, False, False]
    assert rows["corner kick"]["no_dead_at_all"]
    assert r["our_dead_runs"] == 1 and r["our_dead_runs_tagged"] == [1, 1, 1]
    assert r["offsets_s"] == [0.4]                                   # frame 110 - frame 100
    assert r["phases"]["build up"]["n"] == 2 and r["phases"]["build up"]["same"] == 100


def test_team_evidence_and_prior():
    iv = io_tagging.Interval
    sides = {"Home": io_tagging.HOME, "Away": io_tagging.AWAY}
    intervals = [iv("build up", 10, 30, {"team": "Home"}),
                 iv("progression", 25, 40, {"team": "Away"}),          # conflict on 25..29
                 iv("long ball", 50, 60, {"team": "Home"}),            # not team evidence
                 iv("defending in middle third", 60, 70, {"team": "Home"}),
                 iv("set pieces", 70, 80, {"team": "Home", "type": "Corner Kick"}),
                 iv("creating chances", 80, 90, {"team": "N/A"})]
    ev = io_tagging.team_evidence(intervals, sides, 100)
    assert (ev[10:25] == io_tagging.HOME).all() and (ev[30:40] == io_tagging.AWAY).all()
    assert (ev[25:30] == io_tagging.NONE).all() and (ev[40:] == io_tagging.NONE).all()

    logp = np.zeros((100, 3))
    out = io_tagging.add_team_prior(logp, ev, 0.2)
    assert np.array_equal(out[:, 2], logp[:, 2])                       # dead ball never touched
    assert (out[10:25, 1] == -0.2).all() and (out[10:25, 0] == 0).all()
    assert (out[30:40, 0] == -0.2).all() and (out[30:40, 1] == 0).all()
    assert (out[40:] == 0).all() and (logp == 0).all()                 # input not modified


def test_v5_is_v4_without_tags_and_follows_strong_tags(small_bundle):
    from mtah import emission_v5, io_atd
    if not emission_v5.available():
        import pytest
        pytest.skip("v5 config not frozen")
    spec = small_bundle.spec
    tr = io_atd.load_game(small_bundle.path, spec)
    v4, kos = pipeline.label_tracking(tr, spec, "v4")
    none = np.full(spec.n_frames, io_tagging.NONE, dtype=np.int8)
    same, _ = pipeline.label_tracking(tr, spec, "v5", kickoffs=kos, tag_evidence=none)
    assert np.array_equal(same, v4)
    # the synthetic bundle's tagging has kickoffs only, so the file-based evidence is empty too
    assert (emission_v5.tag_evidence(small_bundle.path, spec) == io_tagging.NONE).all()
    home = np.full(spec.n_frames, io_tagging.HOME, dtype=np.int8)
    pushed, _ = pipeline.label_tracking(tr, spec, "v5", kickoffs=kos, tag_evidence=home)
    assert np.sum(pushed == "A") < np.sum(v4 == "A")
    assert np.sum(pushed == "D") <= np.sum(v4 == "D") * 1.05 + 25     # dead ball barely moves
    import pytest
    with pytest.raises(ValueError):
        pipeline.label_tracking(tr, spec, "v5", kickoffs=kos)


def test_weight_rule_refuses_a_weight_that_can_lose():
    from mtah.official import REPO_ROOT as root
    sys.path.insert(0, str(root))
    from ext import tags

    def scenario(gains):
        return {"rows": [{"spell": [0.40 + g]} for g in gains]}
    result = {"weights": [0.0, 0.1, 0.2, 0.4],
              "scenarios": {"clean": scenario([0, 0.004, 0.009, 0.014]),
                            "real-like": scenario([0, 0.003, 0.006, 0.010]),
                            "10% wrong team": scenario([0, 0.001, 0.003, 0.003]),
                            "20% wrong team": scenario([0, 0.003, 0.005, -0.001]),
                            "30% wrong team": scenario([0, 0.002, 0.001, -0.001])}}
    weight, _ = tags.choose_weight(result)
    assert weight == 0.2                       # 0.4 loses when the tags are bad; 0.2 never does
    result["scenarios"]["30% wrong team"] = scenario([0, -0.001, -0.002, -0.004])
    assert tags.choose_weight(result)[0] == 0.0


def test_tagging_stats_tool_runs_on_a_bundle(small_bundle, tmp_path):
    from test_tools import run_tool

    result = pipeline.run(small_bundle.path, small_bundle.spec)
    path = tmp_path / "sub.csv"
    pipeline.write_submission(path, result.labels)
    proc = run_tool(TOOLS / "tagging_stats.py", path, "--data-dir", small_bundle.path,
                    "--spec", small_bundle.path / "game_spec.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tagged intervals" in proc.stdout and "set pieces = tagged restarts" in proc.stdout
