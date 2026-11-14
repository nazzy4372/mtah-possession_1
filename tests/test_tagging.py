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


def test_tagging_stats_tool_runs_on_a_bundle(small_bundle, tmp_path):
    from test_tools import run_tool

    result = pipeline.run(small_bundle.path, small_bundle.spec)
    path = tmp_path / "sub.csv"
    pipeline.write_submission(path, result.labels)
    proc = run_tool(TOOLS / "tagging_stats.py", path, "--data-dir", small_bundle.path,
                    "--spec", small_bundle.path / "game_spec.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tagged intervals" in proc.stdout and "set pieces = tagged restarts" in proc.stdout
