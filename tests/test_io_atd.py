import math

import numpy as np
import pytest

from mtah import io_atd

PITCH = (105.3, 68.0)

META_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
  <Teams><Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
  <Players>
    <Player id="P1" teamId="T2"><Name>Track 1</Name><ShirtNumber>1</ShirtNumber></Player>
    <Player id="P2" teamId="T1"><Name>Track 2</Name><ShirtNumber>2</ShirtNumber></Player>
  </Players>
</Metadata></main>
"""


def write_bundle(tmp_path, lines):
    meta = tmp_path / "m_atd.xml"
    raw = tmp_path / "m_atd.txt"
    meta.write_text(META_TEMPLATE, encoding="utf-8")
    with open(raw, "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return meta, raw


def test_handmade_file_values_units_and_indexing(tmp_path):
    meta, raw = write_bundle(tmp_path, [
        "1:0.5,0.25,36;NaN,NaN,NaN:0.1,0.2,72",
        "2:NaN,NaN,NaN;1,1,0:NaN,NaN,NaN",
        "3:0,0,3.6;0.12345,0.5,NaN:0.5,0.5,0",
    ])
    tr = io_atd.load(meta, raw, PITCH, n_frames=3)
    assert tr.n_frames == 3
    # first <Team> is home; slot order is <Player> order
    assert tr.slot_team.tolist() == [io_atd.AWAY, io_atd.HOME]
    assert (tr.meta.home_name, tr.meta.away_name) == ("Home FC", "Away FC")
    # row i = frame_id i + 1; normalised -> metres; s is already m/s
    assert tr.players_xy[0, 0].tolist() == [0.5 * 105.3, 0.25 * 68.0]
    assert tr.players_speed[0, 0] == 36.0
    assert tr.ball_speed[0] == 72.0
    assert np.isnan(tr.players_xy[0, 1]).all() and np.isnan(tr.ball_xy[1]).all()
    assert tr.players_xy[1, 1].tolist() == [105.3, 68.0]
    assert np.isnan(tr.players_speed[2, 1])
    # bit-exact with Python float(), which the upstream baseline uses
    assert tr.players_xy[2, 1, 0] == float("0.12345") * 105.3
    assert tr.ball_observed.tolist() == [True, False, True]


def test_trailing_separator_is_tolerated(tmp_path):
    meta, raw = write_bundle(tmp_path, ["1:0.5,0.5,1;0.5,0.5,1:0.5,0.5,1;",
                                        "2:0.5,0.5,1;0.5,0.5,1:0.5,0.5,1;"])
    assert io_atd.load(meta, raw, PITCH).n_frames == 2


@pytest.mark.parametrize("lines, fragment", [
    (["1:0.98765,0.5,1;0.5,0.5,1:0.5,0.5,1", "2:0.98765,0.5,1:0.5,0.5,1"], "fields per line"),
    (["1:0.98765,0.5,1;0.5,0.5,1:0.5,0.5,1", "3:0.98765,0.5,1;0.5,0.5,1:0.5,0.5,1"],
     "frame ids"),
    (["1:0.98765,0.5,1;0.5,0.5,1:0.5,0.5,1", "2:0.98765,zz9,1;0.5,0.5,1:0.5,0.5,1"],
     "parseable"),
])
def test_malformed_files_raise_without_quoting_data(tmp_path, lines, fragment):
    meta, raw = write_bundle(tmp_path, lines)
    with pytest.raises(io_atd.ATDFormatError) as info:
        io_atd.load(meta, raw, PITCH)
    message = str(info.value)
    assert fragment in message
    assert "0.98765" not in message and "zz9" not in message


def test_record_count_is_checked(tmp_path):
    meta, raw = write_bundle(tmp_path, ["1:0.5,0.5,1;0.5,0.5,1:0.5,0.5,1"])
    with pytest.raises(io_atd.ATDFormatError):
        io_atd.load(meta, raw, PITCH, n_frames=2)


def test_synthetic_bundle_parses(small_bundle):
    tr = io_atd.load_game(small_bundle.path, small_bundle.spec)
    assert tr.n_frames == small_bundle.spec.n_frames
    assert tr.meta.n_slots == 46
    assert tr.meta.layout_check == "verified"
    assert tr.slot_team.tolist() == small_bundle.truth["slot_team"]
    # ball at rest on the centre spot just before the first kickoff
    k = small_bundle.kickoffs[1]
    assert tr.ball_xy[k - 5].tolist() == [0.5 * 105.3, 0.5 * 68.0]


def test_kloppy_reads_synthetic_bundle_identically(small_bundle):
    """kloppy reads the real Metrica EPTS export; if it reads ours, our format is the real one."""
    from kloppy import metrica

    paths = small_bundle.spec.paths(small_bundle.path)
    dataset = metrica.load_tracking_epts(meta_data=str(paths["atd_meta"]),
                                         raw_data=str(paths["atd_raw"]), coordinates="metrica")
    tr = io_atd.load_game(small_bundle.path, small_bundle.spec)
    assert len(dataset.records) == tr.n_frames
    length, width = tr.pitch
    checked = 0
    for record in dataset.records[::97]:
        row = record.frame_id - 1
        ball = record.ball_coordinates
        if ball is None or math.isnan(ball.x):
            assert np.isnan(tr.ball_xy[row, 0])
        else:
            assert ball.x == pytest.approx(tr.ball_xy[row, 0] / length, abs=1e-9)
            assert ball.y == pytest.approx(tr.ball_xy[row, 1] / width, abs=1e-9)
            checked += 1
        for player, data in record.players_data.items():
            slot = int(player.jersey_no) - 1
            if data.coordinates is not None and not math.isnan(data.coordinates.x):
                assert data.coordinates.x == pytest.approx(tr.players_xy[row, slot, 0] / length,
                                                           abs=1e-9)
    assert checked > 10
