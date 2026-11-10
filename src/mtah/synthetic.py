"""Minimal synthetic match in the exact ATD bundle format.

Purpose: test file format, plumbing, kickoff detection and determinism. NOT football realism --
realistic training and tuning data comes from external games in M3. The model is deliberately
simple:

  * possession runs with a random team and a log-normal length; each ends in a turnover or a
    ball-out, which gives a dead ball of 3-10 s and a restart by a random team;
  * 22 players on formation anchors plus slow random drift; the team in possession pushes up;
  * the ball follows a "carrier" of the team in possession, which changes every few seconds;
  * the kickoff shape before each half's kickoff: ball at rest on the centre spot, each team in
    its own half;
  * light corruption: bursty ball dropout (~40%), players hopping between tracklet slots of their
    team's pool, short player dropouts.

Bundle written (same layout as the real data):
    <out>/tactical/<stem>_atd.txt, <stem>_atd.xml, <stem>_smart_tagging.xml
    <out>/gt.csv          frame,label ground truth (X outside the graded halves)
    <out>/game_spec.json  GameSpec
    <out>/truth.json      kickoff frames and other generator facts for tests

Two presets. ``small``: about 4 minutes with its own GameSpec. ``full``: the BAR-ATH layout
(199,319 frames, the published half blocks); it simulates a few minutes per half and fills the rest
of each half by cycling the simulated post-kickoff frames, which keeps generation to seconds.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import lfilter

from mtah.spec import GameSpec

FPS = 25
PITCH = (105.3, 68.0)
N_SLOTS = 46
HOME_NAME, AWAY_NAME = "Synthetic Home", "Synthetic Away"
TEAM_IDS = ("FIFATMA", "FIFATMB")
PRE_KICKOFF = 3 * FPS            # frames of kickoff shape simulated before each kickoff
NAN_GROUP = "NaN,NaN,NaN"

# Formation anchors in "attacking +x" coordinates; all inside the team's own half.
ANCHOR_X = np.array([5, 20, 20, 20, 20, 35, 35, 35, 35, 48, 48], dtype=float)
ANCHOR_Y = np.array([34, 8, 26, 42, 60, 10, 27, 41, 58, 24, 44], dtype=float)


@dataclass
class HalfPlan:
    half: int
    first: int      # graded block, inclusive
    last: int
    kickoff: int
    kick_team: int  # 0 home, 1 away
    sim_frames: int  # simulated frames after the kickoff (rest of the half is cycled)


def preset(name: str) -> Tuple[GameSpec, List[HalfPlan]]:
    if name == "small":
        halves = ((1, 500, 2749), (2, 3250, 5749))
        spec = GameSpec(name="SYN-SMALL", stem="SYN-SMALL_tactical", n_frames=6000, fps=FPS,
                        pitch=PITCH, halves=halves)
        plans = [HalfPlan(1, 500, 2749, kickoff=500, kick_team=0, sim_frames=2250),
                 HalfPlan(2, 3250, 5749, kickoff=3500, kick_team=1, sim_frames=2250)]
        return spec, plans
    if name == "full":
        spec = GameSpec.from_upstream("BAR-ATH")
        (h1, a1, b1), (h2, a2, b2) = spec.halves
        # Synthetic kickoff offsets, chosen for the test: half 1 is graded from its kickoff, half 2
        # kicks off 80 s into its block.
        plans = [HalfPlan(h1, a1, b1, kickoff=a1, kick_team=0, sim_frames=180 * FPS),
                 HalfPlan(h2, a2, b2, kickoff=a2 + 80 * FPS, kick_team=1, sim_frames=180 * FPS)]
        return spec, plans
    raise ValueError("unknown preset {!r}".format(name))


@dataclass
class HalfSim:
    start: int               # first simulated video frame
    labels: np.ndarray       # (m,) "H"/"A"/"D"
    players_xy: np.ndarray   # (m, 22, 2) metres, players 0-10 home, 11-21 away
    ball_xy: np.ndarray      # (m, 2) metres


def _anchors(team: int, half: int) -> np.ndarray:
    """(11, 2) formation for a team; home attacks +x in half 1, directions swap at half time."""
    attacks_plus = (team == 0) == (half == 1)
    x = ANCHOR_X if attacks_plus else PITCH[0] - ANCHOR_X
    return np.stack([x, ANCHOR_Y], axis=1)


def simulate_half(plan: HalfPlan, rng: np.random.Generator) -> HalfSim:
    start = min(plan.first, plan.kickoff - PRE_KICKOFF)
    n_pre = plan.kickoff - start
    m = n_pre + plan.sim_frames
    centre = np.array([PITCH[0] / 2.0, PITCH[1] / 2.0])

    labels = np.full(m, "D", dtype="<U1")
    in_poss = np.full(m, -1, dtype=np.int8)       # team in possession while alive
    target = np.tile(centre, (m, 1))               # where the ball heads
    carrier = np.full(m, -1, dtype=np.int16)       # index 0..21 of the carrier while alive

    # possession / dead-ball event sequence after the kickoff
    t, team = n_pre, plan.kick_team
    first_carrier = 9 + 11 * team                  # a forward
    while t < m:
        length = int(np.clip(rng.lognormal(np.log(8 * FPS), 0.7), FPS, 40 * FPS))
        stop = min(m, t + length)
        labels[t:stop] = "H" if team == 0 else "A"
        in_poss[t:stop] = team
        c, s = first_carrier, t
        first_carrier = -1
        while s < stop:
            if c < 0:
                c = 11 * team + int(rng.integers(1, 11))
            hold = int(rng.integers(int(1.5 * FPS), 4 * FPS))
            carrier[s:min(stop, s + hold)] = c
            s += hold
            c = -1
        t = stop
        if t >= m:
            break
        if rng.random() < 0.65:                    # turnover: the other team, ball alive
            team = 1 - team
        else:                                      # ball out over a touchline, dead 3-10 s
            dead = int(rng.integers(3 * FPS, 10 * FPS))
            labels[t:min(m, t + dead)] = "D"
            carrier[t:min(m, t + dead)] = -2       # marker: resolve the out point below
            t += dead
            team = int(rng.integers(0, 2))

    # player motion: anchors + possession push + AR(1) drift
    anchors = np.concatenate([_anchors(0, plan.half), _anchors(1, plan.half)])   # (22, 2)
    direction = np.array([1.0 if (k < 11) == (plan.half == 1) else -1.0 for k in range(22)])
    push_raw = np.zeros((m, 22))
    for k in range(22):
        own = 0 if k < 11 else 1
        push_raw[:, k] = np.where(in_poss == own, 20.0, np.where(in_poss >= 0, -3.0, 0.0))
    push = lfilter([0.05], [1.0, -0.95], push_raw, axis=0)
    drift = lfilter([1.0], [1.0, -0.98], rng.normal(0.0, 0.08, size=(m, 22, 2)), axis=0)
    drift[:n_pre] *= 0.2                          # standing still-ish before kickoff
    push[:n_pre] = 0.0
    players = anchors[None] + drift
    players[..., 0] += push * direction[None]
    players[..., 0] = np.clip(players[..., 0], 0.5, PITCH[0] - 0.5)
    players[..., 1] = np.clip(players[..., 1], 0.5, PITCH[1] - 0.5)

    # ball target: carrier while alive, rest point while dead
    rest = centre.copy()
    for i in range(m):
        if carrier[i] >= 0:
            target[i] = players[i, carrier[i]] + (0.6, 0.0)
            rest = target[i]
        elif carrier[i] == -2:
            if i == 0 or carrier[i - 1] != -2:      # ball goes out at the nearest touchline
                y_out = -0.5 if rest[1] < PITCH[1] / 2 else PITCH[1] + 0.5
                rest = np.array([np.clip(rest[0], 1.0, PITCH[0] - 1.0), y_out])
            target[i] = rest
        else:
            target[i] = centre                     # pre-kickoff
    alpha = 0.25
    ball = lfilter([alpha], [1.0, -(1.0 - alpha)], target - centre, axis=0) + centre
    ball[:n_pre] = centre                          # exactly at rest on the spot
    return HalfSim(start=start, labels=labels, players_xy=players, ball_xy=ball)


def corrupt(sim: HalfSim, n_pre: int, slot_pools, rng: np.random.Generator):
    """(slot_xy (m, 46, 2) with NaN, ball_xy with NaN). Players hop between their team's slots."""
    m = len(sim.labels)
    ball = sim.ball_xy.copy()
    # bursty ball dropout after the kickoff, ~40% missing
    missing, state = np.zeros(m, dtype=bool), False
    draws = rng.random(m)
    for i in range(n_pre, m):
        state = (draws[i] < 0.04) if not state else (draws[i] >= 0.06)
        missing[i] = state
    if n_pre > 20:                                  # one short dropout while resting on the spot
        missing[n_pre - 15:n_pre - 12] = True
    ball[missing] = np.nan

    slots = np.full((m, N_SLOTS, 2), np.nan)
    for team in (0, 1):
        pool = list(slot_pools[team])
        current = list(pool[:11])
        free = pool[11:]
        gap_left = [0] * 11
        hop = rng.random((m, 11))
        gap = rng.random((m, 11))
        for i in range(m):
            for j in range(11):
                if i >= n_pre and hop[i, j] < 1.0 / (40 * FPS):
                    k = int(rng.integers(0, len(free)))
                    free[k], current[j] = current[j], free[k]
                if gap_left[j] > 0:
                    gap_left[j] -= 1
                    continue
                if i >= n_pre and gap[i, j] < 1.0 / (60 * FPS):
                    gap_left[j] = int(rng.integers(10, 75))
                    continue
                slots[i, current[j]] = sim.players_xy[i, 11 * team + j]
    return slots, ball


def _speed_kmh(xy: np.ndarray) -> np.ndarray:
    step = np.full(xy.shape[:-1], np.nan)
    step[1:] = np.hypot(xy[1:, ..., 0] - xy[:-1, ..., 0], xy[1:, ..., 1] - xy[:-1, ..., 1])
    return step * FPS * 3.6


def _group(x: float, y: float, s: float) -> str:
    if np.isnan(x) or np.isnan(y):
        return NAN_GROUP
    return "{:.5f},{:.5f},{}".format(x / PITCH[0], y / PITCH[1],
                                     "NaN" if np.isnan(s) else "{:.2f}".format(s))


def format_bodies(slots: np.ndarray, ball: np.ndarray) -> List[str]:
    """Line bodies (everything after '<frame_id>:') for simulated frames."""
    slot_speed = _speed_kmh(slots)
    ball_speed = _speed_kmh(ball)
    bodies = []
    for i in range(len(ball)):
        players = ";".join(_group(slots[i, k, 0], slots[i, k, 1], slot_speed[i, k])
                           for k in range(N_SLOTS))
        bodies.append(players + ":" + _group(ball[i, 0], ball[i, 1], ball_speed[i]))
    return bodies


def metadata_xml(slot_team: np.ndarray, n_frames: int, plans: List[HalfPlan]) -> str:
    """EPTS metadata in the structure kloppy's Metrica EPTS reader expects."""
    players = "\n".join(
        '      <Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
        '<ShirtNumber>{0}</ShirtNumber></Player>'.format(i + 1, TEAM_IDS[int(t)])
        for i, t in enumerate(slot_team))
    channels = "\n".join(
        '      <PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'.format(i + 1, c)
        for i in range(len(slot_team)) for c in ("x", "y", "s"))
    refs = "\n".join(
        '        <SplitRegister separator=",">'
        '<PlayerChannelRef playerChannelId="player{0}_x"/>'
        '<PlayerChannelRef playerChannelId="player{0}_y"/>'
        '<PlayerChannelRef playerChannelId="player{0}_s"/></SplitRegister>'.format(i + 1)
        for i in range(len(slot_team)))
    names = ("first_half", "second_half")
    periods = "\n".join(
        '        <ProviderParameter><Name>{0}_start</Name><Value>{1}</Value></ProviderParameter>\n'
        '        <ProviderParameter><Name>{0}_end</Name><Value>{2}</Value></ProviderParameter>'
        .format(names[i], p.first + 1, p.last + 1) for i, p in enumerate(plans[:2]))
    return """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FileDate>2026-01-01</FileDate>
      <TrackingType>Optical</TrackingType>
      <ProviderName>Metrica Sports</ProviderName>
      <FrameRate>{fps}</FrameRate>
      <ProviderGlobalParameters>
{periods}
      </ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions>
      <Session id="Session_1">
        <SessionType>Match</SessionType>
        <MatchParameters>
          <Score idLocalTeam="{home_id}" idVisitingTeam="{away_id}">
            <LocalTeamScore>0</LocalTeamScore>
            <VisitingTeamScore>0</VisitingTeamScore>
          </Score>
        </MatchParameters>
        <FieldSize><Width>{width}</Width><Height>{height}</Height></FieldSize>
      </Session>
    </Sessions>
    <Teams>
      <Team id="{home_id}"><Name>{home}</Name><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name><Value>left_to_right</Value></ProviderParameter></ProviderTeamsParameters></Team>
      <Team id="{away_id}"><Name>{away}</Name><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name><Value>right_to_left</Value></ProviderParameter></ProviderTeamsParameters></Team>
    </Teams>
    <Players>
{players}
    </Players>
    <Devices>
      <Device id="Device_1">
        <Name>Device 1</Name>
        <Sensors>
          <Sensor id="position">
            <Name>Position</Name>
            <Channels>
              <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
              <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
            </Channels>
          </Sensor>
          <Sensor id="speed">
            <Name>Speed</Name>
            <Channels>
              <Channel id="s"><Name>speed</Name><Unit>km/h</Unit></Channel>
            </Channels>
          </Sensor>
        </Sensors>
      </Device>
    </Devices>
    <PlayerChannels>
{channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification separator=":" startFrame="1" endFrame="{n}">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{refs}
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
        <BallChannelRef channelId="s"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
""".format(fps=FPS, periods=periods, home_id=TEAM_IDS[0], away_id=TEAM_IDS[1],
           width=int(round(PITCH[0])), height=int(round(PITCH[1])), home=HOME_NAME,
           away=AWAY_NAME, players=players, channels=channels, n=n_frames, refs=refs)


def smart_tagging_xml(plans: List[HalfPlan]) -> str:
    """Minimal SportsCode XML: one 'Kick Off (Start)' set piece per half (M2 fills in phases)."""
    instances = []
    for i, p in enumerate(plans):
        instances.append(
            "  <instance><ID>{id}</ID><start>{start:.2f}</start><end>{end:.2f}</end>"
            "<code>SET PIECES</code>"
            "<label><group>Team</group><text>{team}</text></label>"
            "<label><group>Half</group><text>{half}</text></label>"
            "<label><group>Type</group><text>Kick Off (Start)</text></label></instance>".format(
                id=i + 1, start=(p.kickoff - 2 * FPS) / FPS, end=(p.kickoff + 3 * FPS) / FPS,
                team=HOME_NAME if p.kick_team == 0 else AWAY_NAME,
                half="1st Half" if p.half == 1 else "2nd Half"))
    return ('<?xml version="1.0" encoding="utf-8"?>\n<file>\n<ALL_INSTANCES>\n'
            + "\n".join(instances) + "\n</ALL_INSTANCES>\n</file>\n")


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def generate(out_dir, preset_name: str = "small", seed: int = 0) -> Dict:
    """Write a synthetic bundle; return generator facts (also saved as truth.json)."""
    spec, plans = preset(preset_name)
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "tactical").mkdir(parents=True, exist_ok=True)

    slot_team = rng.permutation(np.repeat(np.array([0, 1], dtype=np.int8), N_SLOTS // 2))
    pools = [np.flatnonzero(slot_team == t) for t in (0, 1)]
    for pool in pools:
        rng.shuffle(pool)

    n = spec.n_frames
    body_index = np.full(n, -1, dtype=np.int64)      # index into bodies, -1 = all missing
    gt = np.full(n, "X", dtype="<U1")
    bodies: List[str] = []
    for plan in plans:
        sim = simulate_half(plan, rng)
        n_pre = plan.kickoff - sim.start
        slots, ball = corrupt(sim, n_pre, pools, rng)
        offset = len(bodies)
        bodies.extend(format_bodies(slots, ball))
        m = len(sim.labels)
        frames = np.arange(sim.start, sim.start + m)
        keep = frames <= plan.last
        body_index[frames[keep]] = offset + np.arange(m)[keep]
        labels = sim.labels.copy()
        # cycle the post-kickoff part to fill the rest of the half
        sim_end = sim.start + m - 1
        if sim_end < plan.last:
            rest = np.arange(sim_end + 1, plan.last + 1)
            src = n_pre + (rest - sim_end - 1) % (m - n_pre)
            body_index[rest] = offset + src
        block = np.arange(plan.first, plan.last + 1)
        rel = block - sim.start
        cyc = np.where(rel < m, rel, n_pre + (rel - m) % (m - n_pre))
        gt[block] = labels[cyc]

    nan_body = ";".join([NAN_GROUP] * N_SLOTS) + ":" + NAN_GROUP
    paths = spec.paths(out)
    with open(paths["atd_raw"], "w", encoding="ascii", newline="\n") as handle:
        batch = []
        for f in range(n):
            idx = body_index[f]
            batch.append("{}:{}\n".format(f + 1, bodies[idx] if idx >= 0 else nan_body))
            if len(batch) >= 5000:
                handle.write("".join(batch))
                batch = []
        handle.write("".join(batch))
    _write_text(paths["atd_meta"], metadata_xml(slot_team, n, plans))
    _write_text(paths["smart_tagging"], smart_tagging_xml(plans))
    _write_text(out / "gt.csv", "frame,label\n" + "".join(
        "{},{}\n".format(i, label) for i, label in enumerate(gt)))
    spec.save(out / "game_spec.json")
    truth = {"preset": preset_name, "seed": seed,
             "kickoffs": {str(p.half): p.kickoff for p in plans},
             "slot_team": slot_team.tolist()}
    _write_text(out / "truth.json", json.dumps(truth, indent=2, sort_keys=True) + "\n")
    return truth
