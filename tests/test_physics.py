import numpy as np

from aera.agents.agent import Agent
from aera.config import AgentCfg, WorldCfg
from aera.world.physics import integrate, gait_score
from aera.world.world import World


def make_agent(kind="rover", cfg: AgentCfg | None = None) -> Agent:
    c = cfg or AgentCfg(kind=kind)
    return Agent(c, x=12.0, y=12.0, heading=0.0)


def make_world() -> World:
    return World(WorldCfg(width=24, height=24, seed=5))


def test_rover_moves_forward_with_throttle():
    w, a = make_world(), make_agent("rover")
    x0, y0 = a.x, a.y
    for _ in range(30):
        a.physics.step(a, np.array([0.0, 1.0]), w, 0.1)
        integrate(a, w, 0.1)
    assert (a.x - x0) ** 2 + (a.y - y0) ** 2 > 4.0   # travelled > 2m


def test_rover_stops_at_border_wall():
    w = World(WorldCfg(width=8, height=8, seed=1))
    a = make_agent("rover")
    a.x, a.y = 4.0, 4.0
    for _ in range(120):
        a.physics.step(a, np.array([0.0, 1.0]), w, 0.1)   # heading 0 → east
        integrate(a, w, 0.1)
    assert a.x <= w.w - 0.3   # never through the border


def test_gait_score_prefers_antiphase_hips():
    flail = gait_score(np.array([0.8, 0.7, 0.9, 0.2]))     # same-phase hips
    gait = gait_score(np.array([0.8, 0.4, -0.8, 0.4]))     # antiphase + knees
    assert gait > 0.7
    assert gait > flail


def test_walker_gait_beats_flailing():
    w = World(WorldCfg(width=24, height=24, seed=2))
    walk = make_agent("walker")
    for _ in range(60):
        # alternating gait targets via the CPG-ish pattern
        t = len(walk.cells_seen)  # unused; fixed pattern below
        a = np.array([0.0, 0.8, 0.4, -0.8, 0.4, 0.0])
        walk.physics.step(walk, a, w, 0.1)
        integrate(walk, w, 0.1)
    d_gait = walk.x - 12.0

    flail = make_agent("walker")
    flail.x, flail.y = 12.0, 12.0
    for i in range(60):
        a = np.array([0.0, 0.9, 0.9, 0.9, 0.2, 0.0])   # in-phase → no thrust
        flail.physics.step(flail, a, w, 0.1)
        integrate(flail, w, 0.1)
    d_flail = flail.x - 12.0
    assert d_gait > d_flail + 0.5


def test_pit_blocks_until_jump():
    cfg = WorldCfg(width=20, height=20, seed=9)
    cfg.terrain.pits = 1
    w = World(cfg)
    pit = w.terrain.cells_of(4)[0]            # Terrain.PIT == 4
    px, py = pit[0] + 0.5, pit[1] + 0.5
    a = make_agent("walker")
    a.x, a.y = px - 1.0, py
    for _ in range(40):                        # walk east into the pit
        a.physics.step(a, np.array([0.0, 0.9, 0.5, -0.9, 0.5, 0.0]), w, 0.1)
        integrate(a, w, 0.1)
        assert not (px - 0.3 <= a.x <= px + 0.3 and a.airborne <= 0.0), \
            "ground agent entered a pit cell"
    # now give boots + jump
    a.x, a.y = px - 1.5, py
    a.inventory.add("boots")
    crossed = False
    for _ in range(60):
        a.physics.step(a, np.array([0.0, 0.9, 0.5, -0.9, 0.5, 1.0]), w, 0.1)
        integrate(a, w, 0.1)
        if a.x > px + 0.6:
            crossed = True
            break
    assert crossed, "jumping agent should clear a 1-cell pit"


def test_terrain_friction_slows_mud():
    cfg = WorldCfg(width=24, height=24, seed=4)
    w = World(cfg)
    a1 = make_agent("rover")
    a2 = make_agent("rover")
    w.terrain.grid[:, :] = 0          # all grass
    w.terrain.grid[10:14, 10:14] = 2  # mud square (Terrain.MUD)
    for _ in range(40):
        a1.physics.step(a1, np.array([0.0, 1.0]), w, 0.1)
        integrate(a1, w, 0.1)
    w.terrain.grid[10:14, 10:14] = 0  # back to grass
    a2.x, a2.y = a1.x, a1.y           # same start
    a2.v = a1.v
    for _ in range(40):
        a2.physics.step(a2, np.array([0.0, 1.0]), w, 0.1)
        integrate(a2, w, 0.1)
    assert a2.x > a1.x                # grass run goes further than mud run
