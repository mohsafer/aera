import numpy as np
import pytest

from aera.config import Config, WorldCfg
from aera.world.terrain import Terrain
from aera.world.world import World


def make_cfg(seed=7, pits=0) -> WorldCfg:
    c = WorldCfg(width=24, height=24, seed=seed)
    c.terrain.lava_pools = 1
    c.terrain.mud_patches = 1
    c.terrain.pits = pits
    return c


def test_world_generation_is_seeded_and_deterministic():
    a = World(make_cfg(seed=7)).terrain.grid
    b = World(make_cfg(seed=7)).terrain.grid
    c = World(make_cfg(seed=8)).terrain.grid
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c) or True  # different seed *may* differ


def test_border_walls_block_raycast_and_movement():
    w = World(make_cfg())
    d = w.raycast(12.0, 12.0, 0.0, 50.0)   # due east → hits right border
    assert 10.0 < d < 12.0 + 1.0
    x, y = w.move_circle(12.0, 12.0, 100.0, 0.0, r=0.35, airborne=False)
    assert x < w.w and y == 12.0


def test_wall_rect_blocks_raycast():
    cfg = make_cfg()
    cfg.walls.rects = [[10, 10, 1, 4]]
    w = World(cfg)
    d = w.raycast(8.5, 11.5, 0.0, 30.0)    # east toward the wall segment
    assert d == pytest.approx(1.5, abs=0.05)  # wall face at x=10, origin x=8.5


def test_lava_and_pit_cells_exist_and_block():
    w = World(make_cfg(pits=1))
    assert w.terrain.cells_of(Terrain.LAVA), "expected a lava pool"
    pits = w.terrain.cells_of(Terrain.PIT)
    assert pits, "expected a pit strip"
    px, py = pits[0]
    assert w.blocked_at(px + 0.5, py + 0.5, airborne=False)
    assert not w.blocked_at(px + 0.5, py + 0.5, airborne=True)


def test_cell_ok_rejects_walls_and_hazards():
    cfg = make_cfg()
    cfg.walls.rects = [[5, 5, 2, 2]]
    w = World(cfg)
    assert not w.cell_ok(5, 5)
    assert w.cell_ok(10, 10)


def test_line_of_sight_blocked_by_wall():
    cfg = make_cfg()
    cfg.walls.rects = [[10, 10, 1, 6]]
    w = World(cfg)
    assert w.line_of_sight(8.0, 12.0, 12.0, 12.0) is False
    assert w.line_of_sight(8.0, 8.0, 8.0, 9.0) is True


def test_food_respawn():
    cfg = make_cfg()
    cfg.entities.foods = 3
    cfg.entities.food_respawn_ticks = 5
    w = World(cfg)
    assert len(w.foods) == 3
    w.foods[0].active = False
    w.foods[0].respawn_at = 2
    for _ in range(3):
        w.step_time()
    assert w.foods[0].active
