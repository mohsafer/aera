"""The World: terrain grid + wall rects + entities + geometric queries.

Units are metres; one cell = 1m. The world is *static* except for food
respawn timers; agents live outside this class (see aera/agents) and ask the
world for movement collision (`move_circle`) and sensing geometry
(`raycast`, `line_of_sight`).
"""
from __future__ import annotations

import math

import numpy as np

from ..config import WorldCfg
from .entities import Beacon, Food, Predator, Tool, Workbench, place_free
from .terrain import Terrain, TerrainGrid


def _rects_to_floats(cell_rects: list) -> list[tuple[float, float, float, float]]:
    return [(float(x), float(y), float(w), float(h)) for x, y, w, h in cell_rects]


class World:
    def __init__(self, cfg: WorldCfg):
        self.cfg = cfg
        self.w = float(cfg.width)
        self.h = float(cfg.height)
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = 0

        t = cfg.terrain
        self.terrain = TerrainGrid(cfg.width, cfg.height, self.rng,
                                   mud_patches=t.mud_patches, lava_pools=t.lava_pools,
                                   pits=t.pits, road=t.road)

        # walls: border + configured rects (cell coords → metres)
        self.walls: list[tuple[float, float, float, float]] = []
        if cfg.walls.border:
            self.walls += [(-1.0, -1.0, self.w + 2.0, 1.0),      # top
                           (-1.0, self.h, self.w + 2.0, 1.0),    # bottom
                           (-1.0, -1.0, 1.0, self.h + 2.0),      # left
                           (self.w, -1.0, 1.0, self.h + 2.0)]    # right
        self.walls += _rects_to_floats(cfg.walls.rects)

        self.foods: list[Food] = []
        self.tools: list[Tool] = []
        self.predators: list[Predator] = []
        self.workbenches: list[Workbench] = []
        self.beacon: Beacon | None = None
        self.wind = (0.0, 0.0)   # drift (m/s) applied by integrate; anomalies
        self._spawn_entities()

        self._passable_cells = self._count_passable()

    # ----------------------------------------------------------- spawning
    def _spawn_entities(self) -> None:
        e = self.cfg.entities
        for (x, y) in place_free(e.foods, self, self.rng):
            self.foods.append(Food(x=x, y=y))
        for kind in e.tools:
            spots = place_free(1, self, self.rng, min_spawn_dist=0.0)
            if spots:
                self.tools.append(Tool(x=spots[0][0], y=spots[0][1], kind=kind))
        for kind in e.components:
            spots = place_free(1, self, self.rng, min_spawn_dist=0.0)
            if spots:
                self.tools.append(Tool(x=spots[0][0], y=spots[0][1], kind=kind))
        for _ in range(e.workbenches):
            spots = place_free(1, self, self.rng, min_spawn_dist=3.0)
            if spots:
                self.workbenches.append(Workbench(x=spots[0][0], y=spots[0][1]))
        for _ in range(e.predators):
            spots = place_free(1, self, self.rng, min_spawn_dist=4.0)
            if spots:
                self.predators.append(Predator(x=spots[0][0], y=spots[0][1],
                                               speed=e.predator_speed))
        if e.beacon is not None:
            bx, by = e.beacon
            self.beacon = Beacon(x=float(bx) + 0.5, y=float(by) + 0.5)

    def spawn_predator(self, rng, min_agent_dist: float, agent) -> Predator | None:
        """Anomaly arrival: drop a predator on a free cell at least
        `min_agent_dist` away from the agent (never materialize on top)."""
        for _ in range(40):
            cx = int(rng.integers(1, self.w - 1))
            cy = int(rng.integers(1, self.h - 1))
            if not self.cell_ok(cx, cy):
                continue
            x, y = cx + 0.5, cy + 0.5
            if agent is not None and math.hypot(x - agent.x, y - agent.y) < min_agent_dist:
                continue
            p = Predator(x=x, y=y, speed=self.cfg.entities.predator_speed)
            self.predators.append(p)
            return p
        return None

    def update_predators(self, dt: float, targets: list, tick: int) -> None:
        """Scripted chase AI: pursue the nearest target in aggro range,
        otherwise random-walk. Uses world.rng → deterministic per seed."""
        for p in self.predators:
            if tick < p.stun_until:
                continue
            tx, ty, td = None, None, math.inf
            for a in targets:
                d = math.hypot(a.x - p.x, a.y - p.y)
                if d < td:
                    tx, ty, td = a.x, a.y, d
            if tick < p.fear_until and tx is not None:
                p.heading = math.atan2(p.y - ty, p.x - tx)   # flee the flare
                step = 1.2 * p.speed * dt
            elif tx is not None and td < p.aggro_range:
                p.heading = math.atan2(ty - p.y, tx - p.x)
                step = p.speed * dt
            else:
                if tick >= p.wander_until:
                    p.heading = float(self.rng.uniform(0, 2 * math.pi))
                    p.wander_until = tick + int(self.rng.integers(10, 40))
                step = 0.4 * p.speed * dt
            nx = p.x + math.cos(p.heading) * step
            ny = p.y + math.sin(p.heading) * step
            if self._circle_free(nx, p.y, 0.3, False):
                p.x = nx
            if self._circle_free(p.x, ny, 0.3, False):
                p.y = ny

    def reset_episode(self) -> None:
        """Per-episode reset: foods come back, tools return to pedestals,
        anomaly scars (lava/mud rewrites) heal."""
        self.tick = 0
        self.wind = (0.0, 0.0)
        self.terrain.restore()
        # keep the configured predator population, drop anomaly arrivals
        self.predators = self.predators[:self.cfg.entities.predators]
        for f in self.foods:
            f.active = True
            f.respawn_at = -1
        for tool in self.tools:
            tool.taken = False
        if self.beacon is not None:
            self.beacon.hit = False

    def step_time(self) -> None:
        self.tick += 1
        for f in self.foods:
            if not f.active and f.respawn_at >= 0 and self.tick >= f.respawn_at:
                f.active = True
                f.respawn_at = -1

    # ------------------------------------------------------------ queries
    def cell_ok(self, cx: int, cy: int) -> bool:
        """A cell is ok for spawning if terrain is safe and no wall covers it."""
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return False
        t = Terrain(int(self.terrain.grid[cy, cx]))
        if t in (Terrain.LAVA, Terrain.PIT):
            return False
        return not self._rect_contains(self.walls, cx + 0.5, cy + 0.5)

    def passable_cells(self) -> int:
        return self._passable_cells

    def _count_passable(self) -> int:
        n = 0
        for cy in range(int(self.h)):
            for cx in range(int(self.w)):
                if self.cell_ok(cx, cy):
                    n += 1
        return max(1, n)

    @staticmethod
    def _rect_contains(rects, x: float, y: float) -> bool:
        for (rx, ry, rw, rh) in rects:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return True
        return False

    def blocked_at(self, x: float, y: float, airborne: bool) -> bool:
        """Point blocked for a ground-mover? PIT blocks unless airborne."""
        if self._rect_contains(self.walls, x, y):
            return True
        t = self.terrain.at(x, y)
        return t == Terrain.PIT and not airborne

    def terrain_at(self, x: float, y: float) -> Terrain:
        return self.terrain.at(x, y)

    def move_circle(self, x: float, y: float, dx: float, dy: float,
                    r: float, airborne: bool) -> tuple[float, float]:
        """Axis-separated movement of a circle with push-back against walls
        and pit cells. Cheap, stable, good enough at this scale.

        The displacement is applied in sub-steps of at most r/2 so one large
        step can never leap clean over a 1 m wall or pit cell (tunneling)."""
        dist = math.hypot(dx, dy)
        steps = 1 + int(dist / (0.5 * r))
        sx, sy = dx / steps, dy / steps
        for _ in range(steps):
            nx = x + sx
            if self._circle_free(nx, y, r, airborne):
                x = nx
            ny = y + sy
            if self._circle_free(x, ny, r, airborne):
                y = ny
        return x, y

    def _circle_free(self, x: float, y: float, r: float, airborne: bool) -> bool:
        return not (self.blocked_at(x - r, y - r, airborne) or
                    self.blocked_at(x + r, y - r, airborne) or
                    self.blocked_at(x - r, y + r, airborne) or
                    self.blocked_at(x + r, y + r, airborne))

    # ------------------------------------------------------------- raycast
    def raycast(self, ox: float, oy: float, angle: float,
                max_dist: float) -> float:
        """Distance to the nearest wall (including world bounds) along a ray."""
        best = max_dist
        dx, dy = math.cos(angle), math.sin(angle)
        for (rx, ry, rw, rh) in self.walls:
            t = _ray_slab(ox, oy, dx, dy, rx, ry, rw, rh)
            if t is not None and 0.0 <= t < best:
                best = t
        return best

    def line_of_sight(self, ax: float, ay: float, bx: float, by: float) -> bool:
        d = math.hypot(bx - ax, by - ay)
        if d < 1e-6:
            return True
        return self.raycast(ax, ay, math.atan2(by - ay, bx - ax), d) >= d - 1e-3


def _ray_slab(ox: float, oy: float, dx: float, dy: float,
              rx: float, ry: float, rw: float, rh: float) -> float | None:
    """Ray vs AABB slab test → entry distance or None."""
    tmin, tmax = -math.inf, math.inf
    for o, d, lo, hi in ((ox, dx, rx, rx + rw), (oy, dy, ry, ry + rh)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
    if tmax < 0:
        return None
    return max(tmin, 0.0)
