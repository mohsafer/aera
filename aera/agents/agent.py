"""The agent as an *embodied entity* (not the policy — the policy lives in
aera/rl). The env drives it: senses → action → physics → consequences."""
from __future__ import annotations

import numpy as np

from ..config import AgentCfg
from ..world.physics import PHYSICS, WalkerPhysics


class Agent:
    def __init__(self, cfg: AgentCfg, x: float, y: float, heading: float = 0.0):
        self.cfg = cfg
        self.kind = cfg.kind
        self.x = x
        self.y = y
        self.heading = heading
        self.v = 0.0
        self.effort = 0.0
        self.joints = np.zeros(4, dtype=np.float32)   # walker only
        self.phase = 0.0
        self.airborne = 0.0
        self.jump_cd = 0.0
        self.energy = cfg.max_energy
        self.health = cfg.max_health
        self.inventory: set[str] = set()
        self.cells_seen: set[tuple[int, int]] = set()
        self.foods_eaten = 0
        self.steps_alive = 0
        self.speed_sum = 0.0          # ∫v dt over the episode → mean speed
        self.sensor_noise = 0.0       # set by the env (fog anomaly); senses
        self.events: list[str] = []
        self.physics = PHYSICS[cfg.kind]

    # ------------------------------------------------------------ helpers
    @property
    def is_walker(self) -> bool:
        return isinstance(self.physics, WalkerPhysics)

    def speed(self) -> float:
        return self.v

    def reset(self, x: float, y: float, heading: float) -> None:
        self.x, self.y, self.heading = x, y, heading
        self.v = 0.0
        self.effort = 0.0
        self.joints[:] = 0.0
        self.phase = 0.0
        self.airborne = 0.0
        self.jump_cd = 0.0
        self.energy = self.cfg.max_energy
        self.health = self.cfg.max_health
        self.inventory.clear()
        self.cells_seen.clear()
        self.foods_eaten = 0
        self.steps_alive = 0
        self.speed_sum = 0.0
        self.sensor_noise = 0.0
        self.events.clear()

    def record_cell(self) -> bool:
        """Returns True the first time this cell is visited (novelty)."""
        cell = (int(self.x), int(self.y))
        if cell in self.cells_seen:
            return False
        self.cells_seen.add(cell)
        return True

    def exploration_ratio(self, world) -> float:
        return len(self.cells_seen) / world.passable_cells()
