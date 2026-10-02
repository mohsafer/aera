"""Typed, JSON-serialisable configuration for worlds, agents, rewards, sim.

Every experiment must be reproducible from one JSON file (see configs/).
Dataclasses carry defaults; `Config.load` merges a JSON dict over them so
partial configs are fine.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class TerrainCfg:
    mud_patches: int = 2          # number of mud blobs
    lava_pools: int = 2           # number of lava blobs
    pits: int = 0                 # number of pit strips (crossable only by jumping)
    road: bool = False            # paved path column (fast terrain)


@dataclass
class WallsCfg:
    border: bool = True
    rects: list = field(default_factory=list)   # [cx, cy, w, h] in cells


@dataclass
class EntitiesCfg:
    foods: int = 8
    food_respawn_ticks: int = 300  # -1 = no respawn
    tools: list = field(default_factory=list)   # e.g. ["boots"]
    beacon: list | None = None                  # [cx, cy] or None


@dataclass
class WorldCfg:
    name: str = "Open Field"
    width: int = 32
    height: int = 32
    seed: int = 7
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    walls: WallsCfg = field(default_factory=WallsCfg)
    entities: EntitiesCfg = field(default_factory=EntitiesCfg)


@dataclass
class AgentCfg:
    kind: str = "walker"          # "walker" | "rover"
    radius: float = 0.35
    max_energy: float = 100.0
    max_health: float = 100.0
    hunger_rate: float = 0.8      # energy/sec at rest
    effort_rate: float = 1.1      # extra energy/sec at full effort
    lava_damage: float = 9.0      # hp/sec while on lava
    view_rays: int = 12
    view_range: float = 10.0
    fov_deg: float = 270.0
    cpg: bool = True              # include gait-phase clock in walker obs
    scent: bool = True            # include direction-to-nearest-food in obs


@dataclass
class RewardCfg:
    w_food: float = 3.0
    w_survive: float = 0.01       # per simulated second alive
    w_novelty: float = 0.06       # per newly visited cell
    w_progress: float = 0.05      # per metre closed toward nearest food
    w_damage: float = 0.5         # per hp lost
    w_death: float = 5.0
    w_beacon: float = 10.0
    beacon_terminates: bool = False


@dataclass
class SimCfg:
    dt: float = 0.1               # control timestep (s)
    max_steps: int = 2000


@dataclass
class Config:
    world: WorldCfg = field(default_factory=WorldCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    # Curriculum: list of {"until_episode": N, "set": {attr: value, ...}}.
    # "set" writes attributes onto RewardCfg/AgentCfg by name; the last stage
    # with until_episode == -1 (or the last entry) is the end state.
    curriculum: list = field(default_factory=lambda: [{"until_episode": -1, "set": {}}])

    # ------------------------------------------------------------------ io
    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return Config.from_dict(json.load(f))

    @staticmethod
    def from_dict(d: dict) -> "Config":
        cfg = Config()
        for key, value in (d or {}).items():
            if not hasattr(cfg, key):
                raise KeyError(f"unknown config key: {key}")
            current = getattr(cfg, key)
            if is_dataclass(current) and isinstance(value, dict):
                setattr(cfg, key, _merge(type(current), current, value))
            else:
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ------------------------------------------------------------ curriculum
    def stage_for_episode(self, episode: int) -> dict:
        """Overrides dict that should be active at `episode` (0-based)."""
        active: dict = {}
        for stage in self.curriculum:
            until = int(stage.get("until_episode", -1))
            if until < 0 or episode < until:
                active = dict(active, **stage.get("set", {}))
            else:
                break
        return active

    def apply_overrides(self, overrides: dict) -> None:
        """Write `attr: value` pairs onto RewardCfg / AgentCfg by name."""
        for section in (self.reward, self.agent):
            for attr, val in (overrides or {}).items():
                if hasattr(section, attr):
                    setattr(section, attr, val)


def _merge(cls, base: Any, incoming: dict) -> Any:
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(cls)}
    fields.update(incoming or {})
    return cls(**fields)
