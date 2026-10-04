"""Pickable/placeable world entities (walls live in world.py as plain rects)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Food:
    x: float
    y: float
    active: bool = True
    respawn_at: int = -1        # world tick when it comes back (-1 = never)
    energy: float = 35.0
    score: int = 1


@dataclass
class Tool:
    x: float
    y: float
    kind: str                   # "boots" unlocks the jump action channel
    taken: bool = False


@dataclass
class Predator:
    """Scripted threat (alien): wanders, chases the nearest target within
    aggro range, bites on contact then stuns itself briefly — escape is
    always possible. Updated by World.update_predators, NOT by RL."""
    x: float
    y: float
    heading: float = 0.0
    speed: float = 1.1
    aggro_range: float = 6.0
    stun_until: int = -1
    wander_until: int = 0


@dataclass
class Beacon:
    x: float
    y: float
    z: float = 0.0              # visual height only
    hit: bool = False


def place_free(n: int, world, rng, min_spawn_dist: float = 1.5) -> list[tuple[float, float]]:
    """Scatter `n` points on walkable, safe cells away from walls."""
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < n * 400:
        tries += 1
        cx = int(rng.integers(1, world.w - 1))
        cy = int(rng.integers(1, world.h - 1))
        if not world.cell_ok(cx, cy):
            continue
        p = (cx + 0.5, cy + 0.5)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_spawn_dist ** 2 for q in pts):
            pts.append(p)
    return pts
