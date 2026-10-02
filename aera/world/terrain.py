"""Terrain grid: the ground layer of the world.

One value per 1m cell. Terrain affects locomotion (FRICTION), damages
(LAVA), or blocks ground movement (PIT — crossable only while airborne,
i.e. with the "boots" tool).
"""
from __future__ import annotations

import enum

import numpy as np


class Terrain(enum.IntEnum):
    GRASS = 0
    ROAD = 1
    MUD = 2
    LAVA = 3
    PIT = 4


N_TERRAIN = len(Terrain)

# multiplier applied to horizontal speed while the agent's centre is on the cell
FRICTION = {
    Terrain.GRASS: 1.0,
    Terrain.ROAD: 1.15,
    Terrain.MUD: 0.45,
    Terrain.LAVA: 1.0,
    Terrain.PIT: 0.0,
}
DANGEROUS = frozenset({Terrain.LAVA})
BLOCKING = frozenset({Terrain.PIT})


class TerrainGrid:
    def __init__(self, width: int, height: int, rng: np.random.Generator,
                 mud_patches: int = 2, lava_pools: int = 2, pits: int = 0,
                 road: bool = False):
        self.w = width
        self.h = height
        self._rng = rng
        self.grid = np.full((height, width), Terrain.GRASS.value, dtype=np.int8)
        if road:
            self._road()
        for _ in range(mud_patches):
            self._blob(Terrain.MUD, size=int(0.02 * width * height) + 8)
        for _ in range(lava_pools):
            self._blob(Terrain.LAVA, size=int(0.012 * width * height) + 5)
        for _ in range(pits):
            self._pit_strip()

    # ---------------------------------------------------------------- gen
    def _inner(self) -> tuple[int, int]:
        """Random cell inside the outermost ring (border stays walkable)."""
        return (int(self._rng.integers(1, self.h - 1)),
                int(self._rng.integers(1, self.w - 1)))

    def _road(self) -> None:
        x = int(self._rng.integers(4, self.w - 4))
        self.grid[:, x:x + 2] = Terrain.ROAD.value

    def _blob(self, kind: Terrain, size: int) -> None:
        """Random-walk blob; mostly steps to a neighbour, rarely teleports so
        patches stay contiguous but never hug the border."""
        y, x = self._inner()
        for _ in range(size):
            if 0 < y < self.h - 1 and 0 < x < self.w - 1:
                self.grid[y, x] = kind.value
            if self._rng.random() < 0.12:
                y, x = self._inner()
            else:
                y = int(np.clip(y + int(self._rng.integers(-1, 2)), 0, self.h - 1))
                x = int(np.clip(x + int(self._rng.integers(-1, 2)), 0, self.w - 1))

    def _pit_strip(self) -> None:
        """A short strip of PIT the agent must jump over (boots tool)."""
        if self._rng.random() < 0.5:
            x = int(self._rng.integers(3, self.w - 3))
            y0 = int(self._rng.integers(3, max(4, self.h - 10)))
            length = min(7, self.h - 2 - y0)
            self.grid[y0:y0 + length, x] = Terrain.PIT.value
        else:
            y = int(self._rng.integers(3, self.h - 3))
            x0 = int(self._rng.integers(3, max(4, self.w - 10)))
            length = min(7, self.w - 2 - x0)
            self.grid[y, x0:x0 + length] = Terrain.PIT.value

    # ------------------------------------------------------------ queries
    def at(self, x: float, y: float) -> Terrain:
        cx = int(np.clip(int(x), 0, self.w - 1))
        cy = int(np.clip(int(y), 0, self.h - 1))
        return Terrain(int(self.grid[cy, cx]))

    def cells_of(self, kind: Terrain) -> list[tuple[int, int]]:
        ys, xs = np.nonzero(self.grid == kind.value)
        return list(zip(xs.tolist(), ys.tolist()))
