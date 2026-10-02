"""Egocentric senses: build the observation vector the policy sees.

Layout (also exposed via `obs_layout` for HUD/inspection):

  [0 : R*6]            R rays x [dist_norm, wall, lava, food, tool, beacon]
  [proprio]            speed, sin/cos heading, (walker: sin/cos phase, 4 joints),
                       airborne, jump_cd, energy, health, effort
  [3x3 terrain one-hot] 9 cells x N_TERRAIN
  [inventory]          tool flags (4 reserved slots, boots first)
  [scent]              sin, cos, dist_norm toward nearest active food (opt-in)
  [beacon]             sin, cos, dist_norm toward beacon (zeros if none)

Walls occlude entity sightings. Lava is "seen" by marching the ray through
terrain cells, so a ray pointing at lava reports danger at the lava's edge.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import AgentCfg
from ..world.terrain import N_TERRAIN, Terrain
from ..world.world import World

INVENTORY_SLOTS = ("boots", "key", "lens", "grapple")   # reserved; v1 uses boots
N_INVENTORY = len(INVENTORY_SLOTS)


def action_dim(kind: str) -> int:
    return 2 if kind == "rover" else 6


def action_space_low_high(kind: str) -> tuple[np.ndarray, np.ndarray]:
    d = action_dim(kind)
    return np.full(d, -1.0, np.float32), np.full(d, 1.0, np.float32)


def proprio_dim(cfg: AgentCfg) -> int:
    d = 3                                   # speed, sin h, cos h
    if cfg.kind == "walker":
        d += 4 + (2 if cfg.cpg else 0)      # joints (+ sin/cos gait phase)
    d += 5                                  # airborne, jump_cd, energy, health, effort
    return d


def obs_dim(cfg: AgentCfg) -> int:
    return (cfg.view_rays * 6 + proprio_dim(cfg) + 9 * N_TERRAIN
            + N_INVENTORY + (3 if cfg.scent else 0) + 3)


def obs_layout(cfg: AgentCfg) -> list[tuple[str, int]]:
    seg = [("rays", cfg.view_rays * 6), ("proprio", proprio_dim(cfg)),
           ("terrain3x3", 9 * N_TERRAIN), ("inventory", N_INVENTORY)]
    if cfg.scent:
        seg.append(("scent", 3))
    seg.append(("beacon", 3))
    return seg


def build_obs(agent, world: World, cfg: AgentCfg) -> np.ndarray:
    out = np.zeros(obs_dim(cfg), dtype=np.float32)
    o = 0

    # ---- vision rays -----------------------------------------------------
    fov = math.radians(cfg.fov_deg)
    r0 = agent.heading - fov / 2.0
    step_ang = fov / max(1, cfg.view_rays - 1)
    entities = _visible_entities(agent, world, cfg)
    for i in range(cfg.view_rays):
        ang = r0 + i * step_ang if cfg.view_rays > 1 else agent.heading
        d_wall = world.raycast(agent.x, agent.y, ang, cfg.view_range)
        d_lava = _lava_distance(agent, world, ang, d_wall)
        d_ent, ent_type = _entity_on_ray(agent, entities, ang, step_ang, d_wall)
        dist = min(d_wall, d_lava, d_ent)
        out[o + 0] = dist / cfg.view_range
        out[o + 1] = 1.0 if dist == d_wall else 0.0
        out[o + 2] = 1.0 if dist == d_lava and d_lava < d_wall else 0.0
        out[o + 3] = 1.0 if ent_type == "food" else 0.0
        out[o + 4] = 1.0 if ent_type == "tool" else 0.0
        out[o + 5] = 1.0 if ent_type == "beacon" else 0.0
        o += 6

    # ---- proprioception ----------------------------------------------------
    out[o] = agent.v / 2.6
    out[o + 1] = math.sin(agent.heading)
    out[o + 2] = math.cos(agent.heading)
    p = o + 3
    if cfg.kind == "walker":
        if cfg.cpg:
            out[p] = math.sin(agent.phase)
            out[p + 1] = math.cos(agent.phase)
            p += 2
        out[p:p + 4] = agent.joints
        p += 4
    out[p + 0] = 1.0 if agent.airborne > 0.0 else 0.0
    out[p + 1] = agent.jump_cd / 2.0
    out[p + 2] = agent.energy / cfg.max_energy
    out[p + 3] = agent.health / cfg.max_health
    out[p + 4] = agent.effort
    o += proprio_dim(cfg)

    # ---- 3x3 terrain patch -------------------------------------------------
    base = o
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            t = int(world.terrain_at(agent.x + dx, agent.y + dy))
            out[base + t] = 1.0
            base += N_TERRAIN
    o += 9 * N_TERRAIN

    # ---- inventory -----------------------------------------------------------
    for k, name in enumerate(INVENTORY_SLOTS):
        out[o + k] = 1.0 if name in agent.inventory else 0.0
    o += N_INVENTORY

    # ---- scent (reward-as-observation: direction to nearest food) ----------
    if cfg.scent:
        d, ang = _nearest_food(agent, world)
        if d is not None and d < 8.0:
            rel = ang - agent.heading
            out[o] = math.sin(rel)
            out[o + 1] = math.cos(rel)
            out[o + 2] = d / 8.0
        o += 3

    # ---- beacon direction ----------------------------------------------------
    if world.beacon is not None:
        dx, dy = world.beacon.x - agent.x, world.beacon.y - agent.y
        d = math.hypot(dx, dy)
        rel = math.atan2(dy, dx) - agent.heading
        out[o] = math.sin(rel)
        out[o + 1] = math.cos(rel)
        out[o + 2] = min(1.0, d / max(world.w, world.h))
    return out


# ------------------------------------------------------------------ helpers
def _visible_entities(agent, world: World, cfg: AgentCfg) -> list:
    """Foods/tools/beacon within view range and line of sight."""
    ents = []
    for f in world.foods:
        if f.active:
            ents.append(("food", f.x, f.y))
    for t in world.tools:
        if not t.taken:
            ents.append(("tool", t.x, t.y))
    if world.beacon is not None:
        ents.append(("beacon", world.beacon.x, world.beacon.y))
    out = []
    for (kind, ex, ey) in ents:
        d = math.hypot(ex - agent.x, ey - agent.y)
        if d <= cfg.view_range and world.line_of_sight(agent.x, agent.y, ex, ey):
            out.append((kind, ex, ey, d))
    return out


def _entity_on_ray(agent, entities, ang: float, half_bin: float, d_wall: float):
    best_d, best_kind = math.inf, ""
    half = half_bin / 2.0 + 0.05
    for (kind, ex, ey, d) in entities:
        a = math.atan2(ey - agent.y, ex - agent.x)
        diff = abs((a - ang + math.pi) % (2 * math.pi) - math.pi)
        if diff <= half and d < best_d and d <= d_wall + 1e-3:
            best_d, best_kind = d, kind
    return best_d, best_kind


def _lava_distance(agent, world: World, ang: float, d_wall: float) -> float:
    step = 0.4
    d = step
    while d < d_wall:
        if world.terrain_at(agent.x + math.cos(ang) * d,
                            agent.y + math.sin(ang) * d) == Terrain.LAVA:
            return d
        d += step
    return math.inf


def _nearest_food(agent, world: World):
    best = (None, None)
    bd = math.inf
    for f in world.foods:
        if not f.active:
            continue
        d = math.hypot(f.x - agent.x, f.y - agent.y)
        if d < bd:
            bd = d
            best = (d, math.atan2(f.y - agent.y, f.x - agent.x))
    return best
