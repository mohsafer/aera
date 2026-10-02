"""HUD panels drawn at full window resolution on top of the chunky 3D view:
minimap with fog-of-war, agent vitals, event feed, reward chart, help."""
from __future__ import annotations

import math
from collections import deque

import pygame

from .renderer3d import TERRAIN_COLORS as TC

INK = (232, 230, 220)
DIM = (140, 140, 135)
PANEL = (18, 20, 26)
GOOD = (120, 220, 120)
WARN = (240, 180, 70)
BAD = (240, 90, 70)


class Chart:
    def __init__(self, maxlen=240):
        self.series: deque = deque(maxlen=maxlen)

    def push(self, value: float):
        self.series.append(value)

    def draw(self, surf, rect, label="episode return"):
        pygame.draw.rect(surf, PANEL, rect)
        pygame.draw.rect(surf, DIM, rect, 1)
        if len(self.series) < 2:
            return
        vals = list(self.series)
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        pts = []
        for i, v in enumerate(vals):
            x = rect.x + 4 + i * (rect.w - 8) / (len(vals) - 1)
            y = rect.y + rect.h - 6 - (v - lo) / (hi - lo) * (rect.h - 22)
            pts.append((x, y))
        pygame.draw.lines(surf, GOOD, False, pts, 2)
        f = pygame.font.SysFont("consolas,menlo,monospace", 13)
        surf.blit(f.render(f"{label}  [{lo:.0f} … {hi:.0f}]", True, DIM),
                  (rect.x + 6, rect.y + 4))


def draw_minimap(surf, world, agent, rect, font):
    pygame.draw.rect(surf, PANEL, rect)
    pygame.draw.rect(surf, DIM, rect, 1)
    sx = rect.w / world.w
    sy = rect.h / world.h
    for gy in range(int(world.h)):
        for gx in range(int(world.w)):
            t = int(world.terrain.grid[gy, gx])
            c = TC[t][0]
            if agent is not None and (gx, gy) not in agent.cells_seen:
                c = tuple(v // 4 for v in c)   # fog of war
            pygame.draw.rect(surf, c,
                             (rect.x + gx * sx, rect.y + gy * sy, sx + 0.5, sy + 0.5))
    for (rx, ry, rw, rh) in world.walls:
        pygame.draw.rect(surf, (30, 30, 34),
                         (rect.x + rx * sx, rect.y + ry * sy, rw * sx, rh * sy))
    for f in world.foods:
        if f.active:
            pygame.draw.circle(surf, GOOD, (rect.x + f.x * sx, rect.y + f.y * sy), 2)
    for tool in world.tools:
        if not tool.taken:
            pygame.draw.circle(surf, WARN, (rect.x + tool.x * sx, rect.y + tool.y * sy), 3)
    if world.beacon is not None:
        pygame.draw.circle(surf, (88, 224, 240), (rect.x + world.beacon.x * sx,
                                                  rect.y + world.beacon.y * sy), 3)
    if agent is not None:
        pos = (rect.x + agent.x * sx, rect.y + agent.y * sy)
        pygame.draw.circle(surf, INK, pos, 3)
        pygame.draw.line(surf, INK, pos,
                         (pos[0] + 7 * math.cos(agent.heading),
                          pos[1] + 7 * math.sin(agent.heading)))
    surf.blit(font.render(world.cfg.name, True, DIM), (rect.x + 6, rect.y + 3))


def draw_vitals(surf, env, rect, font, big):
    pygame.draw.rect(surf, PANEL, rect)
    pygame.draw.rect(surf, DIM, rect, 1)
    a = env.agent
    x, y = rect.x + 10, rect.y + 8

    def bar(label, frac, color):
        nonlocal y
        surf.blit(font.render(label, True, DIM), (x, y))
        pygame.draw.rect(surf, (50, 54, 60), (x + 92, y + 2, rect.w - 112, 12))
        pygame.draw.rect(surf, color, (x + 92, y + 2, max(0, (rect.w - 112) * frac), 12))
        y += 20

    inv = ",".join(a.inventory) or "-"
    surf.blit(big.render(f"agent: {a.kind}   ep {env.episode:5d}", True, INK), (x, y)); y += 26
    bar("energy", a.energy / a.cfg.max_energy, WARN)
    bar("health", a.health / a.cfg.max_health, BAD)
    if a.is_walker:
        gait = float(max(0.0, min(1.0, 1.0 - abs(a.joints[0] + a.joints[2]) / 2)))
        bar("gait", gait, GOOD)
    bar("explored", a.exploration_ratio(env.world), GOOD)
    stats = (f"speed {a.v:4.2f} m/s   foods {a.foods_eaten:3d}   "
             f"step {env.steps:5d}   ret {env.ep_ret:7.1f}")
    surf.blit(font.render(stats, True, INK), (x, y)); y += 20
    surf.blit(font.render(f"inventory: {inv}", True, DIM), (x, y)); y += 20


def draw_feed(surf, events: deque, rect, font):
    pygame.draw.rect(surf, PANEL, rect)
    pygame.draw.rect(surf, DIM, rect, 1)
    y = rect.y + 6
    surf.blit(font.render("event feed", True, DIM), (rect.x + 8, y)); y += 20
    for e in list(events)[-9:]:
        surf.blit(font.render(f"› {e}", True, INK), (rect.x + 8, y)); y += 18


HELP = [
    "AERA — controls",
    "space  pause / resume          s  single step (paused)",
    "v      cycle camera (orbit-chase-top-fp)",
    "f      follow agent / world    arrows / drag  orbit",
    "wheel  zoom                    h  toggle this help",
    "esc/q  quit",
]


def draw_help(surf, font):
    w, h = 420, 26 + len(HELP) * 22
    r = pygame.Rect(0, 0, w, h)
    r.center = (surf.get_width() // 2, surf.get_height() // 2)
    pygame.draw.rect(surf, (10, 12, 18), r)
    pygame.draw.rect(surf, DIM, r, 1)
    y = r.y + 10
    for line in HELP:
        surf.blit(font.render(line, True, INK if not line.startswith("AERA") else WARN),
                  (r.x + 14, y))
        y += 22
