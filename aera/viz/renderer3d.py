"""Software low-poly 3D renderer — the 90s look.

Painter's algorithm: collect world-space polygons (ground tiles, wall boxes,
entities, articulated agent bodies), transform to camera space, cull behind
the near plane, sort back-to-front, flat-shade with Lambert lighting and
distance fog, and draw with pygame. Rendered small and upscaled with nearest
neighbour for chunky pixels. Works headless (no display needed) so the same
path serves the live viewer, GIF recording, and tests.
"""
from __future__ import annotations

import math
import os

import numpy as np

import pygame

from .camera import Camera


def _ensure_video() -> None:
    """Make pygame Surfaces usable, headless-safe (dummy driver fallback)."""
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        try:
            pygame.display.init()
        except pygame.error:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.init()

# ------------------------------------------------------------------ palette
SKY_TOP = (34, 26, 58)
SKY_HOR = (214, 138, 84)
FOG = (140, 104, 96)

TERRAIN_COLORS = {
    0: ((92, 156, 74), (84, 146, 68)),      # grass (checker pair)
    1: ((132, 132, 138), (126, 126, 132)),  # road
    2: ((112, 86, 54), (104, 80, 50)),      # mud
    3: ((238, 92, 32), (238, 92, 32)),      # lava (pulses)
    4: ((16, 13, 20), (12, 10, 16)),        # pit
}
WALL = (172, 152, 128)
WALL_SIDE = (140, 122, 102)
FOOD = (96, 224, 96)
TOOL = (250, 210, 64)
BEACON = (88, 224, 240)
SKIN = (222, 192, 152)
ROVER = (196, 84, 72)
TIRE = (44, 44, 50)

LIGHT = (0.45, -0.5, -0.75)   # direction *to* the light (z-up world)
NEAR = 0.18
FAR = 60.0


def render_frame(env, camera: Camera | None = None, size=(426, 240),
                 time_s: float = 0.0) -> np.ndarray:
    """Convenience wrapper: env → RGB uint8 array (H, W, 3)."""
    surf = render_scene(env.world, [env.agent], camera or Camera("chase"),
                        size, time_s)
    arr = pygame.surfarray.array3d(surf)
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8)


def render_scene(world, agents, camera: Camera, size=(426, 240),
                 time_s: float = 0.0) -> pygame.Surface:
    """Render the world into a pygame Surface (the low-res chunky buffer)."""
    _ensure_video()
    W, H = size
    surf = pygame.Surface(size)
    agent = agents[0] if agents else None
    eye, right, up, fwd = camera.basis(agent, world)

    _sky(surf)
    faces = _gather_faces(world, agents, eye, camera, time_s)

    tan_f = 0.62
    aspect = W / H
    polys = []
    for pts, color in faces:
        scr = []
        depth = 0.0
        ok = True
        for (x, y, z) in pts:
            q = (x - eye[0], y - eye[1], z - eye[2])
            cz = q[0] * fwd[0] + q[1] * fwd[1] + q[2] * fwd[2]
            if cz < NEAR:
                ok = False
                break
            cx = q[0] * right[0] + q[1] * right[1] + q[2] * right[2]
            cy = q[0] * up[0] + q[1] * up[1] + q[2] * up[2]
            sx = (cx / (cz * tan_f * aspect)) * (W / 2) + W / 2
            sy = H / 2 - (cy / (cz * tan_f)) * (H / 2)
            scr.append((sx, sy))
            depth += cz
        if ok:
            polys.append((depth / len(pts), scr, color))

    polys.sort(key=lambda p: -p[0])
    for depth, scr, color in polys:
        f = min(1.0, depth / FAR) ** 1.5
        c = _fog_blend(color, f)
        pygame.draw.polygon(surf, c, scr)
    return surf


# ------------------------------------------------------------------ scene
def _gather_faces(world, agents, eye, camera, t):
    faces: list = []
    cx = agents[0].x if agents else world.w / 2
    cy = agents[0].y if agents else world.h / 2
    R = 17
    x0, x1 = int(max(0, cx - R)), int(min(world.w - 1, cx + R))
    y0, y1 = int(max(0, cy - R)), int(min(world.h - 1, cy + R))

    # ground tiles
    for gy in range(y0, y1 + 1):
        for gx in range(x0, x1 + 1):
            kind = int(world.terrain.grid[gy, gx])
            c1, c2 = TERRAIN_COLORS[kind]
            col = c1 if (gx + gy) % 2 == 0 else c2
            if kind == 3:   # lava pulse
                pulse = 0.5 + 0.5 * math.sin(t * 4.0 + gx * 1.7 + gy * 2.3)
                col = (int(238 * (0.7 + 0.3 * pulse)), int(92 + 70 * pulse), 32)
            faces.append(([(gx, gy, 0.0), (gx + 1, gy, 0.0),
                           (gx + 1, gy + 1, 0.0), (gx, gy + 1, 0.0)], col))

    # walls (boxes)
    for (rx, ry, rw, rh) in world.walls:
        if rx + rw < x0 - 1 or rx > x1 + 1 or ry + rh < y0 - 1 or ry > y1 + 1:
            continue
        if rw > world.w + 1:   # border walls look better low
            _box(faces, rx, ry, rw, rh, 0.0, 0.55, WALL, WALL_SIDE)
        else:
            _box(faces, rx, ry, rw, rh, 0.0, 1.25, WALL, WALL_SIDE)

    # entities near the agent
    for i, f in enumerate(world.foods):
        if f.active and abs(f.x - cx) < R and abs(f.y - cy) < R:
            z = 0.18 + 0.06 * math.sin(t * 3 + i)
            _cube(faces, f.x, f.y, z, 0.22, FOOD)
    for i, tool in enumerate(world.tools):
        if not tool.taken and abs(tool.x - cx) < R and abs(tool.y - cy) < R:
            _cube(faces, tool.x, tool.y, 0.1, 0.5, (120, 110, 96))     # pedestal
            z = 0.75 + 0.12 * math.sin(t * 2.5 + i)
            _cube(faces, tool.x, tool.y, z, 0.28, TOOL)
    if world.beacon is not None and world.beacon.x > -1:
        b = world.beacon
        if abs(b.x - cx) < R + 6 and abs(b.y - cy) < R + 6:
            pulse = 0.6 + 0.4 * math.sin(t * 3.0)
            _box(faces, b.x - 0.12, b.y - 0.12, 0.24, 0.24, 0.0, 2.6,
                 BEACON, tuple(int(c * pulse) for c in BEACON))

    for a in agents:
        if a.is_walker:
            _walker(faces, a, t)
        else:
            _rover(faces, a)
    return faces


# ------------------------------------------------------------- primitives
def _box(faces, x, y, w, h, z0, z1, top, side):
    p = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    faces.append(([(p[0][0], p[0][1], z1), (p[1][0], p[1][1], z1),
                   (p[2][0], p[2][1], z1), (p[3][0], p[3][1], z1)], top))
    for i in range(4):
        a, b = p[i], p[(i + 1) % 4]
        faces.append(([(a[0], a[1], z0), (b[0], b[1], z0),
                       (b[0], b[1], z1), (a[0], a[1], z1)], side))


def _cube(faces, x, y, z, s, color):
    _box(faces, x - s / 2, y - s / 2, s, s, z - s / 2, z + s / 2, color,
         tuple(int(c * 0.78) for c in color))


def _segment(faces, a, b, width, color):
    """Thin box between two 3D points (used for limbs)."""
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = math.sqrt(sum(c * c for c in d)) or 1.0
    d = tuple(c / n * width / 2 for c in d)
    # two perpendicular offsets
    perp1 = (-d[1], d[0], 0.0)
    n1 = math.sqrt(sum(c * c for c in perp1)) or 1.0
    perp1 = tuple(c / n1 for c in perp1)
    perp2 = (d[1] * perp1[2] - d[2] * perp1[1],
             d[2] * perp1[0] - d[0] * perp1[2],
             d[0] * perp1[1] - d[1] * perp1[0])
    n2 = math.sqrt(sum(c * c for c in perp2)) or 1.0
    perp2 = tuple(c / n2 for c in perp2)
    w = width / 2
    corners = []
    for (p, u, v, s) in ((a, perp1, perp2, +1), (b, perp1, perp2, -1)):
        for (uu, vv) in ((+w, +w), (-w, +w), (-w, -w), (+w, -w)):
            corners.append((p[0] + u[0] * uu + v[0] * vv,
                            p[1] + u[1] * uu + v[1] * vv,
                            p[2] + u[2] * uu + v[2] * vv))
    idx = [(0, 1, 2, 3), (4, 5, 6, 7)]                 # end caps
    idx += [(0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    for face in idx:
        faces.append(([corners[i] for i in face], color))


def _obb(faces, cx, cy, fx, sy_, hf, hs, z0, z1, top, side):
    """Oriented box: centre (cx,cy), axes f (facing) and s (side)."""
    p = []
    for (df, ds) in ((-hf, -hs), (hf, -hs), (hf, hs), (-hf, hs)):
        p.append((cx + fx[0] * df + sy_[0] * ds, cy + fx[1] * df + sy_[1] * ds))
    faces.append(([(p[0][0], p[0][1], z1), (p[1][0], p[1][1], z1),
                   (p[2][0], p[2][1], z1), (p[3][0], p[3][1], z1)], top))
    for i in range(4):
        a, b = p[i], p[(i + 1) % 4]
        faces.append(([(a[0], a[1], z0), (b[0], b[1], z0),
                       (b[0], b[1], z1), (a[0], a[1], z1)], side))


# ----------------------------------------------------------------- bodies
def _walker(faces, a, t):
    h = a.heading
    f = (math.cos(h), math.sin(h))
    s = (-math.sin(h), math.cos(h))
    px, py = a.x, a.y
    crouch = 1.0 - 0.25 * max(0.0, a.joints[1] + a.joints[3]) / 2.0
    _obb(faces, px, py, f, s, 0.26, 0.20, 0.55 * crouch + 0.1, 1.05 * crouch + 0.1,
         SKIN, tuple(int(c * 0.8) for c in SKIN))
    _cube(faces, px + f[0] * 0.02, py + f[1] * 0.02, 1.05 * crouch + 0.22, 0.26, SKIN)
    for side, (i_hip, i_knee) in ((-1, (0, 1)), (1, (2, 3))):
        sx = px + s[0] * side * 0.15
        sy = py + s[1] * side * 0.15
        hip_a = float(a.joints[i_hip])
        knee_a = float(a.joints[i_knee])
        hip = (sx, sy, 0.55 * crouch + 0.1)
        knee = (sx + f[0] * 0.32 * math.sin(hip_a),
                sy + f[1] * 0.32 * math.sin(hip_a),
                max(0.05, hip[2] - 0.28))
        foot = (knee[0] + f[0] * 0.3 * math.sin(knee_a),
                knee[1] + f[1] * 0.3 * math.sin(knee_a), 0.04)
        _segment(faces, hip, knee, 0.13, tuple(int(c * 0.9) for c in SKIN))
        _segment(faces, knee, foot, 0.11, (60, 48, 40))
        # arms swing opposite to the legs
        sh = (sx, sy, 1.0 * crouch + 0.1)
        elb = (sx - f[0] * 0.28 * math.sin(hip_a), sy - f[1] * 0.28 * math.sin(hip_a),
               sh[2] - 0.26)
        hnd = (elb[0] - f[0] * 0.1, elb[1] - f[1] * 0.1, sh[2] - 0.5)
        _segment(faces, sh, elb, 0.09, tuple(int(c * 0.95) for c in SKIN))
        _segment(faces, elb, hnd, 0.08, SKIN)


def _rover(faces, a):
    h = a.heading
    f = (math.cos(h), math.sin(h))
    s = (-math.sin(h), math.cos(h))
    px, py = a.x, a.y
    _obb(faces, px, py, f, s, 0.38, 0.26, 0.16, 0.44, ROVER,
         tuple(int(c * 0.78) for c in ROVER))
    for df, ds in ((0.26, 0.24), (0.26, -0.24), (-0.26, 0.24), (-0.26, -0.24)):
        wx = px + f[0] * df + s[0] * ds
        wy = py + f[1] * df + s[1] * ds
        _cube(faces, wx, wy, 0.1, 0.16, TIRE)
    _box(faces, px + f[0] * 0.1 - 0.03, py + f[1] * 0.1 - 0.03, 0.06, 0.06,
         0.44, 0.72, (90, 90, 96), (70, 70, 76))
    _cube(faces, px + f[0] * 0.1, py + f[1] * 0.1, 0.78, 0.12, (40, 220, 120))


# ------------------------------------------------------------------ sky/fog
def _sky(surf):
    W, H = surf.get_size()
    steps = 24
    for i in range(steps):
        k = i / (steps - 1)
        col = tuple(int(SKY_TOP[j] + (SKY_HOR[j] - SKY_TOP[j]) * k) for j in range(3))
        pygame.draw.rect(surf, col, (0, int(i * H / steps), W, H // steps + 1))


def _fog_blend(color, f):
    return tuple(int(c * (1 - f) + FOG[i] * f) for i, c in enumerate(color))
