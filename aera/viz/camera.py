"""Free cameras for the 90s-style 3D view: ORBIT / CHASE / TOP / FIRST-PERSON.

The camera exposes (eye, right, up, forward) vectors; the renderer projects
world points with a simple pinhole model. Yaw/pitch/dist are user-controlled
in orbit/top modes; chase auto-follows the agent's heading.
"""
from __future__ import annotations

import math

MODES = ("orbit", "chase", "top", "fp")


class Camera:
    def __init__(self, mode: str = "orbit"):
        self.mode = mode if mode in MODES else "orbit"
        self.yaw = math.radians(45.0)
        self.pitch = math.radians(38.0)
        self.dist = 11.0
        self.yaw_offset = 0.0      # user tweak applied in chase mode
        self.follow = True         # target the agent (vs world centre)

    def cycle_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]

    # ------------------------------------------------------------ basis
    def basis(self, agent, world):
        """Returns (eye, right, up, fwd) 3-vectors."""
        if self.mode == "fp" and agent is not None:
            return self._fp(agent)
        tx, ty = self._target(agent, world)
        if self.mode == "chase" and agent is not None:
            yaw = agent.heading + math.pi + self.yaw_offset
            pitch = 0.48
            dist = min(self.dist, 9.0)
        elif self.mode == "top":
            yaw = self.yaw
            pitch = math.radians(86.0)
            dist = max(world.w, world.h) * 0.75
        else:
            yaw, pitch, dist = self.yaw, self.pitch, self.dist
        eye = (tx + dist * math.cos(pitch) * math.cos(yaw),
               ty + dist * math.cos(pitch) * math.sin(yaw),
               dist * math.sin(pitch) + 0.6)
        look = (tx, ty, 0.6)
        return _basis_from_eye_look(eye, look)

    def _target(self, agent, world):
        if agent is None or not self.follow:
            return world.w / 2.0, world.h / 2.0
        return agent.x, agent.y

    def _fp(self, agent):
        eye = (agent.x, agent.y, 0.9)
        fwd = (math.cos(agent.heading), math.sin(agent.heading), -0.12)
        n = math.sqrt(sum(c * c for c in fwd))
        fwd = tuple(c / n for c in fwd)
        right = (fwd[1], -fwd[0], 0.0)          # right-hand, z-up world
        up = (-fwd[2] * right[1], fwd[2] * right[0], 1.0)
        n = math.sqrt(sum(c * c for c in up))
        up = tuple(c / n for c in up)
        return eye, right, up, fwd


def _basis_from_eye_look(eye, look):
    fwd = tuple(l - e for l, e in zip(look, eye))
    n = math.sqrt(sum(c * c for c in fwd)) or 1.0
    fwd = tuple(c / n for c in fwd)
    right = (fwd[1], -fwd[0], 0.0)
    n = math.sqrt(sum(c * c for c in right)) or 1.0
    right = tuple(c / n for c in right)
    # up = right × fwd (z-up friendly)
    up = (right[1] * fwd[2] - right[2] * fwd[1],
          right[2] * fwd[0] - right[0] * fwd[2],
          right[0] * fwd[1] - right[1] * fwd[0])
    n = math.sqrt(sum(c * c for c in up)) or 1.0
    up = tuple(c / n for c in up)
    return eye, right, up, fwd
