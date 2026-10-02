"""Locomotion models — the swappable physics layer.

v1 uses a *procedural* abstraction (see log.md D2): the rover is a
differential-drive point mass; the walker has 4 joint targets whose
coordination gates forward thrust ("learning to walk" is really learning to
coordinate the gait). A PyBullet/MuJoCo adapter can replace these without
touching env/senses/viz.
"""
from __future__ import annotations

import math

from .terrain import FRICTION
from .world import World

# ------------------------------------------------------------ walker gait
JOINT_NAMES = ("hip_l", "knee_l", "hip_r", "knee_r")


def gait_score(joints) -> float:
    """Coordination of leg joints in [0, 1].

    1.0 requires hips in antiphase (hip_l ≈ -hip_r), knees in phase, and
    meaningful swing amplitude — i.e. a stepping pattern. Random flailing
    scores ~0; a learned gait scores high.
    """
    hip_l, knee_l, hip_r, knee_r = joints
    antiphase = max(0.0, 1.0 - abs(hip_l + hip_r) / 2.0)
    inphase = max(0.0, 1.0 - abs(knee_l - knee_r) / 2.0)
    amp = (abs(hip_l) + abs(hip_r)) / 2.0
    return float(antiphase * inphase * min(1.0, amp / 0.5))


class RoverPhysics:
    """Differential drive: action = [steer, throttle] in [-1, 1]."""
    ACTION_DIM = 2
    MAX_SPEED = 2.6

    def step(self, agent, action, world: World, dt: float) -> None:
        steer = float(np_clip1(action[0]))
        throttle = float(np_clip1(action[1]))
        agent.v += (2.8 * throttle - 1.6 * agent.v) * dt
        agent.v = max(-1.0, min(self.MAX_SPEED, agent.v))
        agent.heading += 2.4 * steer * dt * (0.35 + 0.65 * min(1.0, abs(agent.v) / 1.2))
        agent.effort = abs(throttle)


class WalkerPhysics:
    """Action = [turn, hip_l, knee_l, hip_r, knee_r, jump] in [-1, 1]."""
    ACTION_DIM = 6
    MAX_SPEED = 1.8

    def step(self, agent, action, world: World, dt: float) -> None:
        turn = float(np_clip1(action[0]))
        targets = [float(np_clip1(a)) for a in action[1:5]]
        k = min(1.0, 10.0 * dt)
        for i in range(4):
            agent.joints[i] += (targets[i] - agent.joints[i]) * k

        c = gait_score(agent.joints)
        amp = (abs(agent.joints[0]) + abs(agent.joints[2])) / 2.0
        # small "shamble" floor speed keeps the reward gradient alive early
        agent.v += ((c * 1.7 + 0.12 * amp) - 1.4 * agent.v) * dt
        agent.v = max(0.0, min(self.MAX_SPEED, agent.v))
        agent.heading += 2.0 * turn * dt * (0.3 + 0.7 * c)
        agent.effort = 0.15 + 0.85 * c

        # tool-gated ability: jump channel does nothing without boots
        if action[5] > 0.5 and "boots" in agent.inventory and agent.jump_cd <= 0.0:
            agent.airborne = 0.6
            agent.jump_cd = 2.0
            agent.events.append("jumped")

        agent.phase = (agent.phase + 2.0 * math.pi * 1.4 * dt) % (2.0 * math.pi)


def np_clip1(x) -> float:
    return max(-1.0, min(1.0, float(x)))


PHYSICS = {"rover": RoverPhysics(), "walker": WalkerPhysics()}


def integrate(agent, world: World, dt: float) -> None:
    """Apply terrain friction, move the agent's circle, update timers."""
    if agent.airborne > 0.0:
        agent.airborne = max(0.0, agent.airborne - dt)
    if agent.jump_cd > 0.0:
        agent.jump_cd = max(0.0, agent.jump_cd - dt)

    speed_mult = FRICTION[world.terrain_at(agent.x, agent.y)]
    if agent.airborne > 0.0:
        speed_mult = max(speed_mult, 1.3)   # a leap clears pits and mud
    dist = agent.v * speed_mult * dt
    dx = math.cos(agent.heading) * dist
    dy = math.sin(agent.heading) * dist
    agent.x, agent.y = world.move_circle(agent.x, agent.y, dx, dy,
                                         agent.cfg.radius, agent.airborne > 0.0)
