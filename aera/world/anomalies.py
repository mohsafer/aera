"""World anomalies: seeded, episodic stress events for trained agents.

An AnomalyDirector owns a threaded rng (deterministic per config+seed) and a
list of active effects. The env calls `director.step(world, agent, tick)` once
per env step, BEFORE physics, so effects land in the same tick they trigger.

Effects touch only channels that already exist:
  terrain cells   (lava_surge carves a blob; reset_episode restores the grid)
  food activity   (famine hides food and blocks respawn for a while)
  wind vector     (world.wind, applied by physics.integrate as drift)
  agent kick      (quake shoves v/heading for a short burst)
  sensor noise    (agent.sensor_noise, a per-step multiplicative fog on rays)

Rewards, the observation layout, and the action space are untouched, so
Stable-Baselines3 and the built-in PPO run unchanged. All randomness comes
from the director's own rng — never the global state.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import AnomalyCfg

# effect durations, in env ticks (10 Hz → 20..60 s)
_MIN_DUR, _MAX_DUR = 200, 600
_QUAKE_DUR = 30


class _Effect:
    __slots__ = ("kind", "until", "data")

    def __init__(self, kind: str, until: int, data: dict):
        self.kind = kind
        self.until = until
        self.data = data


class AnomalyDirector:
    def __init__(self, cfg: AnomalyCfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.active: list[_Effect] = []
        self.obs_noise = 0.0            # consumed by senses.build_obs via agent
        self.pending: list[str] = []    # story lines from manual triggers
        self.max_active = 2

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """New episode: clear effects (terrain is restored by the world)."""
        self.active.clear()
        self.pending.clear()
        self.obs_noise = 0.0

    def step(self, world, agent, tick: int) -> list[str]:
        events = list(self.pending)
        self.pending.clear()
        for eff in list(self.active):
            if tick >= eff.until:
                self._expire(eff, world, agent, events)
        if (self.rng.random() < self.cfg.event_prob
                and len(self.active) < self.max_active):
            kind = str(self.rng.choice(self.cfg.kinds))
            ev = self._apply(kind, world, agent, tick,
                             int(self.rng.integers(_MIN_DUR, _MAX_DUR + 1)))
            if ev:
                events.append(ev)
        self._sustain(world, agent, tick)
        return events

    # -------------------------------------------------------------- triggers
    def trigger(self, kind: str, world, agent, tick: int) -> str | None:
        """Force an effect (tests, curriculum scripts). The story line is
        queued and flushed by the next step() so it reaches the HUD feed."""
        I = self.cfg.intensity
        dur = int(self.rng.integers(_MIN_DUR, _MAX_DUR + 1))
        msg = self._apply(kind, world, agent, tick, dur)
        if msg:
            self.pending.append(msg)
        return msg

    def _apply(self, kind: str, world, agent, tick: int, dur: int) -> str | None:
        I = self.cfg.intensity
        dur = int(self.rng.integers(_MIN_DUR, _MAX_DUR + 1))
        if kind == "wind":
            ang = float(self.rng.uniform(0, 2 * math.pi))
            phase = float(self.rng.uniform(0, 2 * math.pi))
            self.active.append(_Effect("wind", tick + dur,
                                       {"ang": ang, "phase": phase, "mag": 1.1 * I}))
            return "a storm rolls in — the wind pushes you!"
        if kind == "fog":
            self.active.append(_Effect("fog", tick + dur, {"sigma": 0.3 * I}))
            return "fog descends — your vision blurs"
        if kind == "famine":
            for f in world.foods:
                f.active = False
                f.respawn_at = -1
            self.active.append(_Effect("famine", tick + dur, {}))
            return "famine — the food is gone"
        if kind == "lava_surge":
            world.terrain.carve_blob(self.rng, 3, size=int(0.01 * world.w * world.h) + 6)
            self.active.append(_Effect("lava_surge", tick + 1, {}))
            return "the ground splits — new lava!"
        if kind == "terrain_shift":
            world.terrain.carve_blob(self.rng, 2, size=int(0.02 * world.w * world.h) + 8)
            self.active.append(_Effect("terrain_shift", tick + 1, {}))
            return "the terrain shifts underfoot"
        if kind == "quake":
            self.active.append(_Effect("quake", tick + _QUAKE_DUR,
                                       {"mag": 2.5 * I}))
            return "earthquake!"
        return None

    # -------------------------------------------------------------- sustain
    def _sustain(self, world, agent, tick: int) -> None:
        self.obs_noise = 0.0
        wx, wy = 0.0, 0.0
        for eff in self.active:
            if eff.kind == "wind":
                gust = 0.6 + 0.4 * math.sin(tick * 0.09 + eff.data["phase"])
                wx += eff.data["mag"] * gust * math.cos(eff.data["ang"])
                wy += eff.data["mag"] * gust * math.sin(eff.data["ang"])
            elif eff.kind == "fog":
                # one pre-drawn bias per step: deterministic, no rng in senses
                self.obs_noise = eff.data["sigma"] * float(self.rng.standard_normal())
            elif eff.kind == "famine":
                for f in world.foods:      # keep respawn suppressed
                    if f.active:
                        f.active = False
                        f.respawn_at = -1
            elif eff.kind == "quake":
                agent.v = min(2.6, agent.v + eff.data["mag"] * 0.3)
                agent.heading += float(self.rng.uniform(-0.3, 0.3))
        world.wind = (wx, wy)

    def _expire(self, eff: _Effect, world, agent, events: list[str]) -> None:
        self.active.remove(eff)
        if eff.kind == "famine":
            # food trickles back on a stagger, via the normal step_time path
            for i, f in enumerate(world.foods):
                if not f.active:
                    f.respawn_at = world.tick + 10 * i
            events.append("the famine ends — food returns")
        elif eff.kind == "wind":
            events.append("the storm passes")
        elif eff.kind == "fog":
            events.append("the fog lifts")
