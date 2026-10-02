"""AeraEnv — the Gymnasium-compatible environment.

    env = AeraEnv(Config.load("configs/field_open.json"))
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())

`info["sub"]` carries the per-term reward breakdown and `info["events"]` the
tick's story events ("found food", "jumped", ...) — both are for humans
(students, HUD), not the policy. Works with or without gymnasium installed
(see aera/gym_compat.py).
"""
from __future__ import annotations

import math

import numpy as np

from .agents.agent import Agent
from .agents.senses import action_dim, build_obs, obs_dim
from .config import Config
from .gym_compat import HAS_GYM, gym, spaces
from .world.entities import place_free
from .world.physics import integrate
from .world.terrain import Terrain
from .world.world import World


class AeraEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    render_mode = None

    def __init__(self, config: Config, render_mode: str | None = None):
        assert config.agent.kind in ("walker", "rover"), config.agent.kind
        self.config = config
        self.render_mode = render_mode

        adim = action_dim(config.agent.kind)
        self.action_space = spaces.Box(-1.0, 1.0, (adim,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (obs_dim(config.agent),), np.float32)

        self.world = World(config.world)
        self.rng = np.random.default_rng(config.world.seed + 1)
        self.agent = Agent(config.agent, *self._spawn_point())

        self.episode = 0
        self.steps = 0
        self.ep_ret = 0.0
        self.ep_events: list[str] = []
        self.last_events: list[str] = []   # per-step events (HUD feed)
        self._prev_food_dist: float | None = None
        self._last_damage_tick = -99

    # -------------------------------------------------------------- helpers
    def _spawn_point(self) -> tuple[float, float]:
        p = place_free(1, self.world, self.rng, min_spawn_dist=0.0)
        return p[0] if p else (self.world.w / 2.0, self.world.h / 2.0)

    # ----------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if HAS_GYM:   # gymnasium seeding bookkeeping; the shim has no reset
            super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.world.reset_episode()
        x, y = self._spawn_point()
        heading = float(self.rng.uniform(0, 2 * math.pi))
        self.agent.reset(x, y, heading)
        # curriculum for the episode that is about to start
        self.config.apply_overrides(self.config.stage_for_episode(self.episode))
        self.steps = 0
        self.ep_ret = 0.0
        self.ep_events.clear()
        self.last_events = []
        self._prev_food_dist = None
        self.agent.record_cell()
        obs = build_obs(self.agent, self.world, self.config.agent)
        return obs, self._info([])

    def step(self, action):
        agent, world, cfg = self.agent, self.world, self.config
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        events: list[str] = []

        agent.physics.step(agent, action, world, cfg.sim.dt)
        integrate(agent, world, cfg.sim.dt)
        agent.speed_sum += agent.v * cfg.sim.dt
        world.step_time()
        agent.steps_alive += 1
        self.steps += 1

        sub = self._consequences(events)
        reward = float(sum(sub.values()))
        self.ep_ret += reward

        success = bool(world.beacon and world.beacon.hit
                       and cfg.reward.beacon_terminates)
        terminated = agent.health <= 0 or agent.energy <= 0 or success
        truncated = self.steps >= cfg.sim.max_steps
        if terminated and not success:
            events.append("starved" if agent.energy <= 0 else "died")
            sub["death"] = -cfg.reward.w_death
            reward += sub["death"]
            self.ep_ret += sub["death"]

        obs = build_obs(agent, world, cfg.agent)
        step_events = agent.events + events   # physics events (jumped) + world events
        self.ep_events.extend(step_events)
        self.last_events = step_events
        agent.events.clear()
        info = self._info(step_events, sub)
        if terminated or truncated:
            info["episode"] = self._episode_stats()
            self.episode += 1
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------- consequences
    def _consequences(self, events: list[str]) -> dict[str, float]:
        a, w, cfg, dt = self.agent, self.world, self.config, self.config.sim.dt
        rw = cfg.reward
        sub = {"food": 0.0, "survive": 0.0, "novelty": 0.0,
               "progress": 0.0, "damage": 0.0, "beacon": 0.0}

        # terrain
        if w.terrain_at(a.x, a.y) == Terrain.LAVA and cfg.agent.lava_damage > 0:
            dmg = cfg.agent.lava_damage * dt
            a.health = max(0.0, a.health - dmg)
            sub["damage"] = -rw.w_damage * dmg
            if w.tick - self._last_damage_tick > 20:
                events.append("burning!")
                self._last_damage_tick = w.tick

        # energy budget
        a.energy = max(0.0, a.energy - (cfg.agent.hunger_rate
                                        + cfg.agent.effort_rate * a.effort) * dt)

        # novelty
        if a.record_cell():
            sub["novelty"] = rw.w_novelty

        # food
        for f in w.foods:
            if f.active and math.hypot(f.x - a.x, f.y - a.y) < 0.6:
                f.active = False
                f.respawn_at = (w.tick + cfg.world.entities.food_respawn_ticks
                                if cfg.world.entities.food_respawn_ticks > 0 else -1)
                a.foods_eaten += 1
                a.energy = min(cfg.agent.max_energy, a.energy + f.energy)
                sub["food"] = rw.w_food
                events.append("found food")

        # tools
        for tool in w.tools:
            if not tool.taken and math.hypot(tool.x - a.x, tool.y - a.y) < 0.7:
                tool.taken = True
                a.inventory.add(tool.kind)
                events.append(f"picked up {tool.kind}!")

        # beacon
        if w.beacon is not None and not w.beacon.hit:
            if math.hypot(w.beacon.x - a.x, w.beacon.y - a.y) < 0.8:
                w.beacon.hit = True
                sub["beacon"] = rw.w_beacon
                events.append("reached the beacon!")

        # survival + progress shaping
        sub["survive"] = rw.w_survive * dt
        d, _ = self._nearest_active_food()
        if d is not None:
            if self._prev_food_dist is not None:
                sub["progress"] = rw.w_progress * (self._prev_food_dist - d)
            self._prev_food_dist = d
        return sub

    def _nearest_active_food(self):
        bd, ba = math.inf, None
        for f in self.world.foods:
            if f.active:
                d = math.hypot(f.x - self.agent.x, f.y - self.agent.y)
                if d < bd:
                    bd, ba = d, f
        return (bd if ba is not None else None), ba

    # ------------------------------------------------------------- info
    def _info(self, events: list[str], sub: dict | None = None) -> dict:
        a = self.agent
        gait = 0.0
        if a.is_walker:
            gait = float(np.clip(1.0 - abs(a.joints[0] + a.joints[2]) / 2, 0, 1))
        return {
            "events": events,
            "sub": sub or {},
            "score": a.foods_eaten * 10 + self.ep_ret,
            "agent": {
                "x": a.x, "y": a.y, "speed": a.v, "energy": a.energy,
                "health": a.health, "foods": a.foods_eaten,
                "inventory": sorted(a.inventory),
                "explored": a.exploration_ratio(self.world),
                "gait": gait,
            },
        }

    def _episode_stats(self) -> dict:
        a = self.agent
        return {
            "episode": self.episode,
            "steps": self.steps,
            "ret": self.ep_ret,
            "score": a.foods_eaten * 10,
            "foods": a.foods_eaten,
            "mean_speed": a.speed_sum / max(1, self.steps),
            "explored": a.exploration_ratio(self.world),
            "inventory": sorted(a.inventory),
            "beacon": bool(self.world.beacon and self.world.beacon.hit),
            "events": list(self.ep_events),
        }

    # ------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        from .viz.renderer3d import render_frame
        return render_frame(self, camera=None, size=(426, 240))

    def close(self):
        pass

    # -------------------------------------------------------------- viewer
    def get_state(self) -> dict:
        return {"world": self.world, "agent": self.agent,
                "config": self.config, "episode": self.episode,
                "ep_ret": self.ep_ret, "steps": self.steps}
