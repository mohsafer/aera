"""The training loop: collect → GAE → PPO update → log → checkpoint, with an
optional live viewer attached to the very env being trained.
"""
from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

from ..agents.senses import action_dim, obs_dim
from ..env import AeraEnv
from ..rl.ppo_numpy import PPO
from .metrics import JsonlLogger, MilestoneTracker


class Trainer:
    def __init__(self, config, out_dir: str, total_steps: int = 200_000,
                 rollout: int = 2048, seed: int = 0, viewer=None,
                 save_every_updates: int = 20, init_from: str | None = None):
        self.config = config
        self.out = out_dir
        self.total_steps = total_steps
        self.rollout = rollout
        self.seed = seed
        self.viewer = viewer
        self.save_every = save_every_updates
        # warm-start weights (+ RunningNorm) from a policy_*.npz. NOT a resume:
        # the curriculum episode counter and RNG start fresh.
        self.init_from = init_from

        os.makedirs(out_dir, exist_ok=True)
        self.logger = JsonlLogger(os.path.join(out_dir, "metrics.jsonl"))
        self.episodes: deque = deque(maxlen=20)
        self.milestones = MilestoneTracker(config.agent.kind)

    # -------------------------------------------------------------- train
    def train(self) -> str:
        env = AeraEnv(self.config)
        obs_dim_ = obs_dim(self.config.agent)
        act_dim_ = action_dim(self.config.agent.kind)
        ppo = PPO(obs_dim_, act_dim_, seed=self.seed)
        if self.init_from:
            ppo.load(self.init_from)
            print(f"warm-started policy from {self.init_from} "
                  f"(curriculum restarts at episode 0)")
        self.config.save(os.path.join(self.out, "config_used.json"))

        obs, _ = env.reset(seed=self.seed)
        hp = ppo.hyper
        global_step, update = 0, 0
        t0 = time.time()

        while global_step < self.total_steps:
            buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
            for _ in range(self.rollout):
                a, logp, val = ppo.act(obs)
                nobs, rew, term, trunc, info = env.step(a)
                if self.viewer is not None:
                    self.viewer.tick(env)
                    if getattr(self.viewer, "quit_requested", False):
                        global_step = self.total_steps + 1   # flush & save
                        break
                buf["obs"].append(obs); buf["act"].append(a)
                buf["logp"].append(logp); buf["val"].append(val)
                buf["rew"].append(rew); buf["done"].append(term or trunc)
                obs = nobs
                global_step += 1
                if term or trunc:
                    self._on_episode_end(info["episode"], env)
                    obs, _ = env.reset()

            if global_step > self.total_steps:
                break

            # bootstrapped returns / GAE(λ)
            with np.errstate(all="ignore"):
                vals = np.array(buf["val"] + [ppo.value(obs)])
                rews = np.array(buf["rew"])
                dones = np.array(buf["done"], float)
                adv = np.zeros_like(rews)
                gae = 0.0
                for t in reversed(range(len(rews))):
                    delta = rews[t] + hp["gamma"] * vals[t + 1] * (1 - dones[t]) - vals[t]
                    gae = delta + hp["gamma"] * hp["lam"] * (1 - dones[t]) * gae
                    adv[t] = gae
                ret = adv + vals[:-1]

            stats = ppo.update(np.array(buf["obs"]), np.array(buf["act"]),
                               np.array(buf["logp"]), adv, ret)
            update += 1
            self.logger.log({"type": "update", "update": update,
                             "step": global_step, **stats,
                             "eps_per_sec": self._eps_rate(t0, global_step)})

            mean_ret = (sum(e["ret"] for e in self.episodes) / len(self.episodes)
                        if self.episodes else 0.0)
            print(f"[update {update:4d}] step {global_step:7d} "
                  f"ret(20ep) {mean_ret:8.2f} pi_loss {stats['pi_loss']:+.4f} "
                  f"v_loss {stats['v_loss']:.4f} clip {stats['clipfrac']:.3f}",
                  flush=True)

            if update % self.save_every == 0:
                ppo.save(os.path.join(self.out, "policy_final.npz"))
            if len(self.episodes) == self.episodes.maxlen:
                best = getattr(self, "_best_ret", -1e9)
                if mean_ret > best:
                    self._best_ret = mean_ret
                    ppo.save(os.path.join(self.out, "policy_best.npz"))

        ppo.save(os.path.join(self.out, "policy_final.npz"))
        print(f"done in {(time.time() - t0) / 60:.1f} min → {self.out}")
        return self.out

    # ------------------------------------------------------------- events
    def _on_episode_end(self, ep: dict, env: AeraEnv) -> None:
        self.episodes.append(ep)
        self.logger.log({"type": "episode", **ep})
        if self.viewer is not None:
            self.viewer.chart.push(ep["ret"])
        for name in self.milestones.update(ep):
            print(f"\n★ SKILL UNLOCKED: {name} (episode {ep['episode']})\n", flush=True)
            self.logger.log({"type": "milestone", "skill": name,
                             "episode": ep["episode"]})

    @staticmethod
    def _eps_rate(t0: float, steps: int) -> float:
        return steps / max(1e-6, time.time() - t0)


def load_policy(path: str, config) -> PPO:
    ppo = PPO(obs_dim(config.agent), action_dim(config.agent.kind))
    ppo.load(path)
    return ppo
