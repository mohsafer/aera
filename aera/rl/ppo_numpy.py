"""A compact, dependency-free PPO (numpy only) — the reference trainer.

Deliberately readable rather than maximally fast: a 2x64 tanh MLP trunk with
a Gaussian policy head and a value head, GAE(λ), clipped surrogate objective,
Adam, running observation normalization. Saves/loads .npz checkpoints.

For serious training swap in Stable-Baselines3 PPO — the env is a standard
Gymnasium env, so `PPO("MlpPolicy", AeraEnv(config), ...)` just works.
"""
from __future__ import annotations

import math

import numpy as np


class RunningNorm:
    """Running mean/std observation normalizer."""

    def __init__(self, dim: int, eps: float = 1e-4, clip: float = 5.0):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = eps
        self.clip = clip

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                       -self.clip, self.clip).astype(np.float32)

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip: float = 0.2, epochs: int = 4, minibatch: int = 256,
                 ent_coef: float = 0.005, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, seed: int = 0):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.hyper = dict(lr=lr, gamma=gamma, lam=lam, clip=clip, epochs=epochs,
                          minibatch=minibatch, ent_coef=ent_coef,
                          vf_coef=vf_coef, max_grad_norm=max_grad_norm)
        self.norm = RunningNorm(obs_dim)
        # threaded generator, never the global np.random state (AGENTS.md §2)
        self.rng = np.random.default_rng(seed)
        rng = self.rng

        def lin(fan_in, fan_out):
            w = rng.normal(0, 1.0 / math.sqrt(fan_in), size=(fan_in, fan_out))
            return [w.astype(np.float64), np.zeros(fan_out, np.float64)]

        self.W1, self.b1 = lin(obs_dim, hidden)
        self.W2, self.b2 = lin(hidden, hidden)
        self.Wm, self.bm = lin(hidden, act_dim)
        self.Wv, self.bv = lin(hidden, 1)
        self.logstd = np.zeros(act_dim, np.float64)   # std starts at 1 (exp(0))

        self._adam: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._t = 0

    # ------------------------------------------------------------ forward
    def _forward(self, obs: np.ndarray):
        """obs already normalized. Returns (z1, z2, mu, v)."""
        z1 = np.tanh(obs @ self.W1 + self.b1)
        z2 = np.tanh(z1 @ self.W2 + self.b2)
        mu = z2 @ self.Wm + self.bm
        v = (z2 @ self.Wv + self.bv)[:, 0]
        return z1, z2, mu, v

    @staticmethod
    def _logp(a, mu, logstd):
        std = np.exp(logstd)
        return (-0.5 * ((a - mu) / std) ** 2).sum(-1) - logstd.sum() \
            - 0.5 * a.shape[-1] * math.log(2 * math.pi)

    # ------------------------------------------------------------- acting
    def act(self, raw_obs: np.ndarray, deterministic: bool = False):
        o = self.norm.normalize(np.asarray(raw_obs, np.float32)[None])
        _, _, mu, v = self._forward(o)
        mu, v = mu[0], float(v[0])
        if deterministic:
            a = mu
        else:
            a = mu + np.exp(self.logstd) * self.rng.standard_normal(mu.shape)
        a = np.clip(a, -1.0, 1.0)          # env clips too; keeps logp sane
        logp = self._logp(a[None], mu[None], self.logstd)[0]
        return a.astype(np.float32), float(logp), v

    def value(self, raw_obs: np.ndarray) -> float:
        o = self.norm.normalize(np.asarray(raw_obs, np.float32)[None])
        return float(self._forward(o)[3][0])

    # ------------------------------------------------------------- update
    def update(self, obs: np.ndarray, act: np.ndarray, logp_old: np.ndarray,
               adv: np.ndarray, ret: np.ndarray) -> dict:
        h = self.hyper
        self.norm.update(obs)
        o = self.norm.normalize(obs)
        n = len(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(n)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clipfrac": 0.0}
        batches = 0
        for _ in range(h["epochs"]):
            self.rng.shuffle(idx)
            for s in range(0, n, h["minibatch"]):
                b = idx[s:s + h["minibatch"]]
                grads, st = self._grads(o[b], act[b], logp_old[b], adv[b], ret[b])
                self._apply(grads)
                for k in stats:
                    stats[k] += st.get(k, 0.0)
                batches += 1
        self._t += 1
        return {k: v / max(1, batches) for k, v in stats.items()}

    def _grads(self, o, a, logp_old, adv, ret):
        h = self.hyper
        z1, z2, mu, v = self._forward(o)
        std = np.exp(self.logstd)

        logp = self._logp(a, mu, self.logstd)
        ratio = np.exp(logp - logp_old)
        clipped = (ratio > 1 + h["clip"]) & (adv > 0) | (ratio < 1 - h["clip"]) & (adv < 0)
        g = np.where(clipped, 0.0, -adv * ratio)            # d(surrogate)/dlogp
        stats = {"pi_loss": float((-np.minimum(ratio * adv, np.clip(ratio, 1 - h["clip"], 1 + h["clip"]) * adv)).mean()),
                 "v_loss": float(0.5 * ((v - ret) ** 2).mean()),
                 "entropy": float(self.logstd.sum() + 0.5 * self.act_dim * math.log(2 * math.pi * math.e)),
                 "clipfrac": float(clipped.mean())}

        # policy head gradients
        dlogp_dmu = (a - mu) / std**2
        dlogp_dlogstd = ((a - mu) / std) ** 2 - 1.0
        g_mu = (g[:, None] * dlogp_dmu) / len(o)
        g_logstd = (g[:, None] * dlogp_dlogstd).sum(0) / len(o) - h["ent_coef"]

        grads: dict[str, np.ndarray] = {}
        grads["Wm"] = z2.T @ g_mu
        grads["bm"] = g_mu.sum(0)
        dz2 = g_mu @ self.Wm.T

        # value head gradients
        g_v = (h["vf_coef"] * (v - ret) / len(o))[:, None]
        grads["Wv"] = z2.T @ g_v
        grads["bv"] = g_v.sum(0)
        dz2 = dz2 + g_v @ self.Wv.T

        # trunk
        dz2 = dz2 * (1 - z2**2)
        grads["W2"] = z1.T @ dz2
        grads["b2"] = dz2.sum(0)
        dz1 = dz2 @ self.W2.T
        dz1 = dz1 * (1 - z1**2)
        grads["W1"] = o.T @ dz1
        grads["b1"] = dz1.sum(0)
        grads["logstd"] = g_logstd

        # global-norm grad clip
        gn = math.sqrt(sum(float((g_ ** 2).sum()) for g_ in grads.values()))
        if gn > h["max_grad_norm"]:
            for k in grads:
                grads[k] *= h["max_grad_norm"] / (gn + 1e-12)
        return grads, stats

    def _apply(self, grads: dict):
        self._t += 1
        lr = self.hyper["lr"]
        for name, g in grads.items():
            m, vv = self._adam.get(name, (np.zeros_like(g), np.zeros_like(g)))
            m = 0.9 * m + 0.1 * g
            vv = 0.999 * vv + 0.001 * g * g
            mh = m / (1 - 0.9**self._t)
            vh = vv / (1 - 0.999**self._t)
            self._set(name, self._get(name) - lr * mh / (np.sqrt(vh) + 1e-8))
            self._adam[name] = (m, vv)

    _PARAMS = ("W1", "b1", "W2", "b2", "Wm", "bm", "Wv", "bv", "logstd")

    def _get(self, name) -> np.ndarray:
        return getattr(self, name)

    def _set(self, name, val):
        setattr(self, name, val)

    # ---------------------------------------------------------------- io
    def save(self, path: str) -> None:
        data = {k: self._get(k) for k in self._PARAMS}
        data["norm_mean"] = self.norm.mean
        data["norm_var"] = self.norm.var
        data["norm_count"] = np.array([self.norm.count])
        np.savez_compressed(path, **data)

    def load(self, path: str) -> None:
        data = np.load(path)
        for k in self._PARAMS:
            self._set(k, data[k])
        self.norm.mean = data["norm_mean"]
        self.norm.var = data["norm_var"]
        self.norm.count = float(data["norm_count"][0])
