"""Gymnasium compatibility: use real gymnasium when present, otherwise a tiny
shim with the same surface so the sim still runs standalone.

The shim is intentionally minimal (it only needs to satisfy AeraEnv and the
built-in trainer). Install gymnasium for full baseline compatibility
(Stable-Baselines3, CleanRL, ...).
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on env
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYM = True
except Exception:  # pragma: no cover
    HAS_GYM = False

    class _Box:  # minimal stand-in for gymnasium.spaces.Box
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.broadcast_to(np.asarray(low, dtype), shape).astype(dtype)
            self.high = np.broadcast_to(np.asarray(high, dtype), shape).astype(dtype)
            self.shape = tuple(shape)
            self.dtype = dtype

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and bool(
                np.all(x >= self.low - 1e-6) and np.all(x <= self.high + 1e-6)
            )

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _Env:
        metadata: dict = {}
        render_mode: str | None = None
        observation_space = None
        action_space = None

        def reset(self, *, seed=None, options=None):  # pragma: no cover
            raise NotImplementedError

        def step(self, action):  # pragma: no cover
            raise NotImplementedError

        def render(self):  # pragma: no cover
            return None

    class gym:  # namespace shim
        Env = _Env

    class spaces:  # namespace shim
        Box = _Box


spaces = globals().get("spaces")
gym = globals().get("gym")
