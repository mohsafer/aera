"""Headless recording: run a rollout and save it as a GIF (Pillow).

Works on servers without a display — the renderer draws into an offscreen
surface only. Used by `python -m aera.scripts.watch --record out.gif`.
"""
from __future__ import annotations

from ..viz.camera import Camera
from ..viz.renderer3d import render_frame


def record_gif(env, policy, path: str, steps: int = 400, fps: int = 20,
               mode: str = "chase", size=(640, 360), seed: int = 0) -> str:
    from PIL import Image

    frames = []
    obs, _ = env.reset(seed=seed)
    cam = Camera(mode)
    for t in range(steps):
        if policy is None:
            a = env.action_space.sample()
        else:
            a = policy.act(obs, deterministic=True)[0]
        obs, _, term, trunc, _info = env.step(a)
        if t % 2 == 0:   # 2 sim steps per frame → smaller files
            arr = render_frame(env, cam, size=size, time_s=t * env.config.sim.dt)
            frames.append(Image.fromarray(arr))
        if term or trunc:
            obs, _ = env.reset()
    if not frames:
        raise RuntimeError("no frames captured")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path
