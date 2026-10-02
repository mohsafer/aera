import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless-safe rendering

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aera.config import Config  # noqa: E402
from aera.env import AeraEnv  # noqa: E402

pygame = pytest.importorskip("pygame")


@pytest.fixture(scope="module")
def env():
    cfg = Config.load("configs/field_small.json")
    return AeraEnv(cfg)


def test_render_frame_offscreen(env):
    env.reset(seed=0)
    from aera.viz.camera import Camera
    from aera.viz.renderer3d import render_frame
    arr = render_frame(env, Camera("chase"), size=(320, 180), time_s=1.0)
    assert arr.shape == (180, 320, 3) and arr.dtype == np.uint8
    assert arr.std() > 5.0, "rendered frame is blank"


def test_all_camera_modes_render(env):
    from aera.viz.camera import Camera
    from aera.viz.renderer3d import render_frame
    env.reset(seed=0)
    for mode in ("orbit", "chase", "top", "fp"):
        arr = render_frame(env, Camera(mode), size=(160, 120), time_s=0.0)
        assert arr.shape == (120, 160, 3)
        assert arr.std() > 1.0


def test_render_survives_full_episode(env):
    from aera.viz.camera import Camera
    from aera.viz.renderer3d import render_frame
    obs, _ = env.reset(seed=1)
    cam = Camera("orbit")
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        arr = render_frame(env, cam, size=(120, 90))
        assert np.isfinite(arr).all()
        if term or trunc:
            env.reset()
            break
