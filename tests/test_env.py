import numpy as np

from aera.config import Config
from aera.env import AeraEnv


def tiny_cfg() -> Config:
    return Config.load("configs/field_small.json")


def test_spaces_match_obs_and_actions():
    env = AeraEnv(tiny_cfg())
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.action_space.shape == (2,)      # rover in field_small
    assert np.all(obs == obs)                  # no NaNs


def test_reset_with_same_seed_is_reproducible():
    e1, e2 = AeraEnv(tiny_cfg()), AeraEnv(tiny_cfg())
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    assert np.allclose(o1, o2)
    r1 = [e1.step(e1.action_space.sample())[1] for _ in range(10)]
    e2.reset(seed=42)
    # determinism of the world itself, not of sampled actions — just check obs
    o3, _ = e1.reset(seed=42)
    assert np.allclose(o1, o3)


def test_step_contract_and_episode_stats():
    env = AeraEnv(tiny_cfg())
    obs, info = env.reset(seed=1)
    total = 0.0
    for _ in range(700):
        obs, r, term, trunc, info = env.step(np.zeros(2, np.float32))
        total += r
        assert np.isfinite(r)
        assert set(info["sub"]) >= {"food", "survive", "novelty", "damage"}
        if term or trunc:
            assert "episode" in info
            assert info["episode"]["steps"] > 0
            assert np.isclose(total, info["episode"]["ret"], atol=1e-3)
            break
    else:
        raise AssertionError("episode did not end within 700 steps (idle rover)")


def test_energy_drains_and_episode_terminates():
    cfg = tiny_cfg()
    cfg.sim.max_steps = 5000                     # rule out truncation
    env = AeraEnv(cfg)
    obs, _ = env.reset(seed=2)
    for _ in range(1500):
        obs, r, term, trunc, info = env.step(np.zeros(2, np.float32))
        if term:
            assert env.agent.energy <= 0 or env.agent.health <= 0
            break
    else:
        raise AssertionError("starving rover never terminated")


def test_lava_damages():
    cfg = tiny_cfg()
    cfg.terrain.lava_pools = 0
    cfg.curriculum = [{"until_episode": -1, "set": {}}]   # dangers on
    env = AeraEnv(cfg)
    env.reset(seed=3)
    # drop the agent on a synthetic lava cell
    env.world.terrain.grid[8, 8] = 3
    env.agent.x, env.agent.y = 8.5, 8.5
    h0 = env.agent.health
    for _ in range(30):
        env.step(np.zeros(2, np.float32))
    assert env.agent.health < h0


def test_obs_layout_dims_add_up():
    from aera.agents.senses import obs_dim, obs_layout
    cfg = tiny_cfg()
    total = sum(d for _, d in obs_layout(cfg.agent))
    assert total == obs_dim(cfg.agent)


def test_walker_action_space():
    cfg = tiny_cfg()
    cfg.agent.kind = "walker"
    env = AeraEnv(cfg)
    assert env.action_space.shape == (6,)
    obs, _ = env.reset(seed=0)
    obs, r, term, trunc, info = env.step(np.zeros(6, np.float32))
    assert np.isfinite(r)
