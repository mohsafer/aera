import numpy as np

from aera.rl.ppo_numpy import PPO, RunningNorm


def test_running_norm_normalizes():
    n = RunningNorm(4)
    x = np.random.default_rng(0).normal(5.0, 2.0, size=(2000, 4))
    n.update(x)
    z = n.normalize(np.array([5.0, 5.0, 5.0, 5.0], np.float32))
    assert np.allclose(z, 0.0, atol=0.3)


def test_act_shapes_and_determinism():
    ppo = PPO(obs_dim=10, act_dim=3, seed=0)
    obs = np.zeros(10, np.float32)
    a1, logp1, v1 = ppo.act(obs, deterministic=True)
    a2, logp2, v2 = ppo.act(obs, deterministic=True)
    assert a1.shape == (3,) and np.all(a1 >= -1) and np.all(a1 <= 1)
    assert np.allclose(a1, a2) and logp1 == logp2 and v1 == v2


def test_update_runs_and_learns_value():
    """PPO update on a synthetic 1-step bandit: value should chase the return."""
    ppo = PPO(obs_dim=4, act_dim=2, hidden=32, epochs=6, minibatch=64, seed=0)
    rng = np.random.default_rng(1)
    obs = rng.normal(0, 1, size=(512, 4)).astype(np.float32)
    act = rng.uniform(-1, 1, size=(512, 2)).astype(np.float32)
    logp = np.zeros(512)
    ret = (obs.sum(1) * 2.0).astype(np.float64)     # learnable target
    adv = ret - ret.mean()
    v_before = np.array([ppo.value(o) for o in obs[:32]])
    for _ in range(5):
        stats = ppo.update(obs, act, logp, adv, ret)
        assert all(np.isfinite(v) for v in stats.values())
    v_after = np.array([ppo.value(o) for o in obs[:32]])
    err_before = np.abs(v_before - ret[:32]).mean()
    err_after = np.abs(v_after - ret[:32]).mean()
    assert err_after < err_before


def test_save_load_roundtrip():
    ppo = PPO(obs_dim=8, act_dim=2, seed=0)
    obs = np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32)
    ppo.update(obs, np.zeros((64, 2)), np.zeros(64), np.zeros(64), np.zeros(64))
    a1, _, _ = ppo.act(obs[0], deterministic=True)
    ppo.save("/tmp/aera_test_policy.npz")
    other = PPO(obs_dim=8, act_dim=2, seed=99)
    other.load("/tmp/aera_test_policy.npz")
    a2, _, _ = other.act(obs[0], deterministic=True)
    assert np.allclose(a1, a2, atol=1e-5)


def test_full_state_roundtrip_continues_optimization():
    """state_dict must carry Adam moments + RNG, not just weights: a loaded
    PPO takes identical actions AND identical subsequent updates."""
    obs = np.random.default_rng(3).normal(size=(128, 6)).astype(np.float32)
    act = np.random.default_rng(4).uniform(-1, 1, size=(128, 2)).astype(np.float32)
    ppo = PPO(obs_dim=6, act_dim=2, hidden=32, seed=0)
    for _ in range(3):
        ppo.update(obs, act, np.zeros(128), np.ones(128), np.ones(128))
    snap = ppo.state_dict()          # snapshot BEFORE acting: same sampler state
    a1, logp1, v1 = ppo.act(obs[7])
    s1 = ppo.update(obs, act, np.zeros(128), np.ones(128), np.ones(128))

    fresh = PPO(obs_dim=6, act_dim=2, hidden=32, seed=42)
    fresh.load_state_dict(snap)
    a2, logp2, v2 = fresh.act(obs[7])
    s2 = fresh.update(obs, act, np.zeros(128), np.ones(128), np.ones(128))
    assert np.allclose(a1, a2) and logp1 == logp2 and v1 == v2
    assert np.isclose(s1["v_loss"], s2["v_loss"]) and np.isclose(s1["pi_loss"], s2["pi_loss"])
