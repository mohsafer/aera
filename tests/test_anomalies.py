import numpy as np

from aera.agents.agent import Agent
from aera.agents.senses import build_obs
from aera.config import AgentCfg, AnomalyCfg, Config, WorldCfg
from aera.env import AeraEnv
from aera.world.anomalies import AnomalyDirector
from aera.world.physics import integrate
from aera.world.world import World


def make_world() -> World:
    return World(WorldCfg(width=24, height=24, seed=5))


def make_director(seed=0, event_prob=0.0) -> AnomalyDirector:
    return AnomalyDirector(AnomalyCfg(enabled=True, event_prob=event_prob),
                           np.random.default_rng(seed))


def test_schedule_is_deterministic():
    cfg = AnomalyCfg(enabled=True, event_prob=0.5)
    d1 = AnomalyDirector(cfg, np.random.default_rng(9))
    d2 = AnomalyDirector(cfg, np.random.default_rng(9))
    a1 = Agent(AgentCfg(kind="rover"), 12, 12)
    a2 = Agent(AgentCfg(kind="rover"), 12, 12)
    w1, w2 = make_world(), make_world()
    seq1 = [d1.step(w1, a1, t) for t in range(300)]
    seq2 = [d2.step(w2, a2, t) for t in range(300)]
    assert seq1 == seq2 and any(seq1), "same seed must replay the same schedule"


def test_famine_hides_food_then_returns_it():
    w = make_world()
    assert w.foods and any(f.active for f in w.foods)
    a = Agent(AgentCfg(kind="rover"), 12, 12)
    d = make_director(seed=1)
    d.trigger("famine", w, a, w.tick)
    assert not any(f.active for f in w.foods)
    for _ in range(900):               # past the max famine duration
        d.step(w, a, w.tick)
        w.step_time()                  # the env's clock drives food revival
    assert any(f.active for f in w.foods), "famine must end"


def test_lava_surge_scars_and_reset_heals():
    w = make_world()
    before = w.terrain.grid.copy()
    d = make_director(seed=2)
    d.trigger("lava_surge", w, None, 0)
    assert (w.terrain.grid != before).any(), "surge must carve lava"
    w.reset_episode()
    assert np.array_equal(w.terrain.grid, before), "reset must heal the grid"


def test_wind_drifts_a_stationary_agent():
    w = make_world()
    w.terrain.grid[:, :] = 0            # flat grass, no hazards
    calm, windy = Agent(AgentCfg(kind="rover"), 4, 12), Agent(AgentCfg(kind="rover"), 4, 12)
    for _ in range(100):
        calm.physics.step(calm, np.zeros(2), w, 0.1)
        integrate(calm, w, 0.1)
    w.wind = (1.0, 0.0)
    for _ in range(100):
        windy.physics.step(windy, np.zeros(2), w, 0.1)
        integrate(windy, w, 0.1)
    assert windy.x > calm.x + 1.0, "1 m/s crosswind must drift the rover"


def test_fog_noise_blurs_the_ray_channel():
    w = make_world()
    cfg = AgentCfg(kind="rover")
    a = Agent(cfg, 12, 12)
    a.sensor_noise = 0.0
    clear = build_obs(a, w, cfg)[: cfg.view_rays * 6].reshape(cfg.view_rays, 6)
    a.sensor_noise = 0.5
    foggy = build_obs(a, w, cfg)[: cfg.view_rays * 6].reshape(cfg.view_rays, 6)
    assert not np.allclose(clear[:, 0], foggy[:, 0]), "dist channel must blur"
    assert np.allclose(clear[:, 1:], foggy[:, 1:]), "flags must stay intact"
    assert foggy[:, 0].max() <= 1.5


def test_env_with_anomalies_runs_and_reports_events():
    cfg = Config.load("configs/field_small.json")
    cfg.world.anomalies.enabled = True
    cfg.world.anomalies.event_prob = 0.0     # trigger manually, keep it hermetic
    env = AeraEnv(cfg)
    obs, _ = env.reset(seed=0)
    env.director.trigger("quake", env.world, env.agent, env.world.tick)
    saw_event = False
    for _ in range(120):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r) and np.all(obs == obs)
        saw_event = saw_event or any("earthquake" in e for e in info["events"])
        if term or trunc:
            obs, _ = env.reset()
    assert saw_event, "triggered anomaly must reach the event feed"
