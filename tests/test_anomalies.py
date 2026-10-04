import numpy as np

from aera.agents.agent import Agent
from aera.agents.senses import build_obs, obs_dim, obs_layout
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


def test_predator_chases_and_bites():
    w = make_world()
    w.terrain.grid[:, :] = 0
    from aera.world.entities import Predator
    from aera.config import EntitiesCfg
    w.cfg.entities.predator_damage = 6.0
    p = Predator(x=10.0, y=12.0, speed=1.5)
    w.predators.append(p)
    a = Agent(AgentCfg(kind="rover"), 12.0, 12.0)
    d0 = a.health
    for t in range(40):
        w.update_predators(0.1, [a], t)
    assert p.x > 10.5, "predator must close in on the target"
    # contact bite: within 0.7 m → damage + predator stun
    p.x, p.y, p.stun_until = a.x - 0.4, a.y, -1
    tick = 40
    cfg = Config.load("configs/field_small.json")
    from aera.env import AeraEnv
    env = AeraEnv(cfg)
    env.world, env.agent = w, a          # reuse the constructed world/agent
    events = []
    env._consequences(events)
    assert a.health < d0 and p.stun_until > 0 and events, "bite: damage + stun"


def test_alien_drop_spawns_far_and_queues_event():
    w = make_world()
    a = Agent(AgentCfg(kind="rover"), 6.5, 6.5)
    d = make_director(seed=4)
    n0 = len(w.predators)
    msg = d.trigger("alien_drop", w, a, 0)
    assert msg and len(w.predators) == n0 + 1
    p = w.predators[-1]
    assert np.hypot(p.x - a.x, p.y - a.y) >= 6.0
    assert any("meteor" in e for e in d.step(w, a, 1)), "queued event flushes"


def test_holes_open_and_heal():
    w = make_world()
    before = w.terrain.grid.copy()
    d = make_director(seed=5)
    d.trigger("holes", w, None, 0)
    assert (w.terrain.grid == 4).sum() > 0, "holes must carve PIT cells"
    w.reset_episode()
    assert np.array_equal(w.terrain.grid, before)


def test_threat_sense_extends_obs_and_flags_predator():
    cfg_plain = AgentCfg(kind="rover")
    cfg_threat = AgentCfg(kind="rover", threat_sense=True)
    assert obs_dim(cfg_threat) > obs_dim(cfg_plain)
    w = make_world()
    from aera.world.entities import Predator
    p = Predator(x=13.0, y=12.0)
    w.predators.append(p)
    a = Agent(cfg_threat, 12.0, 12.0, heading=0.0)
    obs = build_obs(a, w, cfg_threat)
    layout = obs_layout(cfg_threat)
    names = [n for n, _ in layout]
    assert "fear" in names
    fear_at = sum(d for n, d in layout[:names.index("fear")])
    fear = obs[fear_at:fear_at + 4]
    assert abs(fear[1] - 1.0) < 0.1, "predator due east → cos ≈ 1"
    ray_block = obs[: cfg_threat.view_rays * 7].reshape(cfg_threat.view_rays, 7)
    assert ray_block[:, 6].sum() >= 1.0, "predator ahead must flag a ray"
