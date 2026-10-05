import numpy as np

from aera.config import Config
from aera.env import AeraEnv
from aera.minds.base import RuleMind, build_summary
from aera.training.metrics import MilestoneTracker
from aera.world.entities import Predator, RECIPES, Workbench


def make_env() -> AeraEnv:
    cfg = Config.load("configs/field_alien.json")
    cfg.sim.max_steps = 5000
    env = AeraEnv(cfg)
    env.reset(seed=0)
    return env


def test_craft_at_workbench_consumes_components():
    env = make_env()
    a, w = env.agent, env.world
    w.tools.clear()                       # hermetic: no pickups nearby
    w.workbenches = [Workbench(x=a.x + 0.5, y=a.y)]
    a.inventory.update(("scrap", "crystal"))
    events = []
    env._consequences(events)
    assert "shield" in a.inventory
    assert "scrap" not in a.inventory and "crystal" not in a.inventory
    assert any("crafted shield" in e for e in events)


def test_craft_requires_all_components():
    env = make_env()
    a, w = env.agent, env.world
    w.tools.clear()
    w.workbenches = [Workbench(x=a.x + 0.5, y=a.y)]
    a.inventory.add("scrap")              # shield needs scrap+crystal
    events = []
    env._consequences(events)
    assert "shield" not in a.inventory and not any("crafted" in e for e in events)


def test_shield_halves_bite_damage():
    bites = []
    for shield in (False, True):
        env = make_env()
        a, w = env.agent, env.world
        w.tools.clear()
        if shield:
            a.inventory.add("shield")
        w.predators.append(Predator(x=a.x + 0.3, y=a.y))
        h0 = a.health
        env._consequences([])
        bites.append(h0 - a.health)
    assert np.isclose(bites[1], bites[0] / 2), (bites, "shield must halve the bite")


def test_lantern_cancels_fog():
    env = make_env()
    env.world.tools.clear()
    from aera.world.anomalies import AnomalyDirector
    from aera.config import AnomalyCfg
    d = AnomalyDirector(AnomalyCfg(enabled=True, event_prob=0.0),
                        np.random.default_rng(3))
    d.trigger("fog", env.world, env.agent, env.world.tick)
    env.agent.inventory.add("lantern")
    d.step(env.world, env.agent, env.world.tick)
    assert d.obs_noise == 0.0, "lantern must keep vision clean"
    env.agent.inventory.discard("lantern")
    d.step(env.world, env.agent, env.world.tick)
    assert d.obs_noise != 0.0, "without the lantern fog returns"


def test_flare_scares_predators():
    env = make_env()
    a, w = env.agent, env.world
    w.tools.clear()
    a.inventory.add("flare")
    p = Predator(x=a.x + 2.0, y=a.y)
    w.predators.append(p)
    events = []
    env._consequences(events)
    assert "flare" not in a.inventory and p.fear_until > w.tick
    assert any("flare" in e for e in events)
    x0 = p.x
    for _ in range(30):
        w.update_predators(0.1, [a], w.tick)
    assert p.x > x0 + 0.5, "predator must run AWAY while feared"


def test_engineer_milestone_fires():
    tr = MilestoneTracker("walker")
    ep = {"episode": 0, "steps": 500, "ret": 1.0, "foods": 1, "mean_speed": 0.2,
          "explored": 0.05, "explored_norm": 0.2, "inventory": ["shield"],
          "beacon": False, "events": []}
    fired = []
    for _ in range(15):
        fired += tr.update(ep)
    assert "Engineer" in fired


def test_craft_sense_extends_inventory_flags():
    from aera.agents.senses import build_obs, n_inventory
    env = make_env()
    cfg = env.config.agent
    assert cfg.craft_sense and n_inventory(cfg) == 7
    obs = build_obs(env.agent, env.world, cfg)
    assert obs.shape[0] > 0
    assert env.agent.inventory is not None


def test_mind_suggests_crafting_near_bench():
    env = make_env()
    a, w = env.agent, env.world
    w.tools.clear()
    w.workbenches = [Workbench(x=a.x + 1.0, y=a.y)]
    a.inventory.update(("scrap", "crystal"))
    s = build_summary(env)
    assert s["components_held"] == 2 and s["nearest_workbench_dist"] < 2.0
    assert RuleMind().decide(s).goal == "craft"
    assert RECIPES["shield"] == ("scrap", "crystal")
