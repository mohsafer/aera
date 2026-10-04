import json

import numpy as np

from aera.config import Config
from aera.env import AeraEnv
from aera.minds.base import GOALS, LLMMind, RuleMind, build_summary
from aera.training.trainer import Trainer, make_mind


def make_env() -> AeraEnv:
    cfg = Config.load("configs/field_small.json")
    cfg.agent.mind_goal = True
    return AeraEnv(cfg)


def test_rule_mind_prefers_fleeing_over_everything():
    env = make_env()
    env.reset(seed=0)
    # put a predator right next to the agent
    from aera.world.entities import Predator
    env.world.predators.append(Predator(x=env.agent.x + 1.0, y=env.agent.y))
    s = build_summary(env)
    t = RuleMind().decide(s)
    assert t.goal == "flee"
    s["nearest_threat_dist"] = None
    assert RuleMind().decide(s).goal != "flee"


def test_rule_mind_is_deterministic():
    env = make_env()
    env.reset(seed=1)
    mind = RuleMind()
    s = build_summary(env)
    thoughts = [mind.decide(s) for _ in range(5)]
    assert all(t.goal == thoughts[0].goal and t.rationale == thoughts[0].rationale
               for t in thoughts)


def test_goal_slot_in_obs():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] > 0
    # the goal block is the last len(GOALS) floats and one-hot at 'explore'
    block = obs[-len(GOALS):]
    assert np.isclose(block.sum(), 1.0)
    assert np.isclose(block[GOALS.index("explore")], 1.0)
    env.agent.goal = "flee"
    from aera.agents.senses import build_obs
    obs2 = build_obs(env.agent, env.world, env.config.agent)
    assert np.isclose(obs2[-len(GOALS):][GOALS.index("flee")], 1.0)
    # obs layout dims still add up
    from aera.agents.senses import obs_dim, obs_layout
    cfg = env.config.agent
    assert sum(d for _, d in obs_layout(cfg)) == obs_dim(cfg)


def test_make_mind_and_llm_fallback():
    assert make_mind("none", None) is None
    assert isinstance(make_mind("rule", None), RuleMind)
    llm = make_mind("llm", "http://127.0.0.1:1/v1/chat/completions")  # nothing there
    t = llm.decide({"nearest_threat_dist": None})
    assert t.goal == "explore" and "offline" in t.rationale


def test_trainer_runs_with_mind_and_logs_thoughts(tmp_path):
    cfg = Config.load("configs/field_small.json")
    cfg.agent.mind_goal = True
    cfg.sim.max_steps = 300
    out = str(tmp_path / "run")
    Trainer(cfg, out, total_steps=1200, rollout=600, seed=0,
            save_every_updates=1, mind=make_mind("rule", None),
            mind_interval=25).train()
    recs = [json.loads(l) for l in open(f"{out}/metrics.jsonl")]
    thoughts = [r for r in recs if r.get("type") == "thought"]
    assert thoughts, "mind must record thoughts"
    assert all(t["goal"] in GOALS for t in thoughts)
    assert all("summary" in t for t in thoughts)
