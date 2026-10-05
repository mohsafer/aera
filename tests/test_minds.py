import json

import numpy as np

from aera.config import Config
from aera.env import AeraEnv
from aera.minds.base import GOALS, LLMMind, RuleMind, build_summary
from aera.training.metrics import JsonlLogger
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


def test_llm_full_path_with_mock_endpoint(tmp_path):
    """Spin a canned OpenAI-compatible endpoint and verify the full path:
    request → parse → goal, then the cache absorbs the second call."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    hits = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            hits["n"] += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0
            user = body["messages"][1]["content"]
            assert "previous_thought" in user and "anomalies_active" in user
            reply = json.dumps({"goal": "flee", "rationale": "mock says run"})
            payload = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    cache = str(tmp_path / "mind_cache.json")
    llm = LLMMind(url=f"http://127.0.0.1:{port}/v1/chat/completions",
                  model="mock", cache_path=cache)
    s = build_summary(make_env())
    t1 = llm.decide(s, since_last=["storm!"])
    assert t1.goal == "flee" and t1.rationale == "mock says run"
    assert llm.previous is t1                     # continuity
    t2 = llm.decide(s)                            # same summary → cache
    assert t2.goal == "flee" and hits["n"] == 1
    llm2 = LLMMind(url=f"http://127.0.0.1:{port}/v1/chat/completions",
                   model="mock", cache_path=cache)
    t3 = llm2.decide(s)                           # cache survives restart
    assert t3.goal == "flee" and hits["n"] == 1
    server.shutdown()


def test_mindhook_thinks_on_threat_without_waiting_for_timer(tmp_path):
    import math as m
    from aera.training.trainer import MindHook
    from aera.world.entities import Predator

    cfg = Config.load("configs/field_small.json")
    cfg.agent.mind_goal = True
    env = AeraEnv(cfg)
    env.reset(seed=0)
    hook = MindHook(RuleMind(), interval=1000, logger=JsonlLogger(str(tmp_path / "m.jsonl")))
    hook.before_act(env, 0)                       # episode start → thinks
    assert env.agent.goal == "explore"
    assert hook.last_think_step == 0
    env.steps = 5                                 # far from the timer
    n_thoughts = hook.count
    hook.before_act(env, 5)
    assert hook.count == n_thoughts               # sticky: no re-think
    env.world.predators.append(Predator(x=env.agent.x + 1.0, y=env.agent.y))
    hook.before_act(env, 6)                       # urgent → thinks NOW
    assert env.agent.goal == "flee"
    assert hook.count == n_thoughts + 1
    hook.flush(env)
    assert any("THOUGHT" in e for e in env.last_events)
