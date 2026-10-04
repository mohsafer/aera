"""Minds: a slow reasoning layer above the fast RL policy.

The RL policy acts at 10 Hz; a Mind thinks every K steps. It receives a
compact world summary (numbers only — the same facts the policy sees, plus
meta-state like the active anomaly events) and returns a Thought: a goal
token from a small fixed set plus a one-line rationale for the event feed.

The goal conditions the policy through a one-hot slot in the observation
(`AgentCfg.mind_goal`), so the policy learns WHAT each goal means and the
mind decides WHICH goal applies. Everything is pluggable:

  RuleMind — deterministic heuristics, zero deps, fully reproducible.
  LLMMind  — any OpenAI-compatible chat endpoint (AERA_LLM_URL env or
             --mind-endpoint), temperature 0. Recorded, not tape-replayable;
             falls back to "explore" if the endpoint is unreachable.

The env itself never knows a mind exists — the mind lives in the trainer /
watch loop (see training/trainer.py), which keeps the Gymnasium contract
pure (AGENTS.md §2).
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass

GOALS = ("forage", "flee", "shelter", "craft", "navigate", "explore")


@dataclass
class Thought:
    goal: str                 # one of GOALS
    rationale: str            # one line for the event feed


def build_summary(env) -> dict:
    """Compact numeric world snapshot handed to a Mind (humans can read it too)."""
    a, w = env.agent, env.world
    d_food, _a = env._nearest_active_food()
    threat = math.inf
    for p in w.predators:
        threat = min(threat, math.hypot(p.x - a.x, p.y - a.y))
    beacon_d = math.inf
    if w.beacon is not None and not w.beacon.hit:
        beacon_d = math.hypot(w.beacon.x - a.x, w.beacon.y - a.y)
    d_tool = math.inf
    for t in w.tools:
        if not t.taken:
            d_tool = min(d_tool, math.hypot(t.x - a.x, t.y - a.y))
    return {
        "step": env.steps,
        "kind": a.kind,
        "energy": round(a.energy / a.cfg.max_energy, 2),
        "health": round(a.health / a.cfg.max_health, 2),
        "speed": round(a.v, 2),
        "foods_eaten": a.foods_eaten,
        "nearest_food_dist": None if d_food is None else round(d_food, 1),
        "nearest_tool_dist": None if d_tool is math.inf else round(d_tool, 1),
        "nearest_threat_dist": None if threat is math.inf else round(threat, 1),
        "n_threats": len(w.predators),
        "beacon_dist": None if beacon_d is math.inf else round(beacon_d, 1),
        "inventory": sorted(a.inventory),
        "recent_events": list(env.last_events)[-3:],
    }


class Mind:
    name = "mind"

    def decide(self, summary: dict) -> Thought:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RuleMind(Mind):
    """Deterministic survival heuristics — the baseline 'reasoning' that the
    LLMMind must beat in the mind-audit A/B runs."""

    name = "rule"

    def decide(self, summary: dict) -> Thought:
        threat = summary["nearest_threat_dist"]
        if threat is not None and threat < 2.5:
            return Thought("flee", f"threat {threat}m away — run")
        if summary["health"] < 0.35:
            return Thought("shelter", "hurt — avoid hazards, survive")
        if summary["energy"] < 0.35:
            fd = summary["nearest_food_dist"]
            if fd is not None:
                return Thought("forage", f"hungry, food {fd}m — eat")
            return Thought("forage", "starving — search for food")
        if ("boots" not in summary["inventory"]
                and summary.get("nearest_tool_dist") is not None
                and summary.get("nearest_tool_dist") < 8.0):
            return Thought("craft", "tool nearby — gear up")
        bd = summary["beacon_dist"]
        if bd is not None and bd < 6.0:
            return Thought("navigate", f"beacon {bd}m — finish the job")
        return Thought("explore", "no urgent need — explore")


class LLMMind(Mind):
    """Ask an OpenAI-compatible chat endpoint for a thought. Endpoint from
    AERA_LLM_URL (e.g. http://localhost:11434/v1/chat/completions for a local
    ollama, or any /v1/chat/completions), model from AERA_LLM_MODEL, optional
    AERA_LLM_API_KEY. Temperature 0; any failure falls back to explore."""

    name = "llm"

    def __init__(self, url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 5.0):
        self.url = url or os.environ.get("AERA_LLM_URL", "")
        self.model = model or os.environ.get("AERA_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("AERA_LLM_API_KEY", "")
        self.timeout = timeout
        if not self.url:
            raise ValueError("LLMMind needs AERA_LLM_URL (or --mind-endpoint)")

    def _prompt(self, summary: dict) -> dict:
        system = (
            "You are the survival mind of a small robot in a hazardous field. "
            "Pick ONE goal for the next few seconds and answer ONLY with JSON: "
            '{"goal": "<forage|flee|shelter|craft|navigate|explore>", '
            '"rationale": "<max 60 chars>"}. Rules: flee if a threat is close; '
            "forage when hungry; shelter when badly hurt; craft when a tool "
            "matters; navigate when the beacon is near; otherwise explore."
        )
        return {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(summary)}],
        }

    def decide(self, summary: dict) -> Thought:
        try:
            req = urllib.request.Request(
                self.url, data=json.dumps(self._prompt(summary)).encode(),
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {self.api_key}"}
                            if self.api_key else {})})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
            text = payload["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("no JSON object in reply")
            parsed = json.loads(text[start:end + 1])
            goal = parsed.get("goal", "explore")
            if goal not in GOALS:
                goal = "explore"
            return Thought(goal, str(parsed.get("rationale", ""))[:80])
        except Exception as e:
            return Thought("explore", f"mind offline ({type(e).__name__}) — explore")
