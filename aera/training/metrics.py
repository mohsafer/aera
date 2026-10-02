"""Experiment telemetry: JSONL logs and skill-milestone detection.

Every run directory gets `metrics.jsonl` with one line per episode
(`type: episode`) and per PPO update (`type: update`). Milestones are the
human-facing "skills" the agent unlocks — they drive the ★ announcements.
"""
from __future__ import annotations

import json
import os
from collections import deque


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def log(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")


class MilestoneTracker:
    """Rolling-window skill detection. Each skill fires once per run."""

    WINDOW = 15   # episodes averaged for the verdict

    RULES = {
        "Walking":  lambda s, kind: s["mean_speed"] >= (1.2 if kind == "rover" else 0.7),
        "Forager":  lambda s, kind: s["foods"] >= 4,
        "Survivor": lambda s, kind: s["steps"] >= 900,
        "Explorer": lambda s, kind: s["explored"] >= 0.5,
        "ToolUser": lambda s, kind: "boots" in s["inventory"] and "jumped" in s["events"],
        "Navigator": lambda s, kind: s["beacon"],
    }

    def __init__(self, agent_kind: str):
        self.kind = agent_kind
        self.history: deque = deque(maxlen=self.WINDOW)
        self.unlocked: dict[str, int] = {}

    def update(self, ep: dict) -> list[str]:
        self.history.append(ep)
        if len(self.history) < self.WINDOW:
            return []
        avg = {
            "mean_speed": sum(e["mean_speed"] for e in self.history) / len(self.history),
            "foods": sum(e["foods"] for e in self.history) / len(self.history),
            "steps": sorted(e["steps"] for e in self.history)[len(self.history) // 2],
            "explored": sum(e["explored"] for e in self.history) / len(self.history),
            "inventory": self.history[-1]["inventory"],
            "events": [ev for e in self.history for ev in e["events"]],
            "beacon": any(e["beacon"] for e in self.history),
        }
        new = []
        for name, rule in self.RULES.items():
            if name not in self.unlocked and rule(avg, self.kind):
                self.unlocked[name] = self.history[-1]["episode"]
                new.append(name)
        return new
