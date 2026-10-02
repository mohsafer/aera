# AERA — Artificial Environment for Reinforcement Agents

A synthetic world where an agent **wakes up in a field knowing nothing** and
learns — with reinforcement learning — to walk (or drive), forage for food,
survive lava and hunger, pick up tools that unlock new abilities (jump!),
and navigate to the beacon. Watchable the whole time in chunky, retro-90s
3D: free orbit camera, first-person agent view, fog-of-war minimap, live
event feed, and skill milestones (*★ SKILL UNLOCKED: Walking*).

Built for students and RL tinkerers: the environment is a standard
**Gymnasium** env (Stable-Baselines3 / CleanRL plug straight in), the world
is one JSON config, and a dependency-free **numpy PPO** is included so you
can train out of the box.

```text
      ┌────────────┐   obs (rays/proprio/terrain)   ┌────────────┐
      │   WORLD    │ ─────────────────────────────▶ │  POLICY    │
      │ terrain    │                                │ PPO (numpy │
      │ walls/pits │ ◀───────────────────────────── │  or SB3)   │
      │ food/lava  │   action (steer/gait/jump)     └────────────┘
      │ tools/beacon│                                      │
      └─────┬──────┘         episode return, sub-rewards    │
            │ 3D render (painter's algorithm, fog)          ▼
      ┌─────▼─────────────────────────────────────────────────────┐
      │  VIEWER: orbit/chase/top/fp cams · minimap · event feed   │
      │          vitals · reward chart · ★ skill milestones       │
      └───────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# if this machine has no pip/ensurepip (Debian default image):
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install -r requirements.txt

# train (headless) and watch the replay
.venv/bin/python -m aera train --config configs/field_open.json --agent walker --steps 300000
.venv/bin/python -m aera watch --ckpt runs/walker_field_open_s0/policy_best.npz

# or watch training live (needs a display)
.venv/bin/python -m aera train --config configs/field_open.json --watch

# no display? record a GIF of a trained (or random) policy
.venv/bin/python -m aera watch --ckpt runs/... --record demo.gif
```

Controls in the viewer: **space** pause · **v** cycle camera (orbit/chase/
top/first-person) · **f** follow · **arrows/drag** orbit · **wheel** zoom ·
**h** help.

## The world

| element | behaviour |
|---|---|
| grass / road / mud | normal / fast / slow ground |
| walls | block movement *and* vision |
| lava | damage while standing on it |
| pits | impassable on foot — unless you're airborne (boots!) |
| food | restores energy, scores points, respawns |
| boots (tool) | unlocks the jump action channel |
| beacon | goal — big reward |

Agents: **rover** (differential drive — learn to steer) and **walker**
(4 joint targets whose *coordination* produces thrust — literally learn to
walk; watch the gait meter climb from flailing to stepping).

Episodes end by starving, burning, timeout, or the beacon. Curriculum stages
in the config ramp hunger and danger as episodes accumulate, and every
episode logs to `runs/<name>/metrics.jsonl`.

## Status & roadmap

- [x] v0.1 — world, two bodies, senses, rewards+curriculum, numpy PPO,
      3D viewer/recorder, milestones, tests
- [ ] first long training runs (walker: walking; rover: foraging)
- [ ] optional Stable-Baselines3 script as the "modern baseline" proof
- [ ] multi-agent coexistence (renderer already takes N agents)
- [ ] tool *crafting* (combine items at a workbench), more abilities
      (dash, dig), PyBullet physics adapter

