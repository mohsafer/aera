# AERA — User Guide

Everything you need to *use* AERA: install it, watch an agent wake up, train
it, stress it, and read its artifacts. For the architecture and design
rationale see `AGENTS.md` (reference) and `log.md` (decisions); this guide is
the hands-on manual.

---

## 1. Install

```bash
python3 -m venv --without-pip .venv        # only needed on images without pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install -r requirements.txt
```

Core = numpy + pygame-ce + gymnasium (+ pillow for GIFs, matplotlib for
plots). Python ≥ 3.10. Verify:

```bash
.venv/bin/python -m pytest -q       # 36 tests, ~40 s, no display needed
```

Optional extra (modern baseline):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu stable-baselines3
```

## 2. The 60-second tour

```bash
.venv/bin/python -m aera watch                    # random rover in the open field
```

A retro-90s 3D window opens: chunky low-poly world on the left, HUD on the
right (energy/health bars, fog-of-war minimap, reward chart, event feed).
Controls: **space** pause · **v** cycle camera (orbit → chase → top →
first-person) · **f** follow agent/world · **arrows/drag** orbit · **wheel**
zoom · **h** help · **esc/q** quit.

The agent you're watching knows *nothing* — it flails. The point of AERA is
watching it learn not to.

## 3. Concepts in five paragraphs

**The world** is a 1 m-cell grid (grass / road / mud / lava / pits) plus
axis-aligned walls. Food restores energy and scores; a *boots* tool unlocks
the otherwise-inert jump action channel (pits are impassable on foot); the
beacon is the goal. Everything is generated from one seed — same config +
same seed ⇒ the same world, the same training, byte for byte.

**The agent** is a `gymnasium.Env` (`aera/env.py::AeraEnv`): `reset(seed) →
(obs, info)`, `step(action) → (obs, reward, terminated, truncated, info)`.
Two bodies: **rover** (actions `[steer, throttle]`) and **walker** (actions
`[turn, hip_l, knee_l, hip_r, knee_r, jump]` — thrust is gated by gait
*coordination*, so it must literally learn to walk). Observations are
egocentric: distance rays with wall/lava/food/tool/beacon flags,
proprioception (speed, heading, joints, energy, health…), a 3×3 terrain
patch, inventory flags, a scent direction to the nearest food, and the
beacon bearing. `aera.agents.senses.obs_layout(cfg)` prints what each index
means.

**Rewards** are transparent: every step's `info["sub"]` shows the breakdown
(food / survive / novelty / progress / damage / beacon / death). A
**curriculum** ramps danger across episodes via `curriculum` stages in the
config (first safe, then hunger, then lava).

**Milestones** are the human-facing skill tracker: Walking, Forager,
Survivor, Explorer, ToolUser, Navigator. Each is a rolling-15-episode rule
(definitions in `aera/training/metrics.py::RULES`) printed as
`★ SKILL UNLOCKED` and logged. Explorer is *speed-normalized*: it measures
the fraction of cells reachable at your body's cruise speed that you
actually visited (≥ 0.15), so one rule fits both bodies and camping can't
game it.

**Anomalies** are episodic stress events that test robustness once a policy
works: storms (wind drift), fog (noisy vision), famine (food vanishes and
trickles back), lava surges (the ground splits open), terrain shifts (mud
swallows the road), and earthquakes (random shoving). They are seeded and
deterministic, off by default, and fire through the event feed
("a storm rolls in — the wind pushes you!"). Enable them in any config:

```json
"world": { ..., "anomalies": { "enabled": true, "event_prob": 0.004, "intensity": 1.0 } }
```

`kinds` selects the subset; `intensity` scales all of them. Every effect
heals at episode reset (the terrain un-scar, food returns). Rewards and the
observation layout don't change — the agent just has to *cope*.

## 4. Training

```bash
# canonical walker run (~2 h at 10 Hz on one core)
.venv/bin/python -m aera train --config configs/field_open.json --steps 600000

# rover sanity run
.venv/bin/python -m aera train --config configs/field_small.json --steps 50000 --agent rover

# stress run: same world, anomalies on
.venv/bin/python -m aera train --config configs/field_anomalies.json --steps 600000

# watch training live (needs a display / ssh -X)
.venv/bin/python -m aera train --config configs/field_open.json --watch
```

Useful flags: `--seed N` (whole run deterministic), `--out DIR` (run
directory, default `runs/<kind>_<world>_s<seed>`), `--rollout N` (steps per
PPO update), `--fps N` (viewer cap).

**Continuing training** — two different tools for two different jobs:

- `--init runs/<name>/policy_final.npz` *warm-starts* the weights (and obs
  normalizer) but restarts the curriculum at episode 0 and resets the
  optimizer. Use it to transplant a policy into a new experiment — and pair
  it with a smaller learning rate, or a converged policy can be destroyed by
  the fresh Adam moments (this actually happened; see `log.md`):

  ```bash
  .venv/bin/python -m aera train --config configs/field_open.json \
      --init runs/walker_field_open_s0/policy_final.npz --lr 1e-4 --steps 400000
  ```

- `--resume runs/<name>` *continues* where a run stopped: weights + obs
  normalizer + **Adam moments** + the curriculum's episode counter + both RNG
  states (the trainer writes `trainer_state.npz` alongside its checkpoints).
  Use it after a pause or to extend a finished run's budget:

  ```bash
  .venv/bin/python -m aera train --config configs/field_open.json \
      --steps 1000000 --resume runs/walker_field_open_s0
  ```

  A resume continues at an episode boundary, so the step count may differ
  from an uninterrupted run by up to one rollout — it's a faithful restart,
  not a bit-exact tape replay.

## 5. Watching results

```bash
# replay a trained policy in a window
.venv/bin/python -m aera watch --ckpt runs/walker_field_open_s0/policy_best.npz

# headless: render a GIF instead (works on servers, no display)
.venv/bin/python -m aera watch --ckpt runs/walker_field_open_s0/policy_best.npz \
    --record demo.gif --frames 240 --camera chase
```

Cameras: `orbit` free view · `chase` behind the agent · `top` map view ·
`fp` first-person. GIF recording samples every other sim step at 20 fps.

**On a headless server**, browse artifacts from your laptop:

```bash
cd runs && python -m http.server 8000 --bind 127.0.0.1
ssh -L 8000:localhost:8000 user@server        # on YOUR machine
# then open http://localhost:8000
```

## 6. Artifacts & plots

Every run directory contains:

| file | what it is |
|---|---|
| `metrics.jsonl` | one record per episode (`type: episode`) and per PPO update (`type: update`) |
| `policy_final.npz` | latest policy (every 20 updates + at the end) |
| `policy_best.npz` | best rolling-20-episode return |
| `trainer_state.npz` | full continuation state for `--resume` |
| `config_used.json` | the exact config — a run is reproducible from this + seed |
| `demo.gif` | recorded replay (if you made one) |
| `curves_episodes.png` | return, food, speed, length, exploration, tool-held |
| `curves_updates.png` | PPO loss curves (built-in runs) |

Plot them any time:

```bash
.venv/bin/python -m aera plot runs/walker_field_open_s0
```

SB3 runs (`monitor.csv`, `logs/progress.csv`, `sb3_ckpt_*.zip` checkpoints
every 50k steps) plot with the same command.

### Reading metrics.jsonl

```python
import json
for line in open("runs/walker_field_open_s0/metrics.jsonl"):
    rec = json.loads(line)
    if rec["type"] == "episode":
        print(rec["episode"], round(rec["ret"], 1), round(rec["mean_speed"], 2), "m/s")
```

Per-episode keys: `steps`, `ret`, `score`, `foods`, `mean_speed` (true m/s),
`explored` (fraction of passable cells), `explored_norm` (speed-normalized),
`inventory`, `beacon`, `events` (story lines — including anomalies).

## 7. Minds — give the agent thoughts

A **Mind** is a slow reasoning layer above the fast RL policy: every
`--mind-interval` steps (default 25 = 2.5 sim-seconds) it reads a compact
world summary (energy, health, threat/food/beacon distances, inventory) and
emits a *thought* — a goal (`forage / flee / shelter / craft / navigate /
explore`) plus a one-line rationale. The goal enters the policy through a
one-hot slot in the observation; the rationale streams into the event feed
(`★ THOUGHT: forage — starving`) and every thought lands in
`metrics.jsonl` (`type: thought`) for auditing.

```bash
# deterministic rule-based mind (no dependencies)
.venv/bin/python -m aera train --config configs/field_anomalies.json --mind rule

# LLM mind — any OpenAI-compatible chat endpoint (local ollama or hosted)
export AERA_LLM_URL=http://localhost:11434/v1/chat/completions   # ollama
export AERA_LLM_MODEL=llama3.1
.venv/bin/python -m aera train --config configs/field_anomalies.json --mind llm

# watch a minded policy (thoughts appear live in the stream feed)
.venv/bin/python -m aera watch --ckpt ... --mind rule --stream 8889
```

Notes: the environment stays a pure Gymnasium env — the mind lives in the
trainer/watch loop, so SB3 users can ignore it. LLM runs are recorded but
not bit-reproducible (documented exception to the determinism invariant).
`python -m aera plot <run>` renders `curves_mind.png` (goal over steps +
per-goal counts) when thought records exist. To measure whether thoughts
*help*, run the same seed with `--mind rule` vs `--mind none` and compare
returns — the honest A/B.

## 8. Stable-Baselines3

```bash
.venv/bin/python -m aera sb3 --config configs/field_open.json --steps 300000
```

or programmatically (the env is a standard Gymnasium env):

```python
from stable_baselines3 import PPO
from aera.config import Config
from aera.env import AeraEnv
env = AeraEnv(Config.load("configs/field_open.json"))
PPO("MlpPolicy", env, verbose=1).learn(300_000)
```

Measured on this project: SB3 beats the built-in PPO ~7× on the rover smoke
world at equal steps and beats its 600k-step walker score in 300k steps. The
built-in numpy PPO is the readable reference; SB3 is the serious optimizer.

## 9. Config reference (one JSON per world)

```jsonc
{
  "world": {
    "name": "Open Field", "width": 32, "height": 32, "seed": 7,
    "terrain": { "mud_patches": 2, "lava_pools": 1, "pits": 1, "road": false },
    "walls":   { "border": true, "rects": [[cx, cy, w, h], ...] },
    "entities":{ "foods": 8, "food_respawn_ticks": 300, "tools": ["boots"],
                 "beacon": [28, 28] },            // beacon: null to disable
    "anomalies": { "enabled": true, "kinds": ["wind", "fog", "famine",
                   "lava_surge", "terrain_shift", "quake"],
                   "event_prob": 0.004, "intensity": 1.0 }
  },
  "agent":   { "kind": "walker", "view_rays": 12, "view_range": 10.0,
               "cpg": true, "scent": true },
  "sim":     { "dt": 0.1, "max_steps": 2000 },
  "reward":  { "w_food": 3.0, "w_survive": 0.01, "w_novelty": 0.06,
               "w_progress": 0.05, "w_damage": 0.5, "w_death": 5.0,
               "w_beacon": 10.0, "beacon_terminates": false },
  "curriculum": [
    { "until_episode": 40,  "set": { "hunger_rate": 0.0, "lava_damage": 0.0 } },
    { "until_episode": 150, "set": { "hunger_rate": 0.8, "lava_damage": 0.0 } },
    { "until_episode": -1,  "set": { "hunger_rate": 0.8, "lava_damage": 9.0 } }
  ]
}
```

Notes: walls are `[cx, cy, w, h]` in cells; curriculum stages apply
cumulatively and override any `reward`/`agent` attribute by name (to undo a
stage, set the value again); shipped worlds are `field_small` (fast smoke),
`field_open` (canonical), `field_curriculum` (40×40 with walls/pits/tools),
`field_anomalies` (field_open + all stress events).

## 10. Python API

```python
from aera.config import Config
from aera.env import AeraEnv

env = AeraEnv(Config.load("configs/field_open.json"))
obs, info = env.reset(seed=0)
for _ in range(1000):
    action = env.action_space.sample()          # or your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    if info["events"]:
        print(info["events"])                   # story feed (humans only)
    print(info["sub"])                          # reward breakdown (humans only)
    if terminated or truncated:
        print(info["episode"])                  # per-episode stats
        obs, _ = env.reset()
```

The policy sees only `obs` and scalar `reward` — `info["sub"]` and
`info["events"]` are the human channel. Render a frame without a window:
`env.render()` (with `render_mode="rgb_array"`) → RGB uint8 array.

## 11. Troubleshooting

| symptom | cause / fix |
|---|---|
| `pygame.error: No available video device` | headless box — use `--record` for GIFs, or `SDL_VIDEODRIVER=dummy` for render-only work |
| plot says "no metrics.jsonl" | SB3 run? it uses `monitor.csv` — same command works; too-early run? wait for the first episode |
| warm-started run collapses / `nan` in losses | lr too high for fine-tuning — use `--lr 1e-4` (see `log.md`, warm-start divergence) |
| agent drifts with no throttle | a storm (wind anomaly) is active — check the event feed |
| no food anywhere | famine anomaly; it ends after 20–60 s and food trickles back |
| rover learned weird extra actions | fixed in v0.1.1 (action_dim bug); retrain — old rover checkpoints have a 6-dim head |
| milestone never fires | thresholds are rolling-15 rules; check the definitions in `aera/training/metrics.py::RULES` against your run's curves |

## 12. Extending (quick recipes)

Full recipes in `AGENTS.md §8`; the short version:

- **New body** — physics model in `world/physics.py` (register in `PHYSICS`),
  action layout in `senses.py`, render model in `renderer3d.py`, allow it in
  `env.py`.
- **New terrain** — `Terrain` + `FRICTION` in `terrain.py`, palette in
  `renderer3d.TERRAIN_COLORS` (+ HUD `TC`), behaviour in `env._consequences`.
- **New tool** — slot in `senses.INVENTORY_SLOTS`, gate the ability in
  physics on `agent.inventory`.
- **New anomaly** — one branch in `AnomalyDirector._apply` (trigger), plus
  any per-tick effect in `_sustain`; keep randomness on `self.rng`.
- **New milestone** — one lambda in `metrics.MilestoneTracker.RULES`.
- **New world** — copy a JSON in `configs/`; everything is data.
