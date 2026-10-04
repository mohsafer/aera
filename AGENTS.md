# AGENTS.md — AERA complete reference

Everything a coding agent (or a student, or a future you) needs to work on
this repo. The design *story* and decision history live in `log.md`; the
pitch lives in `README.md`. This file is the technical contract.

---

## 1. Mission

AERA (Artificial Environment for Reinforcement Agents) is a synthetic world
where an embodied agent **wakes up knowing nothing** and must learn, through
reinforcement learning, to:

1. **Locomote** — coordinate a body (walk) or drive wheels (rover),
2. **Perceive** — sense walls, terrain, food, dangers through egocentric rays,
3. **Forage** — find food to survive hunger (energy budget),
4. **Survive** — avoid lava / pits / starvation,
5. **Use tools** — pick up *boots* → unlock the jump channel → cross pits,
6. **Navigate** — reach the beacon (goal).

All of it must be *watchable*: retro-90s low-poly 3D, free cameras
(orbit / chase / top / first-person), fog-of-war minimap, live event feed,
skill milestones ("★ SKILL UNLOCKED: Walking"), and reward-curve charts.

## 2. Hard requirements / invariants

- **Gymnasium API compatibility.** `aera/env.py::AeraEnv` must stay a valid
  `gym.Env`: `reset(seed=…) → (obs, info)`, `step(a) → (obs, r, terminated,
  truncated, info)`, `observation_space`, `action_space`. This is what makes
  Stable-Baselines3 / CleanRL / Tianshou plug in unchanged. The internal
  trainer never assumes gymnasium is installed (`aera/gym_compat.py` shim).
- **Light dependencies.** Core = `numpy + pygame + gymnasium` (+ `pillow`
  for GIF export, `matplotlib` for plots). No torch/mujoco in the core path.
  The built-in PPO is numpy-only by design.
- **Determinism.** World generation, spawning, and the learner are all
  seeded. Same config + same seed → same experiment. Never use global
  `random`/`np.random` state; thread an `np.random.Generator`. (Verified in
  practice: a re-run reproduced a paused run's trainer output byte-for-byte
  through 124 updates.)
- **Everything reproducible from one JSON.** A run = `configs/*.json` +
  seed. `runs/<name>/config_used.json` is written by the trainer.
- **Humans get a separate channel.** `info["sub"]` (reward breakdown) and
  `info["events"]`/`env.last_events` (story events) are for students/HUD —
  the policy must only ever see `obs` and scalar reward.

## 3. Module map

```
aera/
  config.py            Config dataclasses (World/Agent/Reward/Sim) + JSON load,
                       curriculum stages (stage_for_episode / apply_overrides)
  gym_compat.py        import gymnasium if present, else minimal shim (HAS_GYM)
  env.py               AeraEnv — gym API, physics call, consequences, rewards,
                       curriculum, events, episode stats, render()
  world/
    terrain.py         Terrain enum (GRASS/ROAD/MUD/LAVA/PIT), grid gen
                       (blobs, pit strips, road), FRICTION table, pristine
                       snapshot + restore() (anomaly scars heal on reset)
    anomalies.py       AnomalyDirector: seeded episodic stress events
                       (wind/fog/famine/lava_surge/terrain_shift/quake) —
                       touches terrain cells, food activity, world.wind,
                       agent kicks, agent.sensor_noise; rewards untouched
    entities.py        Food / Tool / Beacon dataclasses, place_free scatter
    world.py           World: walls (AABB rects incl. border), collision
                       (move_circle — sub-stepped, axis-separated), raycast
                       (slab test), line_of_sight, food respawn, passable-cell count
    physics.py         Locomotion models (swappable!): RoverPhysics (diff
                       drive), WalkerPhysics (4 joints + gait_score gate),
                       jump tool gating, integrate() w/ terrain friction
  agents/
    agent.py           Agent entity: pose, energy/health, joints, inventory,
                       cells_seen (novelty), events buffer, speed_sum
    senses.py          build_obs: rays×6ch, proprio, 3×3 terrain one-hot,
                       inventory, scent, beacon → float32 vector; obs_dim /
                       obs_layout for introspection
  rl/
    ppo_numpy.py       PPO (2×64 tanh trunk, Gaussian head, GAE, clipping,
                       Adam, RunningNorm, save/load npz; threaded RNG)
  training/
    metrics.py         JsonlLogger + MilestoneTracker (skill rules)
    trainer.py         Trainer: collect→GAE→update loop, checkpoints,
                       optional --init warm-start, optional live Viewer hook,
                       load_policy()
  viz/
    camera.py          Camera (orbit/chase/top/fp) → (eye, right, up, fwd)
    renderer3d.py      software low-poly renderer: face gathering, painter's
                       sort, Lambert + fog, sky gradient; walker/rover models
    hud.py             minimap (fog of war), vitals, event feed, chart, help
    viewer.py          Viewer: pygame window; .tick(env) hook for trainer,
                       .play(env, policy) standalone; input handling
    record.py          record_gif: headless rollout → GIF via Pillow
  scripts/
    train.py           CLI  → python -m aera train
    watch.py           CLI  → python -m aera watch (or --record out.gif)
    sb3.py             CLI  → python -m aera sb3 (optional SB3 baseline;
                       Monitor CSV + logger CSV + 50k-step checkpoints)
    plot.py            CLI  → python -m aera plot <run_dir> → curve PNGs
configs/               field_small (tests/smoke), field_open (canonical),
                       field_curriculum (walls+pits+tools, 40×40),
                       field_anomalies (field_open + all stress events)
tests/                 pytest suite (world, physics, env, ppo, renderer, e2e)
```

## 4. Data flow (one step)

```
policy(obs) → action ∈ [-1,1]^d
  → Agent.physics.step (locomotion model; may append "jumped" to agent.events)
  → integrate() (terrain friction, circle-vs-walls/pits movement)
  → world.step_time (food respawn)
  → env._consequences (lava damage, hunger, novelty, food/tool/beacon pickup,
                       shaping) → sub-rewards dict
  → reward = Σ sub (+ death penalty), terminated/truncated, obs = build_obs
  → info["events"]=step_events, info["sub"], info["episode"] on done
```

Episode ends on: health ≤ 0 (lava), energy ≤ 0 (starved), beacon (if
`beacon_terminates`), or `sim.max_steps` → truncated.

## 5. Observation & action contracts

**Action** (Box −1..1): rover `[steer, throttle]`; walker
`[turn, hip_l, knee_l, hip_r, knee_r, jump]`. The jump channel is inert
until the agent holds the *boots* tool — capability gating by design.

**Obs** (float32, layout via `senses.obs_layout(cfg)`):
`rays` (N×6: dist, wall, lava, food, tool, beacon — walls occlude, lava is
ray-marched) · `proprio` (speed, heading sin/cos, [phase sin/cos + 4 joints
for walker], airborne, jump_cd, energy, health, effort) · `terrain3x3`
(9×5 one-hot) · `inventory` (4 reserved flags) · `scent` (sin/cos/dist to
nearest food, opt-in) · `beacon` (sin/cos/dist).

**Learning-to-walk mechanism** (`world/physics.py::gait_score`): thrust is
gated by hips-in-antiphase × knees-in-phase × swing-amplitude. Random
flailing ≈ 0 movement; alternating gait → full speed. A small "shamble"
floor speed keeps gradients alive early, and the CPG clock in the obs
(`agent.cfg.cpg`) lets the policy lock onto a rhythm quickly. Disable cpg
for research-grade difficulty. This is an abstraction over real physics —
swap `WalkerPhysics` for a PyBullet adapter later; nothing else changes.

**Milestone speed units**: `episode["mean_speed"]` is true m/s (speed_sum
is ∫v·dt; divide by simulated seconds). Milestone thresholds are calibrated
against m/s (rover 1.2, walker 0.7).

## 6. Reward terms (all in `RewardCfg`, mirrored into `info["sub"]`)

`w_food` per food (+energy refill) · `w_survive`×dt · `w_novelty` per new
cell (count-based exploration) · `w_progress`×Δdist to nearest food
(shaping; reward-as-observation trick mirrors it into `scent`) ·
`w_damage`×hp lost · `w_beacon` once · `w_death` at terminal failure.

**Curriculum**: `config.curriculum` = list of `{until_episode, set}` stages
applied cumulatively on every reset; `set` writes attributes by name onto
RewardCfg/AgentCfg (e.g. `hunger_rate`, `lava_damage`). To "undo" a stage's
override you must re-set the value explicitly in the next stage.

**Anomalies** (`world/anomalies.py`, off by default, `world.anomalies` in
config): episodic stress events from a threaded director rng — wind (world
wind drift in integrate), fog (per-step multiplicative bias on ray dist via
agent.sensor_noise), famine (food hidden, staggered return), lava_surge /
terrain_shift (carve_blob scarring; reset heals via terrain.restore()),
quake (v/heading kicks). Rewards and obs layout unchanged. See
docs/USER_GUIDE.md §3.

**Milestones** (`training/metrics.py::MilestoneTracker`): rolling 15-episode
rules → Walking / Forager / Survivor / Explorer / ToolUser / Navigator;
printed as `★ SKILL UNLOCKED` and logged to metrics.jsonl. Explorer is
speed-normalized: `explored_norm = explored / min(1, MAX_SPEED × seconds /
passable_cells)` (≈1 novel cell per metre at cruise speed), threshold 0.15 —
one rule across bodies and world sizes, immune to camping (a camper's
reachable area is computed at body cruise speed, not its own). Raw
`explored` stays in the episode record for plotting.

## 7. Running things

### Bootstrap (server images without pip/ensurepip)

```bash
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install -r requirements.txt
```

### Train / watch / record / plot

```bash
.venv/bin/python -m aera train --config configs/field_open.json --agent rover --steps 300000
.venv/bin/python -m aera train --config configs/field_open.json --watch          # live 3D
.venv/bin/python -m aera train --init runs/walker_field_open_s0/policy_final.npz \
    --steps 400000                     # warm-start (--lr recommended, see log.md)
.venv/bin/python -m aera train --steps 1000000 \
    --resume runs/walker_field_open_s0 # true resume: optimizer + curriculum + RNG
.venv/bin/python -m aera watch --ckpt runs/rover_field_open_s0/policy_best.npz
.venv/bin/python -m aera watch --ckpt ... --record demo.gif                      # headless
.venv/bin/python -m aera plot runs/walker_field_open_s0                          # curve PNGs
.venv/bin/python -m pytest -q                                                    # tests
```

Run artifacts in `runs/<name>/`: `metrics.jsonl` (per-episode + per-update),
`policy_final.npz`, `policy_best.npz`, `trainer_state.npz` (for `--resume`),
`config_used.json`, `demo.gif`, `curves_episodes.png`, `curves_updates.png`
(SB3 runs additionally: `monitor.csv`, `logs/progress.csv`, `sb3_ckpt_*.zip`).

### Headless servers

Viewer needs a display; use `--record` (offscreen renderer) or run training
without `--watch`. To browse artifacts from a laptop: `python -m http.server
8000 --bind 127.0.0.1` inside `runs/` + `ssh -L 8000:localhost:8000`.
Live `--watch` works over `ssh -X` (needs a local X server).

### Use a modern baseline instead of the built-in PPO

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu stable-baselines3
python -m aera sb3 --config configs/field_open.json --steps 300000
```

or programmatically:

```python
from stable_baselines3 import PPO
from aera.config import Config; from aera.env import AeraEnv
env = AeraEnv(Config.load("configs/field_open.json"))
PPO("MlpPolicy", env, verbose=1).learn(1_000_000)
```

Measured on this project (see log.md): SB3 PPO reaches ~7× the built-in
PPO's return on the rover smoke world at equal steps, and beats the
built-in's 600k-step walker score in 300k steps. Use SB3 for serious runs;
the numpy PPO is the readable reference.

## 8. Extension recipes

- **New agent body**: add a model in `world/physics.py` (register in
  `PHYSICS`), an action layout in `agents/senses.py::action_dim` /
  `proprio_dim`, a render model in `viz/renderer3d.py::_<name>`, and allow
  the kind in `env.py`.
- **New terrain type**: extend `Terrain` + `FRICTION`, generation in
  `TerrainGrid`, palette in `renderer3d.TERRAIN_COLORS` (+ HUD `TC`), and
  any blocking/damage behavior in `World.blocked_at` / `env._consequences`.
- **New tool/ability**: add a kind to `INVENTORY_SLOTS` (senses), a pickup
  rule (env `_consequences` already handles generic tools), gate the ability
  in physics on `agent.inventory`, optionally a milestone rule.
- **New milestone**: one lambda in `metrics.MilestoneTracker.RULES`.
- **New world**: copy a `configs/*.json` — terrain/walls/entities/curriculum
  are all data.

## 9. Code conventions

- Python 3.10+ (server image runs 3.10.12); type hints on public functions;
  dataclasses for config-like records; no global mutable state.
- `np.float32` at the env boundary; floats inside physics are fine.
  NB: never hand numpy scalars to `pygame.draw.*` — cast to `float` (the
  renderer does this at the projection boundary).
- Keep the human channel (`info["sub"]`, events) out of learning code.
- Units: metres & seconds; one cell = 1 m; `sim.dt = 0.1` (10 Hz control).
- Comments state constraints, not narration.

## 10. Current status & known simplifications

- **Implemented & exercised (v0.2, post server session 2026-10-02)**: full
  package above, 36/36 tests, both bodies, tools (boots→jump), beacon,
  curriculum, milestones (Explorer speed-normalized), world anomalies, 3D
  viewer/recorder, plotting, SB3 baseline script, `train --init` warm-start
  and `train --resume` true continuation. Hands-on manual:
  docs/USER_GUIDE.md.
- **Training results** (see log.md for details): walker 600k on field_open —
  5/6 milestones (Walking@271, no gait tuning needed), ret ~41, 0.75 m/s;
  +400k warm-start continuation (≈1M total): ret ~44 via foraging consistency,
  speed flat (gait plateaued at ~0.4 coordination). Rover peaks at 1.09 m/s
  but the constant-lr reference PPO oscillates (use SB3 for the rover Walking
  milestone). Explorer re-scaled to speed-normalized form (0.15 of
  cruise-reachable) → all 6 milestones attainable; raw exploration plateaus
  at ~0.13/episode.
- **Simplifications (deliberate, see log.md D2/D8)**: locomotion is
  procedural (not articulated-body physics); single agent per env; tool
  *crafting* is designed-for but not implemented; novelty is cell-count.
- **Next candidates**: two-agent coexistence, tool crafting.
