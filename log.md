# AERA — design log

Journal of design decisions for the synthetic RL world ("agent wakes up in an
empty field, learns to walk, forage, survive, and use tools"). Newest entries at
the bottom. This file is my thinking notebook — the stable reference manual
lives in `AGENTS.md`, the pitch in `README.md`.

---

## 2026-10-02 — first design pass

### What we are building (restated)

An abstract synthetic world where one or more agents (rover / humanoid) are
embodied, wake up knowing nothing, and must learn — through RL — to
understand the environment: locomotion, obstacles (walls, pits), resources
(food/energy), dangers (lava), navigation, and eventually *tool use* (e.g.
spring boots that unlock jumping across pits) and goal completion (beacon).
It must be visual (retro 90s 3D look, free camera), student-friendly
(live viewer, event feed, skill milestones, metrics), and compatible with
modern RL baselines (Gymnasium API → Stable-Baselines3 / CleanRL plug in
directly).

### Architecture decisions and the alternatives I rejected

**D1. Language & RL ecosystem: Python + Gymnasium API.**
The whole modern RL baseline world (SB3, CleanRL, Tianshou, RLlib) speaks
Gymnasium. The env must be a first-class `gymnasium.Env`. Fallback: a tiny
internal shim (`aera/gym_compat.py`) so the sim still runs on machines without
gymnasium (our trainer only calls `reset/step/observation_space/action_space`).

**D2. Physics: swappable "locomotion abstraction layer", not PyBullet (v1).**
Rejected options:
- *PyBullet/MuJoCo*: real articulated-body physics, but heavy deps, slow to
  install (we hit this today — the sandbox has no preinstalled stack), and
  harder for students to read.
- *Pure point-mass*: trivial to learn but "learning to walk" becomes fake.
Chosen: a **procedural locomotion model**. The walker has 4 joint targets
(hip/knee × L/R) driven by the policy; joint angles follow targets with a lag;
a *gait score* (hips antiphase × knees in-phase × swing amplitude) gates
forward thrust. Bad gaits → flailing in place; alternating gait → walking.
Plus a small "shamble" floor speed so there is no dead gradient early on.
A central-pattern-generator clock (sin/cos of phase) is included in the
observation by default (`cpg: true`) — policies learn to track it quickly; it
can be disabled for harder research runs. This gives us *visible* learning-to-
walk with zero heavy dependencies, and `aera/world/physics.py` is an interface
— a PyBullet adapter can slot in later without touching env/senses/viz.

**D3. Renderer: software 3D (painter's algorithm) in pygame, not raycasting,
not matplotlib, not WebGL.**
- *Raycaster (Wolfenstein-style)*: cheap but locks the camera to first-person.
- *matplotlib 3D*: slow, ugly, no game feel.
- *Three.js web viewer*: nice but splits the project into two codebases.
Chosen: low-res buffer (~426×240) rendered with flat-shaded polygons sorted
back-to-front, distance fog, chunky upscale → deliberate PS1/90s look.
Cameras: ORBIT (free arbitrary view), CHASE, TOP, FIRST-PERSON (the agent's
egocentric view). The renderer is deterministic and headless-capable
(`render_frame → np.ndarray`), so the same code path does live viewing,
offscreen recording (GIF via Pillow), and tests with `SDL_VIDEODRIVER=dummy`.

**D4. Built-in learner: self-contained numpy PPO (no torch required).**
Keeps the repo installable in seconds (`numpy + pygame + gymnasium`), and is
honest about being a *reference* trainer. Because the env is Gymnasium,
SB3/CleanRL can be swapped in on the server (torch was too heavy to install
here today; that's explicitly the server's job, see AGENTS.md § Bootstrap).

**D5. World model: top-down cell field + rect walls, meters as units.**
Terrain grid (1 m cells): GRASS / ROAD (fast) / MUD (slow) / LAVA (damage) /
PIT (impassable unless airborne). Walls = axis-aligned rects (collision =
circle-vs-AABB, axis-separated; raycasts = slab method). Entities: food
(energy + score), tools (boots), beacon (goal). Deterministic seeded
generation → reproducible worlds for fair experiments.

**D6. Senses: egocentric, inspectable.**
Per-ray channels [dist, wall, lava, food, tool, beacon] × N rays (walls
occlude; lava detected by ray-marching terrain), proprioception (speed,
heading, joints, phase, energy, health), 3×3 terrain patch one-hot, inventory
flags, scent vector to nearest food (reward-as-observation trick, documented),
beacon direction. The layout is exposed programmatically (`obs_layout()`) so
students/HUD can show *what the agent feels*.

**D7. Reward: transparent sub-rewards + curriculum.**
Every step returns `info["sub"] = {food, survive, novelty, progress, damage,
beacon, death}` — students see *why* the reward moved. Curriculum stages
(`configs`) ramp danger: first episodes are safe (learn locomotion), then
hunger, then lava, then tools/goal. Count-based novelty bonus per visited cell
drives exploration. Milestones ("skills") are detected from rolling stats:
Walking, Forager, Survivor, Explorer, ToolUser (crossed a pit), Beacon —
printed as ★ SKILL UNLOCKED and logged.

**D8. Multi-agent: deferred by design.** v1 = one agent per env; the world
API already takes a list of agents in the renderer, so parallel envs
(SyncVectorEnv / self-play later) is an extension, not a rewrite.

### Things to remember about this environment (for the server session)

- Debian box, Python 3.13.5, **no pip, no ensurepip** → bootstrap venv with
  `python3 -m venv --without-pip .venv && curl get-pip.py`. Exact commands in
  AGENTS.md.
- Only Pillow was preinstalled system-wide; network to PyPI works.
- I stopped mid-install per user request; nothing heavy was installed here.

### Open questions / next steps

1. Train on the server: rover (easy sanity) then walker; tune gait constants.
2. Optional: install `torch --index-url .../cpu` + `stable-baselines3` and add
   an SB3 training script as the "modern baseline" proof.
3. Second agent kind + two-agent coexistence (render both, shared world).
4. Tool *crafting* (combine items at a workbench) — hooks exist via
   `Tool.kind`, not implemented in v1.
5. Maybe: WebAppendix dashboard or tensorboard optional logger.

---

## 2026-10-02 — v0.1 implemented (initial code, no heavy jobs per user)

User stopped the installs mid-way ("we will continue on the server") and asked
for initial code + AGENTS.md + this log + a repo name. So today's session
delivered the **complete v0.1 codebase, untested at runtime** (33 modules
compile; pytest + training runs deferred to the server).

What exists now:
- `aera/` package exactly as designed: world (terrain/walls/raycast/collision),
  procedural locomotion (rover diff-drive, walker gait-gated thrust), egocentric
  senses (12 rays × 6 channels, proprio, 3×3 terrain, scent, beacon),
  Gymnasium-compatible `AeraEnv` with transparent sub-rewards + curriculum,
  numpy PPO (GAE/clip/Adam/obs-norm), trainer with JSONL metrics + ★ skill
  milestones + checkpoints, painter's-algorithm 3D renderer with 4 cameras,
  pygame viewer, headless GIF recorder, CLI (`python -m aera train|watch`).
- 3 configs: `field_small` (smoke/tests), `field_open` (canonical wake-up
  world), `field_curriculum` (40×40 walls/pits/tools).
- 6 test files covering determinism, collision/raycast, gait vs flail,
  pit-blocking-until-jump, gym contract, PPO value learning + save/load,
  headless rendering, trainer end-to-end smoke.

Bugs caught during self-review (worth remembering): event double-logging
(ep_events extended twice), "jumped" never reaching episode events (broke the
ToolUser milestone), spawn creating a throwaway World, shim-incompatible
`super().reset()`, `_rng` init order in TerrainGrid, missing `speed_sum`,
`cpg` flag ignored in obs, headless pygame needs `SDL_VIDEODRIVER=dummy`
fallback (renderer now auto-falls back), energy-starvation test outlived the
600-step truncation.

Server checklist is in AGENTS.md §7/§10: bootstrap venv (no ensurepip on this
image), `pytest -q`, rover 50k smoke on field_small, walker run on field_open
with `--watch`, optional SB3 proof.

Repo name candidates I weighed: `aera` (short, brandable, CLI-friendly —
my pick), `aera-rl` (if the base name is taken), `wakefield` (narrative:
agent wakes in a field), `primordial-arena` (evocative but long).

---

## 2026-10-02 — server session: v0.1 debugged, trained, baseline-proven

The server turned out to be a different box than the sandbox (Python 3.10.12,
40 cores, 187 GB RAM, still no pip). Bootstrapped the venv per the checklist.
**The "untested at runtime" caveat was honest: the first `pytest -q` failed
12 tests + 3 errors.** All fixes below are first-runtime discoveries, on top
of (not overlapping) the self-review list from the v0.1 entry.

Bugs found and fixed (each is a lesson):
- `config.py::_merge` merged only one level deep → JSON-loaded
  `world.terrain` stayed a plain dict and crashed World/env/trainer/renderer.
  Fix: recursive merge over nested dataclass fields. Lesson: config plumbing
  needs a test that loads every shipped JSON, not just defaults.
- PPO `g_logstd` gradient missed a `g[:, None]` broadcast → crash on first
  update. Also: PPO used global `np.random` (twice) — violates §2 determinism;
  now a threaded `self.rng`.
- `move_circle` accepted-or-rejected the whole displacement → a large step
  tunneled clean through the 1 m border wall (the border test caught it).
  Fix: sub-steps of ≤ r/2. Lesson: accept/reject collision is only safe when
  steps are smaller than the thinnest obstacle.
- `episode["mean_speed"]` was metres-per-step (∫v·dt / steps), not m/s —
  every speed milestone threshold was unreachable by exactly 10×. Caught by
  noticing the walker "hit the beacon" at 0.03 m/s, which is impossible.
  Diagnosing that also revealed the walker was already foraging well — the
  metric, not the agent, was broken.
- pygame-ce 2.5.8 **rejects `np.float32` scalars in `draw.polygon`** while
  accepting np.float64 and Python floats ("points must be number pairs").
  Walker joint math (np.float32) propagated into screen coords. Fixed once at
  the projection boundary with `float()` casts. Lesson: cast at the API
  boundary, not at each call site.
- `action_dim(self.config.agent)` passed the AgentCfg *object* where a kind
  string was expected → `2 if kind == "rover" else 6` returned 6 for every
  rover, so all rover runs trained a 6-dim policy head (training silently
  "worked" because the env ignores extra action dims!). Lesson: quiet shape
  mismatches need a shape assertion, not vigilance.

Training results (all reproducible: config + seed):
- **Walker 600k on field_open: 5/6 milestones** — Navigator@14, ToolUser@19,
  Survivor@31, Forager@44, **Walking@271** (rolling ≥0.7 m/s). Final ret ~41,
  speed ~0.75 m/s, 8-12 foods/episode, boots fetched deliberately from ep ~250.
  **No gait tuning needed** — D2's procedural gait works as designed. A pause
  at 254k + rerun reproduced the trainer output byte-for-byte → §2 determinism
  is real, not aspirational.
- **Rover 200k: peak then oscillation** — rolling ret 8.3 / 1.09 m/s around
  ep 120-140, then drifts back to ~4. The constant-lr, no-annealing reference
  PPO can't hold its peak; rover Walking (≥1.2 m/s) is an optimizer problem,
  not an env problem.
- **SB3 baseline proof: decisive.** Rover: ret 18.0 vs 2.5 at 30k (7×).
  Walker: **46.3 vs ~41 with half the steps**. D4's "reference trainer is
  honest but not serious" framing confirmed quantitatively.
- **Explorer milestone is miscalibrated**: per-episode exploration plateaus
  at ~0.12; even a perfect 0.75 m/s walker can visit at most ~0.17 of a 32×32
  world in 2000 steps. Options: re-scale the rule, make it speed-normalized
  (cells per metre), or cumulative across episodes. Decision pending — this
  is a semantic change, not a bug fix.

Tooling added along the way: `python -m aera sb3` (Monitor CSV + logger CSV +
50k-step checkpoints — added after realizing a paused `learn()` left NO
policy behind), `python -m aera plot <run_dir>` (episode + PPO-loss curves
from metrics.jsonl or SB3 CSVs; SB3's monitor `r` column is already the
episode total — don't accumulate it), `train --init` warm-start (weights +
obs-norm only; curriculum restarts — a true resume needs episode/RNG state in
the npz), README hero image.

Viewing from a laptop, settled on: GIFs + curve PNGs per run dir over an SSH
tunnel (`http.server --bind 127.0.0.1` + `ssh -L`); live `--watch` over
`ssh -X`. The retro render makes GIFs genuinely watchable, which quietly
became the primary review interface — the viewer is for demos, the GIFs are
for work.

Next: the Explorer decision, true trainer resume, two-agent coexistence
(D8), tool crafting, and maybe an SB3 walker on `field_curriculum` where the
walls force real navigation.

---

## 2026-10-02 — warm-start divergence: a NaN war story

Added `train --init` (warm-start weights + obs-norm from a checkpoint) and
immediately got burned by it: continuing the 600k walker for 400k steps at
the default lr **collapsed from ret ~47 to −4**, with `pi_loss/v_loss = nan`
from update 181 onward. The milestones all "re-fired" at episode 14 (the
rolling window filled while the loaded policy was still good), which makes
the crash look deceptively gentle in the logs — check the end of the curve,
not the beginning.

Root cause chain, worth remembering because each link is a classic:
1. Reset Adam moments + a converged sharp policy (loaded logstd ≈ small) +
   full lr 3e-4 → the first updates overshoot (`pi_loss +1.18` — a *positive*
   policy loss means the surrogate ratio exploded).
2. PPO ratio = exp(logp − logp_old): with a sharp Gaussian, small parameter
   shifts produce huge log-ratios; eventually `exp` overflows to `inf`.
3. The surrogate gradient becomes `inf`, and the global-norm clip scales it
   by `max_norm / inf = 0` → **inf × 0 = NaN** → Adam poisons every weight.
4. NaN weights → the walker "forgets" walking; episodes end by starvation.

Fixes (all in `rl/ppo_numpy.py` + `train --lr`):
- clamp the log-ratio to ±30 before `exp`;
- drop the whole update if any gradient is non-finite (NaN can't be
  un-poisoned, only refused);
- expose `--lr`; fine-tuning recipe is `--init <npz> --lr 1e-4`.
The deeper lesson: from-scratch hyperparameters are not fine-tuning
hyperparameters. The optimizer-state reset is the third state (after weights
and obs-norm) that a checkpoint format should carry if it wants true resumes.

---

## 2026-10-02 — Explorer milestone redesigned (decision: re-scale + speed-normalize)

The last open calibration question is settled. Old rule: rolling
`explored ≥ 0.5` of passable cells per episode — structurally impossible
here (a perfect 0.75 m/s walker covers ≤ ~0.17 of a 32×32 world in 2000
steps; measured plateau 0.12-0.13). New rule, per decision:

    explored_norm = explored / min(1, MAX_SPEED × episode_seconds / passable)
    Explorer ⟺ rolling explored_norm ≥ 0.15

i.e. "visit 15% of the cells your *body class* could reach at cruise speed"
(≈1 novel cell per metre of travel, `MAX_SPEED` from physics, not the
agent's own speed — self-referential normalization would reward camping).
Validated by replaying existing runs through the new tracker: the 600k
walker fires Explorer@14 (norm ≈ 0.33), the warm-start continuation fires
all 6/6, and the rover gets a fair exploration bar for the first time
(fires@34 on its 200k run). A camper scores ~0.05-0.1 and stays locked out.
Raw `explored` stays in the record; the plot prefers `explored_norm` when
present so old runs still graph. Lesson: when a metric is structurally
unreachable, it isn't measuring skill — re-derive what the number *should*
mean before moving the threshold.
