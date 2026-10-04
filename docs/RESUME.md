# Server session state — 2026-10-02 (v0.2 complete)

All work committed through `09d1f83` (local; `origin/main` is behind — push
when ready). Tests: **36/36**. Full record below; hands-on manual:
`docs/USER_GUIDE.md`.

## Headline results (all under `runs/`)

| run | budget | result |
|---|---|---|
| `walker_field_open_s0` | **600k steps, complete** | ret −3 → ~41 (last-10 mean 41.1); speed 0.3 → **0.75 m/s**; food 0 → 8-12/ep; **milestones 5/6**: Navigator@14, ToolUser@19, Survivor@31, Forager@44, **Walking@271**. No gait tuning needed — walking emerged inside budget. |
| `rover_field_small_200k_s0` | 200k steps | peak rolling ret 8.3 / speed **1.09 m/s** (ep ~120-140), then oscillates down to ~4-5 — the constant-lr reference PPO can't hold its peak; rover Walking (≥1.2 m/s rolling) needs a stronger optimizer, not env changes |
| `rover_field_small_s0` | 50k steps | ret 1.1 → ~3.5, drives ~0.7-1.1 m/s; no milestone |
| `sb3_rover_field_small_s0` | 30k steps | last-10 mean ret **18.0** vs built-in PPO **2.5** at the same budget |
| `sb3_walker_field_open_s0` | 300k steps (complete) | last-10 mean ret **46.3** — beats the built-in walker's ~41 with **half** the steps; checkpoint `sb3_policy.zip` saved |
| `walker_field_open_s0_warm400k` | +400k (≈1M total, complete) | warm-start recipe validated (`--init --lr 1e-4`, no NaN after the ratio-overflow fix): ret 34 → **44.2** via foraging consistency (+1 food/ep); speed flat at ~0.71-0.75 m/s — the gait has plateaued at partial coordination; exploration still ~0.13 |
| `walker_field_open_s0_warm400k_diverged` | (first attempt) | NaN collapse at update 181 — kept as the log.md war-story evidence; root cause + fix in `rl/ppo_numpy.py` |

- The paused first attempt is kept as `runs/walker_field_open_s0_paused254k`
  (deterministic prefix of the final run; the rerun reproduced it exactly —
  verified byte-identical trainer output through update 124).
- Plots: `curves_episodes.png` + `curves_updates.png` per run; replays in
  `demo.gif`. Walker exploration plateaus at ~0.12 — the **Explorer milestone
  (≥0.5) is unreachable** at this world size/episode budget; either shrink the
  threshold (≈0.15), raise `max_steps`, or strengthen `w_novelty` (design call,
  not a bug).
- README hero image: `docs/screenshot.png` (rover | walker frames from the GIFs).

## Code changed (uncommitted working tree on `main`, base `beee896`)

Fixes: `config.py::_merge` recursive nested-dataclass merge; PPO `g_logstd`
broadcast + threaded RNG (no global np.random); `move_circle` sub-stepping
(no wall tunneling); `mean_speed` now true m/s; renderer casts screen points
to plain floats (pygame-ce rejects np.float32); `action_dim(kind)` call sites;
viewer chart fed during `--watch`. Tests: 3 expectation fixes + missing import.

Features: `python -m aera sb3` (Monitor CSV + logger CSV), `python -m aera
plot <run_dir>`, README hero + artifacts section, `.gitignore` (runs/, .venv/).

## Suggested next steps

1. `git push` — 6 local commits ahead of `origin/main` (user's earlier
   commits were pushed; the session's are not).
2. Train on the stress world: `python -m aera train --config
   configs/field_anomalies.json --steps 600000` and compare curves against
   `runs/walker_field_open_s0` (robustness evidence for the log).
3. Longer-term: two-agent coexistence, tool crafting.

## Viewing from a laptop

`python -m http.server 8000 --bind 127.0.0.1` inside `runs/` (left running)
+ `ssh -L 8000:localhost:8000 <user>@node0.quickhttpnode15.cloudfaas-pg0.wisc.cloudlab.us`
→ http://localhost:8000. Live viewer: `ssh -X` + `python -m aera train --watch`.
