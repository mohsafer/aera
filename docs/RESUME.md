# AERA — project status & transfer guide (2026-10-04)

Snapshot of everything done on the training server, and exactly how to
continue on a new machine. Read this first, then `README.md` (pitch) and
`AGENTS.md` (technical contract). Hands-on manual: `docs/USER_GUIDE.md`.

## Current state (one paragraph)

v0.3, **45/45 tests green**, repo clean at commit `b25da49`. The full loop
works end-to-end: seeded deterministic world → numpy PPO + SB3 → live browser
streaming (video + event feed + live PPO curves) → plotting with mind-audit
charts. Walker fully trained on the clean world (600k + 400k continuation,
ret ~44, 0.75 m/s, 6/6 milestones under the speed-normalized Explorer rule);
walker mid-training on the ALIEN world (anomalies + predators), paused
resumably at step 137k/300k. Minds layer implemented (RuleMind deterministic;
LLMMind for any OpenAI-compatible endpoint). Demo GIFs, curve sheets, and
checkpoint files for every run are in `runs/` (14 MB — included in the
transfer archive, gitignored in the repo).

## What's in the transfer archive

- `aera.bundle` — the full git history (all commits incl. unpushed). Clone:
  `git clone aera.bundle aera && cd aera && git checkout main`
- `runs/` — every training artifact: checkpoints (`policy_best.npz`,
  `policy_final.npz`, `trainer_state.npz` for resume), `metrics.jsonl`,
  curve PNGs, demo GIFs, SB3 zips.
- `README-TRANSFER.txt` — this file's short version.

## New-server setup (no pip/ensurepip images included)

```bash
git clone aera.bundle aera && cd aera
python3 -m venv --without-pip .venv      # if pip is missing
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q            # expect 45 passed
tar xf aera-runs.tar.gz                  # → runs/ (do this inside the repo)
```

Optional extras: `pip install torch --index-url https://download.pytorch.org/whl/cpu
stable-baselines3` (baseline runs), `matplotlib` is in requirements (plots).

## Continue the paused alien-world run (137k/300k)

```bash
.venv/bin/python -m aera train --config configs/field_anomalies.json \
    --steps 300000 --resume runs/walker_field_anomalies_s0 --stream 8888
```

`trainer_state.npz` carries policy + obs-norm + Adam moments + curriculum
episode counter + RNG states: it continues at the exact episode boundary
(last save: step ~137k; up to 20 updates since the previous save are lost —
lower `--save-every` via Trainer for finer checkpoints). Watch it live via
`ssh -L 8888:localhost:8888` → http://localhost:8888 (video + curves + feed).

## Run inventory (runs/)

| run | state | note |
|---|---|---|
| `walker_field_anomalies_s0` | **COMPLETE 300k** | alien frontier: ret −2 vs +34 clean, speed 0.46 vs 0.73 m/s, foods 0.5 vs 8.2 — anomalies make the task ~2× harder; Survivor@159; comparison chart `comparison_alien_vs_clean.png` + `demo.gif` in the run dir |
| `walker_field_open_s0` | complete 600k | ret ~41, 0.75 m/s, 6/6 milestones (Walking@271) |
| `walker_field_open_s0_warm400k` | complete (+400k) | ret ~44 via foraging; speed plateaued |
| `walker_field_open_s0_warm400k_diverged` | evidence | NaN war story (log.md) |
| `rover_field_small_s0` / `_200k_s0` | complete | peak 1.09 m/s; reference-PPO oscillation |
| `sb3_rover_field_small_s0` | complete 30k | ret 18 vs built-in 2.5 |
| `sb3_walker_field_open_s0` | complete 300k | ret 46.3 vs built-in ~41 |
| `*_prefix1/_oldunits/_nologs/_paused254k` | historical | pre-fix evidence, kept for the record |

## Mind-audit: the open experiment

`python -m aera plot runs/<run>` renders `curves_mind.png` (goals over steps
+ counts) when thought records exist. The honest test that reasoning helps:
same seed, `--mind rule` vs `--mind none`, compare returns; then LLMMind
(export `AERA_LLM_URL`, `AERA_LLM_MODEL`) once an endpoint is chosen.

## The headline comparison

`runs/walker_field_anomalies_s0/comparison_alien_vs_clean.png`: the same
walker architecture on the clean field (600k) reaches ret ~40 at 0.75 m/s;
on the alien frontier (300k, all anomalies + predator) it hovers at ~0 with
0.46 m/s — it survives and navigates but foraging stays hard. Natural next
experiments: train the alien world with `--mind rule` (goal-conditioned) and
with crafting emphasis, or resume the alien run to 600k for parity.

## Housekeeping

- `origin` = https://github.com/mohsafer/aera — server commits through
  `7c301c4` are pushed; everything after (`f08b843`…`b25da49`) is local-only
  but INCLUDED in the bundle. Optionally `git push --all` after transfer.
- Ports used on the old server: 8000 (runs/ browsing), 8888/8889 (live
  streams), 9000 (user service) — all loopback-bound, nothing exposed.
- Python on the old box: 3.10.12. Code targets ≥ 3.10.
