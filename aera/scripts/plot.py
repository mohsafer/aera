"""Plot training metrics from a run directory into PNGs.

    python -m aera plot runs/walker_field_open_s0

Reads `metrics.jsonl` (built-in trainer: per-episode and per-update records)
or, if absent, a Stable-Baselines3 `monitor.csv` (columns r/l/t). Writes
`curves_episodes.png` and `curves_updates.png` (the latter only if update
records exist) next to the metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def _load_jsonl(path: str) -> tuple[list[dict], list[dict]]:
    episodes: list[dict] = []
    updates: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "episode":
                episodes.append(rec)
            elif rec.get("type") == "update":
                updates.append(rec)
    return episodes, updates


def _load_monitor_csv(path: str) -> list[dict]:
    """SB3 Monitor format: comment lines start with '#', header 'r,l,t',
    one row per episode where r IS the episode return total."""
    episodes: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if row and not row[0].startswith("#")]
    for i, row in enumerate(rows):
        try:
            r, l = float(row[0]), float(row[1])
        except (ValueError, IndexError):
            continue   # header or malformed row
        episodes.append({"episode": i, "ret": r, "steps": int(l)})
    return episodes


def _load_progress_csv(path: str) -> list[dict]:
    """SB3 logger CSV (logs/progress.csv): one row per log cadence with
    columns like train/loss, train/value_loss, rollout/ep_rew_mean."""
    with open(path, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _rolling(xs: list[float], k: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(xs):
        acc += v
        if i >= k:
            acc -= xs[i - k]
        out.append(acc / min(i + 1, k))
    return out


def _panel(ax, ys, title, ylabel, xlabel="episode", rolling=15):
    import numpy as np
    xs = list(range(len(ys)))
    ax.plot(xs, ys, lw=0.8, alpha=0.45, color="#c45448")
    if rolling > 1 and len(ys) >= rolling // 2:
        ax.plot(xs, _rolling([y for y in ys], rolling), lw=1.8,
                color="#20242c", label=f"rolling {rolling}")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    if all(np.isfinite(ys)):
        lo, hi = min(ys), max(ys)
        if hi > lo:
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))


def plot_run(run_dir: str, window: int = 15) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jl = os.path.join(run_dir, "metrics.jsonl")
    mon = os.path.join(run_dir, "monitor.csv")
    if os.path.exists(jl):
        episodes, updates = _load_jsonl(jl)
    elif os.path.exists(mon):
        episodes, updates = _load_monitor_csv(mon), []
    else:
        raise SystemExit(f"no metrics.jsonl or monitor.csv in {run_dir}")
    if not episodes:
        raise SystemExit(f"no episode records yet in {run_dir}")

    written: list[str] = []

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    fig.suptitle(f"{os.path.basename(run_dir)} — {len(episodes)} episodes", fontsize=12)
    panels = [
        ("episode return", [e.get("ret") for e in episodes]),
        ("food per episode", [e.get("foods") for e in episodes]),
        ("mean speed (m/s)", [e.get("mean_speed") for e in episodes]),
        ("episode length", [e.get("steps") for e in episodes]),
        ("exploration (fraction of passable cells)", [e.get("explored") for e in episodes]),
        ("tool held (boots)",
         [(float(bool(e["inventory"])) if "inventory" in e else None) for e in episodes]),
    ]
    for ax, (title, ys) in zip(axes.flat, panels):
        if any(y is None for y in ys):     # stat not recorded (e.g. SB3 monitor)
            ax.axis("off")
            continue
        _panel(ax, ys, title, title.split("(")[0].strip(), rolling=window)
    fig.tight_layout()
    out_ep = os.path.join(run_dir, "curves_episodes.png")
    fig.savefig(out_ep, dpi=130)
    plt.close(fig)
    written.append(out_ep)

    if updates:
        _plot_update_curves(run_dir, [(u.get("step", i * 2048), u) for i, u in enumerate(updates)],
                            keys=[("pi_loss", "policy loss"), ("v_loss", "value loss"),
                                  ("entropy", "policy entropy"), ("clipfrac", "clip fraction")],
                            x_label="env steps")
        written.append(os.path.join(run_dir, "curves_updates.png"))
    else:
        prog_path = os.path.join(run_dir, "logs", "progress.csv")
        if os.path.exists(prog_path):
            rows = _load_progress_csv(prog_path)
            x_key = "time/total_timesteps" if rows and "time/total_timesteps" in rows[0] else None
            series = [(k, t) for k, t in
                      (("train/loss", "total loss"), ("train/value_loss", "value loss"),
                       ("train/policy_gradient_loss", "policy gradient loss"),
                       ("train/entropy_loss", "entropy loss"), ("train/approx_kl", "approx KL"),
                       ("train/clip_fraction", "clip fraction"))
                      if rows and k in rows[0]]
            if series:
                _plot_update_curves(run_dir,
                                    [((float(r[x_key]) if x_key else i), r) for i, r in enumerate(rows)],
                                    keys=series, x_label="env steps")
                written.append(os.path.join(run_dir, "curves_updates.png"))

    return written


def _plot_update_curves(run_dir: str, rows, keys, x_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4))
    fig.suptitle("PPO updates", fontsize=12)
    if n == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, keys):
        xs = [x for x, _ in rows]
        ys = [r.get(key) for _, r in rows]
        pts = [(x, float(y)) for x, y in zip(xs, ys) if y not in (None, "")]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.1, color="#20242c")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "curves_updates.png"), dpi=130)
    plt.close(fig)


def main(argv=None) -> list[str]:
    p = argparse.ArgumentParser(description="Plot AERA training metrics")
    p.add_argument("run_dir", help="run directory containing metrics.jsonl (or monitor.csv)")
    p.add_argument("--window", type=int, default=15, help="rolling-mean window")
    args = p.parse_args(argv)
    paths = plot_run(args.run_dir, window=args.window)
    for path in paths:
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    main()
