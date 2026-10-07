"""Compare multiple runs side by side — the mind-audit A/B tool.

    python -m aera compare runs/a runs/b [runs/c ...]

Renders a comparison PNG (episode return, food, speed — rolling means) next
to the runs' metrics, and prints a last-N-episode stats table. Use it for
same-seed A/Bs (mind vs no-mind, clean vs alien world).
"""
from __future__ import annotations

import argparse
import json
import os

COLORS = ("#2f7d32", "#c45448", "#3b6ea5", "#f0b446", "#7b4ea3")


def _load_episodes(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no metrics.jsonl in {run_dir}")
    eps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "episode":
                eps.append(rec)
    if not eps:
        raise SystemExit(f"no episode records in {run_dir}")
    return eps


def _rolling(xs: list[float], k: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def compare(run_dirs: list[str], window: int = 15, tail: int = 40) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {d: _load_episodes(d) for d in run_dirs}
    labels = [os.path.basename(d.rstrip("/")) for d in run_dirs]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    fig.suptitle("run comparison (rolling %d)" % window, fontsize=12)
    panels = (("ret", "episode return"), ("foods", "food per episode"),
              ("mean_speed", "mean speed (m/s)"))
    for ax, (key, title) in zip(axes, panels):
        for color, label, (run, eps) in zip(COLORS, labels, runs.items()):
            ys = [e.get(key) for e in eps]
            if any(y is None for y in ys):
                continue
            ax.plot(range(len(ys)), ys, lw=0.7, alpha=0.3, color=color)
            ax.plot(range(len(ys)), _rolling(ys, window), lw=1.9,
                    color=color, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("episode", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(run_dirs[0].rstrip("/")) or ".",
                           f"comparison_{'_vs_'.join(l.replace(' ', '-') for l in labels)}.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    print(f"{'run':<44}{'eps':>5}{'ret':>9}{'foods':>8}{'m/s':>7}{'steps':>8}")
    for label, eps in zip(labels, runs.values()):
        t = eps[-tail:]
        print(f"{label:<44}{len(eps):>5}"
              f"{sum(e['ret'] for e in t)/len(t):>9.1f}"
              f"{sum(e.get('foods', 0) for e in t)/len(t):>8.2f}"
              f"{sum(e.get('mean_speed', 0) for e in t)/len(t):>7.2f}"
              f"{sorted(e['steps'] for e in t)[len(t)//2]:>8}")
    return out_png


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description="Compare AERA runs side by side")
    p.add_argument("runs", nargs="+", help="run directories to compare")
    p.add_argument("--window", type=int, default=15, help="rolling-mean window")
    p.add_argument("--tail", type=int, default=40,
                   help="episodes averaged in the stats table")
    args = p.parse_args(argv)
    if len(args.runs) < 2:
        raise SystemExit("give at least two run directories")
    out = compare(args.runs, window=args.window, tail=args.tail)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
