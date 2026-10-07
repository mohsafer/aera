"""Train an agent: python -m aera.scripts.train --config configs/field_open.json

Add --watch to watch the world live while the policy trains (same env the
trainer is learning from), or run headless on a server and watch later with
aera.scripts.watch.
"""
from __future__ import annotations

import argparse
import os

from ..config import Config
from ..training.trainer import Trainer, make_mind


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description="AERA trainer (built-in PPO)")
    p.add_argument("--config", default="configs/field_open.json",
                   help="world/agent/reward JSON (see configs/)")
    p.add_argument("--agent", choices=("walker", "rover"), default=None,
                   help="override the agent kind from the config")
    p.add_argument("--steps", type=int, default=200_000,
                   help="total env steps")
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--out", default=None, help="run directory (default runs/<name>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--watch", action="store_true",
                   help="open the live 3D viewer during training")
    p.add_argument("--stream", type=int, default=None, metavar="PORT",
                   help="serve a live browser view (MJPEG) on 127.0.0.1:PORT "
                        "during training — watch via your SSH tunnel")
    p.add_argument("--fps", type=int, default=30, help="viewer frame cap")
    p.add_argument("--init", default=None, metavar="NPZ",
                   help="warm-start from a policy_*.npz (weights + obs-norm only; "
                        "the curriculum restarts at episode 0)")
    p.add_argument("--lr", type=float, default=None,
                   help="PPO learning rate (default 3e-4; use e.g. 1e-4 when "
                        "fine-tuning with --init)")
    p.add_argument("--resume", default=None, metavar="RUNDIR",
                   help="continue from a run directory's trainer_state.npz "
                        "(policy + optimizer + curriculum episode + RNG state)")
    p.add_argument("--mind", choices=("none", "rule", "llm"), default="none",
                   help="attach a Mind: 'rule' = deterministic heuristics, "
                        "'llm' = OpenAI-compatible endpoint (AERA_LLM_URL)")
    p.add_argument("--mind-interval", type=int, default=25,
                   help="env steps between thoughts (25 = every 2.5 s)")
    p.add_argument("--mind-start", type=int, default=0, metavar="N",
                   help="mind curriculum: stay silent (constant explore goal) "
                        "until env step N, then start thinking")
    p.add_argument("--mind-endpoint", default=None, metavar="URL",
                   help="chat completions URL for --mind llm "
                        "(default env AERA_LLM_URL)")
    args = p.parse_args(argv)

    config = Config.load(args.config)
    if args.agent:
        config.agent.kind = args.agent
    if args.mind != "none":
        config.agent.mind_goal = True     # obs gains the goal one-hot slot

    name = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join("runs", f"{config.agent.kind}_{name}_s{args.seed}")
    os.makedirs(out, exist_ok=True)

    viewer = None
    if args.watch:
        from ..viz.viewer import Viewer
        viewer = Viewer(fps=args.fps)
    elif args.stream:
        from ..viz.streamer import StreamViewer
        viewer = StreamViewer(port=args.stream)

    print(f"AERA training → {out}  (agent={config.agent.kind}, "
          f"world={config.world.name}, steps={args.steps}, mind={args.mind})")
    cache = (os.path.join(out, "mind_cache.json") if args.mind == "llm" else None)
    Trainer(config, out, total_steps=args.steps, rollout=args.rollout,
            seed=args.seed, viewer=viewer, init_from=args.init,
            lr=args.lr, resume_from=args.resume,
            mind=make_mind(args.mind, args.mind_endpoint, cache_path=cache),
            mind_interval=args.mind_interval,
            mind_start=args.mind_start).train()
    return out


if __name__ == "__main__":
    main()
