"""Watch the world: random policy, or a trained checkpoint.

    python -m aera.scripts.watch --config configs/field_open.json --agent walker
    python -m aera.scripts.watch --ckpt runs/walker_field_open_s0/policy_best.npz
    python -m aera.scripts.watch --ckpt ... --record demo.gif   # headless GIF

--record works on servers without a display (offscreen renderer + Pillow).
"""
from __future__ import annotations

import argparse

from ..config import Config
from ..env import AeraEnv


def main(argv=None):
    p = argparse.ArgumentParser(description="AERA viewer / recorder")
    p.add_argument("--config", default="configs/field_open.json")
    p.add_argument("--agent", choices=("walker", "rover"), default=None)
    p.add_argument("--ckpt", default=None,
                   help="path to policy_*.npz; omit for a random policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera", default="chase",
                   choices=("orbit", "chase", "top", "fp"))
    p.add_argument("--record", default=None, metavar="GIF",
                   help="render headless to a GIF instead of opening a window")
    p.add_argument("--frames", type=int, default=300)
    args = p.parse_args(argv)

    config = Config.load(args.config)
    if args.agent:
        config.agent.kind = args.agent
    env = AeraEnv(config)

    policy = None
    if args.ckpt:
        from ..training.trainer import load_policy
        policy = load_policy(args.ckpt, config)
        print(f"loaded policy: {args.ckpt}")

    if args.record:
        from ..viz.record import record_gif
        path = record_gif(env, policy, args.record, steps=args.frames,
                          mode=args.camera, seed=args.seed)
        print(f"wrote {path}")
        return path

    from ..viz.viewer import Viewer
    Viewer().play(env, policy, seed=args.seed)
    return None


if __name__ == "__main__":
    main()
