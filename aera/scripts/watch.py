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
    p.add_argument("--stream", type=int, default=None, metavar="PORT",
                   help="serve a live browser view (MJPEG) on 127.0.0.1:PORT "
                        "instead of opening a window")
    p.add_argument("--mind", choices=("none", "rule", "llm"), default="none",
                   help="attach a Mind during playback (thoughts hit the feed)")
    p.add_argument("--mind-interval", type=int, default=25)
    p.add_argument("--mind-endpoint", default=None, metavar="URL")
    p.add_argument("--frames", type=int, default=300)
    args = p.parse_args(argv)

    config = Config.load(args.config)
    if args.agent:
        config.agent.kind = args.agent
    if args.mind != "none":
        config.agent.mind_goal = True
    env = AeraEnv(config)
    from ..minds.base import build_summary
    from ..training.trainer import make_mind
    mind = make_mind(args.mind, args.mind_endpoint)

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

    if args.stream:
        from ..viz.streamer import StreamViewer
        print(f"live view → http://localhost:{args.stream}/ "
              f"(via ssh -L {args.stream}:localhost:{args.stream} user@host)")
        StreamViewer(port=args.stream).play(env, policy, seed=args.seed,
                                            mind=mind,
                                            mind_interval=args.mind_interval)
        return None

    from ..viz.viewer import Viewer
    Viewer().play(env, policy, seed=args.seed, mind=mind,
                  mind_interval=args.mind_interval)
    return None


if __name__ == "__main__":
    main()
