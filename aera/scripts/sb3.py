"""Optional modern-baseline proof: Stable-Baselines3 PPO on AeraEnv.

Requires the optional stack:  pip install torch --index-url https://download.pytorch.org/whl/cpu stable-baselines3
(the built-in trainer stays torch-free; this script is a standalone extra).

    python -m aera sb3 --config configs/field_small.json --steps 100000
"""
from __future__ import annotations

import argparse
import os


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description="Stable-Baselines3 PPO baseline")
    p.add_argument("--config", default="configs/field_small.json")
    p.add_argument("--agent", choices=("walker", "rover"), default=None)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--out", default=None, help="run directory (default runs/sb3_<name>)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        from stable_baselines3 import PPO
    except ImportError as e:  # pragma: no cover - optional extra
        raise SystemExit(
            "stable-baselines3 is not installed. Run: pip install "
            "torch --index-url https://download.pytorch.org/whl/cpu stable-baselines3"
        ) from e

    from ..config import Config
    from ..env import AeraEnv

    config = Config.load(args.config)
    if args.agent:
        config.agent.kind = args.agent
    name = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join("runs", f"sb3_{config.agent.kind}_{name}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    config.save(os.path.join(out, "config_used.json"))

    env = AeraEnv(config)
    # Monitor CSV (r/l/t per episode) → plottable with `python -m aera plot <out>`
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(env, os.path.join(out, "monitor.csv"))
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log=None)
    # CSV logger → logs/progress.csv (losses per update) for `python -m aera plot`
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(os.path.join(out, "logs"), ["csv", "stdout"]))
    # periodic checkpoints so an interrupted learn() still leaves a policy behind
    from stable_baselines3.common.callbacks import CheckpointCallback
    ckpt = CheckpointCallback(save_freq=50_000, save_path=out, save_replay_buffer=False,
                              name_prefix="sb3_ckpt")
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=ckpt)
    model.save(os.path.join(out, "sb3_policy"))
    print(f"saved SB3 policy → {os.path.join(out, 'sb3_policy.zip')}")
    return out


if __name__ == "__main__":
    main()
