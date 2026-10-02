"""End-to-end smoke: the trainer runs, logs, and checkpoints (small run)."""
import json
import os

from aera.config import Config
from aera.training.trainer import Trainer, load_policy


def test_trainer_smoke(tmp_path):
    cfg = Config.load("configs/field_small.json")
    cfg.sim.max_steps = 300
    out = str(tmp_path / "run")
    Trainer(cfg, out, total_steps=1200, rollout=600, seed=0,
            save_every_updates=1).train()

    metrics = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    types = {m["type"] for m in metrics}
    assert {"update", "episode"} <= types
    assert os.path.exists(os.path.join(out, "policy_final.npz"))
    assert os.path.exists(os.path.join(out, "config_used.json"))

    # checkpoint loads into a fresh policy with matching spaces
    policy = load_policy(os.path.join(out, "policy_final.npz"), cfg)
    env_out = None
    from aera.env import AeraEnv
    env = AeraEnv(cfg)
    obs, _ = env.reset(seed=0)
    a, logp, v = policy.act(obs, deterministic=True)
    assert a.shape == env.action_space.shape and np.isfinite(v)
