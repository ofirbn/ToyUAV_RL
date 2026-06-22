"""
M4A — train a PPO expert specialized for FlightMode.APPROACH only.

Reuses train.py's model-building/checkpoint machinery but writes to an
isolated save path (models/experts/approach.zip) instead of the default
training path (models/latest.zip), so the default `python main.py` train
flow is completely unaffected.

The env is locked to APPROACH every reset via curriculum_phase="approach"
(see sim/curriculum.py: get_locked_mode) — no env/reward code changes.

Run via:  python tools/train_expert_approach.py --timesteps 300000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.fixed_wing_env import FixedWingEnv
from train import _build_model, LogCallback

N_ENVS   = 4
OUT_PATH = "models/experts/approach"   # PPO.save()/.load() path, no .zip suffix


def main():
    parser = argparse.ArgumentParser(
        description="Train a single-mode PPO expert for FlightMode.APPROACH")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--out", type=str, default=OUT_PATH,
                         help="Save path without .zip extension")
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--force-new", action="store_true",
                         help="Ignore any existing checkpoint at --out and start fresh")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env = make_vec_env(
        FixedWingEnv,
        n_envs      = N_ENVS,
        vec_env_cls = DummyVecEnv,
        env_kwargs  = {
            "training_mode":    True,
            "curriculum_phase": "approach",   # hard-locks FlightMode.APPROACH each reset
        },
    )

    cfg   = {"force_new": "true" if args.force_new else "false"}
    model = _build_model(env, cfg, args.out)

    print(f"\n{'='*55}")
    print("  ToyUAV RL — APPROACH Expert Training (M4A)")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Envs      : {N_ENVS}  (DummyVecEnv)")
    print(f"  Save path : {args.out}.zip")
    print(f"{'='*55}\n")

    callback = LogCallback(
        best_reward_path  = args.out + "_best_reward",
        best_success_path = args.out + "_best_success",
        save_every        = args.save_every,
    )

    model.learn(
        total_timesteps     = args.timesteps,
        reset_num_timesteps = False,
        callback            = callback,
    )

    model.save(args.out)
    print(f"\n[TRAIN] Saved -> {args.out}.zip")
    print("[TRAIN] Done.")


if __name__ == "__main__":
    main()
