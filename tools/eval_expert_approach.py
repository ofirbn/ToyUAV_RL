"""
M4A — evaluate the APPROACH expert checkpoint (models/experts/approach.zip)
in isolation, without pygame/visualize.

Runs N deterministic episodes with the env locked to FlightMode.APPROACH
(curriculum_phase="approach") and reports crash/success/stall rates.

Run via:  python tools/eval_expert_approach.py --episodes 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.fixed_wing_env import FixedWingEnv

MODEL_PATH = "models/experts/approach"   # PPO.load() path, no .zip suffix


def main():
    parser = argparse.ArgumentParser(description="Evaluate the APPROACH expert")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                         help="Checkpoint path without .zip extension")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    zip_path = args.model if args.model.endswith(".zip") else args.model + ".zip"
    if not os.path.exists(zip_path):
        print(f"[EVAL] No checkpoint at {zip_path}. Train it first with "
              f"tools/train_expert_approach.py.")
        sys.exit(1)

    model = PPO.load(args.model)
    env   = FixedWingEnv(training_mode=True, curriculum_phase="approach")

    rewards, crashes, successes, stalls = [], 0, 0, 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        crashed = info.get("crashed", False)
        stalled = info.get("stalled_this_ep", False)
        crashes   += int(crashed)
        successes += int(not crashed)
        stalls    += int(stalled)
        rewards.append(ep_reward)

        print(f"  ep {ep+1:>3}/{args.episodes}  reward={ep_reward:+8.2f}  "
              f"{'CRASH' if crashed else 'ok'}{'  STALL' if stalled else ''}")

    n = args.episodes
    print(f"\n{'='*55}")
    print(f"  APPROACH expert eval — {n} episodes")
    print(f"  mean_reward  : {np.mean(rewards):+.2f}")
    print(f"  success_rate : {successes / n:.0%}")
    print(f"  crash_rate   : {crashes / n:.0%}")
    print(f"  stall_rate   : {stalls / n:.0%}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
