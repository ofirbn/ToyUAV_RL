"""
Record teacher autopilot demonstrations for behavior cloning.

Usage:
    python ml/record_teacher_dataset.py --phase stabilize --episodes 500
    python ml/record_teacher_dataset.py --phase waypoint  --episodes 1000 --seed 42
    python ml/record_teacher_dataset.py --phase loiter    --episodes 800

Supported phases: stabilize, altitude_hold, heading_hold, waypoint, loiter,
                  approach, recovery

Output: results/demos/<phase>_demo_NNN.npz
    Arrays: obs (N,31), actions (N,4), phases (N,), rewards (N,), dones (N,)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np

from envs.fixed_wing_env import FixedWingEnv
from controllers.teacher_autopilot import TeacherAutopilot


VALID_PHASES = [
    "stabilize", "altitude_hold", "heading_hold",
    "waypoint", "loiter", "approach", "recovery",
]


def record(phase: str, n_episodes: int, seed: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    env     = FixedWingEnv(training_mode=True, curriculum_phase=phase)
    teacher = TeacherAutopilot()

    all_obs     = []
    all_actions = []
    all_phases  = []
    all_rewards = []
    all_dones   = []

    ep         = 0
    total_steps = 0
    total_crashes = 0
    total_success = 0

    while ep < n_episodes:
        rng_seed = (seed + ep) if seed >= 0 else None
        obs, _   = env.reset(seed=rng_seed)
        done = truncated = False

        while not (done or truncated):
            action = teacher.act(obs, phase, target=env._target)
            next_obs, reward, done, truncated, info = env.step(action)

            all_obs.append(obs.copy())
            all_actions.append(action.copy())
            all_phases.append(int(env._mode))
            all_rewards.append(float(reward))
            all_dones.append(bool(done or truncated))

            obs = next_obs
            total_steps += 1

        if info.get("crashed", False):
            total_crashes += 1
        if info.get("episode_success", False):
            total_success += 1

        ep += 1
        if ep % 50 == 0 or ep == n_episodes:
            crash_r   = total_crashes / ep
            success_r = total_success / ep
            print(f"[RECORD] {ep:4d}/{n_episodes}  steps={total_steps:,}  "
                  f"crash={crash_r:.0%}  success={success_r:.0%}")

    env.close()

    # Find next free filename
    idx = 1
    while True:
        path = os.path.join(out_dir, f"{phase}_demo_{idx:03d}.npz")
        if not os.path.exists(path):
            break
        idx += 1

    np.savez(
        path,
        obs     = np.array(all_obs,     dtype=np.float32),
        actions = np.array(all_actions, dtype=np.float32),
        phases  = np.array(all_phases,  dtype=np.int32),
        rewards = np.array(all_rewards, dtype=np.float32),
        dones   = np.array(all_dones,   dtype=bool),
    )

    print(f"\n[RECORD] Saved {total_steps:,} steps -> {path}")
    print(f"[RECORD] crash={total_crashes/n_episodes:.0%}  "
          f"success={total_success/n_episodes:.0%}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Record teacher demos for BC")
    parser.add_argument("--phase",    required=True, choices=VALID_PHASES,
                        help="Flight phase to record")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to record")
    parser.add_argument("--seed",     type=int, default=0,
                        help="RNG seed base (-1 = random)")
    parser.add_argument("--out-dir",  default="results/demos",
                        help="Output directory for .npz files")
    args = parser.parse_args()

    record(args.phase, args.episodes, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
