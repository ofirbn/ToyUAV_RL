"""
READ-ONLY analysis: teacher -> BC -> PPO pipeline for the STABILIZE expert.

This script does NOT modify the pipeline, env, rewards, or any training code. It
only:
  - evaluates the TeacherAutopilot, the BC checkpoint, and PPO snapshots, and
  - trains PPO *from the BC checkpoint* exactly as the real pipeline does
    (PPO.load(models/bc/stabilize_bc, env=stabilize_env) — same call
    train._build_model makes for a BC warm-start), saving snapshots to a temp
    dir so models/experts/ is untouched.

Metrics per stage (STABILIZE phase, deterministic actions, shared seeds):
  success_rate, crash_rate, mean_reward, attitude_err (mean |roll|+|pitch|, deg),
  control_smoothness (mean ||Δaction|| per step; lower = smoother),
  recovery_capability (success/crash on the RECOVERY phase = large-upset start).
"""

import argparse
import math
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.fixed_wing_env import FixedWingEnv
from sim.flight_modes import FlightMode
from controllers.teacher_autopilot import TeacherAutopilot

N_ENVS  = 4                       # match the real expert pipeline
BC_PATH = "models/bc/stabilize_bc"


# ── policies ──────────────────────────────────────────────────────────────────

def teacher_policy():
    t = TeacherAutopilot()
    return lambda obs: t.act(obs, FlightMode.STABILIZE)


def model_policy(path):
    m = PPO.load(path)
    return lambda obs: m.predict(obs, deterministic=True)[0]


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(policy_fn, phase, episodes, seed):
    env = FixedWingEnv(training_mode=True, curriculum_phase=phase)
    rewards, succ, crash, attitude, smooth, lengths = [], [], [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = trunc = False
        R = 0.0
        prev_a = None
        att, ad = [], []

        while not (done or trunc):
            a = np.asarray(policy_fn(obs), dtype=np.float32).reshape(-1)
            obs, r, done, trunc, info = env.step(a)
            R += r
            s = env._state
            att.append(abs(math.degrees(s.roll)) + abs(math.degrees(s.pitch)))
            if prev_a is not None:
                ad.append(float(np.linalg.norm(a - prev_a)))
            prev_a = a

        crashed = info.get("crashed", False)
        success = info.get("episode_success", not crashed)
        rewards.append(R)
        succ.append(int(success))
        crash.append(int(crashed))
        attitude.append(float(np.mean(att)) if att else 0.0)
        smooth.append(float(np.mean(ad)) if ad else 0.0)
        lengths.append(env.steps)

    env.close()
    return {
        "n":        episodes,
        "success":  100.0 * float(np.mean(succ)),
        "crash":    100.0 * float(np.mean(crash)),
        "reward":   float(np.mean(rewards)),
        "attitude": float(np.mean(attitude)),
        "smooth":   float(np.mean(smooth)),
        "length":   float(np.mean(lengths)),
    }


# ── PPO-from-BC with snapshots (faithful to the real warm-start) ──────────────

def train_snapshots(milestones, outdir, log_every=True):
    env = make_vec_env(
        FixedWingEnv, n_envs=N_ENVS, vec_env_cls=DummyVecEnv,
        env_kwargs={"training_mode": True, "curriculum_phase": "stabilize"},
    )
    model = PPO.load(BC_PATH, env=env)     # identical to train._build_model BC warm-start
    snaps = {}

    p0 = os.path.join(outdir, "ppo_0")
    model.save(p0)
    snaps[0] = (p0, int(model.num_timesteps))
    print(f"[SNAP] ppo_0 saved (num_timesteps={model.num_timesteps})", flush=True)

    for m in milestones:
        delta = max(1, m - int(model.num_timesteps))
        model.learn(total_timesteps=delta, reset_num_timesteps=False)
        p = os.path.join(outdir, f"ppo_{m}")
        model.save(p)
        snaps[m] = (p, int(model.num_timesteps))
        print(f"[SNAP] ppo_{m} saved (actual num_timesteps={model.num_timesteps})",
              flush=True)

    env.close()
    return snaps


# ── report ────────────────────────────────────────────────────────────────────

_COLS = [
    ("success",  "Success %",        "{:.1f}"),
    ("crash",    "Crash %",          "{:.1f}"),
    ("reward",   "Mean reward",      "{:+.1f}"),
    ("attitude", "Attitude err deg", "{:.2f}"),
    ("smooth",   "Ctrl-delta(smooth)", "{:.4f}"),
    ("length",   "Mean ep length",   "{:.0f}"),
]


def print_table(title, rows):
    print(f"\n{'='*96}")
    print(f"  {title}")
    print(f"{'='*96}")
    hdr = f"  {'Stage':<22}" + "".join(f"{lbl:>18}" for _, lbl, _ in _COLS)
    print(hdr)
    for label, res in rows:
        line = f"  {label:<22}"
        for key, _, fmt in _COLS:
            line += f"{fmt.format(res[key]):>18}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--rec-episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--milestones", type=int, nargs="+",
                    default=[10_000, 50_000, 100_000, 300_000])
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    outdir = args.outdir or tempfile.mkdtemp(prefix="stab_snaps_")
    print(f"[ANALYZE] snapshot dir: {outdir}", flush=True)

    # 1) Train PPO-from-BC snapshots first (so everything below is evaluable).
    snaps = train_snapshots(args.milestones, outdir)

    # 2) Build the ordered list of (label, policy_fn) stages.
    final_m = args.milestones[-1]
    stages = [
        ("Teacher (PID)",        teacher_policy()),
        ("BC (deterministic)",   model_policy(BC_PATH)),
        ("PPO @ 0 updates",      model_policy(snaps[0][0])),
    ]
    for m in args.milestones:
        tag = "final" if m == final_m else f"{m//1000}k"
        stages.append((f"PPO @ {tag} ({snaps[m][1]} steps)", model_policy(snaps[m][0])))

    # 3) Evaluate on STABILIZE and on RECOVERY (recovery capability).
    stab_rows, rec_rows = [], []
    for label, fn in stages:
        print(f"[EVAL] {label}  (stabilize)…", flush=True)
        stab_rows.append((label, evaluate(fn, "stabilize", args.episodes, args.seed)))
    for label, fn in stages:
        print(f"[EVAL] {label}  (recovery)…", flush=True)
        rec_rows.append((label, evaluate(fn, "recovery", args.rec_episodes, args.seed)))

    print_table(f"STABILIZE  ({args.episodes} episodes, seed {args.seed})", stab_rows)
    print_table(f"RECOVERY-capability probe  ({args.rec_episodes} episodes)", rec_rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"snaps": {str(k): v[1] for k, v in snaps.items()},
                       "stabilize": {l: r for l, r in stab_rows},
                       "recovery":  {l: r for l, r in rec_rows}}, f, indent=2)
        print(f"\n[ANALYZE] wrote {args.json}")


if __name__ == "__main__":
    main()
