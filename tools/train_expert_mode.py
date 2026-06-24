"""
Generalized single-mode PPO expert trainer.

Mode-agnostic version of tools/train_expert_approach.py: train a PPO expert
specialized for ANY FlightMode and save it to the isolated expert path
models/experts/<file>.zip, leaving the default `python main.py` train flow
(models/latest.zip) completely unaffected.

Reuses the existing teacher -> BC -> PPO machinery from train.py:
  - _build_model() handles resume / force-new / BC warm-start.
  - LogCallback() handles periodic + best_reward/best_success checkpoints.

The env is locked to the requested mode via its curriculum phase
(sim/curriculum.py: get_locked_mode) — no env / reward / physics changes.

CLI:
    python tools/train_expert_mode.py --mode landing --timesteps 300000
    python tools/train_expert_mode.py --mode loiter --init-from-bc models/bc/all_bc
    python tools/train_expert_mode.py --list-modes

Programmatic (used by main.py when config mode=train_expert):
    from tools.train_expert_mode import train_expert
    train_expert(cfg)   # reads cfg['expert_mode'], cfg['timesteps'], ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.fixed_wing_env import FixedWingEnv
from sim.curriculum import get_phase_weighted_modes, get_locked_mode
from train import _build_model, LogCallback
from tools.eval_expert_mode import (resolve_mode, default_model_path,
                                     phase_for, EXPERT_MODES, _EXPERTS_DIR)

N_ENVS = 4


def train_expert_mode(mode, timesteps: int, out: str = None,
                      save_every: int = 50_000, force_new: bool = False,
                      init_from_bc: str = None, n_envs: int = N_ENVS):
    """Train one PPO expert for `mode` and save it to `out` (default
    models/experts/<file>). Returns the resolved output path (no .zip)."""
    phase = phase_for(mode)
    out   = out or default_model_path(mode)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    env = make_vec_env(
        FixedWingEnv,
        n_envs      = n_envs,
        vec_env_cls = DummyVecEnv,
        env_kwargs  = {
            "training_mode":    True,
            "curriculum_phase": phase,   # locks (or weights) the mode each reset
        },
    )

    cfg = {"force_new": "true" if force_new else "false"}
    if init_from_bc:
        cfg["init_from_bc"] = init_from_bc
    model = _build_model(env, cfg, out)

    weighted = get_phase_weighted_modes(phase) is not None
    print(f"\n{'='*60}")
    print("  ToyUAV RL — Single-Mode Expert Training")
    print(f"  Mode      : {mode.name}  (phase='{phase}')")
    if weighted:
        print(f"  NOTE      : phase '{phase}' samples multiple modes by design.")
    elif get_locked_mode(phase) != mode:
        print(f"  WARNING   : phase '{phase}' does not hard-lock {mode.name}.")
    print(f"  Timesteps : {timesteps:,}")
    print(f"  Envs      : {n_envs}  (DummyVecEnv)")
    print(f"  Warm-start: {init_from_bc or '(none)'}")
    print(f"  Save path : {out}.zip")
    print(f"{'='*60}\n")

    callback = LogCallback(
        best_reward_path  = out + "_best_reward",
        best_success_path = out + "_best_success",
        save_every        = save_every,
    )

    model.learn(
        total_timesteps     = timesteps,
        reset_num_timesteps = False,
        callback            = callback,
    )

    model.save(out)
    print(f"\n[TRAIN] Saved expert -> {out}.zip")
    print("[TRAIN] Done.")
    return out


def train_expert(cfg: dict):
    """main.py entry point: train an expert from a parsed config.txt dict.

    Reads: expert_mode (default 'approach'), timesteps, save_every, force_new,
    init_from_bc, expert_out (optional explicit save path)."""
    mode = resolve_mode(cfg.get("expert_mode", "approach"))
    return train_expert_mode(
        mode         = mode,
        timesteps    = int(cfg.get("timesteps", 300_000)),
        out          = cfg.get("expert_out") or None,
        save_every   = int(cfg.get("save_every", 50_000)),
        force_new    = cfg.get("force_new", "false").lower() == "true",
        init_from_bc = cfg.get("init_from_bc") or None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a single-mode PPO expert for any FlightMode.")
    parser.add_argument("--mode", type=str, default=None,
                        help="FlightMode to train (stabilize, recovery, "
                             "altitude_hold, heading_hold, waypoint, loiter, "
                             "approach, landing).")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--out", type=str, default=None,
                        help="Save path without .zip (default models/experts/<mode>).")
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--force-new", action="store_true",
                        help="Ignore any existing checkpoint at --out and start fresh.")
    parser.add_argument("--init-from-bc", type=str, default=None,
                        help="Warm-start PPO from a behavior-cloned checkpoint "
                             "(e.g. models/bc/all_bc).")
    parser.add_argument("--list-modes", action="store_true",
                        help="List trainable modes and exit.")
    args = parser.parse_args(argv)

    if args.list_modes:
        print("Trainable expert modes (mode -> default save path):")
        for m in EXPERT_MODES:
            print(f"  {m.name.lower():<14} -> {default_model_path(m)}.zip")
        return

    if not args.mode:
        print("[TRAIN] --mode is required (e.g. --mode landing). "
              "Use --list-modes to see options.")
        sys.exit(2)

    train_expert_mode(
        mode         = resolve_mode(args.mode),
        timesteps    = args.timesteps,
        out          = args.out,
        save_every   = args.save_every,
        force_new    = args.force_new,
        init_from_bc = args.init_from_bc,
    )


if __name__ == "__main__":
    main()
