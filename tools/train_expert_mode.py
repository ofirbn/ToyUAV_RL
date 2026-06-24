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
                      init_from_bc: str = None, init_from: str = None,
                      n_envs: int = N_ENVS):
    """Train one PPO expert for `mode` and save it to `out` (default
    models/experts/<file>). Returns the resolved output path (no .zip).

    Initialization precedence (in _build_model): resume an existing expert
    checkpoint at `out` (unless force_new) -> seed from `init_from` (e.g.
    models/latest) -> warm-start from `init_from_bc` -> fresh random weights."""
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
    if init_from:
        cfg["init_from"] = init_from
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
    print(f"  Seed from : {init_from or '(none)'}")
    print(f"  BC init   : {init_from_bc or '(none)'}")
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


def _auto_bc_path(mode):
    """Conventional BC checkpoint for `mode` (models/bc/<phase>_bc), or None.
    Used to warm-start each expert when training all of them and no explicit
    --init-from-bc was given."""
    cand = os.path.join("models", "bc", phase_for(mode) + "_bc")
    return cand if os.path.exists(cand + ".zip") else None


def _norm_init_from(value):
    """Normalize an expert_init_from / --init-from value to a load path or None.
    Treats empty / '(auto)' / 'none' as 'no explicit seed' and strips .zip."""
    if not value:
        return None
    v = str(value).strip()
    if v.lower() in ("", "none", "auto") or v.lower().startswith("(auto"):
        return None
    return v[:-4] if v.endswith(".zip") else v


def train_all_experts(timesteps: int, save_every: int = 50_000,
                      force_new: bool = False, init_from_bc: str = None,
                      init_from: str = None, n_envs: int = N_ENVS):
    """Train every expert in EXPERT_MODES sequentially, each to its own
    models/experts/<mode>.zip. Each expert runs for `timesteps` steps. If
    init_from is given (e.g. models/latest), every fresh expert seeds from it;
    otherwise, when no explicit init_from_bc is given, each mode warm-starts
    from its conventional BC checkpoint if one exists. Returns (mode, out)."""
    results = []
    total = len(EXPERT_MODES)
    for i, mode in enumerate(EXPERT_MODES):
        bc = init_from_bc or _auto_bc_path(mode)
        print(f"\n{'#'*60}")
        print(f"#  EXPERT {i+1}/{total}: {mode.name}")
        print(f"{'#'*60}")
        out = train_expert_mode(
            mode         = mode,
            timesteps    = timesteps,
            out          = None,
            save_every   = save_every,
            force_new    = force_new,
            init_from_bc = bc,
            init_from    = init_from,
            n_envs       = n_envs,
        )
        results.append((mode, out))

    print(f"\n{'='*60}")
    print(f"  ALL {total} EXPERTS TRAINED")
    for mode, out in results:
        print(f"  {mode.name:<14} -> {out}.zip")
    print(f"{'='*60}")
    return results


def train_expert(cfg: dict):
    """main.py entry point: train one expert, or all, from a parsed config.txt.

    Reads: expert_mode ('all' or a mode name; default 'approach'), timesteps,
    save_every, force_new, init_from_bc, expert_init_from (seed a fresh expert
    from this model, e.g. models/latest), expert_out (optional explicit path,
    single-mode only)."""
    common = dict(
        timesteps    = int(cfg.get("timesteps", 300_000)),
        save_every   = int(cfg.get("save_every", 50_000)),
        force_new    = cfg.get("force_new", "false").lower() == "true",
        init_from_bc = cfg.get("init_from_bc") or None,
        init_from    = _norm_init_from(cfg.get("expert_init_from")),
    )
    em = str(cfg.get("expert_mode", "approach")).lower()
    if em == "all":
        return train_all_experts(**common)
    return train_expert_mode(mode=resolve_mode(em),
                             out=cfg.get("expert_out") or None, **common)


def train_expert_visual(cfg: dict):
    """main.py entry for mode=train_expert_visual: train ONE expert with the
    live pygame dashboard (reuses train.train_visual). expert_mode='all' is not
    shown live — it falls back to headless all-expert training."""
    em = str(cfg.get("expert_mode", "approach")).lower()
    init_from = _norm_init_from(cfg.get("expert_init_from"))

    if em == "all":
        # Train every expert sequentially in one live dashboard window.
        jobs = [{
            "label":        m.name,
            "phase":        phase_for(m),
            "save_path":    default_model_path(m),
            "init_from":    init_from,
            "init_from_bc": _auto_bc_path(m),
        } for m in EXPERT_MODES]
        print(f"[TRAIN] Visual expert training: ALL {len(jobs)} experts "
              f"(sequential, one window)")
        from train import train_visual
        train_visual(cfg, skip_config_screen=True, expert_jobs=jobs)
        return [j["save_path"] for j in jobs]

    mode  = resolve_mode(em)
    out   = cfg.get("expert_out") or default_model_path(mode)
    phase = phase_for(mode)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Drive the existing live dashboard: curriculum off, env locked to the mode,
    # final checkpoint saved to the isolated expert path. Auto BC warm-start
    # unless the caller set one explicitly.
    vis_cfg = {
        **cfg,
        "curriculum":       "false",
        "curriculum_phase": phase,
        "model":            out + ".zip",   # so _model_dir -> models/experts
    }
    if not vis_cfg.get("init_from_bc"):
        vis_cfg["init_from_bc"] = _auto_bc_path(mode)
    vis_cfg["init_from"] = _norm_init_from(cfg.get("expert_init_from"))

    print(f"[TRAIN] Visual expert training: {mode.name} "
          f"(phase='{phase}') -> {out}.zip")
    from train import train_visual
    train_visual(vis_cfg, save_path=out, skip_config_screen=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a single-mode PPO expert for any FlightMode.")
    parser.add_argument("--mode", type=str, default=None,
                        help="FlightMode to train (stabilize, recovery, "
                             "altitude_hold, heading_hold, waypoint, loiter, "
                             "approach, landing), or 'all' to train every "
                             "expert sequentially.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--out", type=str, default=None,
                        help="Save path without .zip (default models/experts/<mode>).")
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--force-new", action="store_true",
                        help="Ignore any existing checkpoint at --out and start fresh.")
    parser.add_argument("--init-from-bc", type=str, default=None,
                        help="Warm-start PPO from a behavior-cloned checkpoint "
                             "(e.g. models/bc/all_bc).")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Seed a fresh expert from this model checkpoint "
                             "(e.g. models/latest). Ignored if the expert is "
                             "resumed; takes priority over --init-from-bc.")
    parser.add_argument("--list-modes", action="store_true",
                        help="List trainable modes and exit.")
    args = parser.parse_args(argv)

    if args.list_modes:
        print("Trainable expert modes (mode -> default save path):")
        for m in EXPERT_MODES:
            print(f"  {m.name.lower():<14} -> {default_model_path(m)}.zip")
        return

    if not args.mode:
        print("[TRAIN] --mode is required (e.g. --mode landing, or --mode all). "
              "Use --list-modes to see options.")
        sys.exit(2)

    if args.mode.lower() == "all":
        if args.out:
            print("[TRAIN] --out is ignored with --mode all; each expert saves "
                  "to its own models/experts/<mode>.zip.")
        train_all_experts(
            timesteps    = args.timesteps,
            save_every   = args.save_every,
            force_new    = args.force_new,
            init_from_bc = args.init_from_bc,
            init_from    = _norm_init_from(args.init_from),
        )
        return

    train_expert_mode(
        mode         = resolve_mode(args.mode),
        timesteps    = args.timesteps,
        out          = args.out,
        save_every   = args.save_every,
        force_new    = args.force_new,
        init_from_bc = args.init_from_bc,
        init_from    = _norm_init_from(args.init_from),
    )


if __name__ == "__main__":
    main()
