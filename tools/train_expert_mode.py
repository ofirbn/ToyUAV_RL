"""
Generalized adaptive PPO expert trainer.

Trains a PPO expert specialized for ANY FlightMode and saves it to the isolated
expert path models/experts/<file>.zip, leaving the default `python main.py`
train flow (models/latest.zip) completely unaffected.

Training strategy (all modes): teacher demos -> behavior cloning (existing
ml/ tools) -> PPO warm-start from BC -> PERIODIC EVALUATION with multi-objective
plateau early-stopping and best-checkpoint selection (tools/expert_eval.py).
There is no hard-coded PPO step count; expert_max_timesteps is only a safety cap.

The env is locked to the requested mode via its curriculum phase
(sim/curriculum.py: get_locked_mode) — no env / reward / physics changes.

CLI:
    python tools/train_expert_mode.py --mode landing
    python tools/train_expert_mode.py --mode loiter --init-from-bc models/bc/all_bc
    python tools/train_expert_mode.py --mode all --max-timesteps 800000
    python tools/train_expert_mode.py --list-modes

Programmatic (used by main.py when config mode=train_expert):
    from tools.train_expert_mode import train_expert
    train_expert(cfg)   # reads expert_mode, expert_* adaptive params, ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.fixed_wing_env import FixedWingEnv
from sim.curriculum import get_phase_weighted_modes, get_locked_mode
from train import _build_model
from tools.eval_expert_mode import (resolve_mode, default_model_path,
                                     phase_for, EXPERT_MODES, _EXPERTS_DIR)
from tools.expert_eval import (AdaptiveExpertMonitor, parse_expert_cfg,
                               finalize_expert)

N_ENVS = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def _auto_bc_path(mode):
    """Conventional BC checkpoint for `mode` (models/bc/<phase>_bc), or None."""
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


# ── headless adaptive training ────────────────────────────────────────────────

def train_expert_mode(mode, params: dict, out: str = None,
                      force_new: bool = False, init_from: str = None,
                      init_from_bc: str = None, n_envs: int = N_ENVS,
                      shared_state=None):
    """Adaptively train one PPO expert for `mode`, saving the routed checkpoint
    to `out`.zip (default models/experts/<file>.zip). Returns the path (no .zip).

    `params` comes from tools.expert_eval.parse_expert_cfg(cfg, mode).
    Init precedence (in _build_model): resume existing <out>.zip (unless
    force_new) -> seed from init_from -> BC warm-start -> fresh."""
    phase = phase_for(mode)
    out   = out or default_model_path(mode)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    env = make_vec_env(
        FixedWingEnv, n_envs=n_envs, vec_env_cls=DummyVecEnv,
        env_kwargs={"training_mode": True, "curriculum_phase": phase},
    )

    cfg_build = {"force_new": "true" if force_new else "false"}
    if init_from_bc:
        cfg_build["init_from_bc"] = init_from_bc
    if init_from:
        cfg_build["init_from"] = init_from
    model = _build_model(env, cfg_build, out)

    weighted = get_phase_weighted_modes(phase) is not None
    print(f"\n{'='*62}")
    print("  ToyUAV RL — Adaptive Single-Mode Expert Training")
    print(f"  Mode        : {mode.name}  (phase='{phase}')")
    if weighted:
        print(f"  NOTE        : phase '{phase}' samples multiple modes by design.")
    elif get_locked_mode(phase) != mode:
        print(f"  WARNING     : phase '{phase}' does not hard-lock {mode.name}.")
    print(f"  Seed from   : {init_from or '(none)'}")
    print(f"  BC init     : {init_from_bc or '(none)'}")
    print(f"  Eval        : every {params['eval_interval']:,} steps, "
          f"{params['eval_episodes']} episodes")
    print(f"  Early stop  : patience {params['patience']} (plateau) | "
          f"target succ>={params['success_target']:.0%}/crash<={params['crash_target']:.0%}")
    print(f"  Max steps   : {params['max_timesteps']:,}  (safety cap)")
    print(f"  Save path   : {out}.zip  (best -> _best.zip)")
    print(f"{'='*62}\n")

    monitor = AdaptiveExpertMonitor(mode, out, params, shared_state=shared_state)
    model.learn(total_timesteps=params["max_timesteps"],
                reset_num_timesteps=False, callback=monitor)
    env.close()

    if monitor.stop_reason:
        print(f"[TRAIN] {mode.name}: stopped — {monitor.stop_reason}")
    else:
        print(f"[TRAIN] {mode.name}: reached max_timesteps safety cap.")
    finalize_expert(out, mode)
    print(f"[TRAIN] {mode.name}: done.")
    return out


def train_all_experts(cfg: dict, force_new: bool = False, init_from: str = None,
                      init_from_bc: str = None, n_envs: int = N_ENVS):
    """Adaptively train every expert in EXPERT_MODES sequentially, each to its
    own models/experts/<mode>.zip. Params are resolved per mode from `cfg`."""
    results = []
    total = len(EXPERT_MODES)
    for i, mode in enumerate(EXPERT_MODES):
        params = parse_expert_cfg(cfg, mode)
        bc = init_from_bc or _auto_bc_path(mode)
        print(f"\n{'#'*62}")
        print(f"#  EXPERT {i+1}/{total}: {mode.name}")
        print(f"{'#'*62}")
        out = train_expert_mode(mode, params, out=None, force_new=force_new,
                                init_from=init_from, init_from_bc=bc, n_envs=n_envs)
        results.append((mode, out))

    print(f"\n{'='*62}")
    print(f"  ALL {total} EXPERTS TRAINED")
    for mode, out in results:
        print(f"  {mode.name:<14} -> {out}.zip")
    print(f"{'='*62}")
    return results


def train_expert(cfg: dict):
    """main.py entry (headless): train one expert, or all, from a config dict."""
    force_new    = cfg.get("force_new", "false").lower() == "true"
    init_from    = _norm_init_from(cfg.get("expert_init_from"))
    init_from_bc = cfg.get("init_from_bc") or None

    em = str(cfg.get("expert_mode", "approach")).lower()
    if em == "all":
        return train_all_experts(cfg, force_new=force_new, init_from=init_from,
                                 init_from_bc=init_from_bc)

    mode   = resolve_mode(em)
    params = parse_expert_cfg(cfg, mode)
    bc     = init_from_bc or _auto_bc_path(mode)
    return train_expert_mode(mode, params, out=cfg.get("expert_out") or None,
                             force_new=force_new, init_from=init_from,
                             init_from_bc=bc)


# ── visual adaptive training (delegates to train.train_visual via expert_jobs) ─

def _build_jobs(cfg):
    """Build the expert_jobs list (1 for a single mode, all 8 for 'all'),
    each carrying its mode, phase, save path, seed/BC sources, and adaptive
    params — consumed by train.train_visual's expert_jobs loop."""
    em           = str(cfg.get("expert_mode", "approach")).lower()
    init_from    = _norm_init_from(cfg.get("expert_init_from"))
    init_from_bc = cfg.get("init_from_bc") or None
    expert_out   = cfg.get("expert_out") or None

    modes = list(EXPERT_MODES) if em == "all" else [resolve_mode(em)]
    jobs = []
    for m in modes:
        out = (expert_out if (em != "all" and expert_out) else default_model_path(m))
        jobs.append({
            "label":        m.name,
            "mode":         m,
            "phase":        phase_for(m),
            "save_path":    out,
            "init_from":    init_from,
            "init_from_bc": init_from_bc or _auto_bc_path(m),
            "params":       parse_expert_cfg(cfg, m),
        })
    return jobs


def train_expert_visual(cfg: dict):
    """main.py entry for mode=train_expert_visual: adaptive expert training with
    the live pygame dashboard. One expert (single mode) or all 8 sequentially,
    each evaluated/early-stopped exactly like the headless path."""
    jobs = _build_jobs(cfg)
    if len(jobs) > 1:
        print(f"[TRAIN] Visual adaptive expert training: ALL {len(jobs)} experts "
              f"(sequential, one window)")
    else:
        print(f"[TRAIN] Visual adaptive expert training: {jobs[0]['label']}")
    from train import train_visual
    train_visual(cfg, skip_config_screen=True, expert_jobs=jobs)
    return [j["save_path"] for j in jobs]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Adaptively train a single-mode PPO expert for any FlightMode.")
    parser.add_argument("--mode", type=str, default=None,
                        help="FlightMode to train (stabilize, recovery, "
                             "altitude_hold, heading_hold, waypoint, loiter, "
                             "approach, landing), or 'all'.")
    parser.add_argument("--out", type=str, default=None,
                        help="Save path without .zip (default models/experts/<mode>).")
    parser.add_argument("--max-timesteps", type=int, default=1_000_000,
                        help="Safety cap on PPO steps (NOT the primary stop rule).")
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=None,
                        help="Override the per-mode reward improvement delta.")
    parser.add_argument("--force-new", action="store_true",
                        help="Ignore any existing checkpoint at --out and start fresh.")
    parser.add_argument("--init-from-bc", type=str, default=None,
                        help="Warm-start PPO from a BC checkpoint (e.g. models/bc/all_bc).")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Seed a fresh expert from this model (e.g. models/latest). "
                             "Ignored on resume; priority over --init-from-bc.")
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

    if args.mode.lower() == "all" and args.out:
        print("[TRAIN] --out is ignored with --mode all; each expert saves to "
              "its own models/experts/<mode>.zip.")

    cfg = {
        "expert_mode":                  args.mode,
        "expert_max_timesteps":         args.max_timesteps,
        "expert_eval_interval_steps":   args.eval_interval,
        "expert_eval_episodes":         args.eval_episodes,
        "expert_early_stop_patience":   args.patience,
        "force_new":                    "true" if args.force_new else "false",
        "init_from_bc":                 args.init_from_bc,
        "expert_init_from":             args.init_from,
        "expert_out":                   args.out,
    }
    if args.min_delta is not None:
        cfg["expert_min_improvement_delta"] = args.min_delta

    train_expert(cfg)


if __name__ == "__main__":
    main()
