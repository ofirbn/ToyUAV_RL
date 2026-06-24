"""
M5A — generalized per-mode expert evaluator.

Evaluate ONE expert/checkpoint in isolation for ANY FlightMode, with the env
hard-locked to that mode's curriculum phase. This is the mode-agnostic
generalization of tools/eval_expert_approach.py: same deterministic-rollout
machinery, same comparison table and example-trajectory output, but the env
phase, default checkpoint, success condition and the mode-specific accuracy
metrics are all selected from the requested mode.

Nothing here modifies routing, rewards, physics or training — it only *reads*
the env's own state/target/info and the env's own `episode_success` signal.

Single-model usage:
    python tools/eval_expert_mode.py --mode approach --episodes 50
    python tools/eval_expert_mode.py --mode stabilize \
        --model models/experts/stabilize --episodes 50

Multi-model comparison (identical seeds / initial conditions per episode):
    python tools/eval_expert_mode.py --mode approach --episodes 50 \
        --model models/latest --model models/experts/approach

List the modes this tool understands:
    python tools/eval_expert_mode.py --list-modes
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.fixed_wing_env import FixedWingEnv
from sim.flight_modes import FlightMode
from sim.curriculum import get_locked_mode, get_phase_weighted_modes


# ── Mode registry ─────────────────────────────────────────────────────────────
# For each evaluatable FlightMode: the curriculum phase that activates it under
# FixedWingEnv(training_mode=True, curriculum_phase=...), and the conventional
# expert checkpoint filename under models/experts/ (mirrors
# controllers.mission_manager.MissionManager._MODE_FILENAMES).

_MODE_SPECS = {
    FlightMode.STABILIZE:     {"phase": "stabilize",     "file": "stabilize"},
    FlightMode.RECOVERY:      {"phase": "recovery",      "file": "recovery"},
    FlightMode.ALTITUDE_HOLD: {"phase": "altitude_hold", "file": "alt_hold"},
    FlightMode.HEADING_HOLD:  {"phase": "heading_hold",  "file": "hdg_hold"},
    FlightMode.WAYPOINT:      {"phase": "waypoint",      "file": "waypoint"},
    FlightMode.LOITER:        {"phase": "loiter",        "file": "loiter"},
    FlightMode.APPROACH:      {"phase": "approach",      "file": "approach"},
    FlightMode.LANDING:       {"phase": "landing",       "file": "landing"},
}

_EXPERTS_DIR = "models/experts"


def _angle_diff(a, b):
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def resolve_mode(token: str) -> FlightMode:
    """Resolve a CLI mode token to a FlightMode. Accepts the enum name
    (e.g. 'ALTITUDE_HOLD'), the curriculum phase ('altitude_hold') or the
    expert filename ('alt_hold'), case-insensitively."""
    t = token.strip().lower()
    for mode, spec in _MODE_SPECS.items():
        if t in (mode.name.lower(), spec["phase"], spec["file"]):
            return mode
    valid = ", ".join(sorted({m.name.lower() for m in _MODE_SPECS}))
    print(f"[EVAL] Unknown mode '{token}'. Valid modes: {valid}")
    sys.exit(2)


def default_model_path(mode: FlightMode) -> str:
    return os.path.join(_EXPERTS_DIR, _MODE_SPECS[mode]["file"])


# ── Mode-specific accuracy metrics ────────────────────────────────────────────
# Each function returns an ordered dict {metric_name: value} for the CURRENT
# step, computed from the same quantities FixedWingEnv._episode_success() uses,
# so reported error metrics line up with the env's own success criterion.
#   - "mean" aggregation: mean over the episode, then mean over episodes.
#   - "final" aggregation: value at the last step, then mean over episodes.
# AGG declares, per metric, which aggregation is meaningful for the summary.

_GLIDESLOPE = math.radians(3.0)


def _step_metrics(mode: FlightMode, s, t, stall_speed: float) -> dict:
    if mode == FlightMode.STABILIZE:
        return {
            "roll_err_deg":  abs(math.degrees(s.roll)),
            "pitch_err_deg": abs(math.degrees(s.pitch)),
        }
    if mode == FlightMode.RECOVERY:
        return {
            "airspeed":     s.airspeed,
            "speed_margin": s.airspeed - stall_speed,
            "roll_err_deg": abs(math.degrees(s.roll)),
        }
    if mode == FlightMode.ALTITUDE_HOLD:
        if t is None:
            return {}
        return {"alt_err_m": abs(s.pos[2] - t.altitude)}
    if mode == FlightMode.HEADING_HOLD:
        if t is None:
            return {}
        return {"hdg_err_deg": abs(math.degrees(_angle_diff(s.yaw, t.heading)))}
    if mode == FlightMode.WAYPOINT:
        if t is None:
            return {}
        dx, dy, dz = s.pos[0] - t.position[0], s.pos[1] - t.position[1], s.pos[2] - t.position[2]
        return {"dist_to_wp_m": math.sqrt(dx * dx + dy * dy + dz * dz)}
    if mode == FlightMode.LOITER:
        if t is None:
            return {}
        r = math.hypot(s.pos[0] - t.position[0], s.pos[1] - t.position[1])
        return {"radial_err_m": abs(r - t.radius), "radius_m": r}
    if mode == FlightMode.APPROACH:
        if t is None:
            return {}
        dx, dy = s.pos[0] - t.position[0], s.pos[1] - t.position[1]
        ideal_alt = math.hypot(dx, dy) * math.tan(_GLIDESLOPE)
        return {
            "lateral_err_m":   abs(dx),
            "heading_err_deg": abs(math.degrees(_angle_diff(s.yaw, t.heading))),
            "glideslope_err_m": abs(s.pos[2] - ideal_alt),
        }
    if mode == FlightMode.LANDING:
        if t is None:
            return {}
        return {
            "altitude_m":      s.pos[2],
            "dist_to_zone_m":  math.hypot(s.pos[0] - t.position[0], s.pos[1] - t.position[1]),
            "vertical_speed":  s.vel[2],
        }
    return {}


# Which aggregation to surface per metric ("mean" | "final"). Metrics not listed
# default to "mean". Touchdown/arrival quantities are most meaningful at the end.
_METRIC_AGG = {
    "dist_to_wp_m":   "final",
    "altitude_m":     "final",
    "dist_to_zone_m": "final",
    "vertical_speed": "final",
}


def _agg_for(metric: str) -> str:
    return _METRIC_AGG.get(metric, "mean")


# ── Rollout ───────────────────────────────────────────────────────────────────

def _resolve_zip(path: str) -> str:
    zip_path = path if path.endswith(".zip") else path + ".zip"
    if not os.path.exists(zip_path):
        print(f"[EVAL] No checkpoint at {zip_path}.")
        sys.exit(1)
    return zip_path


def run_eval(model_path: str, mode: FlightMode, episodes: int, seed: int,
             verbose: bool = True):
    """Run `episodes` deterministic episodes for one model with the env locked
    to `mode`'s curriculum phase. Returns a per-episode result dict, or None if
    the checkpoint's observation space is incompatible with the current env."""
    spec  = _MODE_SPECS[mode]
    phase = spec["phase"]

    zip_path = _resolve_zip(model_path)
    model = PPO.load(model_path)
    env   = FixedWingEnv(training_mode=True, curriculum_phase=phase)

    model_shape = tuple(model.observation_space.shape)
    env_shape   = tuple(env.observation_space.shape)
    compatible  = model_shape == env_shape

    print(f"\n[DIAG] checkpoint : {zip_path}")
    print(f"[DIAG] mode       : {mode.name}  (phase='{phase}')")
    print(f"[DIAG] model obs  : {model_shape}")
    print(f"[DIAG] env obs    : {env_shape}")
    print(f"[DIAG] compatible : {'yes' if compatible else 'no'}")

    if not compatible:
        # Do NOT pad/truncate observations — refuse and let the caller skip it.
        print(f"[EVAL] Skipping model {model_path}: obs shape mismatch, "
              f"model expects {model_shape[-1]}, env provides {env_shape[-1]}")
        env.close()
        return None

    rewards, lengths, successes, crashes, stalls = [], [], [], [], []
    term_reasons, trajectories, actual_modes = [], [], []
    ep_means, ep_finals = [], []   # per-episode {metric: value} (mode-specific)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        actual_mode = env._mode          # actual mode this episode (weighted phases vary)
        done = truncated = False
        ep_reward = 0.0
        step_acc, last_step = {}, {}
        traj = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            s = env._state
            sm = _step_metrics(actual_mode, s, env._target, env._stall_speed)
            for k, v in sm.items():
                step_acc.setdefault(k, []).append(v)
            last_step = sm
            traj.append((round(s.pos[0], 1), round(s.pos[1], 1), round(s.pos[2], 1)))

        crashed = info.get("crashed", False)
        landed  = info.get("landed", False)
        success = info.get("episode_success", not crashed)
        stalled = info.get("stalled_this_ep", False)

        if crashed:
            reason = f"crash:{info.get('crash_type', 'UNKNOWN')}"
        elif landed:
            reason = "landed"
        elif success:
            reason = "success"
        else:
            reason = "timeout"

        rewards.append(ep_reward)
        lengths.append(env.steps)
        successes.append(int(success))
        crashes.append(int(crashed))
        stalls.append(int(stalled))
        term_reasons.append(reason)
        trajectories.append(traj)
        actual_modes.append(actual_mode)
        ep_means.append({k: float(np.mean(v)) for k, v in step_acc.items()})
        ep_finals.append(dict(last_step))

        if verbose:
            tag = "" if actual_mode == mode else f"  [{actual_mode.name}]"
            print(f"  ep {ep+1:>3}/{episodes}  reward={ep_reward:+8.2f}  "
                  f"len={env.steps:>4}  {reason}{tag}")

    env.close()
    return {
        "model_path":   model_path,
        "mode":         mode,
        "phase":        phase,
        "rewards":      rewards,
        "lengths":      lengths,
        "successes":    successes,
        "crashes":      crashes,
        "stalls":       stalls,
        "term_reasons": term_reasons,
        "trajectories": trajectories,
        "actual_modes": actual_modes,
        "ep_means":     ep_means,
        "ep_finals":    ep_finals,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def summarize(result: dict) -> dict:
    n = len(result["rewards"])
    mode = result["mode"]

    reasons = {}
    for r in result["term_reasons"]:
        reasons[r] = reasons.get(r, 0) + 1

    # Off-mode episodes (only possible for weighted phases, e.g. altitude_hold).
    off_counts = {}
    on_idx = []
    for i, am in enumerate(result["actual_modes"]):
        if am == mode:
            on_idx.append(i)
        else:
            off_counts[am.name] = off_counts.get(am.name, 0) + 1

    # Mode-specific metrics aggregated over ON-mode episodes only.
    mode_metrics = {}
    if on_idx:
        means_src  = [result["ep_means"][i]  for i in on_idx]
        finals_src = [result["ep_finals"][i] for i in on_idx]
        keys = []
        for d in means_src:
            for k in d:
                if k not in keys:
                    keys.append(k)
        for k in keys:
            agg = _agg_for(k)
            src = finals_src if agg == "final" else means_src
            vals = [d[k] for d in src if k in d]
            if vals:
                mode_metrics[k] = {"agg": agg, "value": float(np.mean(vals))}

    return {
        "n":            n,
        "n_on_mode":    len(on_idx),
        "off_counts":   off_counts,
        "success_rate": 100.0 * sum(result["successes"]) / n,
        "crash_rate":   100.0 * sum(result["crashes"]) / n,
        "stall_rate":   100.0 * sum(result["stalls"]) / n,
        "mean_reward":  float(np.mean(result["rewards"])),
        "std_reward":   float(np.std(result["rewards"])),
        "mean_length":  float(np.mean(result["lengths"])),
        "term_reasons": reasons,
        "mode_metrics": mode_metrics,
    }


_METRIC_LABELS = {
    "roll_err_deg":    "Mean |roll| (deg)",
    "pitch_err_deg":   "Mean |pitch| (deg)",
    "airspeed":        "Mean airspeed (m/s)",
    "speed_margin":    "Mean speed margin (m/s)",
    "alt_err_m":       "Mean altitude error (m)",
    "hdg_err_deg":     "Mean heading error (deg)",
    "dist_to_wp_m":    "Final dist to waypoint (m)",
    "radial_err_m":    "Mean radial error (m)",
    "radius_m":        "Mean orbit radius (m)",
    "lateral_err_m":   "Mean lateral error (m)",
    "heading_err_deg": "Mean heading error (deg)",
    "glideslope_err_m": "Mean glideslope error (m)",
    "altitude_m":      "Final altitude (m)",
    "dist_to_zone_m":  "Final dist to zone (m)",
    "vertical_speed":  "Touchdown v-speed (m/s)",
}


def _label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric)


def print_summary(label: str, result: dict) -> dict:
    summ = summarize(result)
    print(f"\n{'='*60}")
    print(f"  {label}  ({result['model_path']})  mode={result['mode'].name}")
    print(f"  episodes              : {summ['n']}")
    if summ["off_counts"]:
        print(f"  on-mode episodes      : {summ['n_on_mode']}/{summ['n']}  "
              f"(phase '{result['phase']}' also sampled: {summ['off_counts']})")
    print(f"  success_rate          : {summ['success_rate']:.1f}%")
    print(f"  crash_rate            : {summ['crash_rate']:.1f}%")
    print(f"  stall_rate            : {summ['stall_rate']:.1f}%")
    print(f"  mean_reward           : {summ['mean_reward']:+.2f}  (std {summ['std_reward']:.2f})")
    print(f"  mean_episode_length   : {summ['mean_length']:.1f}")
    for k, m in summ["mode_metrics"].items():
        print(f"  {_label(k):<22}: {m['value']:.2f}")
    print(f"  termination_reasons   : {summ['term_reasons']}")
    print(f"{'='*60}")
    return summ


def print_example_trajectories(label: str, result: dict):
    if not result["rewards"]:
        return
    order = sorted(range(len(result["rewards"])), key=lambda i: result["rewards"][i])
    worst_i, best_i, median_i = order[0], order[-1], order[len(order) // 2]

    print(f"\n  -- {label}: example trajectories (start -> end, 5-pt sample) --")
    for tag, i in (("best", best_i), ("median", median_i), ("worst", worst_i)):
        traj = result["trajectories"][i]
        idxs = sorted(set(np.linspace(0, len(traj) - 1, min(5, len(traj))).astype(int)))
        pts  = " -> ".join(f"({traj[j][0]},{traj[j][1]},{traj[j][2]})" for j in idxs)
        print(f"    {tag:>6} (ep {i+1}, reward={result['rewards'][i]:+.2f}, "
              f"{result['term_reasons'][i]}): {pts}")


def print_comparison(results):
    """results: list of (label, result|None) preserving CLI order."""
    compatible = [(lbl, r) for lbl, r in results if r is not None]
    if len(compatible) < 2 and not any(r is None for _, r in results):
        return

    summ_by_label = {lbl: summarize(r) for lbl, r in compatible}

    # Common rows, plus the union of mode-specific metric keys (insertion order).
    metric_keys = []
    for _, r in compatible:
        for k in summarize(r)["mode_metrics"]:
            if k not in metric_keys:
                metric_keys.append(k)

    print(f"\n{'='*78}")
    print("  COMPARISON")
    print(f"{'='*78}")
    header = f"  {'Metric':<28}" + "".join(f"{lbl:>18}" for lbl, _ in results)
    print(header)

    common_rows = [
        ("Success rate (%)",    "success_rate", "{:.1f}"),
        ("Crash rate (%)",      "crash_rate",   "{:.1f}"),
        ("Stall rate (%)",      "stall_rate",   "{:.1f}"),
        ("Mean reward",         "mean_reward",  "{:+.2f}"),
        ("Mean episode length", "mean_length",  "{:.1f}"),
    ]
    for name, key, fmt in common_rows:
        line = f"  {name:<28}"
        for lbl, r in results:
            cell = "INCOMPAT" if r is None else fmt.format(summ_by_label[lbl][key])
            line += f"{cell:>18}"
        print(line)

    for mk in metric_keys:
        line = f"  {_label(mk):<28}"
        for lbl, r in results:
            if r is None:
                cell = "INCOMPAT"
            else:
                m = summ_by_label[lbl]["mode_metrics"].get(mk)
                cell = "-" if m is None else "{:.2f}".format(m["value"])
            line += f"{cell:>18}"
        print(line)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a single expert in isolation for any FlightMode.")
    parser.add_argument("--mode", type=str, default=None,
                        help="FlightMode to evaluate: stabilize, recovery, "
                             "altitude_hold, heading_hold, waypoint, loiter, "
                             "approach, landing.")
    parser.add_argument("--model", type=str, action="append", default=None,
                        help="Checkpoint path without .zip. Repeatable: pass "
                             "--model multiple times to compare. Defaults to "
                             "models/experts/<mode>.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-modes", action="store_true",
                        help="List evaluatable modes and exit.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_modes:
        print("Evaluatable modes (mode -> phase -> default expert):")
        for mode, spec in _MODE_SPECS.items():
            print(f"  {mode.name.lower():<14} -> phase '{spec['phase']:<13}' "
                  f"-> {os.path.join(_EXPERTS_DIR, spec['file'])}.zip")
        return

    if not args.mode:
        print("[EVAL] --mode is required (e.g. --mode approach). "
              "Use --list-modes to see options.")
        sys.exit(2)

    mode = resolve_mode(args.mode)

    # Warn when the chosen phase does not purely lock the requested mode.
    if get_phase_weighted_modes(_MODE_SPECS[mode]["phase"]) is not None:
        print(f"[EVAL] NOTE: phase '{_MODE_SPECS[mode]['phase']}' samples "
              f"multiple modes by design; mode-specific metrics are computed "
              f"over the {mode.name} episodes only (others reported separately).")
    elif get_locked_mode(_MODE_SPECS[mode]["phase"]) != mode:
        print(f"[EVAL] WARNING: phase '{_MODE_SPECS[mode]['phase']}' does not "
              f"hard-lock {mode.name}; results may include other modes.")

    models = args.model if args.model else [default_model_path(mode)]

    results, incompatible = [], []
    for m in models:
        label  = os.path.basename(m.rstrip("/\\")).replace(".zip", "")
        result = run_eval(m, mode, args.episodes, args.seed)
        if result is None:
            incompatible.append(label)
        results.append((label, result))

    summaries = []
    for label, result in results:
        if result is None:
            continue
        print_summary(label, result)
        print_example_trajectories(label, result)
        summaries.append(label)

    if incompatible:
        print(f"\n[EVAL] Incompatible (skipped): {', '.join(incompatible)}")

    if not summaries:
        print("[EVAL] No compatible models were evaluated.")
        return

    if len(summaries) > 1 or incompatible:
        print_comparison(results)


if __name__ == "__main__":
    main()
