"""
Adaptive expert-training support: periodic evaluation, multi-objective plateau
early-stopping, and best-checkpoint selection. Shared by the headless trainer
(tools/train_expert_mode.py) and the live-dashboard path (train.py expert_jobs).

This module is read-only w.r.t. rewards, physics, the teacher controller, and
MissionManager routing. It only runs the trained policy in a locked eval env and
decides when to stop / which checkpoint to keep.

Early stopping is multi-objective (NOT a single weighted score): an evaluation
counts as progress if it meaningfully improves ANY of {mean_reward, success_rate,
crash_rate, the mode's primary task metric}. A plateau is `patience` consecutive
evals with no improvement on any objective. A separate target rule stops once the
controller is reliably good (success high / crash low) for a few evals. The
weighted `score` is computed only as a logged diagnostic.
"""

import math
import os
import shutil

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.fixed_wing_env import FixedWingEnv
from sim.flight_modes import FlightMode
from tools.eval_expert_mode import _step_metrics, phase_for


# ── per-mode primary metric (lower = better; it's an error) ───────────────────

_PRIMARY = {
    FlightMode.STABILIZE:     (("roll_err_deg", "pitch_err_deg"), "attitude_deg"),
    FlightMode.RECOVERY:      (("roll_err_deg",),                 "roll_deg"),
    FlightMode.ALTITUDE_HOLD: (("alt_err_m",),                    "alt_err_m"),
    FlightMode.HEADING_HOLD:  (("hdg_err_deg",),                  "hdg_err_deg"),
    FlightMode.WAYPOINT:      (("dist_to_wp_m",),                 "dist_wp_m"),
    FlightMode.LOITER:        (("radial_err_m",),                 "radial_err_m"),
    FlightMode.APPROACH:      (("glideslope_err_m",),             "glideslope_m"),
    FlightMode.LANDING:       (("dist_to_zone_m",),               "dist_zone_m"),
}

# Per-mode fallback deltas: (reward_delta, primary_metric_delta).
# reward_delta is overridden by config expert_min_improvement_delta when set.
_MODE_DELTAS = {
    FlightMode.STABILIZE:     (5.0, 0.5),
    FlightMode.RECOVERY:      (5.0, 1.0),
    FlightMode.ALTITUDE_HOLD: (3.0, 1.0),
    FlightMode.HEADING_HOLD:  (3.0, 1.0),
    FlightMode.WAYPOINT:      (3.0, 5.0),
    FlightMode.LOITER:        (3.0, 2.0),
    FlightMode.APPROACH:      (3.0, 1.0),
    FlightMode.LANDING:       (3.0, 2.0),
}


def _primary_value(mode, means: dict) -> float:
    keys, _ = _PRIMARY.get(mode, ((), "metric"))
    return float(sum(means.get(k, 0.0) for k in keys))


def primary_label(mode) -> str:
    return _PRIMARY.get(mode, ((), "metric"))[1]


# ── deterministic evaluation ──────────────────────────────────────────────────

def eval_policy(predict_fn, mode, episodes: int, seed: int) -> dict:
    """Run `episodes` deterministic episodes in an env locked to `mode`.
    `predict_fn(obs) -> action`. Returns metrics (rates as fractions 0..1)."""
    env = FixedWingEnv(training_mode=True, curriculum_phase=phase_for(mode))
    rewards, succ, crash, smooth, lengths = [], [], [], [], []
    metric_acc = {}

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = trunc = False
        R = 0.0
        prev_a = None
        per_step = {}
        ad = []
        while not (done or trunc):
            a = np.asarray(predict_fn(obs), dtype=np.float32).reshape(-1)
            obs, r, done, trunc, info = env.step(a)
            R += r
            sm = _step_metrics(env._mode, env._state, env._target, env._stall_speed)
            for k, v in sm.items():
                per_step.setdefault(k, []).append(v)
            if prev_a is not None:
                ad.append(float(np.linalg.norm(a - prev_a)))
            prev_a = a

        crashed = info.get("crashed", False)
        success = info.get("episode_success", not crashed)
        rewards.append(R)
        succ.append(int(success))
        crash.append(int(crashed))
        smooth.append(float(np.mean(ad)) if ad else 0.0)
        lengths.append(env.steps)
        for k, vals in per_step.items():
            metric_acc.setdefault(k, []).append(float(np.mean(vals)))

    env.close()
    means = {k: float(np.mean(v)) for k, v in metric_acc.items()}
    return {
        "n":       episodes,
        "success": float(np.mean(succ)),
        "crash":   float(np.mean(crash)),
        "reward":  float(np.mean(rewards)),
        "smooth":  float(np.mean(smooth)),
        "length":  float(np.mean(lengths)),
        "primary": _primary_value(mode, means),
        "primary_label": primary_label(mode),
    }


# ── config parsing ────────────────────────────────────────────────────────────

def parse_expert_cfg(cfg: dict, mode) -> dict:
    """Resolve the adaptive-training params for `mode` from a config dict,
    applying per-mode fallbacks where appropriate."""
    rdelta_default, pdelta_default = _MODE_DELTAS.get(mode, (3.0, 1.0))

    def _f(key, d):
        try:
            v = cfg.get(key)
            return float(v) if v not in (None, "") else d
        except (TypeError, ValueError):
            return d

    def _i(key, d):
        try:
            v = cfg.get(key)
            return int(float(v)) if v not in (None, "") else d
        except (TypeError, ValueError):
            return d

    # expert_min_improvement_delta (if set) overrides the per-mode reward delta.
    reward_delta = _f("expert_min_improvement_delta", rdelta_default)

    return {
        "eval_interval":   _i("expert_eval_interval_steps", 20_000),
        "eval_episodes":   _i("expert_eval_episodes",       30),
        "patience":        _i("expert_early_stop_patience", 4),
        "reward_delta":    reward_delta,
        "primary_delta":   _f("expert_primary_delta",       pdelta_default),
        "rate_delta":      _f("expert_rate_delta",          0.02),
        "max_timesteps":   _i("expert_max_timesteps",       1_000_000),
        "success_target":  _f("expert_success_target",      0.90),
        "crash_target":    _f("expert_crash_target",        0.05),
        "target_patience": _i("expert_target_patience",     3),
        "eval_seed":       _i("expert_eval_seed",           9_000),
        # diagnostic-only weighted score (NOT used for stop/selection)
        "success_weight":  _f("expert_success_weight",  100.0),
        "crash_weight":    _f("expert_crash_weight",    200.0),
        "smoothness_weight": _f("expert_smoothness_weight", 500.0),
    }


def diagnostic_score(r: dict, p: dict) -> float:
    return (r["reward"] + p["success_weight"] * r["success"]
            - p["crash_weight"] * r["crash"]
            - p["smoothness_weight"] * r["smooth"])


# ── adaptive monitor callback ─────────────────────────────────────────────────

class AdaptiveExpertMonitor(BaseCallback):
    """Periodically evaluates the policy, keeps the best checkpoint, and stops
    PPO when learning plateaus (multi-objective) or a quality target is met.

    Checkpoints written (prefix = save_prefix):
      <prefix>_best.zip    — best by lexicographic dominance
      <prefix>_latest.zip  — most recent eval snapshot
    Call finalize_expert(save_prefix, mode) after training to copy best -> the
    routed <mode>.zip.
    """

    def __init__(self, mode, save_prefix: str, params: dict,
                 shared_state=None, verbose: int = 0):
        super().__init__(verbose)
        self.mode        = mode
        self.save_prefix = save_prefix
        self.p           = params
        self.ss          = shared_state

        self.best_path   = save_prefix + "_best"
        self.latest_path = save_prefix + "_latest"

        self._started     = False
        self._last_eval   = 0
        self._best_obj    = {"reward": -1e18, "success": -1e18,
                             "crash": 1e18, "primary": 1e18}
        self._best_sel    = None         # selection record
        self._no_improve  = 0
        self._target_hits = 0

        self.should_stop  = False
        self.stop_reason  = None
        self.evals        = []

    # — objective improvement (plateau detection) —
    def _register_improvement(self, r) -> bool:
        improved = False
        if r["reward"]  > self._best_obj["reward"]  + self.p["reward_delta"]:
            self._best_obj["reward"]  = r["reward"];  improved = True
        if r["success"] > self._best_obj["success"] + self.p["rate_delta"]:
            self._best_obj["success"] = r["success"]; improved = True
        if r["crash"]   < self._best_obj["crash"]   - self.p["rate_delta"]:
            self._best_obj["crash"]   = r["crash"];   improved = True
        if r["primary"] < self._best_obj["primary"] - self.p["primary_delta"]:
            self._best_obj["primary"] = r["primary"]; improved = True
        return improved

    # — checkpoint selection (lexicographic, weight-free) —
    def _is_better(self, r) -> bool:
        b = self._best_sel
        if b is None:
            return True
        sd, pd = self.p["rate_delta"], self.p["primary_delta"]
        if r["success"] > b["success"] + sd: return True
        if r["success"] < b["success"] - sd: return False
        if r["crash"]   < b["crash"]   - sd: return True
        if r["crash"]   > b["crash"]   + sd: return False
        if r["primary"] < b["primary"] - pd: return True
        if r["primary"] > b["primary"] + pd: return False
        return r["reward"] > b["reward"]

    def _evaluate(self):
        predict = lambda obs: self.model.predict(obs, deterministic=True)[0]
        if self.ss is not None:
            try: self.ss.push_event(f"EVAL {self.mode.name}…", (200, 200, 120))
            except Exception: pass

        r = eval_policy(predict, self.mode, self.p["eval_episodes"], self.p["eval_seed"])
        r["steps"] = int(self.num_timesteps)
        r["score"] = diagnostic_score(r, self.p)
        self.evals.append(r)

        sel = "       "
        if self._is_better(r):
            self._best_sel = dict(r)
            self.model.save(self.best_path)
            sel = " *BEST*"
        self.model.save(self.latest_path)

        improved = self._register_improvement(r)
        self._no_improve = 0 if improved else self._no_improve + 1
        if r["success"] >= self.p["success_target"] and r["crash"] <= self.p["crash_target"]:
            self._target_hits += 1
        else:
            self._target_hits = 0

        print(f"[EXPERT-EVAL] {self.mode.name:<13} step={r['steps']:>8}  "
              f"succ={r['success']*100:5.1f}%  crash={r['crash']*100:5.1f}%  "
              f"R={r['reward']:+8.1f}  {r['primary_label']}={r['primary']:7.2f}  "
              f"smooth={r['smooth']:.4f}  noimp={self._no_improve}/{self.p['patience']}{sel}",
              flush=True)
        if self.ss is not None:
            try:
                self.ss.push_event(
                    f"{self.mode.name} {r['success']*100:.0f}%/{r['crash']*100:.0f}% "
                    f"R{r['reward']:+.0f}", (120, 200, 255))
            except Exception:
                pass

        if self._target_hits >= self.p["target_patience"]:
            self.should_stop = True
            self.stop_reason = (f"target met ({self._target_hits} consecutive evals: "
                                f"succ>={self.p['success_target']:.0%}, "
                                f"crash<={self.p['crash_target']:.0%})")
        elif self._no_improve >= self.p["patience"]:
            self.should_stop = True
            self.stop_reason = (f"plateau ({self._no_improve} evals with no "
                                f"meaningful multi-objective improvement)")

    def _on_training_start(self) -> None:
        # Baseline eval once (captures the BC starting point as initial best).
        if not self._started:
            self._started = True
            self._last_eval = int(self.num_timesteps)
            self._evaluate()

    def _on_step(self) -> bool:
        ts = int(self.num_timesteps)
        if ts - self._last_eval >= self.p["eval_interval"]:
            self._last_eval = ts
            self._evaluate()
            if self.should_stop:
                print(f"[EXPERT-EVAL] {self.mode.name} EARLY STOP: {self.stop_reason}",
                      flush=True)
                return False
        return True


def finalize_expert(save_prefix: str, mode) -> str:
    """Copy <prefix>_best.zip -> <prefix>.zip (the routed checkpoint). Falls back
    to <prefix>_latest.zip, then leaves any existing <prefix>.zip in place."""
    dest = save_prefix + ".zip"
    for src in (save_prefix + "_best.zip", save_prefix + "_latest.zip"):
        if os.path.exists(src):
            shutil.copyfile(src, dest)
            tag = "best" if src.endswith("_best.zip") else "latest (no best)"
            print(f"[EXPERT] {mode.name}: routed checkpoint <- {tag}  ({dest})",
                  flush=True)
            return dest
    print(f"[EXPERT] {mode.name}: no checkpoint to finalize at {save_prefix}_*.zip",
          flush=True)
    return dest
