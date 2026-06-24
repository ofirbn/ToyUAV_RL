"""
PPO training for the multi-mode fixed-wing UAV.

Edit config.txt, then run:  python main.py

Curriculum auto-progression
────────────────────────────
Set  curriculum=true  in config.txt.  Training starts at the phase named by
curriculum_phase (default: stabilize) and auto-advances through all 9 stages
as the rolling success rate exceeds 80 % over 100 consecutive episodes.

curriculum=false  keeps the old static single-phase behaviour.
"""

import math
import os
import time
import collections
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from envs.fixed_wing_env import FixedWingEnv
from sim.flight_modes import FlightMode, MODE_NAMES


# ══════════════════════════════════════════════════════ constants ══════════════

N_ENVS     = 4
CHUNK_SIZE = 1024   # timesteps per learn() call in train_visual


_EPISODE_WINDOW = 100


def _model_dir(cfg: dict) -> str:
    path = cfg.get("model", "models/latest.zip")
    return os.path.dirname(path) or "models"


def _latest_path(cfg: dict) -> str:
    return os.path.join(_model_dir(cfg), "latest")


def _best_reward_path(cfg: dict) -> str:
    return os.path.join(_model_dir(cfg), "best_reward")


def _best_success_path(cfg: dict) -> str:
    return os.path.join(_model_dir(cfg), "best_success")


def _curriculum_save_path(cfg: dict) -> str:
    return os.path.join(_model_dir(cfg), "curriculum_state.json")


# ── Curriculum manager factory ────────────────────────────────────────────────

def _make_curriculum(cfg: dict):
    """Return a CurriculumManager if curriculum=true, else None."""
    if cfg.get("curriculum", "false").lower() != "true":
        return None
    from sim.curriculum import CurriculumManager
    return CurriculumManager(
        start_stage=cfg.get("curriculum_phase", "stabilize"),
        save_path=_curriculum_save_path(cfg),
    )


# ── VecEnv factory ─────────────────────────────────────────────────────────────

def _make_env(cfg: dict, curriculum_manager=None) -> DummyVecEnv:
    # Parse optional emergency_recovery_enabled override from config
    er_raw = cfg.get("emergency_recovery_enabled", None)
    er_val = (er_raw.lower() == "true") if er_raw is not None else None

    env_kwargs: dict = {
        "training_mode":               True,
        "action_smooth_weight":        float(cfg.get("action_smooth_weight", 0.03)),
        "stall_speed":                 float(cfg.get("stall_speed",          6.0)),
        "emergency_recovery_enabled":  er_val,
    }
    if curriculum_manager is not None:
        env_kwargs["curriculum_manager"] = curriculum_manager
    else:
        env_kwargs["curriculum_phase"] = cfg.get("curriculum_phase", "mixed")

    return make_vec_env(
        FixedWingEnv,
        n_envs      = N_ENVS,
        vec_env_cls = DummyVecEnv,
        env_kwargs  = env_kwargs,
    )


def _build_model(env, cfg: dict, load_path: str) -> PPO:
    force_new  = cfg.get("force_new", "false").lower() == "true"
    bc_path    = cfg.get("init_from_bc", None)
    init_path  = cfg.get("init_from", None)

    # 1. Resume from latest checkpoint (normal flow)
    if not force_new and os.path.exists(load_path + ".zip"):
        try:
            model = PPO.load(load_path, env=env)
            print(f"[TRAIN] Loaded model from {load_path}.zip - continuing training.")
            return model
        except Exception as e:
            print(f"[TRAIN] Could not load model ({e}) — starting fresh.")

    elif force_new:
        print("[TRAIN] force_new=true — starting from scratch.")

    # 1b. Warm-start (seed) from an explicitly selected checkpoint — e.g. seed a
    #     new expert from models/latest.zip. Only reached when NOT resuming, so a
    #     half-trained expert is never overwritten. Takes priority over BC.
    if init_path:
        init_file = init_path if init_path.endswith(".zip") else init_path + ".zip"
        if os.path.exists(init_file):
            try:
                model = PPO.load(init_file[:-4], env=env)
                print(f"[TRAIN] Seeded from selected model {init_file}")
                return model
            except Exception as e:
                print(f"[TRAIN] Could not seed from {init_file} ({e}) — "
                      f"trying BC / fresh.")
        else:
            print(f"[TRAIN] init_from not found: {init_file} — trying BC / fresh.")

    # 2. Warm-start from behavior-cloned weights
    if bc_path is not None:
        bc_file = bc_path if bc_path.endswith(".zip") else bc_path + ".zip"
        if os.path.exists(bc_file):
            try:
                model = PPO.load(bc_path, env=env)
                print(f"[TRAIN] BC warm-start from {bc_file}")
                return model
            except Exception as e:
                print(f"[TRAIN] Could not load BC model ({e}) — starting fresh.")
        else:
            print(f"[TRAIN] BC path not found: {bc_file} — starting fresh.")

    # 3. Fresh model
    return PPO(
        "MlpPolicy",
        env,
        learning_rate = 3e-4,
        gamma         = 0.99,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        ent_coef      = 0.01,
        clip_range    = 0.2,
        policy_kwargs = dict(net_arch=[256, 256]),
        verbose       = 1,
    )


# ══════════════════════════════════════════════════════ callbacks ══════════════

class _BestModelCallback(BaseCallback):
    """Saves best_reward.zip and best_success.zip when rolling stats improve."""

    def __init__(self, best_reward_path: str, best_success_path: str,
                 save_every: int = 100_000):
        super().__init__()
        self._best_reward_path  = best_reward_path
        self._best_success_path = best_success_path
        self._save_every        = save_every

        self._best_mean_r  = -9999.0
        self._best_success = 0.0
        self._last_save    = 0

        self._ep_crash   = collections.deque(maxlen=_EPISODE_WINDOW)
        self._ep_success = collections.deque(maxlen=_EPISODE_WINDOW)

    def _on_step(self) -> bool:
        ts    = self.num_timesteps
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            crashed = info.get("crashed", False)
            self._ep_crash.append(1 if crashed else 0)
            self._ep_success.append(0 if crashed else 1)

        n = len(self._ep_crash)
        if n < 20:
            return True

        mean_r    = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer])) \
                    if self.model.ep_info_buffer else 0.0
        success_r = sum(self._ep_success) / n

        if mean_r > self._best_mean_r:
            self._best_mean_r = mean_r
            self.model.save(self._best_reward_path)

        if success_r > self._best_success:
            self._best_success = success_r
            self.model.save(self._best_success_path)

        if ts - self._last_save >= self._save_every:
            self.model.save(self._best_reward_path.replace("best_reward", "latest"))
            self._last_save = ts

        return True


class LogCallback(BaseCallback):
    LOG_EVERY = 10_000

    def __init__(self, best_reward_path: str, best_success_path: str,
                 save_every: int = 100_000):
        super().__init__()
        self._last_log = 0
        self._bmc = _BestModelCallback(best_reward_path, best_success_path, save_every)

    def init_callback(self, model):
        super().init_callback(model)
        self._bmc.init_callback(model)

    def _on_step(self) -> bool:
        self._bmc._on_step()
        ts = self.num_timesteps
        if ts - self._last_log >= self.LOG_EVERY:
            buf = self.model.ep_info_buffer
            if buf:
                mean_r = float(np.mean([ep['r'] for ep in buf]))
                mean_l = float(np.mean([ep['l'] for ep in buf]))
                # Read curriculum status from env0 if available
                env0   = self.model.env.envs[0].unwrapped
                curr   = getattr(env0, '_curriculum', None)
                curr_s = curr.status_str() if curr else ''
                print(f"[{ts:>9,}]  mean_ep_reward={mean_r:+8.2f}  "
                      f"mean_ep_len={mean_l:.0f}  {curr_s}")
            else:
                print(f"[{ts:>9,}]  (collecting episodes…)")
            self._last_log = ts
        return True


class _LogCapture:
    """
    Injected into SB3's logger.output_formats so we capture train/* metrics
    on every logger.dump() call — before name_to_value is cleared.
    This is necessary because dump() clears name_to_value before _on_step
    can read it during the next rollout collection phase.
    """
    def __init__(self, target: dict):
        self._target = target

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        for k, v in key_values.items():
            if k.startswith("train/"):
                short = k[6:]   # strip "train/" prefix
                try:
                    self._target[short] = float(v)
                except (TypeError, ValueError):
                    pass

    def close(self) -> None:
        pass


class LiveCallback(BaseCallback):
    """
    Updates shared_state every step so the pygame renderer stays live.
    Never calls predict(), env.step(), or render().
    """

    LOG_EVERY  = 10_000
    TRAJ_EVERY = 4

    def __init__(self, shared_state_module, cfg: dict, save_prefix: str = None):
        super().__init__()
        self._ss           = shared_state_module
        self._cfg          = cfg
        self._save_every   = int(cfg.get("save_every", 100_000))
        # When save_prefix is given (e.g. expert training -> models/experts/<mode>),
        # derive periodic/best checkpoints from it instead of the generic
        # <model_dir>/best_reward etc., so experts don't collide on shared names.
        if save_prefix:
            self._best_reward_path  = save_prefix + "_best_reward"
            self._best_success_path = save_prefix + "_best_success"
            self._latest_path       = save_prefix
        else:
            self._best_reward_path  = _best_reward_path(cfg)
            self._best_success_path = _best_success_path(cfg)
            self._latest_path       = _latest_path(cfg)

        self._last_log      = 0
        self._last_save     = 0
        self._episode_count = 0
        self._best_mean_r   = -9999.0
        self._best_success  = 0.0

        self._ep_crash   = collections.deque(maxlen=_EPISODE_WINDOW)
        self._ep_land    = collections.deque(maxlen=_EPISODE_WINDOW)
        self._ep_success = collections.deque(maxlen=_EPISODE_WINDOW)
        self._ep_stall   = collections.deque(maxlen=_EPISODE_WINDOW)

        self._mode_success = {m: collections.deque(maxlen=50) for m in FlightMode}
        self._mode_crash   = {m: collections.deque(maxlen=50) for m in FlightMode}

        self._fps_ts   = 0
        self._fps_time = time.monotonic()
        self._fps_cur  = 0.0
        self._traj_ctr = 0
        self._prev_mode = None

        # Live capture dict — populated by _LogCapture on every logger.dump()
        self._log_capture: dict = {}
        # Stable cache — holds the last non-zero training metrics for the HUD
        self._ppo_cache = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "explained_var": 0.0,
            "learning_rate": 3e-4, "training_iter": 0,
        }

    def init_callback(self, model) -> None:
        super().init_callback(model)
        # Inject capture writer so we get train/* values before name_to_value is cleared
        model.logger.output_formats.append(_LogCapture(self._log_capture))

    def _on_step(self) -> bool:
        env0 = self.model.env.envs[0].unwrapped
        s    = env0._state
        t    = env0._target
        curr = getattr(env0, '_curriculum', None)

        # FPS
        ts  = self.num_timesteps
        now = time.monotonic()
        dt  = now - self._fps_time
        if dt >= 1.0:
            self._fps_cur  = (ts - self._fps_ts) / dt
            self._fps_ts   = ts
            self._fps_time = now

        # PPO metrics — read from _log_capture which is populated by _LogCapture
        # on every logger.dump() (after each training update), so it's always current.
        lv = self._log_capture
        policy_loss = float(lv.get("policy_gradient_loss", 0.0))
        value_loss  = float(lv.get("value_loss",           0.0))
        entropy     = float(lv.get("entropy_loss",         0.0))
        approx_kl   = float(lv.get("approx_kl",           0.0))
        expl_var    = float(lv.get("explained_variance",   0.0))
        lr          = float(lv.get("learning_rate",        3e-4))
        train_iter  = int(float(lv.get("n_updates",        0)))

        # Update stable cache on new training data; use cache otherwise so HUD never
        # reverts to zeros between training updates
        if policy_loss != 0.0 or value_loss != 0.0:
            self._ppo_cache = {
                "policy_loss":   policy_loss,
                "value_loss":    value_loss,
                "entropy":       entropy,
                "approx_kl":     approx_kl,
                "explained_var": expl_var,
                "learning_rate": lr,
                "training_iter": train_iter,
            }
        policy_loss = self._ppo_cache["policy_loss"]
        value_loss  = self._ppo_cache["value_loss"]
        entropy     = self._ppo_cache["entropy"]
        approx_kl   = self._ppo_cache["approx_kl"]
        expl_var    = self._ppo_cache["explained_var"]
        lr          = self._ppo_cache["learning_rate"]
        train_iter  = self._ppo_cache["training_iter"]

        # Episode bookkeeping
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            self._episode_count += 1
            crashed = info.get("crashed",       False)
            landed  = info.get("landed",        False)
            stalled = info.get("stalled_this_ep", False)
            mode    = info.get("mode",          FlightMode.STABILIZE)

            self._ep_crash.append(1 if crashed else 0)
            self._ep_land.append( 1 if landed  else 0)
            self._ep_success.append(0 if crashed else 1)
            self._ep_stall.append( 1 if stalled else 0)

            if isinstance(mode, FlightMode):
                self._mode_success[mode].append(0 if crashed else 1)
                self._mode_crash[mode].append(  1 if crashed else 0)

            # Events (env 0 only)
            if i == 0:
                if crashed:
                    self._ss.push_event("CRASH", (255, 60, 60))
                elif landed:
                    self._ss.push_event("GOOD LANDING", (0, 255, 120))
                elif (info.get("episode_success") and
                      info.get("mode") == FlightMode.WAYPOINT):
                    self._ss.push_event("WAYPOINT REACHED", (0, 200, 255))

                # Curriculum stage advance notification
                if curr is not None and info.get("curriculum_advanced"):
                    self._ss.push_event(
                        f"STAGE -> {curr.stage_name.upper()}", (255, 215, 0))

        # Stall warning (env0)
        if s.airspeed < 6.0 and s.pos[2] > 10:
            self._ss.push_event("STALL WARNING", (255, 80, 0))

        # Mode switch (env0)
        cur_mode = int(env0._mode)
        if self._prev_mode is not None and cur_mode != self._prev_mode:
            self._ss.push_event(f"MODE -> {MODE_NAMES[cur_mode]}", (160, 200, 255))
        self._prev_mode = cur_mode

        # Reward breakdown
        breakdown = infos[0].get("reward_breakdown", {}) if infos else {}

        # Aggregate stats
        mean_r    = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer])) \
                    if self.model.ep_info_buffer else 0.0
        ep_reward = float(env0.episode_reward)
        n         = len(self._ep_crash)
        crash_r   = (sum(self._ep_crash)   / n) if n else 0.0
        land_r    = (sum(self._ep_land)    / n) if n else 0.0
        success_r = (sum(self._ep_success) / n) if n else 0.0
        stall_r   = (sum(self._ep_stall)   / n) if n else 0.0

        # Best model saving
        if n >= 20:
            if mean_r > self._best_mean_r:
                self._best_mean_r = mean_r
                os.makedirs(os.path.dirname(self._best_reward_path) or ".", exist_ok=True)
                self.model.save(self._best_reward_path)

            if success_r > self._best_success:
                self._best_success = success_r
                os.makedirs(os.path.dirname(self._best_success_path) or ".", exist_ok=True)
                self.model.save(self._best_success_path)

        # Periodic latest save + curriculum state save
        if ts - self._last_save >= self._save_every:
            os.makedirs(os.path.dirname(self._latest_path) or ".", exist_ok=True)
            self.model.save(self._latest_path)
            if curr is not None:
                curr.save()
            self._last_save = ts

        # Per-mode success rates
        def _rate(deq):
            return (sum(deq) / len(deq)) if deq else 0.0

        mode_rates = {
            "stabilize_success_rate":     _rate(self._mode_success[FlightMode.STABILIZE]),
            "altitude_hold_success_rate": _rate(self._mode_success[FlightMode.ALTITUDE_HOLD]),
            "heading_hold_success_rate":  _rate(self._mode_success[FlightMode.HEADING_HOLD]),
            "waypoint_success_rate":      _rate(self._mode_success[FlightMode.WAYPOINT]),
            "loiter_success_rate":        _rate(self._mode_success[FlightMode.LOITER]),
            "approach_success_rate":      _rate(self._mode_success[FlightMode.APPROACH]),
            "landing_success_rate":       _rate(self._mode_success[FlightMode.LANDING]),
            "recovery_success_rate":      _rate(self._mode_success[FlightMode.RECOVERY]),
        }

        # Curriculum display info.
        # Use env0._active_phase rather than curr.stage_name so that the
        # stabilize curriculum lock is reflected correctly in the HUD even
        # when the CurriculumManager loaded a later stage from a save file.
        if curr is not None:
            curr_phase        = getattr(env0, '_active_phase', curr.stage_name)
            curr_rate_val     = curr.success_rate()
            curr_stage_idx    = curr.stage_index
            curr_num_stages   = curr.num_stages
            mastery_locked    = not curr.is_mastered()
            mastery_failing   = curr.locked_criteria()
            mastery_details   = curr.mastery_details()
        else:
            curr_phase        = self._cfg.get("curriculum_phase", "mixed")
            curr_rate_val     = success_r
            curr_stage_idx    = 0
            curr_num_stages   = 1
            mastery_locked    = False
            mastery_failing   = []
            mastery_details   = {}

        # Mission segment
        mission_seg   = 0
        mission_total = 0
        if env0._mission is not None:
            mission_seg   = env0._mission.current_index
            mission_total = env0._mission.num_segments

        stall_warning = bool(infos[0].get("stall_warning", False)) if infos else False
        recovery_triggered = bool(infos[0].get("recovery_triggered", False)) if infos else False
        recovery_reason    = infos[0].get("recovery_reason", None) if infos else None

        patch: dict = {
            "pos":          s.pos.tolist(),
            "vel":          s.vel.tolist(),
            "pitch":        float(s.pitch),
            "roll":         float(s.roll),
            "yaw":          float(s.yaw),
            "pitch_rate":   float(s.pitch_rate),
            "roll_rate":    float(s.roll_rate),
            "yaw_rate":     float(s.yaw_rate),
            "throttle_pos":     float(s.throttle_pos),
            "throttle_command": float(infos[0].get("throttle_command", s.throttle_pos)) if infos else float(s.throttle_pos),
            "throttle_actual":  float(infos[0].get("throttle_actual",  s.throttle_pos)) if infos else float(s.throttle_pos),
            "throttle_delta":   float(infos[0].get("throttle_delta",   0.0)) if infos else 0.0,
            "target_airspeed":  float(infos[0].get("target_airspeed",  0.0)) if infos else 0.0,
            "airspeed_error":   float(infos[0].get("airspeed_error",   0.0)) if infos else 0.0,
            "altitude_error":   float(infos[0].get("altitude_error",   0.0)) if infos else 0.0,
            "vertical_speed":   float(infos[0].get("vertical_speed",   0.0)) if infos else 0.0,
            "airspeed":     float(s.airspeed),
            "stall_warning": stall_warning,
            "elevator":     float(env0._prev_action[0]),
            "aileron":      float(env0._prev_action[1]),
            "rudder":       float(env0._prev_action[2]),
            "mode":              int(env0._mode),
            "active_mode":       int(env0._mode),
            "commanded_mode":    int(env0._mode),
            "autonomous_switching_enabled": getattr(env0, '_autonomous_switching', True),
            "recovery_triggered": recovery_triggered,
            "recovery_reason":   recovery_reason,
            "mission_seg":  mission_seg,
            "mission_total": mission_total,
            "reward":           ep_reward,
            "reward_breakdown": breakdown,
            "timesteps":      ts,
            "episode_count":  self._episode_count,
            "mean_ep_reward": mean_r,
            "best_reward":    self._best_mean_r,
            "best_success_rate": self._best_success,
            "policy_loss":    policy_loss,
            "value_loss":     value_loss,
            "entropy":        entropy,
            "approx_kl":      approx_kl,
            "explained_var":  expl_var,
            "learning_rate":  lr,
            "training_iter":  train_iter,
            "fps":            self._fps_cur,
            "crash_rate":     crash_r,
            "landing_rate":   land_r,
            "success_rate":   success_r,
            "stall_rate":     stall_r,
            # Curriculum info
            "curriculum_phase":      curr_phase,
            "curriculum_rate":       curr_rate_val,
            "curriculum_stage_idx":  curr_stage_idx,
            "curriculum_num_stages": curr_num_stages,
            "mastery_locked":        mastery_locked,
            "mastery_failing":       mastery_failing,
            "mastery_details":       mastery_details,
            # Waypoint FSM telemetry — sourced from env0 info so renderer shows live state
            "wp_state":             str(infos[0].get("wp_state",            "approaching")) if infos else "approaching",
            "wp_capture_event":     bool(infos[0].get("wp_capture_event",   False))         if infos else False,
            "wp_captured_count":    int(infos[0].get("wp_captured_count",   0))             if infos else 0,
            "wp_transition_timer":  int(infos[0].get("wp_transition_timer", 0))             if infos else 0,
            "wp_curr_dist":         float(infos[0].get("wp_curr_dist",      -1.0))          if infos else -1.0,
            "wp_arrived":           bool(infos[0].get("wp_arrived",         False))         if infos else False,
            "wp_mission_idx":       int(infos[0].get("wp_mission_idx",      -1))            if infos else -1,
            "wp_leg_start_dist":    float(infos[0].get("wp_leg_start_dist", 0.0))           if infos else 0.0,
            "wp_capture_eligible":  bool(infos[0].get("wp_capture_eligible", True))         if infos else True,
            "wp_cooldown_active":   bool(infos[0].get("wp_cooldown_active",  False))        if infos else False,
            "wp_capture_threshold": float(infos[0].get("wp_capture_threshold", 20.0))      if infos else 20.0,
            # Loiter orbit quality diagnostics
            "loiter_radius_current": float(infos[0].get("loiter_radius_current", 0.0)) if infos else 0.0,
            "loiter_radius_desired": float(infos[0].get("loiter_radius_desired", 0.0)) if infos else 0.0,
            "loiter_radial_error":   float(infos[0].get("loiter_radial_error",   0.0)) if infos else 0.0,
            "loiter_radial_vel":     float(infos[0].get("loiter_radial_vel",     0.0)) if infos else 0.0,
            "loiter_radius_std":     float(infos[0].get("loiter_radius_std",     0.0)) if infos else 0.0,
            "loiter_tang_ratio":     float(infos[0].get("loiter_tang_ratio",     0.0)) if infos else 0.0,
            "ready": True,
        }
        patch.update(mode_rates)

        if t is not None:
            patch["target_position"] = t.position.tolist()
            patch["target_heading"]  = float(t.heading)
            patch["target_altitude"] = float(t.altitude)
            patch["target_radius"]   = float(t.radius)

        self._ss.update(patch)

        # History pushes
        self._traj_ctr += 1
        if self._traj_ctr >= self.TRAJ_EVERY:
            self._traj_ctr = 0
            self._ss.push_history("trajectory", s.pos.tolist())

        if dones is not None and dones[0]:
            self._ss.push_history("reward",      ep_reward)
            self._ss.push_history("mean_reward",  mean_r)
            self._ss.push_history("crash",   float(self._ep_crash[-1])   if self._ep_crash   else 0.0)
            self._ss.push_history("success", float(self._ep_success[-1]) if self._ep_success else 0.0)
            self._ss.push_history("stall",   float(self._ep_stall[-1])   if self._ep_stall   else 0.0)

        if policy_loss != 0.0:
            self._ss.push_history("policy_loss", abs(policy_loss))
            self._ss.push_history("value_loss",  value_loss)

        # Console log
        if ts - self._last_log >= self.LOG_EVERY:
            if self.model.ep_info_buffer:
                curr_str = curr.status_str() if curr else f"phase={curr_phase}"
                print(f"[{ts:>9,}]  mean_ep_reward={mean_r:+8.2f}  "
                      f"crash={crash_r:.0%}  success={success_r:.0%}  "
                      f"stall={stall_r:.0%}  fps={self._fps_cur:.0f}  "
                      f"{curr_str}")
            else:
                print(f"[{ts:>9,}]  (collecting episodes…)")
            self._last_log = ts

        return True


# ══════════════════════════════════════════════════════ train ═════════════════

def train(cfg: dict):
    timesteps         = int(cfg.get("timesteps", 300_000))
    save_every        = int(cfg.get("save_every", 100_000))
    latest_path       = _latest_path(cfg)
    best_reward_path  = _best_reward_path(cfg)
    best_success_path = _best_success_path(cfg)

    os.makedirs(_model_dir(cfg), exist_ok=True)

    curriculum_mgr = _make_curriculum(cfg)

    bc_path      = cfg.get("init_from_bc", None)

    print(f"\n{'='*55}")
    print(f"  ToyUAV RL — Training")
    print(f"  Timesteps  : {timesteps:,}")
    print(f"  Envs       : {N_ENVS}  (DummyVecEnv)")
    if curriculum_mgr is not None:
        print(f"  Curriculum : AUTO  (start={curriculum_mgr.stage_name})")
    else:
        print(f"  Phase      : {cfg.get('curriculum_phase', 'mixed')}  (static)")
    if bc_path:
        print(f"  BC init    : {bc_path}")
    print(f"  Save path  : {latest_path}.zip")
    print(f"{'='*55}\n")

    env   = _make_env(cfg, curriculum_mgr)
    model = _build_model(env, cfg, latest_path)

    callback = LogCallback(best_reward_path, best_success_path, save_every)

    model.learn(
        total_timesteps     = timesteps,
        reset_num_timesteps = False,
        callback            = callback,
    )

    model.save(latest_path)
    if curriculum_mgr is not None:
        curriculum_mgr.save()
    print(f"\n[TRAIN] Saved -> {latest_path}.zip")
    print("[TRAIN] Done.")


# ══════════════════════════════════════════════════════ train_visual ══════════

def train_visual(cfg: dict, save_path: str = None, skip_config_screen: bool = False,
                 expert_jobs: list = None):
    """
    PPO training with live pygame telemetry dashboard.

    Flow:
      1. Config screen  — user sets teacher/curriculum params, clicks START.
      2. Training phase — background thread runs PPO; main thread drives pygame.

    save_path (optional): explicit checkpoint path (without .zip) for the final
        model.save(); defaults to <model_dir>/latest. Used by expert training to
        save to models/experts/<mode>.
    skip_config_screen (optional): when True, skip the in-pygame config screen
        and use cfg as-is (the caller, e.g. the tkinter launcher, already
        configured it).
    expert_jobs (optional): list of {label, phase, save_path, init_from,
        init_from_bc} dicts. When given, the training thread trains each job
        sequentially in this one window (used by visual "train all experts"),
        each for `timesteps` steps, saving to its own save_path. Overrides the
        single-model path.
    """
    import sys
    import threading
    import pygame
    import shared_state as ss
    from render.pygame_renderer import Renderer

    W, H = 1600, 950

    pygame.init()
    pygame.display.set_mode((W, H))
    pygame.display.set_caption("ToyUAV RL — Live Training Dashboard")
    clock    = pygame.time.Clock()
    renderer = Renderer(W, H)
    font     = pygame.font.SysFont("Consolas", 26)

    # ── Phase 1: pre-training config screen ───────────────────────────────────
    if not skip_config_screen:
        renderer.init_config_screen(cfg)
        cfg_action = None
        while cfg_action != "start":
            for event in pygame.event.get():
                result = renderer.handle_config_event(event)
                if result in ("start", "quit"):
                    cfg_action = result
                    break
            if cfg_action == "quit":
                pygame.quit()
                sys.exit(0)
            if cfg_action != "start":
                renderer.render_config_screen()
            clock.tick(60)

        # Merge user choices into cfg and write back to config.txt
        user_settings = renderer.get_config_draft()
        renderer.save_config_to_file(user_settings)
        cfg = {**cfg, **user_settings}

    # ── Phase 2: training setup (uses the now-final cfg) ─────────────────────
    timesteps   = int(cfg.get("timesteps", 300_000))
    latest_path = save_path or _latest_path(cfg)
    os.makedirs(os.path.dirname(latest_path) or _model_dir(cfg), exist_ok=True)

    curriculum_mgr = _make_curriculum(cfg)
    bc_path_vis    = cfg.get("init_from_bc", None)

    print(f"\n{'='*55}")
    print(f"  ToyUAV RL — Live Training Dashboard")
    print(f"  Timesteps  : {timesteps:,}")
    print(f"  Envs       : {N_ENVS}  (DummyVecEnv)")
    print(f"  Chunk      : {CHUNK_SIZE} steps/learn()")
    if curriculum_mgr is not None:
        print(f"  Curriculum : AUTO  (start={curriculum_mgr.stage_name})")
    else:
        print(f"  Phase      : {cfg.get('curriculum_phase', 'mixed')}  (static)")
    if bc_path_vis:
        print(f"  BC init    : {bc_path_vis}")
    print(f"  Window     : {W}×{H}")
    if expert_jobs:
        print(f"  Experts    : {len(expert_jobs)} (sequential)  "
              f"-> {', '.join(j['label'] for j in expert_jobs)}")
    else:
        print(f"  Save path  : {latest_path}.zip")
    print(f"{'='*55}\n")

    stop_event = threading.Event()

    def _train_one(load_path, env_cfg, save_to, save_prefix):
        """Build + train a single model for `timesteps`, rendering live."""
        env      = _make_env(env_cfg, None if expert_jobs else curriculum_mgr)
        model    = _build_model(env, env_cfg, load_path)
        callback = LiveCallback(ss, cfg, save_prefix=save_prefix)
        remaining = timesteps
        while remaining > 0 and not stop_event.is_set():
            chunk = min(CHUNK_SIZE, remaining)
            model.learn(
                total_timesteps     = chunk,
                reset_num_timesteps = False,
                callback            = callback,
            )
            remaining -= chunk
        model.save(save_to)
        try:
            env.close()
        except Exception:
            pass

    def _training_thread():
        if expert_jobs:
            n = len(expert_jobs)
            for i, job in enumerate(expert_jobs):
                if stop_event.is_set():
                    break
                ss.push_event(f"TRAINING {job['label']} ({i+1}/{n})", (120, 200, 255))
                print(f"\n[TRAIN] === Expert {i+1}/{n}: {job['label']} "
                      f"(phase='{job['phase']}') -> {job['save_path']}.zip ===")
                job_cfg = {**cfg, "curriculum": "false",
                           "curriculum_phase": job["phase"],
                           "model": job["save_path"] + ".zip"}
                if job.get("init_from"):
                    job_cfg["init_from"] = job["init_from"]
                if job.get("init_from_bc"):
                    job_cfg["init_from_bc"] = job["init_from_bc"]
                else:
                    job_cfg.pop("init_from_bc", None)
                _train_one(job["save_path"], job_cfg, job["save_path"],
                           job["save_path"])
                print(f"[TRAIN] Saved expert -> {job['save_path']}.zip")
                ss.push_event(f"DONE {job['label']}", (0, 255, 120))
            print("\n[TRAIN] All experts done.")
            ss.update({"training_done": True})
            ss.push_event("ALL EXPERTS COMPLETE", (0, 255, 120))
            return

        _train_one(latest_path, cfg, save_path or latest_path, save_path)
        if curriculum_mgr is not None:
            curriculum_mgr.save()
        print(f"\n[TRAIN] Saved -> {latest_path}.zip")
        print("[TRAIN] Done.")
        ss.update({"training_done": True})
        ss.push_event("TRAINING COMPLETE", (0, 255, 120))

    thread = threading.Thread(target=_training_thread, daemon=True)
    thread.start()

    print("[TRAIN] Window open. F1-F4 = camera modes. ESC = quit.")

    # ── Phase 3: dashboard event loop ─────────────────────────────────────────
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                thread.join(timeout=3.0)
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    stop_event.set()
                    thread.join(timeout=3.0)
                    pygame.quit()
                    sys.exit(0)
                elif k == pygame.K_F1: ss.update({"camera_mode": 0})
                elif k == pygame.K_F2: ss.update({"camera_mode": 1})
                elif k == pygame.K_F3: ss.update({"camera_mode": 2})
                elif k == pygame.K_F4: ss.update({"camera_mode": 3})

        state = ss.read()
        if state.get("ready"):
            renderer.render_state(state)
        else:
            scr = pygame.display.get_surface()
            scr.fill((5, 8, 22))
            msg = font.render("Initializing PPO training…", True, (180, 180, 255))
            scr.blit(msg, (W // 2 - msg.get_width() // 2, H // 2 - 16))
            sub = font.render("F1 Chase  F2 Side  F3 Top  F4 Cockpit  ESC Quit",
                              True, (80, 80, 120))
            scr.blit(sub, (W // 2 - sub.get_width() // 2, H // 2 + 20))
            pygame.display.flip()

        clock.tick(60)


# ══════════════════════════════════════════════════════ pipeline ══════════════

_ALL_RECORD_PHASES = [
    "stabilize", "altitude_hold", "heading_hold",
    "waypoint", "loiter", "approach", "landing",
]


def _record_phase_visual(cfg: dict, ss) -> str:
    """
    Phase 1: teacher flies and records demos; writes shared_state each step.
    Records N episodes per phase for every selected phase, saving one combined .npz.
    """
    import time
    from envs.fixed_wing_env import FixedWingEnv
    from controllers.teacher_autopilot import TeacherAutopilot

    record_phases_cfg = cfg.get("record_phases", "all")
    if record_phases_cfg == "all":
        phases_to_record = _ALL_RECORD_PHASES
    else:
        phases_to_record = [record_phases_cfg]

    n_ep     = int(cfg.get("record_episodes", 100))   # per phase
    seed     = int(cfg.get("record_seed", 0))
    demo_dir = cfg.get("demo_dir", "results/demos")
    teacher  = TeacherAutopilot()

    all_obs, all_actions, all_phase_ids, all_rewards, all_dones = [], [], [], [], []
    total_steps = 0
    total_episodes = len(phases_to_record) * n_ep
    ep_global = 0
    _VIS_EVERY = 10

    for phase in phases_to_record:
        # Always create a curriculum manager so the env uses phase-appropriate
        # difficulty (±7° roll for stabilize, etc.) instead of _DEFAULT_DIFF (±57°).
        from sim.curriculum import CurriculumManager
        curriculum_mgr = CurriculumManager(start_stage=phase,
                                           save_path=None)
        env = FixedWingEnv(training_mode=True,
                           curriculum_phase=phase,
                           curriculum_manager=curriculum_mgr)
        phase_crashes = phase_success = 0

        for ep in range(n_ep):
            obs, _ = env.reset(seed=(seed + ep_global) if seed >= 0 else None)
            done = truncated = False

            while not (done or truncated):
                s = env._state
                t = env._target
                action = teacher.act(obs, phase, target=t)
                next_obs, reward, done, truncated, info = env.step(action)

                all_obs.append(obs.copy())
                all_actions.append(action.copy())
                all_phase_ids.append(int(env._mode))
                all_rewards.append(float(reward))
                all_dones.append(bool(done or truncated))

                if total_steps % _VIS_EVERY == 0:
                    patch = {
                        "pos":          s.pos.tolist(),
                        "vel":          s.vel.tolist(),
                        "pitch":        float(s.pitch),
                        "roll":         float(s.roll),
                        "yaw":          float(s.yaw),
                        "pitch_rate":   float(s.pitch_rate),
                        "roll_rate":    float(s.roll_rate),
                        "yaw_rate":     float(s.yaw_rate),
                        "throttle_pos": float(s.throttle_pos),
                        "airspeed":     float(s.airspeed),
                        "elevator":     float(action[0]),
                        "aileron":      float(action[1]),
                        "rudder":       float(action[2]),
                        "mode":         int(env._mode),
                        "reward":       float(reward),
                        "episode_count": ep_global + 1,
                        "timesteps":    total_steps,
                        "success_rate": phase_success / max(ep, 1),
                        "crash_rate":   phase_crashes / max(ep, 1),
                        "pipeline_phase_label":
                            f"[1/3] RECORDING  {phase}  "
                            f"ep {ep+1}/{n_ep}  "
                            f"({phases_to_record.index(phase)+1}/{len(phases_to_record)} phases)",
                        "ready": True,
                    }
                    if t is not None:
                        patch.update({
                            "target_position": t.position.tolist(),
                            "target_heading":  float(t.heading),
                            "target_altitude": float(t.altitude),
                            "target_radius":   float(t.radius),
                        })
                    ss.update(patch)

                traj_ctr = total_steps % 4
                if traj_ctr == 0:
                    ss.push_history("trajectory", s.pos.tolist())

                time.sleep(0)
                obs = next_obs
                total_steps += 1

            crashed = info.get("crashed", False)
            phase_crashes += int(crashed)
            phase_success += int(not crashed)
            ep_global += 1
            if crashed:
                ss.push_event(f"CRASH ({phase})", (255, 60, 60))

        env.close()
        print(f"[PIPELINE] {phase}: {n_ep} eps, {phase_success} ok, {phase_crashes} crashes")
        ss.push_event(f"{phase} done", (0, 200, 120))

    os.makedirs(demo_dir, exist_ok=True)
    tag = "all" if len(phases_to_record) > 1 else phases_to_record[0]
    idx = 1
    while os.path.exists(os.path.join(demo_dir, f"{tag}_demo_{idx:03d}.npz")):
        idx += 1
    path = os.path.join(demo_dir, f"{tag}_demo_{idx:03d}.npz")
    np.savez(path,
             obs=np.array(all_obs,        dtype=np.float32),
             actions=np.array(all_actions,    dtype=np.float32),
             phases=np.array(all_phase_ids,   dtype=np.int32),
             rewards=np.array(all_rewards,    dtype=np.float32),
             dones=np.array(all_dones,        dtype=bool))
    print(f"[PIPELINE] Demos saved -> {path}  ({total_steps:,} steps, {ep_global} episodes)")
    return path


def _bc_phase_visual(cfg: dict, demo_path: str, out_path: str, ss) -> str:
    """Phase 2: behavioral cloning; writes loss to shared_state each epoch."""
    import torch
    import torch.nn.functional as F
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from envs.fixed_wing_env import FixedWingEnv
    from ml.train_behavior_clone import load_demos

    phase      = cfg.get("curriculum_phase", "stabilize")
    epochs     = int(cfg.get("bc_epochs",     60))
    batch_size = int(cfg.get("bc_batch_size", 512))
    lr         = float(cfg.get("bc_lr",       1e-3))
    val_frac   = 0.10

    obs_all, act_all = load_demos([demo_path])
    n     = len(obs_all)
    perm  = np.random.permutation(n)
    obs_all = obs_all[perm]; act_all = act_all[perm]
    split   = int(n * (1.0 - val_frac))
    obs_tr, act_tr = obs_all[:split], act_all[:split]
    obs_vl, act_vl = obs_all[split:], act_all[split:]

    print(f"[PIPELINE] BC: {n:,} steps  train {split:,}  val {n-split:,}")

    vec_env = DummyVecEnv(
        [lambda: FixedWingEnv(training_mode=True, curriculum_phase=phase)])
    model  = PPO("MlpPolicy", vec_env,
                 policy_kwargs=dict(net_arch=[256, 256]), verbose=0)
    policy = model.policy
    optim  = torch.optim.Adam(policy.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    obs_t_tr = torch.FloatTensor(obs_tr); act_t_tr = torch.FloatTensor(act_tr)
    obs_t_vl = torch.FloatTensor(obs_vl); act_t_vl = torch.FloatTensor(act_vl)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        policy.train()
        perm_ep = torch.randperm(len(obs_t_tr))
        ep_losses = []
        for start in range(0, len(obs_t_tr), batch_size):
            idx  = perm_ep[start:start + batch_size]
            dist = policy.get_distribution(obs_t_tr[idx])
            pred = dist.distribution.mean
            loss = F.mse_loss(pred, act_t_tr[idx])
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optim.step()
            ep_losses.append(loss.item())
        sched.step()

        policy.eval()
        with torch.no_grad():
            val_pred = policy.get_distribution(obs_t_vl).distribution.mean
            val_loss = F.mse_loss(val_pred, act_t_vl).item()

        train_loss = float(np.mean(ep_losses))
        ss.update({
            "pipeline_phase_label":
                f"[2/3] BEHAVIOR CLONING  epoch {epoch}/{epochs}  "
                f"train {train_loss:.4f}  val {val_loss:.4f}",
            "policy_loss":   train_loss,
            "value_loss":    val_loss,
            "timesteps":     epoch,
            "episode_count": epoch,
            "ready":         True,
        })
        ss.push_history("policy_loss", abs(train_loss))
        ss.push_history("value_loss",  val_loss)

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            model.save(out_path)

    vec_env.close()
    print(f"[PIPELINE] BC done  val MSE {best_val:.4f}  -> {out_path}.zip")
    return out_path


def train_pipeline_visual(cfg: dict):
    """
    Full teacher→BC→PPO pipeline with live pygame visualization.

    Phase 1: Teacher autopilot flies and records demos (shown in 3D view).
    Phase 2: Behavior cloning from those demos (loss shown in graphs).
    Phase 3: PPO fine-tunes from BC weights (full telemetry dashboard).
    """
    import sys
    import threading
    import pygame
    import shared_state as ss
    from render.pygame_renderer import Renderer

    W, H = 1600, 950
    pygame.init()
    pygame.display.set_mode((W, H))
    pygame.display.set_caption("ToyUAV RL — Training Pipeline")
    clock    = pygame.time.Clock()
    renderer = Renderer(W, H)
    font_msg = pygame.font.SysFont("Consolas", 22)

    phase    = cfg.get("curriculum_phase", "stabilize")
    demo_dir = cfg.get("demo_dir", "results/demos")
    _demo_base = os.path.splitext(os.path.basename(cfg.get("demo_path", "")))[0]
    _phase_tag = _demo_base.split("_demo_")[0] if "_demo_" in _demo_base else phase
    bc_out   = os.path.join("models", "bc", f"{_phase_tag}_bc")

    def _handle_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit(0)
                elif k == pygame.K_F1: ss.update({"camera_mode": 0})
                elif k == pygame.K_F2: ss.update({"camera_mode": 1})
                elif k == pygame.K_F3: ss.update({"camera_mode": 2})
                elif k == pygame.K_F4: ss.update({"camera_mode": 3})

    def _run_bg(fn):
        """Run fn() in a background thread; drive pygame until it finishes."""
        result  = [None]
        done_ev = threading.Event()
        def _worker():
            result[0] = fn()
            done_ev.set()
        threading.Thread(target=_worker, daemon=True).start()
        while not done_ev.is_set():
            _handle_events()
            state = ss.read()
            if state.get("ready"):
                renderer.render_state(state)   # banner drawn inside, single flip
            else:
                scr = pygame.display.get_surface()
                scr.fill((5, 8, 22))
                lbl = str(state.get("pipeline_phase_label", "Initializing…"))
                msg = font_msg.render(lbl, True, (180, 180, 255))
                scr.blit(msg, (W // 2 - msg.get_width() // 2, H // 2))
                pygame.display.flip()
            clock.tick(60)
        return result[0]

    # ── Phase 1: record (or reuse existing demos) ────────────────────────────
    existing_demo = cfg.get("demo_path", "").strip()
    if existing_demo and os.path.exists(existing_demo):
        demo_path = existing_demo
        print(f"\n{'='*55}")
        print(f"  PIPELINE Phase 1/3 — SKIPPED (reusing existing demos)")
        print(f"  Demo: {demo_path}")
        print(f"{'='*55}\n")
        ss.update({"pipeline_phase_label": f"[1/3] SKIPPED — using {os.path.basename(demo_path)}",
                   "ready": True})
        ss.push_event(f"Reusing: {os.path.basename(demo_path)}", (0, 255, 120))
    else:
        print(f"\n{'='*55}")
        print(f"  PIPELINE Phase 1/3 — Recording teacher demos")
        print(f"  Phase: {phase}  Episodes: {cfg.get('record_episodes', 500)}")
        print(f"{'='*55}\n")
        demo_path = _run_bg(lambda: _record_phase_visual(cfg, ss))
        ss.push_event("RECORDING COMPLETE", (0, 255, 120))

    # ── Phase 2: behavior clone (or reuse existing BC model) ─────────────────
    existing_bc = cfg.get("bc_model_path", "").strip()
    existing_bc_base = existing_bc[:-4] if existing_bc.endswith(".zip") else existing_bc
    if existing_bc and os.path.exists(existing_bc):
        bc_out = existing_bc_base
        print(f"\n{'='*55}")
        print(f"  PIPELINE Phase 2/3 — SKIPPED (reusing existing BC model)")
        print(f"  BC model: {existing_bc}")
        print(f"{'='*55}\n")
        ss.update({"pipeline_phase_label": f"[2/3] SKIPPED — using {os.path.basename(existing_bc)}",
                   "ready": True})
        ss.push_event(f"BC reused: {os.path.basename(existing_bc)}", (0, 255, 120))
    else:
        print(f"\n{'='*55}")
        print(f"  PIPELINE Phase 2/3 — Behavior cloning")
        print(f"  Demo: {demo_path}  Epochs: {cfg.get('bc_epochs', 60)}")
        print(f"{'='*55}\n")
        _run_bg(lambda: _bc_phase_visual(cfg, demo_path, bc_out, ss))
        ss.push_event("BC COMPLETE", (0, 255, 120))

    # ── Phase 3: PPO fine-tune ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PIPELINE Phase 3/3 — PPO fine-tuning from BC weights")
    print(f"  BC model: {bc_out}.zip")
    print(f"{'='*55}\n")

    cfg         = {**cfg, "init_from_bc": bc_out}
    timesteps   = int(cfg.get("timesteps", 300_000))
    latest_path = _latest_path(cfg)
    os.makedirs(_model_dir(cfg), exist_ok=True)
    curriculum_mgr = _make_curriculum(cfg)
    stop_event     = threading.Event()

    ss.update({"pipeline_phase_label": "[3/3] PPO FINE-TUNING — student learns from BC weights"})

    def _ppo_thread():
        env      = _make_env(cfg, curriculum_mgr)
        model    = _build_model(env, cfg, latest_path)
        callback = LiveCallback(ss, cfg)
        remaining = timesteps
        while remaining > 0 and not stop_event.is_set():
            chunk = min(CHUNK_SIZE, remaining)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                        callback=callback)
            remaining -= chunk
        model.save(latest_path)
        if curriculum_mgr is not None:
            curriculum_mgr.save()
        ss.update({"training_done": True})
        ss.push_event("TRAINING COMPLETE", (0, 255, 120))
        print(f"\n[PIPELINE] Done. Model -> {latest_path}.zip")

    threading.Thread(target=_ppo_thread, daemon=True).start()

    while True:
        _handle_events()
        state = ss.read()
        if state.get("ready"):
            renderer.render_state(state)   # banner drawn inside, single flip
        else:
            scr = pygame.display.get_surface()
            scr.fill((5, 8, 22))
            msg = font_msg.render("Initializing PPO…", True, (180, 180, 255))
            scr.blit(msg, (W // 2 - msg.get_width() // 2, H // 2))
            pygame.display.flip()
        clock.tick(60)


# legacy alias
def train_live(cfg_or_ts=None):
    if isinstance(cfg_or_ts, dict):
        train_visual(cfg_or_ts)
    else:
        ts = int(cfg_or_ts) if cfg_or_ts else 300_000
        train_visual({"timesteps": ts})


# ══════════════════════════════════════════════════════ entry ══════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",      type=int,   default=300_000)
    parser.add_argument("--phase",          type=str,   default="mixed",
                        help="curriculum_phase (e.g. waypoint)")
    parser.add_argument("--init-from-bc",   type=str,   default=None,
                        dest="init_from_bc",
                        help="Path to BC .zip for warm-start (without .zip ext)")
    args = parser.parse_args()

    cfg: dict = {
        "timesteps":       str(args.timesteps),
        "curriculum_phase": args.phase,
    }
    if args.init_from_bc:
        cfg["init_from_bc"] = args.init_from_bc

    train(cfg)
