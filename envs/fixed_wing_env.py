"""
FixedWingEnv — multi-mode autonomous fixed-wing UAV environment.

Observation (31-D):
    [0:3]   pos_x, pos_y, pos_z          / 500
    [3:6]   vel_x, vel_y, vel_z           / 20
    [6:9]   roll, pitch, yaw              (radians, ±π)
    [9:12]  pitch_rate, roll_rate, yaw_rate / 5
    [12]    airspeed                       / 20
    [13:16] target_dx, dy, dz             / 500  (target - pos)
    [16]    target_distance               / 500
    [17]    target_heading_error          / π
    [18]    target_altitude_error         / 100
    [19:27] mode one-hot                  (8 values)
    [27:31] prev_action: elevator, aileron, rudder, throttle

Action (4-D, continuous):
    [0] elevator  [-1, 1]
    [1] aileron   [-1, 1]
    [2] rudder    [-1, 1]
    [3] throttle  [ 0, 1]
"""

import collections
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aircraft import AircraftState, AircraftPhysics
from sim.flight_modes import FlightMode, NUM_MODES
from sim.mission_manager import MissionManager, TargetInfo
from sim.rewards import compute_reward
from sim.curriculum import (get_active_modes, StageDifficulty, get_emergency_recovery_default,
                            get_locked_mode, get_phase_weighted_modes, EpisodeData)
from sim.crash_diagnostics import classify_crash, save_crash_report, CrashType


_CRUISE_SPEED  = 10.0
_CRUISE_THR    = 0.55   # physical trim at ~12 m/s with increased drag
_ALT_HOLD_THR  = 0.60   # physical trim at 14 m/s target airspeed
_ALT_HOLD_TGT_AIRSPEED = 14.0
_LOITER_SPEED  = 15.0   # m/s — cruise at healthy energy state, low AoA
_LOITER_THR    = 0.65   # throttle trim for 15 m/s loiter cruise
_MAX_ALTITUDE  = 800.0
_MIN_ALTITUDE  = -1.0

_DEFAULT_DIFF  = StageDifficulty()    # fallback when no curriculum manager


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def _vel_from_heading(yaw: float, speed: float, vz: float = 0.0) -> np.ndarray:
    return np.array([math.sin(yaw) * speed, -math.cos(yaw) * speed, vz],
                    dtype=float)


def _envelope_penalties(state) -> tuple:
    """Returns (total_penalty, breakdown_dict). All terms are positive penalties."""
    pen = 0.0
    bd  = {}

    roll_excess = max(0.0, abs(state.roll) - math.radians(60))
    if roll_excess > 0:
        v = roll_excess / math.radians(30) * 0.3
        bd["env_roll"]  = round(-v, 4)
        pen += v

    pitch_excess = max(0.0, abs(state.pitch) - math.radians(45))
    if pitch_excess > 0:
        v = pitch_excess / math.radians(30) * 0.3
        bd["env_pitch"] = round(-v, 4)
        pen += v

    rrate_excess = max(0.0, abs(state.roll_rate)  - math.radians(90))
    if rrate_excess > 0:
        v = rrate_excess * 0.1
        bd["env_rrate"] = round(-v, 4)
        pen += v

    prate_excess = max(0.0, abs(state.pitch_rate) - math.radians(60))
    if prate_excess > 0:
        v = prate_excess * 0.1
        bd["env_prate"] = round(-v, 4)
        pen += v

    yrate_excess = max(0.0, abs(state.yaw_rate)   - math.radians(45))
    if yrate_excess > 0:
        v = yrate_excess * 0.1
        bd["env_yrate"] = round(-v, 4)
        pen += v

    sink_excess = max(0.0, -state.vel[2] - 5.0)
    if sink_excess > 0:
        v = sink_excess * 0.2
        bd["env_sink"]  = round(-v, 4)
        pen += v

    return pen, bd


class FixedWingEnv(gym.Env):
    metadata = {"render_modes": []}
    MAX_STEPS = 600

    def __init__(self,
                 mission_path:                 str   = None,
                 training_mode:                bool  = True,
                 curriculum_phase:             str   = 'mixed',
                 curriculum_manager                         = None,
                 action_smooth_weight:         float = 0.06,
                 stall_speed:                  float = 6.0,
                 emergency_recovery_enabled          = None):
        super().__init__()

        self._training_mode        = training_mode
        self._curriculum_phase     = curriculum_phase
        self._curriculum           = curriculum_manager   # CurriculumManager | None
        self._action_smooth_weight = action_smooth_weight
        self._stall_speed          = stall_speed
        # None = auto-derive from stage/phase; True/False = explicit override
        self._er_override          = (None if emergency_recovery_enabled is None
                                      else bool(emergency_recovery_enabled))
        # Updated each episode to the effective curriculum phase name
        self._active_phase         = curriculum_phase
        # Tracks previous mode for mode-change logging
        self._prev_mode: FlightMode = None

        self._phys           = AircraftPhysics()
        self._state: AircraftState = None

        self._mission = MissionManager(mission_path) if mission_path else None
        self._target: TargetInfo  = None
        self._mode:   FlightMode  = FlightMode.STABILIZE
        self._forced_mode: FlightMode = None

        self._prev_pos            = np.zeros(3)
        self._prev_action         = np.zeros(4)
        self._prev_throttle_cmd   = 0.0   # effective command after deadband
        self._prev_throttle_actual= 0.0   # state.throttle_pos from previous step
        self.steps                = 0
        self.episode_reward = 0.0
        self._stalled_this_ep  = False
        self._waypoint_reached = False
        self._wp_arrival_steps = 0       # steps spent within arrival radius
        self._prev_mission_wp_idx = -1   # for transition-event detection
        self._prev_wp_dist: float    = 0.0   # leg-local prev distance to active WP
        self._leg_start_dist: float  = 0.0   # distance to WP at leg start
        self._wp_leg_initialized     = False  # True once _prev_wp_dist is valid for current leg
        self._stall_steps      = 0       # consecutive stalled steps before crash
        self._autonomous_switching = True  # False when a curriculum phase lock is active

        # Per-episode accumulators for mastery gate metrics
        self._ep_roll_rate_acc  = 0.0
        self._ep_pitch_rate_acc = 0.0
        self._ep_ctrl_osc_acc   = 0.0
        # Tracks previous ACTUAL surface positions for oscillation measurement
        self._prev_surface = np.zeros(3)   # [elevator_pos, aileron_pos, rudder_pos]

        # Rolling telemetry buffer — last 60 steps (6 s) saved on crash
        self._telemetry_buf: collections.deque = collections.deque(maxlen=60)

        # obs: 27 base + 4 prev_action = 31-D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0, -1.0,  0.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
        )

        self.reset()

    # ================================================================= reset ==

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._forced_mode      = None
        self._prev_action      = np.zeros(4)
        self.steps             = 0
        self.episode_reward    = 0.0
        self._stalled_this_ep  = False
        self._waypoint_reached = False
        self._wp_arrival_steps = 0
        self._prev_mission_wp_idx = -1
        self._prev_wp_dist     = 0.0
        self._leg_start_dist   = 0.0
        self._wp_leg_initialized = False
        self._stall_steps      = 0
        self._ep_roll_rate_acc  = 0.0
        self._ep_pitch_rate_acc = 0.0
        self._ep_ctrl_osc_acc   = 0.0
        self._prev_surface      = np.zeros(3)
        self._telemetry_buf.clear()

        if self._training_mode:
            self._reset_training()
        else:
            self._reset_mission()

        self._prev_pos             = self._state.pos.copy()
        self._prev_throttle_cmd    = float(self._state.throttle_pos)
        self._prev_throttle_actual = float(self._state.throttle_pos)
        return self._build_obs(), {}

    # -------------------------------------------------------------- training --

    def _reset_training(self):
        # Resolve active modes + difficulty from curriculum manager or phase string
        if self._curriculum is not None:
            active_modes  = self._curriculum.get_active_modes()
            diff          = self._curriculum.get_difficulty()
            self._active_phase = self._curriculum.stage_name
            stage_er      = self._curriculum.get_emergency_recovery_enabled()
        else:
            active_modes  = get_active_modes(self._curriculum_phase)
            diff          = _DEFAULT_DIFF
            self._active_phase = self._curriculum_phase
            stage_er      = get_emergency_recovery_default(self._curriculum_phase)

        # Resolve effective emergency_recovery_enabled (config override takes priority)
        emer_recovery = self._er_override if self._er_override is not None else stage_er

        # ── Phase mode selection ──────────────────────────────────────────────
        locked_mode    = get_locked_mode(self._active_phase)
        weighted_modes = get_phase_weighted_modes(self._active_phase)

        if weighted_modes is not None:
            # Weighted sampling: e.g. altitude_hold → 70% ALT_HOLD + 30% STABILIZE.
            # Bypasses the hard single-mode lock so replay diversity is preserved.
            modes_w, probs_w = weighted_modes
            mode = FlightMode(self.np_random.choice([m.value for m in modes_w], p=probs_w))
            emer_recovery              = False
            self._autonomous_switching = True   # no mission transitions in training
        elif locked_mode is not None:
            # Hard single-mode lock — only one mode per episode
            active_modes               = [locked_mode]
            emer_recovery              = False
            self._autonomous_switching = False
            mode                       = locked_mode
        else:
            # Multi-mode phase: uniform sampling after RECOVERY filter
            self._autonomous_switching = True
            if not emer_recovery:
                active_modes = [m for m in active_modes if m != FlightMode.RECOVERY]
                if not active_modes:
                    active_modes = [FlightMode.STABILIZE]
            mode = FlightMode(self.np_random.choice([m.value for m in active_modes]))

        # Log mode changes between episodes
        if self._prev_mode is not None and mode != self._prev_mode:
            print(f"[MODE CHANGE] {self._prev_mode.name} -> {mode.name}"
                  f"  reason=episode_reset  phase={self._active_phase}")
        self._prev_mode = mode

        # Invariant: locked phases must activate exactly the locked mode
        if not self._autonomous_switching:
            _locked = get_locked_mode(self._active_phase)
            assert mode == _locked, (
                f"curriculum lock violated at reset: phase={self._active_phase} "
                f"expected={_locked.name} selected={mode.name}"
            )

        # Mission manager must be disabled for all isolated curriculum phases.
        # (self._mission is always None in training mode; this assertion makes
        # the contract explicit and will catch any future regression.)
        if self._active_phase != 'mixed':
            assert self._mission is None, (
                f"[ASSERT] Mission manager must be disabled during isolated "
                f"phase '{self._active_phase}' — set mission_path=None"
            )

        # Debug: print curriculum status on every reset
        if self._curriculum is not None:
            print(
                f"[RESET] CURRICULUM ENABLED | "
                f"PHASE={self._active_phase.upper()} | "
                f"MODE={mode.name} | "
                f"MISSION_MANAGER={self._mission is not None} | "
                f"AUTO_SWITCH={self._autonomous_switching}"
            )

        self._mode = mode

        alt = float(self.np_random.uniform(*diff.alt_range))
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        spd = float(self.np_random.uniform(*diff.spd_range))

        if mode == FlightMode.STABILIZE:
            roll  = float(self.np_random.uniform(-diff.roll_perturb,  diff.roll_perturb))
            pitch = float(self.np_random.uniform(-diff.pitch_perturb, diff.pitch_perturb))
            self._state = AircraftState(
                pos=np.array([0., 0., alt]),
                vel=_vel_from_heading(yaw, spd),
                pitch=pitch, roll=roll, yaw=yaw, throttle_pos=_CRUISE_THR,
            )
            self._target = TargetInfo(FlightMode.STABILIZE,
                                      position=[0, 0, alt], heading=yaw)

        elif mode == FlightMode.ALTITUDE_HOLD:
            d_alt = float(self.np_random.uniform(*diff.alt_delta_range))
            d_alt *= float(self.np_random.choice([-1, 1]))
            target_alt = float(np.clip(alt + d_alt, 30, 500))
            self._state = AircraftState(
                pos=np.array([0., 0., alt]),
                vel=_vel_from_heading(yaw, spd),
                yaw=yaw, throttle_pos=_ALT_HOLD_THR,
            )
            self._target = TargetInfo(FlightMode.ALTITUDE_HOLD,
                                      position=[0, 0, target_alt],
                                      heading=yaw, altitude=target_alt)

        elif mode == FlightMode.HEADING_HOLD:
            d_yaw      = float(self.np_random.uniform(*diff.hdg_delta_range))
            d_yaw     *= float(self.np_random.choice([-1, 1]))
            target_yaw = yaw + d_yaw
            self._state = AircraftState(
                pos=np.array([0., 0., alt]),
                vel=_vel_from_heading(yaw, spd),
                yaw=yaw, throttle_pos=_CRUISE_THR,
            )
            self._target = TargetInfo(FlightMode.HEADING_HOLD,
                                      position=[0, 0, alt],
                                      heading=target_yaw, altitude=alt)

        elif mode == FlightMode.WAYPOINT:
            dist = float(self.np_random.uniform(*diff.wp_dist_range))
            wx = dist * math.sin(yaw)
            wy = -dist * math.cos(yaw)
            wz = alt + float(self.np_random.uniform(-20, 30))
            wz = float(np.clip(wz, 30, 400))
            self._state = AircraftState(
                pos=np.array([0., 0., alt]),
                vel=_vel_from_heading(yaw, spd),
                yaw=yaw, throttle_pos=_CRUISE_THR,
            )
            self._target = TargetInfo(FlightMode.WAYPOINT,
                                      position=[wx, wy, wz],
                                      heading=yaw, altitude=wz)

        elif mode == FlightMode.LOITER:
            radius   = float(self.np_random.uniform(*diff.loiter_rad_range))
            left_yaw = yaw - math.pi / 2
            cx = math.sin(left_yaw) * radius
            cy = -math.cos(left_yaw) * radius
            self._state = AircraftState(
                pos=np.array([0., 0., alt]),
                vel=_vel_from_heading(yaw, _LOITER_SPEED),
                yaw=yaw, throttle_pos=_LOITER_THR,
            )
            self._target = TargetInfo(FlightMode.LOITER,
                                      position=[cx, cy, alt],
                                      altitude=alt, radius=radius)

        elif mode == FlightMode.APPROACH:
            rwy_x, rwy_y = 0.0, 0.0
            hdg  = math.radians(180.0)
            dist = float(self.np_random.uniform(*diff.approach_dist_range))
            sx   = rwy_x - math.sin(hdg) * dist + float(self.np_random.uniform(-30, 30))
            sy   = rwy_y + math.cos(hdg) * dist
            sz   = dist * math.tan(math.radians(3.0)) + float(self.np_random.uniform(-10, 10))
            sz   = float(np.clip(sz, 15, 200))
            self._state = AircraftState(
                pos=np.array([sx, sy, sz]),
                vel=_vel_from_heading(hdg, spd, vz=-spd * math.tan(math.radians(3.0))),
                pitch=-math.radians(3.0), yaw=hdg, throttle_pos=_CRUISE_THR * 0.8,
            )
            gs_dist = 400.0
            app_x   = rwy_x - math.sin(hdg) * gs_dist
            app_y   = rwy_y + math.cos(hdg) * gs_dist
            app_z   = gs_dist * math.tan(math.radians(3.0))
            self._target = TargetInfo(FlightMode.APPROACH,
                                      position=[app_x, app_y, app_z],
                                      heading=hdg, altitude=app_z)

        elif mode == FlightMode.LANDING:
            rwy_x, rwy_y = 0.0, 0.0
            hdg  = math.radians(180.0)
            dist = float(self.np_random.uniform(*diff.landing_dist_range))
            sx   = rwy_x - math.sin(hdg) * dist + float(self.np_random.uniform(-5, 5))
            sy   = rwy_y + math.cos(hdg) * dist
            sz   = dist * math.tan(math.radians(3.0))
            self._state = AircraftState(
                pos=np.array([sx, sy, sz]),
                vel=_vel_from_heading(hdg, 8.0, vz=-0.5),
                pitch=-math.radians(2.0), yaw=hdg, throttle_pos=0.2,
            )
            self._target = TargetInfo(FlightMode.LANDING,
                                      position=[rwy_x, rwy_y, 0.0],
                                      heading=hdg, altitude=0.0)

        elif mode == FlightMode.RECOVERY:
            low_spd = float(self.np_random.uniform(4, 7))
            roll    = float(self.np_random.choice([-1, 1])) * \
                      float(self.np_random.uniform(0.5, 1.2))
            pitch   = float(self.np_random.uniform(-0.4, 0.6))
            self._state = AircraftState(
                pos=np.array([0., 0., alt]),
                vel=_vel_from_heading(yaw, low_spd),
                pitch=pitch, roll=roll, yaw=yaw, throttle_pos=0.15,
            )
            self._target = TargetInfo(FlightMode.RECOVERY,
                                      position=[0, 0, alt],
                                      heading=yaw, altitude=alt)

    # --------------------------------------------------------------- mission --

    def _reset_mission(self):
        if self._mission is None:
            yaw = 0.0
            self._state = AircraftState(
                pos=np.array([0., 200., 60.]),
                vel=_vel_from_heading(yaw, _CRUISE_SPEED),
                yaw=yaw, throttle_pos=_CRUISE_THR,
            )
            self._mode   = FlightMode.STABILIZE
            self._target = TargetInfo(FlightMode.STABILIZE,
                                      position=[0, 200, 60], heading=yaw)
            return

        self._mission.reset()
        yaw = math.radians(180.0)
        self._state = AircraftState(
            pos=np.array([0., 0., 5.]),
            vel=_vel_from_heading(yaw, 0.0),
            yaw=yaw, throttle_pos=0.0,
        )
        self._target = self._mission.get_target(self._state.pos, self._state.yaw)
        self._mode   = self._target.mode if self._target else FlightMode.STABILIZE

    # ================================================================== step ==

    def step(self, action):
        self.steps += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        elevator, aileron, rudder, throttle_raw = action

        # Throttle deadband: ignore command changes < 0.02 (prevents twitchy micro-adjustments)
        if abs(float(throttle_raw) - self._prev_throttle_cmd) < 0.02:
            throttle_effective = self._prev_throttle_cmd
        else:
            throttle_effective = float(throttle_raw)
        self._prev_throttle_cmd = throttle_effective

        actuators = np.array([throttle_effective, elevator, aileron, rudder, 0.0])
        self._prev_pos          = self._state.pos.copy()
        prev_throttle_actual    = self._prev_throttle_actual
        self._state             = self._phys.step(self._state, actuators, dt=0.1)
        throttle_delta          = abs(float(self._state.throttle_pos) - prev_throttle_actual)
        self._prev_throttle_actual = float(self._state.throttle_pos)

        # Rolling telemetry — recorded before crash detection so final step is captured
        self._telemetry_buf.append({
            'step':       self.steps,
            'pos':        [round(float(v), 1) for v in self._state.pos],
            'airspeed':   round(float(self._state.airspeed), 2),
            'roll_deg':   round(math.degrees(float(self._state.roll)),  1),
            'pitch_deg':  round(math.degrees(float(self._state.pitch)), 1),
            'yaw_deg':    round(math.degrees(float(self._state.yaw)),   1),
            'roll_rate':  round(float(self._state.roll_rate),  3),
            'pitch_rate': round(float(self._state.pitch_rate), 3),
            'vz':         round(float(self._state.vel[2]), 2),
            'action':     [round(float(x), 3) for x in action],
        })

        # Capture arrival at CURRENT target before mission.update() may advance it.
        # After update, self._target points to the NEW waypoint, so any distance
        # computed there would reference the wrong leg.
        prev_step_mode = self._mode
        wp_arrived_pre_update = False
        dist_to_wp_pre_update = None
        if prev_step_mode == FlightMode.WAYPOINT and self._target is not None:
            dist_to_wp_pre_update = float(np.linalg.norm(self._state.pos - self._target.position))
            wp_arrived_pre_update = dist_to_wp_pre_update < 20.0

        # Mission mode update
        wp_switch_event = False
        if self._forced_mode is not None:
            self._mode = self._forced_mode
        elif self._mission is not None:
            done_mission = self._mission.update(self._state.pos, self._state.yaw)
            new_idx = self._mission.current_index
            if self._prev_mission_wp_idx >= 0 and new_idx != self._prev_mission_wp_idx:
                wp_switch_event = True
            self._prev_mission_wp_idx = new_idx
            t = self._mission.get_target(self._state.pos, self._state.yaw)
            if t is not None:
                self._target = t
                self._mode   = t.mode
            if done_mission:
                self._mode = FlightMode.STABILIZE

        # Waypoint proximity tracking — use pre-update distance so we credit
        # arrival at the waypoint we were actually flying toward.
        if wp_arrived_pre_update:
            self._waypoint_reached = True
            self._wp_arrival_steps += 1

        # Leg-local waypoint progress bookkeeping.
        # _prev_wp_dist is reset on any leg transition so that the first step of
        # the new leg computes progress = 0 instead of a spurious large negative.
        wp_leg_reset = False
        curr_wp_dist = None
        if self._mode == FlightMode.WAYPOINT and self._target is not None:
            if wp_switch_event:
                # Target just changed — compute distance to the NEW waypoint.
                curr_wp_dist = float(np.linalg.norm(self._state.pos - self._target.position))
            else:
                # Same target — reuse pre-update distance if available.
                curr_wp_dist = (dist_to_wp_pre_update
                                if dist_to_wp_pre_update is not None
                                else float(np.linalg.norm(self._state.pos - self._target.position)))
            if (not self._wp_leg_initialized
                    or wp_switch_event
                    or prev_step_mode != FlightMode.WAYPOINT):
                self._prev_wp_dist   = curr_wp_dist
                self._leg_start_dist = curr_wp_dist
                self._wp_leg_initialized = True
                wp_leg_reset = True
        else:
            self._wp_leg_initialized = False

        # State checks
        crashed  = False
        landed   = False
        alt      = float(self._state.pos[2])
        airspeed = float(self._state.airspeed)

        stall_warning = (airspeed < self._stall_speed) and (alt > 30)
        if stall_warning:
            self._stalled_this_ep = True

        # Stall timer: crash only after sustained stall (15 steps = 1.5 s)
        if self._mode == FlightMode.RECOVERY:
            self._stall_steps = 0   # recovery allows low speed
        elif stall_warning:
            self._stall_steps += 1
        else:
            self._stall_steps = max(0, self._stall_steps - 1)

        if alt <= _MIN_ALTITUDE:
            if self._mode == FlightMode.LANDING:
                landed  = True
            else:
                crashed = True

        if alt > _MAX_ALTITUDE:
            crashed = True

        if self._stall_steps >= 15:
            crashed = True

        # Base reward
        reward, breakdown = compute_reward(
            mode         = self._mode,
            state        = self._state,
            target       = self._target,
            prev_pos     = self._prev_pos,
            crashed      = crashed,
            landed       = landed,
            prev_wp_dist = (self._prev_wp_dist if curr_wp_dist is not None else None),
            # Pass pre-update arrival only on transition steps; otherwise let the
            # reward function detect arrival from curr_dist < 20 (training mode).
            wp_arrived   = (wp_arrived_pre_update if wp_switch_event else None),
        )

        # Advance leg-local prev distance for next step.
        if curr_wp_dist is not None:
            self._prev_wp_dist = curr_wp_dist

        # Strong throttle actuator derivative penalty in ALT_HOLD — discourages hunting.
        # throttle_delta is tiny (≤0.005/step) so weight of 50 gives max ~0.25/step.
        if self._mode == FlightMode.ALTITUDE_HOLD and throttle_delta > 0:
            thr_deriv_pen = throttle_delta * 50.0
            breakdown["thr_deriv"] = round(-thr_deriv_pen, 4)
            reward -= thr_deriv_pen

        # Action smoothness penalty (penalizes rapid control changes)
        delta      = action - self._prev_action
        smooth_pen = float(np.linalg.norm(delta)) * self._action_smooth_weight

        # Accumulate per-episode mastery metrics
        self._ep_roll_rate_acc  += abs(float(self._state.roll_rate))
        self._ep_pitch_rate_acc += abs(float(self._state.pitch_rate))

        # Oscillation: actual surface deflection deltas after physics lag.
        # Raw PPO action deltas are NOT used — the actuator rate-limiter means
        # PPO can jump ±0.3 while surfaces only move ±0.06, so raw deltas
        # always appear large even during smooth flight.
        actual_surf = np.array([self._state.elevator_pos,
                                self._state.aileron_pos,
                                self._state.rudder_pos])
        surf_delta  = actual_surf - self._prev_surface
        # Deadzone: micro-corrections < 0.01 are normal trim activity, ignore them
        surf_delta  = np.where(np.abs(surf_delta) < 0.01, 0.0, surf_delta)
        osc_step    = float(np.linalg.norm(surf_delta))
        self._ep_ctrl_osc_acc += osc_step
        self._prev_surface = actual_surf

        breakdown["smoothness"] = round(-smooth_pen, 4)
        reward -= smooth_pen

        # Surface saturation penalty (penalizes holding surfaces near ±1)
        elev, ail, rud = float(action[0]), float(action[1]), float(action[2])
        sat_pen = 0.0
        for surf_val in (elev, ail, rud):
            excess = max(0.0, abs(surf_val) - 0.85)
            sat_pen += excess * 0.6
        if sat_pen > 0:
            breakdown["saturation"] = round(-sat_pen, 4)
            reward -= sat_pen

        # High angular rate penalty (discourages sign-flipping oscillations)
        rate_pen = (abs(self._state.roll_rate)  * 0.08 +
                    abs(self._state.pitch_rate) * 0.06 +
                    abs(self._state.yaw_rate)   * 0.04)
        if rate_pen > 0.02:
            breakdown["ang_rate"] = round(-rate_pen, 4)
            reward -= rate_pen

        # Stall penalty (per-step, while stalled but not already crashed)
        if stall_warning and not crashed:
            breakdown["stall"] = -3.0
            reward -= 3.0

        # Flight envelope penalties
        env_pen, env_bd = _envelope_penalties(self._state)
        if env_bd:
            breakdown.update(env_bd)
        reward -= env_pen

        self._prev_action = action.copy()
        self.episode_reward += reward

        done      = crashed or landed
        truncated = (not done) and (self.steps >= self.MAX_STEPS)

        if self._mission is not None and self._mission.is_done and not done:
            truncated = True

        # In training mode (single-waypoint, no mission), terminate immediately
        # on arrival so the agent never experiences negative post-arrival progress.
        if (self._mission is None and
                self._mode == FlightMode.WAYPOINT and
                self._waypoint_reached and not done):
            truncated = True

        # Strict curriculum lock assertion: locked phases must never run any mode
        # other than the single mode they are locked to.
        if self._training_mode and not self._autonomous_switching:
            _locked = get_locked_mode(self._active_phase)
            if _locked is not None:
                assert self._mode == _locked, (
                    f"[ASSERT] curriculum lock violated in step: "
                    f"phase={self._active_phase}  "
                    f"expected={_locked.name}  actual={self._mode.name}"
                )

        # ── Episode outcome ───────────────────────────────────────────────────
        _ep_steps = max(1, self.steps)
        info = {
            'mode':               self._mode,
            'crashed':            crashed,
            'landed':             landed,
            'airspeed':           airspeed,
            'altitude':           alt,
            'stall_warning':      stall_warning,
            'stalled_this_ep':    self._stalled_this_ep,
            'reward_breakdown':   breakdown,
            # Oscillation debug: step-level and rolling episode mean
            'ctrl_osc_step':      osc_step,
            'ctrl_osc_ep_mean':   self._ep_ctrl_osc_acc / _ep_steps,
            # Throttle telemetry
            'throttle_command':   self._prev_throttle_cmd,
            'throttle_actual':    float(self._state.throttle_pos),
            'throttle_delta':     throttle_delta,
            'target_airspeed':    (_ALT_HOLD_TGT_AIRSPEED
                                   if self._mode == FlightMode.ALTITUDE_HOLD else 0.0),
            'airspeed_error':     (airspeed - _ALT_HOLD_TGT_AIRSPEED
                                   if self._mode == FlightMode.ALTITUDE_HOLD else 0.0),
            # Altitude / vertical-speed telemetry
            'target_altitude':    (float(self._target.altitude)
                                   if self._target is not None else 0.0),
            'altitude_error':     (float(self._state.pos[2] - self._target.altitude)
                                   if self._target is not None else 0.0),
            'vertical_speed':     float(self._state.vel[2]),
            # Curriculum / mode telemetry
            'curriculum_phase':              self._active_phase,
            'autonomous_switching_enabled':  self._autonomous_switching,
            'commanded_mode':     self._mode,
            'active_mode':        self._mode,
            'recovery_triggered': False,
            'recovery_reason':    None,
            # Waypoint debug telemetry
            'wp_prev_dist':    breakdown.get('wp_prev_dist',  0.0),
            'wp_curr_dist':    breakdown.get('wp_curr_dist',  0.0),
            'wp_progress':     breakdown.get('progress',      0.0),
            'wp_arrived':           self._waypoint_reached,
            'wp_arrived_this_step': wp_arrived_pre_update,
            'wp_arrival_steps':     self._wp_arrival_steps,
            'wp_switch_event':      wp_switch_event,
            'wp_leg_reset':         wp_leg_reset,
            'wp_leg_start_dist':    round(self._leg_start_dist, 1),
            'wp_mission_idx':       (self._mission.current_index if self._mission else -1),
        }

        if done or truncated:
            success = self._episode_success(crashed, landed)
            info['episode_success'] = success

            if self._curriculum is not None:
                steps = max(1, self.steps)
                ep_data = EpisodeData(
                    success=success,
                    crashed=crashed,
                    stalled=self._stalled_this_ep,
                    ep_length=self.steps,
                    mean_roll_rate=self._ep_roll_rate_acc  / steps,
                    mean_pitch_rate=self._ep_pitch_rate_acc / steps,
                    ctrl_oscillation=self._ep_ctrl_osc_acc  / steps,
                )
                advanced = self._curriculum.record_episode(ep_data)
                info['curriculum_advanced'] = advanced
                info['curriculum_stage']    = self._curriculum.stage_name

        # Crash diagnostics — classify and save telemetry replay
        if crashed:
            crash_type = classify_crash(
                stall_steps = self._stall_steps,
                state       = self._state,
                info        = info,
                stall_speed = self._stall_speed,
            )
            info['crash_type'] = crash_type.value
            save_crash_report(
                crash_type    = crash_type,
                telemetry_buf = list(self._telemetry_buf),
                state         = self._state,
                info          = info,
                mode_name     = self._mode.name,
            )

        return self._build_obs(), float(reward), done, truncated, info

    # ─────────────────────────────────── per-mode episode success condition ──

    def _episode_success(self, crashed: bool, landed: bool) -> bool:
        """Return True when the agent meaningfully achieved the episode goal."""
        if crashed:
            return False

        s = self._state
        t = self._target
        m = self._mode

        if m == FlightMode.STABILIZE:
            return (abs(s.roll)  < math.radians(25) and
                    abs(s.pitch) < math.radians(25))

        elif m == FlightMode.RECOVERY:
            return (s.airspeed > self._stall_speed + 1.0 and
                    abs(s.roll) < math.radians(35))

        elif m == FlightMode.ALTITUDE_HOLD:
            if t is None:
                return False
            alt_err = s.pos[2] - t.altitude
            success = abs(alt_err) < 25.0
            if not success:
                spd_err = s.airspeed - _ALT_HOLD_TGT_AIRSPEED
                failing = []
                if abs(alt_err) >= 25.0:
                    failing.append(f"altitude_error={alt_err:+.1f}m (need <25)")
                if abs(spd_err) > 4.0:
                    failing.append(f"speed_err={spd_err:+.1f}m/s (need <4)")
                if abs(s.vel[2]) > 3.0:
                    failing.append(f"vspd={s.vel[2]:+.2f}m/s (need <3)")
                print(f"[ALT_HOLD FAIL] {' | '.join(failing)}")
            return success

        elif m == FlightMode.HEADING_HOLD:
            return (t is not None and
                    abs(_angle_diff(s.yaw, t.heading)) < math.radians(25))

        elif m == FlightMode.WAYPOINT:
            return self._waypoint_reached

        elif m == FlightMode.LOITER:
            if t is None:
                return False
            dx = s.pos[0] - t.position[0]
            dy = s.pos[1] - t.position[1]
            return abs(math.sqrt(dx * dx + dy * dy) - t.radius) < 20.0

        elif m == FlightMode.APPROACH:
            if t is None:
                return False
            dx    = s.pos[0] - t.position[0]
            dy    = s.pos[1] - t.position[1]
            horiz = math.sqrt(dx * dx + dy * dy)
            ideal_alt = horiz * math.tan(math.radians(3.0))
            return abs(s.pos[2] - ideal_alt) < 15.0

        elif m == FlightMode.LANDING:
            return landed

        return True   # non-crash is good enough for unknown modes

    # ============================================================== build obs ==

    def _build_obs(self) -> np.ndarray:
        s = self._state
        t = self._target

        pos_n  = s.pos          / 500.0
        vel_n  = s.vel          / 20.0
        rates  = np.array([s.pitch_rate, s.roll_rate, s.yaw_rate]) / 5.0
        spd_n  = min(s.airspeed / 20.0, 3.0)

        if t is not None:
            delta  = (t.position - s.pos) / 500.0
            dist_n = min(float(np.linalg.norm(t.position - s.pos)) / 500.0, 4.0)
            hdg_e  = _angle_diff(t.heading, s.yaw) / math.pi
            alt_e  = (s.pos[2] - t.altitude) / 100.0
        else:
            delta  = np.zeros(3)
            dist_n = 0.0
            hdg_e  = 0.0
            alt_e  = 0.0

        mode_oh = np.zeros(NUM_MODES)
        mode_oh[int(self._mode)] = 1.0

        obs = np.concatenate([
            pos_n,                                           # 3
            vel_n,                                           # 3
            [s.roll, s.pitch, s.yaw],                       # 3
            rates,                                           # 3
            [spd_n],                                         # 1
            delta,                                           # 3
            [dist_n, hdg_e, alt_e],                         # 3
            mode_oh,                                         # 8
            self._prev_action,                               # 4
        ]).astype(np.float32)

        return obs

    # ========================================================= keyboard API ==

    def force_mode(self, mode: FlightMode):
        self._forced_mode = mode
        self._mode        = mode
        pos = self._state.pos.copy()
        yaw = self._state.yaw

        if mode == FlightMode.STABILIZE:
            self._target = TargetInfo(mode, position=pos, heading=yaw)

        elif mode == FlightMode.ALTITUDE_HOLD:
            self._target = TargetInfo(mode, position=pos,
                                      heading=yaw, altitude=pos[2])

        elif mode == FlightMode.HEADING_HOLD:
            self._target = TargetInfo(mode, position=pos,
                                      heading=yaw, altitude=pos[2])

        elif mode == FlightMode.WAYPOINT:
            dist = 200.0
            wx = pos[0] + math.sin(yaw) * dist
            wy = pos[1] - math.cos(yaw) * dist
            self._target = TargetInfo(mode,
                                      position=[wx, wy, pos[2]],
                                      heading=yaw, altitude=pos[2])

        elif mode == FlightMode.LOITER:
            radius = 60.0
            left   = yaw - math.pi / 2
            cx = pos[0] + math.sin(left) * radius
            cy = pos[1] - math.cos(left) * radius
            self._target = TargetInfo(mode,
                                      position=[cx, cy, pos[2]],
                                      altitude=pos[2], radius=radius)

        elif mode == FlightMode.APPROACH:
            hdg     = math.radians(180.0)
            gs_dist = 400.0
            self._target = TargetInfo(mode,
                                      position=[-gs_dist * math.sin(hdg),
                                                 gs_dist * math.cos(hdg),
                                                 gs_dist * math.tan(math.radians(3))],
                                      heading=hdg,
                                      altitude=gs_dist * math.tan(math.radians(3)))

        elif mode == FlightMode.LANDING:
            hdg = math.radians(180.0)
            self._target = TargetInfo(mode,
                                      position=[0.0, 0.0, 0.0],
                                      heading=hdg, altitude=0.0)

        elif mode == FlightMode.RECOVERY:
            self._target = TargetInfo(mode, position=pos,
                                      heading=yaw, altitude=pos[2])

    # =========================================================== properties ==

    @property
    def pos(self) -> np.ndarray:
        return self._state.pos.astype(np.float32)

    @property
    def vel(self) -> np.ndarray:
        return self._state.vel.astype(np.float32)

    @property
    def mode(self) -> FlightMode:
        return self._mode

    @property
    def target(self) -> TargetInfo:
        return self._target

    @property
    def mission(self) -> MissionManager:
        return self._mission
