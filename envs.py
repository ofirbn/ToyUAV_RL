"""
Two-level flight environments.

PilotageEnv  — goal-conditioned low-level pilot.
               Observation = [flight_state | goal_command].
               One unified reward = distance to goal.
               No scenario one-hot, no conflicting reward functions.
               Goal cmd = [target_speed, target_alt, target_vz,
                           target_yaw, target_roll]

LandingEnv   — high-level navigation: outputs flight commands consumed by a
               frozen pilotage brain; rewards are approach/landing metrics.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aircraft import AircraftPhysics, AircraftState, SimplePhysics

# ── Physics toggle ────────────────────────────────────────────────────────────
# Set True  → arcade kinematics (decoupled, no aero, no gravity).
#             Use this to verify reward structure and RL convergence first.
# Set False → full aerodynamic model.
SIMPLE_PHYSICS = True
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# PilotageEnv  — goal-conditioned pilot
# ============================================================

class PilotageEnv(gym.Env):
    """
    Observation (15-D):
        airspeed, altitude,
        pitch, roll, yaw (normalised to ±π),
        pitch_rate, roll_rate, yaw_rate,
        throttle_pos, flap_pos,
        cmd_speed, cmd_alt, cmd_vz, cmd_yaw, cmd_roll

    Action (5-D):
        throttle [0,1]  elevator [-1,1]  aileron [-1,1]
        rudder [-1,1]   flaps [0,1]  (flaps locked at 0 during pilotage)

    Unified reward:
        1.0 - speed_err - alt_err - vz_err - yaw_err - roll_err
        minus throttle penalty below 30%

    Episode modes (randomly chosen each reset):
        level     — hold speed/alt/heading
        climb     — climb to target altitude at target vz
        descent   — descend to target altitude at target vz
        turn      — reach target heading while holding alt
        speed     — change speed while holding alt
        recovery  — recover from unusual attitude to wings-level
    """

    CONVERGENCE_REWARD = 250.0
    CONVERGENCE_WINDOW = 40
    MAX_TIMESTEPS      = 3_000_000
    MAX_STEPS          = 400

    # kept for LandingEnv._pilot_obs compat (single-element list, no padding)
    SCENARIOS = ['goal_conditioned']

    _last_scenario: str = ''

    # mode probabilities
    _MODES  = ['level', 'climb', 'descent', 'turn', 'speed', 'recovery']
    _PROBS  = [0.30,    0.15,    0.15,      0.15,   0.10,    0.15]

    def __init__(self, active_scenarios=None):
        super().__init__()
        self._phys = SimplePhysics() if SIMPLE_PHYSICS else AircraftPhysics()
        # active_scenarios restricts which modes are sampled each episode.
        # Pass ['level'] to train only level-flight, etc.
        self._active_modes = (list(active_scenarios)
                              if active_scenarios else self._MODES)

        # obs: 10 state + 5 goal = 15-D
        obs_high = np.array([
            30, 700,                           # airspeed, altitude
            math.pi/2, math.pi, math.pi,      # pitch, roll, yaw
            6, 6, 6,                           # angular rates
            1.0, 1.0,                          # throttle_pos, flap_pos
            30, 700, 6, math.pi, math.pi,      # cmd: speed, alt, vz, yaw, roll
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low  = np.array([0, -1, -1, -1, 0], dtype=np.float32),
            high = np.array([1,  1,  1,  1, 1], dtype=np.float32),
        )

        self._state      = None
        self._cmd        = np.zeros(5)   # [speed, alt, vz, yaw, roll]
        self._scenario   = ''
        self._prev_action = np.zeros(5)  # for smoothness penalty
        self.steps       = 0
        self.last_grade  = ''
        self.last_score  = 0.0
        self.reset()

    # ----------------------------------------------------------
    # Observation
    # ----------------------------------------------------------

    def _obs(self):
        s = self._state
        yaw     = math.atan2(math.sin(s.yaw),         math.cos(s.yaw))
        cmd_yaw = math.atan2(math.sin(self._cmd[3]),   math.cos(self._cmd[3]))
        return np.array([
            s.airspeed, s.pos[2],
            s.pitch, s.roll, yaw,
            s.pitch_rate, s.roll_rate, s.yaw_rate,
            s.throttle_pos, s.flap_pos,
            self._cmd[0], self._cmd[1], self._cmd[2], cmd_yaw, self._cmd[4],
        ], dtype=np.float32)

    # ----------------------------------------------------------
    # Unified reward
    # ----------------------------------------------------------

    def _reward(self):
        s = self._state
        cmd_speed, cmd_alt, cmd_vz, cmd_yaw, cmd_roll = self._cmd

        speed_err = abs(s.airspeed - cmd_speed) / 3.0   # /5 let policy hold start speed and still converge
        # /150 prevents saturation: 80m climb starts at 0.47 not -1.0,
        # so the policy has gradient throughout the manoeuvre.
        alt_err   = abs(s.pos[2]   - cmd_alt)   / 150.0
        yaw_err   = abs(math.atan2(math.sin(s.yaw - cmd_yaw),
                                   math.cos(s.yaw - cmd_yaw))) / math.pi * 0.5
        roll_err  = abs(s.roll - cmd_roll) / math.pi
        # vz_err and vz_pen removed: for climb/descent they contradict each other
        # (cmd_vz stays fixed but vz should go to 0 once altitude is reached).
        # alt_err already drives the elevator to maintain/reach altitude.

        r = 1.0 - speed_err - alt_err - yaw_err - roll_err
        r = float(np.clip(r, -1.0, 1.0))

        # throttle below cruise minimum → severe penalty
        r -= max(0.0, 0.30 - s.throttle_pos) * 4.0
        return r

    # ----------------------------------------------------------
    # Gym interface
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        alt = float(np.random.uniform(80, 400))
        yaw = float(np.random.uniform(-math.pi, math.pi))
        mode = str(np.random.choice(self._active_modes))

        if mode == 'level':
            target_spd = float(np.random.uniform(9, 13))
            start_spd  = target_spd * float(np.random.uniform(0.80, 0.90))
            self._state = AircraftState(
                pos=np.array([0.0, 200.0, alt]),
                vel=np.array([math.sin(yaw)*start_spd, -math.cos(yaw)*start_spd, 0.0]),
                pitch=0.0, roll=0.0, yaw=yaw, throttle_pos=start_spd / 15.0,
            )
            self._cmd = np.array([target_spd, alt, 0.0, yaw, 0.0])

        elif mode == 'climb':
            spd   = float(np.random.uniform(9, 13))
            d_alt = float(np.random.uniform(30, 120))
            rate  = float(np.random.uniform(1.5, 4.0))
            self._state = AircraftState(
                pos=np.array([0.0, 200.0, alt]),
                vel=np.array([math.sin(yaw)*spd, -math.cos(yaw)*spd, 0.0]),
                pitch=0.0, roll=0.0, yaw=yaw, throttle_pos=spd / 15.0,
            )
            self._cmd = np.array([spd, alt + d_alt, rate, yaw, 0.0])

        elif mode == 'descent':
            spd   = float(np.random.uniform(8, 12))
            d_alt = float(np.random.uniform(30, 100))
            rate  = float(np.random.uniform(1.0, 3.5))
            self._state = AircraftState(
                pos=np.array([0.0, 200.0, alt]),
                vel=np.array([math.sin(yaw)*spd, -math.cos(yaw)*spd, 0.0]),
                pitch=0.05, roll=0.0, yaw=yaw, throttle_pos=0.25,
            )
            self._cmd = np.array([spd, max(30.0, alt - d_alt), -rate, yaw, 0.0])

        elif mode == 'turn':
            spd   = float(np.random.uniform(9, 13))
            d_yaw = float(np.random.choice([-1, 1])) * float(np.random.uniform(40, 120))
            self._state = AircraftState(
                pos=np.array([0.0, 200.0, alt]),
                vel=np.array([math.sin(yaw)*spd, -math.cos(yaw)*spd, 0.0]),
                pitch=0.0, roll=0.0, yaw=yaw, throttle_pos=spd / 15.0,
            )
            # cmd_roll=0: in simple physics aileron drives yaw directly,
            # no bank angle needed as intermediate goal
            self._cmd = np.array([spd, alt, 0.0, yaw + math.radians(d_yaw), 0.0])

        elif mode == 'speed':
            spd0 = float(np.random.uniform(8, 10))
            spd1 = float(np.clip(
                spd0 + float(np.random.choice([-1, 1])) * float(np.random.uniform(2, 4)),
                7.5, 14.0))
            self._state = AircraftState(
                pos=np.array([0.0, 200.0, alt]),
                vel=np.array([math.sin(yaw)*spd0, -math.cos(yaw)*spd0, 0.0]),
                pitch=0.10, roll=0.0, yaw=yaw, throttle_pos=0.30,
            )
            self._cmd = np.array([spd1, alt, 0.0, yaw, 0.0])

        elif mode == 'recovery':
            spd   = float(np.random.uniform(7, 12))
            pitch = float(np.random.uniform(-0.5, 0.5))
            roll  = float(np.random.choice([-1, 1])) * float(np.random.uniform(0.5, 1.2))
            self._state = AircraftState(
                pos=np.array([0.0, 200.0, alt]),
                vel=np.array([math.sin(yaw)*spd, -math.cos(yaw)*spd, 0.0]),
                pitch=pitch, roll=roll, yaw=yaw, throttle_pos=0.30,
            )
            cmd_spd = float(np.random.uniform(9, 12))
            self._cmd = np.array([cmd_spd, alt, 0.0, yaw, 0.0])

        self._scenario    = mode
        PilotageEnv._last_scenario = mode
        self._prev_action = np.zeros(5)
        self.steps        = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        actuators = np.array(action, dtype=float)
        actuators[4] = 0.0   # flaps locked at 0 during pilotage
        self._state = self._phys.step(self._state, actuators, dt=0.1)
        s = self._state

        reward = self._reward()

        # Smoothness penalty — discourages rapid action changes (limit-cycle hunting).
        # Each channel weighted by how much it affects the rewarded goal.
        delta = actuators - self._prev_action
        reward -= (abs(delta[0]) * 0.5    # throttle  → speed
                 + abs(delta[1]) * 0.3    # elevator  → altitude
                 + abs(delta[2]) * 0.3)   # aileron   → heading
        self._prev_action = actuators.copy()

        done = False
        if not SIMPLE_PHYSICS:
            # attitude limits only matter with real aerodynamics
            if abs(s.pitch) > math.radians(75):
                reward -= 20.0; done = True
            if abs(s.roll) > math.radians(100):
                reward -= 20.0; done = True
        if s.pos[2] < 10.0:  done = True
        if s.pos[2] > 700.0: done = True

        truncated = (not done) and (self.steps >= self.MAX_STEPS)
        return self._obs(), reward, done, truncated, {}

    @property
    def pos(self): return self._state.pos.astype(np.float32)
    @property
    def vel(self): return self._state.vel.astype(np.float32)


# ============================================================
# LandingEnv
# ============================================================

class LandingEnv(gym.Env):
    """
    High-level navigation brain.

    Observation (7-D):
        pos_x, pos_y, pos_z,
        airspeed, yaw, pitch, roll

    Action (3-D): commands forwarded to the pilotage brain as a 5-D goal:
        target_speed  [CMD_SPEED_LO, CMD_SPEED_HI]
        target_vz     [CMD_VZ_LO,    CMD_VZ_HI]
        target_roll   [CMD_ROLL_LO,  CMD_ROLL_HI]

    The pilotage model then decides the actual actuators.
    """

    CONVERGENCE_REWARD  = 800.0
    CONVERGENCE_WINDOW  = 40
    MAX_TIMESTEPS       = 5_000_000

    RWY_X  =  13.0
    RWY_Y0 = -25.0
    RWY_Y1 =  95.0

    CMD_SPEED_LO, CMD_SPEED_HI =  7.5, 14.0
    CMD_VZ_LO,    CMD_VZ_HI    = -4.0,  1.0   # mostly descending for landing
    CMD_ROLL_LO,  CMD_ROLL_HI  = math.radians(-35), math.radians(35)

    MAX_STEPS = 2000

    def __init__(self, pilotage_model):
        super().__init__()
        self._pilot = pilotage_model
        self._phys  = AircraftPhysics()

        obs_high = np.array(
            [400, 800, 250, 20, math.pi * 2, math.pi / 2, math.pi],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low  = np.array([self.CMD_SPEED_LO, self.CMD_VZ_LO,  self.CMD_ROLL_LO],
                            dtype=np.float32),
            high = np.array([self.CMD_SPEED_HI, self.CMD_VZ_HI,  self.CMD_ROLL_HI],
                            dtype=np.float32),
        )

        self._state       = None
        self._prev_horiz  = 0.0
        self._prev_alt    = 0.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_vel1   = -12.0
        self.steps        = 0
        self.last_grade   = ""
        self.last_score   = 0.0
        self.reset()

    # ---- helpers ----

    def _runway_horiz_dist(self):
        cx = float(np.clip(self._state.pos[0], -self.RWY_X,  self.RWY_X))
        cy = float(np.clip(self._state.pos[1],  self.RWY_Y0, self.RWY_Y1))
        return math.sqrt((float(self._state.pos[0]) - cx) ** 2 +
                         (float(self._state.pos[1]) - cy) ** 2)

    def _obs(self):
        s = self._state
        return np.array([
            s.pos[0], s.pos[1], s.pos[2],
            s.airspeed, s.yaw, s.pitch, s.roll,
        ], dtype=np.float32)

    def _pilot_obs(self, cmd):
        """Build the 15-D goal-conditioned observation for the pilotage policy."""
        s   = self._state
        yaw = math.atan2(math.sin(s.yaw), math.cos(s.yaw))
        # cmd from landing brain: [speed, vz, roll]
        # map to pilot goal: [speed, alt, vz, yaw=0 (face runway), roll]
        return np.array([
            s.airspeed, s.pos[2],
            s.pitch, s.roll, yaw,
            s.pitch_rate, s.roll_rate, s.yaw_rate,
            s.throttle_pos, s.flap_pos,
            cmd[0],      # target speed
            s.pos[2],    # target alt = hold current (descent handled via vz)
            cmd[1],      # target vz
            0.0,         # target yaw = runway heading
            cmd[2],      # target roll
        ], dtype=np.float32)

    # ---- gym interface ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pos_x = float(np.random.uniform(-120, 120))
        pos_y = float(np.random.uniform(150, 700))
        horiz = math.sqrt(pos_x ** 2 + max(0.0, pos_y - 95.0) ** 2)
        horiz = max(horiz, 50.0)
        alt_lo = max(horiz * math.tan(math.radians(3)),   20.0)
        alt_hi = min(horiz * math.tan(math.radians(25)), 220.0)
        alt_hi = max(alt_hi, alt_lo + 10.0)
        pos_z  = float(np.random.uniform(alt_lo, alt_hi))

        self._state = AircraftState(
            pos          = np.array([pos_x, pos_y, pos_z]),
            vel          = np.array([float(np.random.uniform(-2, 2)), -12.0, -1.0]),
            yaw          = 0.0,
            throttle_pos = 0.6,
        )
        self._prev_horiz  = self._runway_horiz_dist()
        self._prev_alt    = pos_z
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_vel1   = -12.0
        self.steps        = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1

        cmd = np.clip(action, self.action_space.low, self.action_space.high)

        pilot_obs    = self._pilot_obs(cmd)
        actuators, _ = self._pilot.predict(pilot_obs, deterministic=True)

        self._state = self._phys.step(self._state, actuators, dt=0.1)

        altitude   = float(self._state.pos[2])
        curr_horiz = self._runway_horiz_dist()
        curr_alt   = max(0.0, altitude)
        pos_x      = float(self._state.pos[0])
        pos_y      = float(self._state.pos[1])
        vz         = float(self._state.vel[2])
        vy         = float(self._state.vel[1])

        reward  = (self._prev_horiz - curr_horiz) * 3.0
        reward += (self._prev_alt   - curr_alt)   * 1.0
        reward -= abs(pos_x) * 0.02

        delta   = cmd - self._prev_action
        reward -= (abs(float(delta[0])) * 0.08
                 + abs(vy - self._prev_vel1) * 0.05
                 + abs(float(delta[2])) * 0.05)

        stall_margin = abs(vy) - 4.0
        if stall_margin < 4.0:
            reward -= (4.0 - stall_margin) * 0.15

        spd_h = math.sqrt(float(self._state.vel[0]) ** 2 + vy ** 2)
        if spd_h > 0.5:
            cross = abs(float(self._state.vel[0])) / spd_h
            aw    = max(0.0, 1.0 - altitude / 180.0)
            reward -= cross * 0.15 * (0.2 + 0.8 * aw)

        if vz > 0:
            reward -= vz * 0.5
        else:
            aw_vz = max(0.0, 1.0 - altitude / 180.0)
            reward -= abs(vz) * 0.08 * (0.1 + 0.9 * aw_vz)

        reward -= max(0.0, abs(pos_x) - self.RWY_X) ** 2 * 0.003
        reward -= max(0.0, self.RWY_Y0 - pos_y)     ** 2 * 0.003
        reward -= max(0.0, pos_y - 750.0)            ** 2 * 0.003

        self._prev_horiz  = curr_horiz
        self._prev_alt    = curr_alt
        self._prev_action = cmd.copy()
        self._prev_vel1   = vy

        done = False; truncated = False

        def _abort(msg):
            nonlocal done
            self.last_grade = "*CRASHED*"; self.last_score = 0.0
            print(msg); done = True

        if altitude > 1000:
            _abort(f"CEILING BREACH           alt={altitude:.1f}")
            return self._obs(), reward - 100, done, truncated, {}
        if abs(pos_x) > 400:
            _abort(f"OUT OF BOUNDS (lateral)    x={pos_x:+.1f}")
            return self._obs(), reward - 100, done, truncated, {}
        if pos_y < -400:
            _abort(f"OUT OF BOUNDS (overshoot)  y={pos_y:.1f}")
            return self._obs(), reward - 100, done, truncated, {}
        if pos_y > 1200:
            _abort(f"OUT OF BOUNDS (retreat)    y={pos_y:.1f}")
            return self._obs(), reward - 100, done, truncated, {}

        if altitude <= 0:
            on_x      = abs(pos_x) < self.RWY_X
            on_y      = self.RWY_Y0 <= pos_y <= self.RWY_Y1
            hard_slam = on_x and on_y and abs(vz) > 5.0

            if on_x and on_y and not hard_slam:
                x_q = max(0.0, 1.0 - abs(pos_x) / self.RWY_X)
                y_q = max(0.0, 1.0 - abs(pos_y - 82.0) / 30.0)
                spd_h2 = math.sqrt(float(self._state.vel[0])**2 + vy**2)
                h_q  = (max(0.0, -vy / spd_h2) if spd_h2 > 0.5 else 1.0)
                vz_q = max(0.0, 1.0 - abs(vz) / 5.0)
                combined = x_q * y_q * h_q * vz_q
                reward += 50 + 550 * (combined ** 2)

                if   combined == 1.0:  grade = "PERFECT SCORE LANDING"
                elif combined >= 0.95: grade = "EXCELLENT LANDING"
                elif combined >= 0.90: grade = "VERY GOOD LANDING"
                elif combined >= 0.80: grade = "GOOD LANDING"
                elif combined >= 0.70: grade = "ALMOST GOOD LANDING"
                elif combined >= 0.50: grade = "FAIR LANDING"
                elif combined >= 0.25: grade = "POOR LANDING"
                else:                  grade = "BAD LANDING"

                self.last_grade = grade; self.last_score = combined
                yaw_deg = math.degrees(math.atan2(float(self._state.vel[0]), -vy))
                print(f"{grade:<22}  x={pos_x:+5.1f}  y={pos_y:5.1f}  "
                      f"hdg={yaw_deg:+6.1f}°  vz={vz:+5.2f}  score={combined:.3f}")
            else:
                reward -= 100
                self.last_grade = "*CRASHED*"; self.last_score = 0.0
                if hard_slam:
                    print(f"HARD SLAM (on strip)     x={pos_x:+5.1f}  y={pos_y:5.1f}  vz={vz:+5.2f}")
                else:
                    print(f"CRASH                    x={pos_x:+5.1f}  y={pos_y:5.1f}")
            done = True

        if self.steps >= self.MAX_STEPS:
            truncated = True

        return self._obs(), reward, done, truncated, {}

    @property
    def pos(self): return self._state.pos.astype(np.float32)
    @property
    def vel(self): return self._state.vel.astype(np.float32)
