"""
Two-level flight environments.

PilotageEnv  — low-level pilot training through structured flight scenarios.
               Each episode drills one specific skill from a syllabus of six.
               The brain learns to operate real actuators
               (throttle, elevator, aileron, rudder, flaps).

LandingEnv   — high-level navigation: outputs flight commands consumed by a
               frozen pilotage brain; rewards are approach/landing metrics.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aircraft import AircraftPhysics, AircraftState


# ============================================================
# PilotageEnv  — structured flight training syllabus
# ============================================================

class PilotageEnv(gym.Env):
    """
    Observation (11-D):
        airspeed, altitude,
        pitch, roll, yaw,
        pitch_rate, roll_rate, yaw_rate,
        target_speed, target_pitch, target_roll

    Action (5-D):
        throttle [0,1]  elevator [-1,1]  aileron [-1,1]
        rudder   [-1,1]  flaps   [0,1]

    Six training scenarios (one per episode, chosen at random):
      1. straight_level   — hold altitude, heading, airspeed
      2. rate_climb       — climb at commanded rate to target altitude
      3. coordinated_turn — turn to target heading at commanded bank angle
      4. rate_descent     — descend at commanded rate to target altitude
      5. speed_change     — accelerate / decelerate while holding altitude
      6. recovery         — escape from an unusual attitude to safe flight

    These scenarios are about FLYING, not landing.
    The landing environment uses the trained brain separately.
    """

    CONVERGENCE_REWARD = 200.0
    CONVERGENCE_WINDOW = 40
    MAX_TIMESTEPS      = 3_000_000

    MAX_STEPS = 400

    # ---- scenario catalogue ----
    SCENARIOS = [
        'straight_level',
        'rate_climb',
        'coordinated_turn',
        'rate_descent',
        'speed_change',
        'recovery',
    ]

    _last_scenario: str = ''   # class-level; any instance writes here on reset

    # curriculum order: simpler/single-axis first, compound manoeuvres last
    CURRICULUM = [
        'recovery',          # 1. stop tumbling — most urgent, clearest signal
        'straight_level',    # 2. hold trim — foundation of all flight
        'rate_climb',        # 3. single-axis pitch up + throttle
        'rate_descent',      # 4. single-axis pitch down
        'speed_change',      # 5. throttle management while holding altitude
        'coordinated_turn',  # 6. coupled roll + rudder + altitude hold
    ]

    def __init__(self, active_scenarios=None):
        """
        active_scenarios: list of scenario names to use in this env instance.
        None means use all SCENARIOS (full curriculum already active).
        Curriculum training creates envs with progressively larger subsets.
        """
        super().__init__()
        self._phys = AircraftPhysics()
        self._active = list(active_scenarios) if active_scenarios else list(self.SCENARIOS)

        n_sc = len(self.SCENARIOS)   # always 6, for consistent obs dim

        # obs = flight state (11) + scenario one-hot (6) = 17-D
        obs_high = np.array([
            20, 700,                          # airspeed, altitude
            math.pi/2, math.pi, math.pi*2,   # pitch, roll, yaw
            6, 6, 6,                          # angular rates
            20, math.pi/2, math.pi,          # target speed, pitch, roll
            *([1.0] * n_sc),                 # scenario one-hot
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low  = np.array([0, -1, -1, -1, 0], dtype=np.float32),
            high = np.array([1,  1,  1,  1, 1], dtype=np.float32),
        )

        self._state: AircraftState = None
        self._scenario  = ''
        self._target    = {}      # scenario-specific goal dict
        self._cmd       = np.zeros(3)  # [target_speed, target_pitch, target_roll]
        self.steps      = 0
        self.last_grade = ""
        self.last_score = 0.0
        self.reset()

    # ----------------------------------------------------------
    # Scenario initialisers
    # ----------------------------------------------------------

    def _init_straight_level(self):
        spd = float(np.random.uniform(8, 13))
        alt = float(np.random.uniform(80, 400))
        yaw = float(np.random.uniform(-math.pi, math.pi))
        self._state = AircraftState(
            pos=np.array([0.0, 200.0, alt]),
            vel=np.array([math.sin(yaw)*-spd, math.cos(yaw)*-spd, 0.0]),
            pitch=0.0, roll=0.0, yaw=yaw, throttle_pos=0.55,
        )
        self._target = {'speed': spd, 'alt': alt, 'yaw': yaw}
        self._cmd = np.array([spd, 0.0, 0.0])

    def _init_rate_climb(self):
        spd   = float(np.random.uniform(9, 13))
        alt0  = float(np.random.uniform(60, 300))
        d_alt = float(np.random.uniform(30, 120))
        rate  = float(np.random.uniform(1.5, 4.0))   # m/s climb rate
        tgt_pitch = math.asin(min(rate / spd, 0.35))
        self._state = AircraftState(
            pos=np.array([0.0, 200.0, alt0]),
            vel=np.array([0.0, -spd, 0.0]),
            pitch=0.0, roll=0.0, yaw=0.0, throttle_pos=0.65,
        )
        self._target = {'alt': alt0 + d_alt, 'rate': rate}
        self._cmd = np.array([spd, tgt_pitch, 0.0])

    def _init_rate_descent(self):
        spd   = float(np.random.uniform(8, 12))
        alt0  = float(np.random.uniform(150, 500))
        d_alt = float(np.random.uniform(30, 100))
        rate  = float(np.random.uniform(1.0, 3.5))
        tgt_pitch = -math.asin(min(rate / spd, 0.3))
        self._state = AircraftState(
            pos=np.array([0.0, 200.0, alt0]),
            vel=np.array([0.0, -spd, 0.0]),
            pitch=0.0, roll=0.0, yaw=0.0, throttle_pos=0.45,
        )
        self._target = {'alt': alt0 - d_alt, 'rate': rate}
        self._cmd = np.array([spd, tgt_pitch, 0.0])

    def _init_coordinated_turn(self):
        spd      = float(np.random.uniform(9, 13))
        alt      = float(np.random.uniform(100, 400))
        yaw0     = float(np.random.uniform(-math.pi, math.pi))
        d_yaw    = float(np.random.choice([-1, 1])) * float(np.random.uniform(40, 120))
        bank     = math.radians(abs(d_yaw) * 0.25)   # proportional bank
        bank     = min(bank, math.radians(35))
        bank    *= math.copysign(1, d_yaw)
        self._state = AircraftState(
            pos=np.array([0.0, 200.0, alt]),
            vel=np.array([math.sin(yaw0)*-spd, math.cos(yaw0)*-spd, 0.0]),
            pitch=0.0, roll=0.0, yaw=yaw0, throttle_pos=0.55,
        )
        tgt_yaw = yaw0 + math.radians(d_yaw)
        self._target = {'yaw': tgt_yaw, 'alt': alt}
        self._cmd = np.array([spd, 0.0, bank])

    def _init_speed_change(self):
        alt  = float(np.random.uniform(100, 400))
        spd0 = float(np.random.uniform(8, 10))
        spd1 = spd0 + float(np.random.choice([-1,1])) * float(np.random.uniform(2, 4))
        spd1 = float(np.clip(spd1, 5.0, 14.0))
        self._state = AircraftState(
            pos=np.array([0.0, 200.0, alt]),
            vel=np.array([0.0, -spd0, 0.0]),
            pitch=0.0, roll=0.0, yaw=0.0, throttle_pos=0.5,
        )
        self._target = {'speed': spd1, 'alt': alt}
        self._cmd = np.array([spd1, 0.0, 0.0])

    def _init_recovery(self):
        spd  = float(np.random.uniform(8, 12))
        alt  = float(np.random.uniform(150, 450))
        # unusual attitude: steep bank and/or pitch
        pitch = float(np.random.uniform(-0.5, 0.5))
        roll  = float(np.random.choice([-1,1])) * float(np.random.uniform(0.5, 1.2))
        self._state = AircraftState(
            pos=np.array([0.0, 200.0, alt]),
            vel=np.array([0.0, -spd, 0.0]),
            pitch=pitch, roll=roll, yaw=0.0, throttle_pos=0.5,
        )
        self._target = {'pitch': 0.0, 'roll': 0.0, 'alt': alt}
        self._cmd = np.array([spd, 0.0, 0.0])   # recover to wings-level

    # ----------------------------------------------------------
    # Observation
    # ----------------------------------------------------------

    def _scenario_onehot(self):
        vec = np.zeros(len(self.SCENARIOS), dtype=np.float32)
        if self._scenario in self.SCENARIOS:
            vec[self.SCENARIOS.index(self._scenario)] = 1.0
        return vec

    def _obs(self):
        s = self._state
        return np.concatenate([
            [s.airspeed, s.pos[2],
             s.pitch, s.roll, s.yaw,
             s.pitch_rate, s.roll_rate, s.yaw_rate,
             self._cmd[0], self._cmd[1], self._cmd[2]],
            self._scenario_onehot(),
        ]).astype(np.float32)

    # ----------------------------------------------------------
    # Scenario-specific reward
    # ----------------------------------------------------------

    def _reward(self):
        """
        All scenarios return reward in the same normalised range.
        Base: 0.0 per step.  Good step: up to +1.  Bad step: down to -1.
        Terminal success bonus: +10 (same for all).
        Terminal safety crash:  -20 (applied in step()).
        Consistent scale lets the shared value function learn reliably.
        """
        s  = self._state
        sc = self._scenario

        if sc == 'straight_level':
            speed_err = abs(s.airspeed - self._target['speed']) / 6.0      # norm by max plausible err
            alt_err   = abs(s.pos[2]   - self._target['alt'])   / 50.0
            r  =  1.0 - speed_err - alt_err
            r -= abs(s.roll)  / math.pi         # 0–1 range
            r -= abs(s.pitch) / (math.pi / 2)
            r  = float(np.clip(r, -1.0, 1.0))

        elif sc == 'rate_climb':
            rate_err = abs(float(s.vel[2]) - self._target['rate']) / 4.0
            r  = 1.0 - rate_err
            r -= abs(s.roll) / math.pi
            r  = float(np.clip(r, -1.0, 1.0))
            if s.pos[2] >= self._target['alt'] - 5:
                r += 10.0   # success

        elif sc == 'rate_descent':
            rate_err = abs(-float(s.vel[2]) - self._target['rate']) / 4.0
            r  = 1.0 - rate_err
            r -= abs(s.roll) / math.pi
            r  = float(np.clip(r, -1.0, 1.0))
            if s.pos[2] <= self._target['alt'] + 5:
                r += 10.0

        elif sc == 'coordinated_turn':
            yaw_err = abs(math.atan2(
                math.sin(s.yaw - self._target['yaw']),
                math.cos(s.yaw - self._target['yaw'])
            )) / math.pi                          # 0–1 range
            alt_err = abs(s.pos[2] - self._target['alt']) / 50.0
            r  = 1.0 - yaw_err - alt_err
            r  = float(np.clip(r, -1.0, 1.0))
            if yaw_err * math.pi < math.radians(5):
                r += 10.0

        elif sc == 'speed_change':
            speed_err = abs(s.airspeed - self._target['speed']) / 6.0
            alt_err   = abs(s.pos[2]   - self._target['alt'])   / 50.0
            r  = 1.0 - speed_err - alt_err
            r -= abs(s.roll) / math.pi
            r  = float(np.clip(r, -1.0, 1.0))
            if speed_err * 6.0 < 0.5:
                r += 10.0

        elif sc == 'recovery':
            pitch_err = abs(s.pitch)      / (math.pi / 2)   # 0–1
            roll_err  = abs(s.roll)       / math.pi
            rate_pen  = (abs(s.pitch_rate) + abs(s.roll_rate)) / 8.0
            r  = 1.0 - pitch_err - roll_err - rate_pen
            r  = float(np.clip(r, -1.0, 1.0))
            if abs(s.pitch) < math.radians(5) and abs(s.roll) < math.radians(10):
                r += 10.0

        else:
            r = 0.0

        return r

    # ----------------------------------------------------------
    # Gym interface
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._scenario = str(np.random.choice(self._active))
        PilotageEnv._last_scenario = self._scenario   # visible to training thread
        {
            'straight_level'  : self._init_straight_level,
            'rate_climb'      : self._init_rate_climb,
            'rate_descent'    : self._init_rate_descent,
            'coordinated_turn': self._init_coordinated_turn,
            'speed_change'    : self._init_speed_change,
            'recovery'        : self._init_recovery,
        }[self._scenario]()
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        self._state = self._phys.step(self._state, action, dt=0.1)
        s = self._state

        reward = self._reward()

        # ---- universal safety termination ----
        done = False
        if abs(s.pitch) > math.radians(75):
            reward -= 20.0; done = True
        if abs(s.roll)  > math.radians(100):
            reward -= 20.0; done = True
        if s.pos[2] < 10.0:
            reward -= 30.0; done = True
        if s.pos[2] > 700.0:
            reward -= 10.0; done = True

        truncated = (not done) and (self.steps >= self.MAX_STEPS)
        return self._obs(), reward, done, truncated, {}

    # ---- visualiser compatibility ----

    @property
    def pos(self):
        return self._state.pos.astype(np.float32)

    @property
    def vel(self):
        return self._state.vel.astype(np.float32)


# ============================================================
# LandingEnv
# ============================================================

class LandingEnv(gym.Env):
    """
    High-level navigation brain.

    Observation (7-D):
        pos_x, pos_y, pos_z,
        airspeed, yaw, pitch, roll

    Action (3-D): commands forwarded to the pilotage brain
        target_speed  [CMD_SPEED_LO, CMD_SPEED_HI]
        target_pitch  [CMD_PITCH_LO, CMD_PITCH_HI]
        target_roll   [CMD_ROLL_LO,  CMD_ROLL_HI]

    The pilotage model then decides the actual actuators.
    Training goal: consistently land on-strip with decent grade.
    Convergence: mean episode reward > CONVERGENCE_REWARD for
                 CONVERGENCE_WINDOW consecutive learn() calls.
    """

    CONVERGENCE_REWARD  = 800.0
    CONVERGENCE_WINDOW  = 40
    MAX_TIMESTEPS       = 5_000_000

    RWY_X  =  13.0
    RWY_Y0 = -25.0
    RWY_Y1 =  95.0

    CMD_SPEED_LO, CMD_SPEED_HI =  7.5, 14.0
    CMD_PITCH_LO, CMD_PITCH_HI = math.radians(-20), math.radians(12)
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
            low  = np.array([self.CMD_SPEED_LO, self.CMD_PITCH_LO, self.CMD_ROLL_LO],
                            dtype=np.float32),
            high = np.array([self.CMD_SPEED_HI, self.CMD_PITCH_HI, self.CMD_ROLL_HI],
                            dtype=np.float32),
        )

        self._state: AircraftState = None
        self._prev_horiz  = 0.0
        self._prev_alt    = 0.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_vel1   = -12.0
        self.steps = 0
        self.last_grade = ""
        self.last_score = 0.0
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
        s = self._state
        base = np.array([
            s.airspeed, s.pos[2],
            s.pitch, s.roll, s.yaw,
            s.pitch_rate, s.roll_rate, s.yaw_rate,
            cmd[0], cmd[1], cmd[2],
        ], dtype=np.float32)
        # pilotage policy was trained with 17-D obs (11 state + 6 scenario one-hot);
        # pad zeros — no active scenario when used as a command-follower
        return np.concatenate([base, np.zeros(len(PilotageEnv.SCENARIOS), dtype=np.float32)])

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

        # pilotage brain converts flight commands → actuators
        pilot_obs    = self._pilot_obs(cmd)
        actuators, _ = self._pilot.predict(pilot_obs, deterministic=True)

        # physics step
        self._state = self._phys.step(self._state, actuators, dt=0.1)

        altitude   = float(self._state.pos[2])
        curr_horiz = self._runway_horiz_dist()
        curr_alt   = max(0.0, altitude)
        pos_x      = float(self._state.pos[0])
        pos_y      = float(self._state.pos[1])
        vz         = float(self._state.vel[2])
        vy         = float(self._state.vel[1])

        # ---- step reward ----
        reward  = (self._prev_horiz - curr_horiz) * 3.0
        reward += (self._prev_alt   - curr_alt)   * 1.0
        reward -= abs(pos_x) * 0.02

        # smoothness
        delta   = cmd - self._prev_action
        reward -= (abs(float(delta[0])) * 0.08
                 + abs(vy - self._prev_vel1) * 0.05
                 + abs(float(delta[2])) * 0.05)

        # stall margin
        stall_margin = abs(vy) - 4.0
        if stall_margin < 4.0:
            reward -= (4.0 - stall_margin) * 0.15

        # heading alignment
        spd_h = math.sqrt(float(self._state.vel[0]) ** 2 + vy ** 2)
        if spd_h > 0.5:
            cross = abs(float(self._state.vel[0])) / spd_h
            aw    = max(0.0, 1.0 - altitude / 180.0)
            reward -= cross * 0.15 * (0.2 + 0.8 * aw)

        # vertical speed
        if vz > 0:
            reward -= vz * 0.5
        else:
            aw_vz = max(0.0, 1.0 - altitude / 180.0)
            reward -= abs(vz) * 0.08 * (0.1 + 0.9 * aw_vz)

        # out-of-bounds soft walls
        reward -= max(0.0, abs(pos_x) - self.RWY_X) ** 2 * 0.003
        reward -= max(0.0, self.RWY_Y0 - pos_y)     ** 2 * 0.003
        reward -= max(0.0, pos_y - 750.0)            ** 2 * 0.003

        self._prev_horiz  = curr_horiz
        self._prev_alt    = curr_alt
        self._prev_action = cmd.copy()
        self._prev_vel1   = vy

        done = False; truncated = False

        # ---- hard aborts ----
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

        # ---- touchdown ----
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

                self.last_grade = grade
                self.last_score = combined
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

    # ---- visualiser compatibility ----

    @property
    def pos(self):
        return self._state.pos.astype(np.float32)

    @property
    def vel(self):
        return self._state.vel.astype(np.float32)
