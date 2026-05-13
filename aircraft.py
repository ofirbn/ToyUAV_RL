import math
import numpy as np


class AircraftState:
    """Mutable flight state passed through physics each step."""

    __slots__ = (
        'pos', 'vel',
        'pitch', 'roll', 'yaw',
        'pitch_rate', 'roll_rate', 'yaw_rate',
        'throttle_pos', 'flap_pos',
        'elevator_pos', 'aileron_pos', 'rudder_pos',
    )

    def __init__(self, pos, vel,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 pitch_rate=0.0, roll_rate=0.0, yaw_rate=0.0,
                 throttle_pos=0.5, flap_pos=0.0,
                 elevator_pos=0.0, aileron_pos=0.0, rudder_pos=0.0):
        self.pos          = np.array(pos,  dtype=float)
        self.vel          = np.array(vel,  dtype=float)
        self.pitch        = float(pitch)
        self.roll         = float(roll)
        self.yaw          = float(yaw)
        self.pitch_rate   = float(pitch_rate)
        self.roll_rate    = float(roll_rate)
        self.yaw_rate     = float(yaw_rate)
        self.throttle_pos = float(throttle_pos)
        self.flap_pos     = float(flap_pos)
        self.elevator_pos = float(elevator_pos)
        self.aileron_pos  = float(aileron_pos)
        self.rudder_pos   = float(rudder_pos)

    def copy(self):
        s = AircraftState.__new__(AircraftState)
        s.pos          = self.pos.copy()
        s.vel          = self.vel.copy()
        s.pitch        = self.pitch
        s.roll         = self.roll
        s.yaw          = self.yaw
        s.pitch_rate   = self.pitch_rate
        s.roll_rate    = self.roll_rate
        s.yaw_rate     = self.yaw_rate
        s.throttle_pos = self.throttle_pos
        s.flap_pos     = self.flap_pos
        s.elevator_pos = self.elevator_pos
        s.aileron_pos  = self.aileron_pos
        s.rudder_pos   = self.rudder_pos
        return s

    @property
    def airspeed(self):
        return float(np.linalg.norm(self.vel))


class AircraftPhysics:
    """
    Simplified but physically motivated fixed-wing UAV dynamics.

    Actuator vector: [throttle(0-1), elevator(-1,1), aileron(-1,1), rudder(-1,1), flaps(0-1)]

    Coordinate system (same as renderer):
        x  = lateral (right)
        y  = forward along approach (runway is at y~0, plane approaches from +y)
        z  = altitude (up)
    Heading yaw=0 means flying in the -y direction (toward runway).
    """

    # ---- airframe constants ----
    MASS       = 2.5     # kg
    WING_AREA  = 0.40    # m²
    RHO        = 1.225   # kg/m³  (sea-level)
    MAX_THRUST = 12.0    # N  (T/W ≈ 0.49)
    G          = 9.81    # m/s²

    # ---- control surface authority (heavy trainer UAV — slow to respond) ----
    ELEV_GAIN  = 0.45    # reduced: slow pitch response
    AIL_GAIN   = 0.70    # reduced: slower roll response, less aggressive bank
    RUDD_GAIN  = 0.06    # very weak: trim sideslip only, bank drives the turn

    # ---- very strong aerodynamic damping — rates decay quickly ----
    PITCH_DAMP = 7.0
    ROLL_DAMP  = 9.0
    YAW_DAMP   = 5.0

    # ---- static stability: strong restoring tendency to wings-level/trim ----
    PITCH_STAB = 2.5     # strong pitch trim tendency
    # ROLL_STAB reduced so bank is sustainable without constant aileron pressure.
    # With AIL_GAIN=0.8: max sustainable bank = AIL_GAIN/ROLL_STAB ≈ 41°.
    ROLL_STAB  = 1.2     # mild dihedral — allows coordinated loiter bank angles

    # ---- strong actuator lag: actual = 0.95*prev + 0.05*cmd per step ----
    ELEV_LAG     = 0.5   # 5% per 0.1s step → heavy lag
    AIL_LAG      = 0.5
    RUDD_LAG     = 0.5
    THROTTLE_LAG = 0.05   # 0.005 per step → 0.995*prev + 0.005*cmd
    FLAP_LAG     = 0.8

    # ---- surface rate limits (max change per second) ----
    SURF_RATE_LIMIT = 0.50  # units/s — slightly slower for smoother response

    # ---- stall model ----
    STALL_ALPHA      = math.radians(13)
    STALL_NOSE_DOWN  = 7.0   # strong nose-down tendency past stall
    Q_REF            = 0.5 * 1.225 * 10.0**2

    # ---- safe limits ----
    MAX_PITCH  = math.pi / 3
    MAX_ROLL   = math.pi / 2
    MIN_SPEED  = 1.0

    def step(self, state: AircraftState, actuators, dt: float = 0.1) -> AircraftState:
        throttle  = float(np.clip(actuators[0], 0.0, 1.0))
        elevator  = float(np.clip(actuators[1], -1.0, 1.0))
        aileron   = float(np.clip(actuators[2], -1.0, 1.0))
        rudder    = float(np.clip(actuators[3], -1.0, 1.0))
        flaps     = float(np.clip(actuators[4],  0.0, 1.0))

        # --- actuator lags with rate limiting (heavy lag: ~5% per step) ---
        max_surf = self.SURF_RATE_LIMIT * dt
        def _lag(cur, cmd, lag):
            delta = float(np.clip((cmd - cur) * lag * dt, -max_surf, max_surf))
            return cur + delta

        state.throttle_pos = float(np.clip(
            state.throttle_pos + (throttle - state.throttle_pos) * self.THROTTLE_LAG * dt,
            0.0, 1.0))
        state.flap_pos     = float(np.clip(
            state.flap_pos + (flaps - state.flap_pos) * self.FLAP_LAG * dt,
            0.0, 1.0))
        state.elevator_pos = float(np.clip(_lag(state.elevator_pos, elevator, self.ELEV_LAG), -1.0, 1.0))
        state.aileron_pos  = float(np.clip(_lag(state.aileron_pos,  aileron,  self.AIL_LAG),  -1.0, 1.0))
        state.rudder_pos   = float(np.clip(_lag(state.rudder_pos,   rudder,   self.RUDD_LAG), -1.0, 1.0))

        # smoothed surface positions drive all angular dynamics
        eff_elev = state.elevator_pos
        eff_ail  = state.aileron_pos
        eff_rudd = state.rudder_pos

        # --- dynamic pressure — gates all aero forces and control authority ---
        v_act   = float(np.linalg.norm(state.vel))
        v       = max(v_act, self.MIN_SPEED)
        q       = 0.5 * self.RHO * v_act ** 2
        q_ratio = q / self.Q_REF   # 1.0 at cruise (10 m/s), 0.25 at half speed

        # --- AoA for stall model (needed before angular dynamics for nose-down) ---
        vh    = math.sqrt(state.vel[0] ** 2 + state.vel[1] ** 2)
        gamma = math.atan2(state.vel[2], max(vh, 0.1))
        alpha = state.pitch - gamma

        # --- angular dynamics: gains + restoring moments scale with q_ratio ---
        p_acc = (eff_elev * self.ELEV_GAIN - self.PITCH_STAB * state.pitch) * q_ratio \
                - self.PITCH_DAMP * state.pitch_rate
        r_acc = (eff_ail  * self.AIL_GAIN  - self.ROLL_STAB  * state.roll)  * q_ratio \
                - self.ROLL_DAMP  * state.roll_rate

        # Coordinated-turn equilibrium: yaw_rate naturally settles at g·tan(bank)/v.
        # Rudder adds a small perturbation about this equilibrium; it cannot generate
        # significant turn curvature independently of bank angle.
        cr_safe  = max(math.cos(state.roll), 0.1)
        coord_yr = (self.G / v) * math.sin(state.roll) / cr_safe   # g·tan(bank)/v
        y_acc = eff_rudd * self.RUDD_GAIN * q_ratio \
                - self.YAW_DAMP * (state.yaw_rate - coord_yr)

        # --- natural nose-down moment in stall (prevents hanging at high AoA) ---
        if alpha > self.STALL_ALPHA:
            stall_excess = alpha - self.STALL_ALPHA
            p_acc -= stall_excess * self.STALL_NOSE_DOWN

        state.pitch_rate += p_acc * dt
        state.roll_rate  += r_acc * dt
        state.yaw_rate   += y_acc * dt

        state.pitch = float(np.clip(state.pitch + state.pitch_rate * dt,
                                    -self.MAX_PITCH, self.MAX_PITCH))
        state.roll  = float(np.clip(state.roll  + state.roll_rate  * dt,
                                    -self.MAX_ROLL,  self.MAX_ROLL))
        # yaw_rate already converges to g·tan(bank)/v via y_acc above;
        # no separate coord-turn shortcut needed.
        state.yaw  += state.yaw_rate * dt

        # --- aerodynamics: CL peaks at STALL_ALPHA then drops smoothly ---
        if alpha <= self.STALL_ALPHA:
            CL = 0.4 + 4.0 * alpha + 1.2 * state.flap_pos
        else:
            excess  = alpha - self.STALL_ALPHA
            CL_peak = 0.4 + 4.0 * self.STALL_ALPHA + 1.2 * state.flap_pos
            # Gentler drop — 1.2× excess (was 2.0×) so stall is recoverable
            CL = CL_peak * max(0.0, 1.0 - 1.2 * excess)
        CL = float(np.clip(CL, -0.2, 2.2))

        CD = 0.135 + 0.06 * CL ** 2 + 0.04 * state.flap_pos
        if alpha > self.STALL_ALPHA:
            # Increased drag in stall — slows aircraft, aids recovery
            CD += 1.2 * (alpha - self.STALL_ALPHA)

        F_lift   = q * self.WING_AREA * CL
        F_drag   = q * self.WING_AREA * CD
        F_thrust = state.throttle_pos * self.MAX_THRUST

        # --- force components in world frame ---
        cy, sy = math.cos(state.yaw), math.sin(state.yaw)
        cp, sp = math.cos(state.pitch), math.sin(state.pitch)
        tx =  sy * cp * F_thrust
        ty = -cy * cp * F_thrust
        tz =  sp      * F_thrust

        cr, sr = math.cos(state.roll), math.sin(state.roll)
        lx =  sr * F_lift
        ly =  0.0
        lz =  cr * F_lift

        vel_norm = state.vel / (np.linalg.norm(state.vel) + 1e-6)
        dx, dy, dz = -F_drag * vel_norm

        gz = -self.G * self.MASS

        # --- sideslip damping (keel / weathervane effect — stronger) ---
        lat_x =  math.cos(state.yaw)
        lat_y =  math.sin(state.yaw)
        v_lat = state.vel[0] * lat_x + state.vel[1] * lat_y
        F_keel = -(q / max(v_act, self.MIN_SPEED)) * self.WING_AREA * 0.5 * v_lat
        kx = F_keel * lat_x
        ky = F_keel * lat_y

        ax = (tx + lx + dx + kx) / self.MASS
        ay = (ty + ly + dy + ky) / self.MASS
        az = (tz + lz + dz + gz) / self.MASS

        state.vel = state.vel + np.array([ax, ay, az]) * dt
        state.pos = state.pos + state.vel * dt

        return state


class SimplePhysics:
    """
    Arcade kinematics for reward / learning debugging.
    Zero aerodynamics, zero gravity, fully decoupled controls.

    throttle [0,1]  → forward speed  0..MAX_SPEED m/s   (first-order lag)
    elevator [-1,1] → climb rate    -MAX_VZ..MAX_VZ m/s  (first-order lag)
    aileron  [-1,1] → yaw rate      -MAX_YAW_RATE..+MAX_YAW_RATE rad/s (first-order lag)

    All three channels have lag so bang-bang policy outputs get filtered
    before they move the aircraft — prevents high-frequency oscillation.
    If the policy does not learn with this model the reward is wrong.
    """

    MAX_SPEED    = 15.0   # m/s forward
    MAX_VZ       =  5.0   # m/s climb/descent
    MAX_YAW_RATE =  1.0   # rad/s at full aileron
    SPD_LAG      =  1.0   # forward speed  time-constant (1/s) — was 2.0
    VZ_LAG       =  1.5   # climb rate     time-constant (1/s) — was 3.0
    YR_LAG       =  2.0   # yaw rate       time-constant (1/s) — was 3.0
    PITCH_SMOOTH =  4.0   # cosmetic pitch smoothing  (1/s)

    def step(self, state: AircraftState, actuators, dt: float = 0.1) -> AircraftState:
        throttle = float(np.clip(actuators[0],  0.0, 1.0))
        elevator = float(np.clip(actuators[1], -1.0, 1.0))
        aileron  = float(np.clip(actuators[2], -1.0, 1.0))

        spd_h  = math.sqrt(state.vel[0] ** 2 + state.vel[1] ** 2)
        cur_vz = float(state.vel[2])

        # All three channels converge toward commanded values with first-order lag
        new_spd = spd_h        + (throttle * self.MAX_SPEED    - spd_h)        * self.SPD_LAG * dt
        new_vz  = cur_vz       + (elevator * self.MAX_VZ       - cur_vz)       * self.VZ_LAG  * dt
        new_yr  = state.yaw_rate + (aileron * self.MAX_YAW_RATE - state.yaw_rate) * self.YR_LAG  * dt

        state.yaw_rate = new_yr
        state.yaw     += new_yr * dt

        state.vel[0] =  math.sin(state.yaw) * new_spd
        state.vel[1] = -math.cos(state.yaw) * new_spd
        state.vel[2] =  new_vz

        # Cosmetic attitude — all smoothed so bang-bang inputs don't flicker.
        target_pitch = math.atan2(new_vz, max(new_spd, 0.1)) * 0.6
        target_roll  = new_yr / max(self.MAX_YAW_RATE, 0.01) * 0.4
        k_cos        = min(1.0, self.PITCH_SMOOTH * dt)
        state.pitch += (target_pitch - state.pitch) * k_cos
        state.roll  += (target_roll  - state.roll)  * k_cos

        # throttle_pos: use same slow lag as speed so it doesn't flicker
        state.throttle_pos += (throttle - state.throttle_pos) * self.SPD_LAG * dt

        state.pos = state.pos + state.vel * dt
        return state
