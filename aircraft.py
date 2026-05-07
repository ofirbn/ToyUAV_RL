import math
import numpy as np


class AircraftState:
    """Mutable flight state passed through physics each step."""

    __slots__ = (
        'pos', 'vel',
        'pitch', 'roll', 'yaw',
        'pitch_rate', 'roll_rate', 'yaw_rate',
        'throttle_pos', 'flap_pos',
    )

    def __init__(self, pos, vel,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 pitch_rate=0.0, roll_rate=0.0, yaw_rate=0.0,
                 throttle_pos=0.5, flap_pos=0.0):
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
    MAX_THRUST = 12.0    # N  (T/W ≈ 0.49 — fixed-wing cannot hover on thrust alone)
    G          = 9.81    # m/s²

    # ---- actuator effectiveness (rated at Q_REF) ----
    ELEV_GAIN  =  2.0    # pitch angular accel per unit elevator (rad/s²)
    AIL_GAIN   =  3.5    # roll  angular accel per unit aileron  (rad/s²)
    RUDD_GAIN  =  0.6    # yaw   angular accel per unit rudder   (rad/s²)

    # ---- aerodynamic damping ----
    PITCH_DAMP = 1.8
    ROLL_DAMP  = 2.5
    YAW_DAMP   = 1.2

    # ---- static stability (restoring moments) ----
    # Without these the aircraft is pitch/roll neutral: a held input spins
    # forever instead of settling at a trim angle.  With them, elevator and
    # aileron act as angle commands: trim_pitch = elevator * ELEV_GAIN/PITCH_STAB
    # and trim_roll = aileron * AIL_GAIN/ROLL_STAB.
    PITCH_STAB = 2.0    # = ELEV_GAIN  →  elevator 0.15 trims to 0.15 rad pitch
    ROLL_STAB  = 3.5    # = AIL_GAIN   →  aileron  0.30 trims to 0.30 rad bank

    # ---- actuator lags (time constant, s⁻¹) ----
    THROTTLE_LAG = 3.0   # throttle responds quickly
    FLAP_LAG     = 0.8   # flaps deploy slowly

    # ---- stall model ----
    STALL_ALPHA = math.radians(14)        # AoA where CL peaks (~0.244 rad)
    Q_REF       = 0.5 * 1.225 * 10.0**2  # dynamic pressure at 10 m/s (61.25 Pa)

    # ---- safe limits ----
    MAX_PITCH  = math.pi / 3      # ±60°
    MAX_ROLL   = math.pi / 2      # ±90°
    MIN_SPEED  = 1.0              # floor for q calculation only — state.vel is never rescaled

    def step(self, state: AircraftState, actuators, dt: float = 0.1) -> AircraftState:
        throttle  = float(np.clip(actuators[0], 0.0, 1.0))
        elevator  = float(np.clip(actuators[1], -1.0, 1.0))
        aileron   = float(np.clip(actuators[2], -1.0, 1.0))
        rudder    = float(np.clip(actuators[3], -1.0, 1.0))
        flaps     = float(np.clip(actuators[4],  0.0, 1.0))

        # --- actuator lags ---
        state.throttle_pos += (throttle - state.throttle_pos) * self.THROTTLE_LAG * dt
        state.flap_pos     += (flaps    - state.flap_pos)     * self.FLAP_LAG     * dt

        # --- dynamic pressure gates all aero forces and control authority ---
        # FIX 1: control surfaces lose effectiveness as airspeed drops;
        #        at low q the elevator can no longer hold a high-AoA attitude.
        v_act   = float(np.linalg.norm(state.vel))
        v       = max(v_act, self.MIN_SPEED)   # floored only for turn-rate / gamma calcs
        q       = 0.5 * self.RHO * v_act ** 2  # actual speed — zero lift at zero airspeed
        q_ratio = q / self.Q_REF   # 1.0 at 10 m/s, ~0.25 at 5 m/s

        # --- angular dynamics: gains + restoring moments scale with q_ratio ---
        p_acc = (elevator * self.ELEV_GAIN - self.PITCH_STAB * state.pitch) * q_ratio \
                - self.PITCH_DAMP * state.pitch_rate
        r_acc = (aileron  * self.AIL_GAIN  - self.ROLL_STAB  * state.roll)  * q_ratio \
                - self.ROLL_DAMP  * state.roll_rate
        y_acc = rudder   * self.RUDD_GAIN * q_ratio - self.YAW_DAMP   * state.yaw_rate

        state.pitch_rate += p_acc * dt
        state.roll_rate  += r_acc * dt
        state.yaw_rate   += y_acc * dt

        state.pitch = float(np.clip(state.pitch + state.pitch_rate * dt,
                                    -self.MAX_PITCH, self.MAX_PITCH))
        state.roll  = float(np.clip(state.roll  + state.roll_rate  * dt,
                                    -self.MAX_ROLL,  self.MAX_ROLL))
        state.yaw  += state.yaw_rate * dt

        # --- coordinated turn ---
        state.yaw += (self.G / v) * math.sin(state.roll) * dt

        # --- aerodynamics ---
        vh    = math.sqrt(state.vel[0] ** 2 + state.vel[1] ** 2)
        gamma = math.atan2(state.vel[2], max(vh, 0.1))
        alpha = state.pitch - gamma

        # FIX 2: CL peaks at STALL_ALPHA then drops — no free lift at high AoA
        if alpha <= self.STALL_ALPHA:
            CL = 0.4 + 4.0 * alpha + 1.2 * state.flap_pos
        else:
            excess  = alpha - self.STALL_ALPHA
            CL_peak = 0.4 + 4.0 * self.STALL_ALPHA + 1.2 * state.flap_pos
            CL      = CL_peak * max(0.0, 1.0 - 2.5 * excess)
        CL = float(np.clip(CL, -0.2, 2.2))

        # Drag uses CL² induced term (physically correct) + higher CD0 so that
        # cruise requires ~30-35% throttle rather than ~5% (which made the tanh
        # output centre of 50% wildly wrong, preventing throttle from being learned).
        CD = 0.08 + 0.06 * CL ** 2 + 0.04 * state.flap_pos
        if alpha > self.STALL_ALPHA:
            CD += 0.8 * (alpha - self.STALL_ALPHA)

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
        # lx sign: positive roll → yaw increases (left turn from -y heading) →
        # centripetal force must be in +x direction → lx must be positive.
        # Old sign (-sr) was wrong: lift fought the turn, forcing the keel to
        # do all the work and braking horizontal speed.
        lx =  sr * cp * F_lift
        ly =  0.0
        lz =  cr * cp * F_lift

        vel_norm = state.vel / (np.linalg.norm(state.vel) + 1e-6)
        dx, dy, dz = -F_drag * vel_norm

        gz = -self.G * self.MASS

        # --- sideslip damping (keel / vertical fin weathervane effect) ---
        # Keeps velocity direction aligned with heading.
        # Physically: F_side = q * S * CY * beta  where beta = v_lat / v
        #           = (q/v) * S * CY * v_lat
        # Must use q/v (not q) — using q directly gave 183 N from a tiny
        # misalignment, braking the plane to 0 horizontal speed in 2 steps.
        lat_x =  math.cos(state.yaw)
        lat_y =  math.sin(state.yaw)
        v_lat = state.vel[0] * lat_x + state.vel[1] * lat_y
        F_keel = -(q / max(v_act, self.MIN_SPEED)) * self.WING_AREA * 0.2 * v_lat
        kx = F_keel * lat_x
        ky = F_keel * lat_y

        ax = (tx + lx + dx + kx) / self.MASS
        ay = (ty + ly + dy + ky) / self.MASS
        az = (tz + lz + dz + gz) / self.MASS

        state.vel = state.vel + np.array([ax, ay, az]) * dt

        # No velocity floor on state.vel — division-by-zero is already handled:
        #   q uses  max(norm, MIN_SPEED),  drag uses  +1e-6,  gamma uses  max(vh, 0.1).
        # Rescaling vel to MIN_SPEED was inflating upward drift into a climb.
        state.pos = state.pos + state.vel * dt

        return state
