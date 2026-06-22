"""
Classical deterministic teacher autopilot for behavior cloning demonstrations.

Each mode uses a PID-style controller tuned to produce stable, sensible flight.
The teacher need not be perfect — just consistent enough for supervised pre-training.

API:
    teacher = TeacherAutopilot()
    action = teacher.act(obs, phase, target=target_info)   # → [elev, ail, rud, thr]
"""

import math
import numpy as np

from sim.flight_modes import FlightMode


def _adiff(a: float, b: float) -> float:
    """Shortest signed difference a − b wrapped to (−π, π]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def _clip(v, lo=-1.0, hi=1.0):
    return float(np.clip(v, lo, hi))


class TeacherAutopilot:
    """
    PID-style autopilot for all 8 flight modes.

    Inputs come from the 31-D observation vector (same format as FixedWingEnv).
    Obs layout:
        [6]  roll       (rad)
        [7]  pitch      (rad)
        [8]  yaw        (rad)
        [9]  pitch_rate / 5
        [10] roll_rate  / 5
        [11] yaw_rate   / 5
        [12] airspeed   / 20
        [13] target_dx  / 500   (target.pos - state.pos, x)
        [14] target_dy  / 500
        [15] target_dz  / 500
        [16] target_dist / 500
        [17] heading_err / π    (target.heading − yaw, wrapped)
        [18] altitude_err / 100 (state.pos_z − target.altitude)
        [19:27] mode one-hot
    """

    # ── shared PID gains ──────────────────────────────────────────────────────
    _KP_ROLL  = 2.5
    _KD_ROLL  = 0.40
    _KP_PITCH = 1.5
    _KD_PITCH = 0.25

    def act(self, obs: np.ndarray, phase, target=None) -> np.ndarray:
        """
        Returns np.float32 array [elevator, aileron, rudder, throttle].

        phase: FlightMode enum OR phase string ('stabilize', 'waypoint', …).
        target: optional TargetInfo — needed for loiter radius.
        """
        roll       = float(obs[6])
        pitch      = float(obs[7])
        yaw        = float(obs[8])
        pitch_rate = float(obs[9])  * 5.0
        roll_rate  = float(obs[10]) * 5.0
        yaw_rate   = float(obs[11]) * 5.0
        airspeed   = float(obs[12]) * 20.0
        altitude   = float(obs[2])  * 500.0   # absolute Z (pos_n[2] * 500)
        vert_speed = float(obs[5])  * 20.0    # vz component
        dx         = float(obs[13]) * 500.0   # target − pos, x
        dy         = float(obs[14]) * 500.0
        dist       = float(obs[16]) * 500.0
        hdg_err    = float(obs[17]) * math.pi  # target.heading − yaw
        alt_err    = float(obs[18]) * 100.0    # pos_z − target_alt

        rud = _clip(-0.30 * yaw_rate)
        ail = _clip(-self._KP_ROLL * roll - self._KD_ROLL * roll_rate)

        # Resolve mode first — safety checks depend on it
        if isinstance(phase, FlightMode):
            mode = phase
        else:
            mode_oh  = obs[19:27]
            mode_int = int(np.argmax(mode_oh))
            mode     = FlightMode(mode_int)

        is_landing = (mode == FlightMode.LANDING or
                      (isinstance(phase, str) and 'landing' in phase))
        is_approach = (mode == FlightMode.APPROACH or
                       (isinstance(phase, str) and 'approach' in phase))
        on_final = is_landing or is_approach

        # ── safety envelope ──────────────────────────────────────────────────
        # Stall guard: skip on final (near-stall speed expected on approach/landing)
        if airspeed < 7.5 and not on_final:
            elev = _clip(self._KP_PITCH * (-0.15 - pitch) - self._KD_PITCH * pitch_rate)
            return np.array([elev, ail, rud, 0.95], dtype=np.float32)

        # Ground proximity: skip on final (intentionally low on approach/landing)
        if not on_final and (altitude < 22.0 or (altitude < 35.0 and vert_speed < -4.0)):
            elev = _clip(self._KP_PITCH * (0.30 - pitch) - self._KD_PITCH * pitch_rate)
            return np.array([elev, ail, rud, 0.90], dtype=np.float32)

        # Extreme bank: recover wings immediately
        if abs(roll) > math.radians(72):
            return self._recovery(roll, pitch, roll_rate, pitch_rate, yaw_rate, airspeed)

        if mode == FlightMode.STABILIZE or (isinstance(phase, str) and 'stabilize' in phase):
            return self._stabilize(roll, pitch, roll_rate, pitch_rate, yaw_rate)

        if mode == FlightMode.ALTITUDE_HOLD or (isinstance(phase, str) and 'altitude' in phase):
            return self._altitude_hold(roll, pitch, roll_rate, pitch_rate, yaw_rate,
                                       airspeed, alt_err)

        if mode == FlightMode.HEADING_HOLD or (isinstance(phase, str) and 'heading' in phase):
            return self._heading_hold(roll, pitch, roll_rate, pitch_rate, yaw_rate, hdg_err)

        if mode == FlightMode.WAYPOINT or (isinstance(phase, str) and 'waypoint' in phase):
            return self._waypoint(roll, pitch, roll_rate, pitch_rate, yaw_rate, hdg_err, alt_err)

        if mode == FlightMode.LOITER or (isinstance(phase, str) and 'loiter' in phase):
            desired_r = float(target.radius) if target is not None else 50.0
            return self._loiter(roll, pitch, roll_rate, pitch_rate, yaw_rate,
                                airspeed, yaw, dx, dy, dist, alt_err, desired_r)

        if mode == FlightMode.APPROACH or (isinstance(phase, str) and 'approach' in phase):
            app_hdg_err = self._approach_hdg(dx, dy, dist, yaw, hdg_err)
            return self._approach(roll, pitch, roll_rate, pitch_rate, yaw_rate,
                                  airspeed, app_hdg_err, alt_err)

        if mode == FlightMode.RECOVERY or (isinstance(phase, str) and 'recovery' in phase):
            return self._recovery(roll, pitch, roll_rate, pitch_rate, yaw_rate, airspeed)

        if is_landing:
            app_hdg_err = self._approach_hdg(dx, dy, dist, yaw, hdg_err)
            return self._landing_ctrl(roll, pitch, roll_rate, pitch_rate, yaw_rate,
                                      airspeed, app_hdg_err, alt_err)

        return self._stabilize(roll, pitch, roll_rate, pitch_rate, yaw_rate)

    # ── per-mode controllers ──────────────────────────────────────────────────

    def _stabilize(self, roll, pitch, roll_rate, pitch_rate, yaw_rate):
        aileron  = _clip(-self._KP_ROLL  * roll  - self._KD_ROLL  * roll_rate)
        elevator = _clip( self._KP_PITCH * (-pitch + math.radians(2)) - self._KD_PITCH * pitch_rate)
        rudder   = _clip(-0.30 * yaw_rate)
        return np.array([elevator, aileron, rudder, 0.55], dtype=np.float32)

    def _altitude_hold(self, roll, pitch, roll_rate, pitch_rate, yaw_rate,
                       airspeed, alt_err):
        desired_pitch = _clip(-alt_err * 0.018, -0.40, 0.35)
        pitch_err     = desired_pitch - pitch
        elevator      = _clip(self._KP_PITCH * pitch_err - self._KD_PITCH * pitch_rate)
        aileron       = _clip(-self._KP_ROLL * roll - self._KD_ROLL * roll_rate)
        rudder        = _clip(-0.30 * yaw_rate)
        throttle      = _clip(0.60 + (14.0 - airspeed) * 0.05, 0.30, 0.90)
        return np.array([elevator, aileron, rudder, throttle], dtype=np.float32)

    def _heading_hold(self, roll, pitch, roll_rate, pitch_rate, yaw_rate, hdg_err):
        desired_bank = _clip(hdg_err * 1.5, -0.52, 0.52)
        roll_err     = desired_bank - roll
        aileron      = _clip(self._KP_ROLL * roll_err - self._KD_ROLL * roll_rate)
        bank_comp    = abs(roll) * 0.12
        elevator     = _clip(self._KP_PITCH * (-pitch + math.radians(2) + bank_comp)
                             - self._KD_PITCH * pitch_rate)
        rudder       = _clip(-0.30 * yaw_rate)
        return np.array([elevator, aileron, rudder, 0.55], dtype=np.float32)

    def _waypoint(self, roll, pitch, roll_rate, pitch_rate, yaw_rate, hdg_err, alt_err):
        desired_bank = _clip(hdg_err * 2.0, -0.65, 0.65)
        roll_err     = desired_bank - roll
        aileron      = _clip(self._KP_ROLL * roll_err - self._KD_ROLL * roll_rate)
        desired_pitch = _clip(-alt_err * 0.012, -0.35, 0.30)
        pitch_err     = desired_pitch - pitch
        elevator      = _clip(self._KP_PITCH * pitch_err - self._KD_PITCH * pitch_rate)
        rudder        = _clip(-0.30 * yaw_rate)
        return np.array([elevator, aileron, rudder, 0.55], dtype=np.float32)

    def _loiter(self, roll, pitch, roll_rate, pitch_rate, yaw_rate,
                airspeed, yaw, dx, dy, dist, alt_err, desired_radius):
        # Aircraft position relative to loiter center
        x_from = -dx
        y_from = -dy

        # Desired tangent heading for CW orbit
        tangent_yaw = math.atan2(y_from, x_from)
        hdg_err_tan = _adiff(tangent_yaw, yaw)

        # Bank toward tangent + mild radial correction
        radius_err   = dist - desired_radius
        bank_base    = _clip(hdg_err_tan * 1.5, -0.60, 0.60)
        bank_radial  = _clip(-radius_err * 0.008, -0.20, 0.20)
        desired_bank = _clip(bank_base + bank_radial, -0.65, 0.65)

        roll_err  = desired_bank - roll
        aileron   = _clip(self._KP_ROLL * roll_err - self._KD_ROLL * roll_rate)
        bank_comp = abs(roll) * 0.14
        desired_pitch = _clip(-alt_err * 0.012, -0.35, 0.25)
        elevator  = _clip(self._KP_PITCH * (desired_pitch - pitch + math.radians(2) + bank_comp)
                          - self._KD_PITCH * pitch_rate)
        rudder    = _clip(-0.30 * yaw_rate)
        throttle  = _clip(0.65 + (15.0 - airspeed) * 0.04, 0.40, 0.90)
        return np.array([elevator, aileron, rudder, throttle], dtype=np.float32)

    def _landing_ctrl(self, roll, pitch, roll_rate, pitch_rate, yaw_rate,
                      airspeed, hdg_err, alt_err):
        """Landing: 3-deg glideslope to touchdown, reduced throttle to slow on short final.

        In this physics model vz is commanded directly by elevator, so the glideslope
        formula (nose-down elevator) naturally carries the aircraft through z=0 to -1 m
        (MIN_ALTITUDE).  A flare (pitch-up) would reverse the elevator sign and cause
        a climb, so we don't do one.
        """
        # alt_err == actual altitude AGL (target.altitude = 0)
        desired_pitch = _clip(-alt_err * 0.015 - math.radians(3), -0.38, 0.05)
        pitch_err     = desired_pitch - pitch
        elevator      = _clip(self._KP_PITCH * pitch_err - self._KD_PITCH * pitch_rate)
        desired_bank  = _clip(hdg_err * 1.5, -0.35, 0.35)
        roll_err      = desired_bank - roll
        aileron       = _clip(self._KP_ROLL * roll_err - self._KD_ROLL * roll_rate)
        rudder        = _clip(-0.30 * yaw_rate)
        throttle      = _clip(0.20 + (8.0 - airspeed) * 0.04, 0.05, 0.40)
        return np.array([elevator, aileron, rudder, throttle], dtype=np.float32)

    def _approach_hdg(self, dx, dy, dist, yaw, hdg_err) -> float:
        """Bearing-to-target heading error when far; runway heading error when close."""
        if dist > 30.0:
            bearing = math.atan2(dx, -dy)
            return _adiff(bearing, yaw)
        return hdg_err

    def _approach(self, roll, pitch, roll_rate, pitch_rate, yaw_rate,
                  airspeed, hdg_err, alt_err,
                  tgt_speed=9.5, tgt_thr=0.38):
        desired_pitch = _clip(-alt_err * 0.015 - math.radians(3), -0.42, 0.10)
        pitch_err     = desired_pitch - pitch
        elevator      = _clip(self._KP_PITCH * pitch_err - self._KD_PITCH * pitch_rate)
        desired_bank  = _clip(hdg_err * 1.5, -0.40, 0.40)
        roll_err      = desired_bank - roll
        aileron       = _clip(self._KP_ROLL * roll_err - self._KD_ROLL * roll_rate)
        rudder        = _clip(-0.30 * yaw_rate)
        throttle      = _clip(tgt_thr + (tgt_speed - airspeed) * 0.04, 0.10, 0.80)
        return np.array([elevator, aileron, rudder, throttle], dtype=np.float32)

    def _recovery(self, roll, pitch, roll_rate, pitch_rate, yaw_rate, airspeed):
        aileron  = _clip(-self._KP_ROLL * roll - self._KD_ROLL * roll_rate)
        elevator = _clip(-0.30) if airspeed < 8.0 else \
                   _clip(-self._KP_PITCH * pitch - self._KD_PITCH * pitch_rate)
        rudder   = _clip(-0.30 * yaw_rate)
        return np.array([elevator, aileron, rudder, 0.80], dtype=np.float32)
