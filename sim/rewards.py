import math
import numpy as np

from sim.flight_modes import FlightMode
from sim.mission_manager import TargetInfo


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def compute_reward(mode: FlightMode,
                   state,
                   target: TargetInfo,
                   prev_pos: np.ndarray,
                   crashed: bool = False,
                   landed: bool  = False,
                   prev_wp_dist: float = None,
                   wp_arrived: bool = None) -> tuple:
    """
    Returns (total_reward: float, breakdown: dict).

    All modes are shaped to roughly [-2, 2] per step.
    breakdown maps each named reward term to its signed contribution.
    """
    if crashed:
        return -50.0, {"crash": -50.0}

    if mode == FlightMode.STABILIZE:
        return _stabilize(state)

    elif mode == FlightMode.ALTITUDE_HOLD:
        return _altitude_hold(state, target)

    elif mode == FlightMode.HEADING_HOLD:
        return _heading_hold(state, target)

    elif mode == FlightMode.WAYPOINT:
        return _waypoint(state, target, prev_pos,
                         prev_dist=prev_wp_dist, arrived_override=wp_arrived)

    elif mode == FlightMode.LOITER:
        return _loiter(state, target)

    elif mode == FlightMode.APPROACH:
        return _approach(state, target, prev_pos)

    elif mode == FlightMode.LANDING:
        return _landing(state, target, landed)

    elif mode == FlightMode.RECOVERY:
        return _recovery(state)

    return 0.0, {}


# ---------------------------------------------------------------- STABILIZE ---

def _stabilize(state) -> tuple:
    roll_r  = 1.0 - abs(state.roll)  / math.pi
    pitch_r = 1.0 - abs(state.pitch) / (math.pi / 2)

    # Strongly penalize angular rates — discourages twitchy / oscillating behavior
    roll_rate_p  = abs(state.roll_rate)  * 0.6
    pitch_rate_p = abs(state.pitch_rate) * 0.4
    yaw_rate_p   = abs(state.yaw_rate)   * 0.2
    rate_p = roll_rate_p + pitch_rate_p + yaw_rate_p

    # Small reward for maintaining safe airspeed
    spd_r = min(state.airspeed / 10.0, 1.0) * 0.15

    bd = {
        "roll_level":   round(0.5  * roll_r,  4),
        "pitch_level":  round(0.25 * pitch_r, 4),
        "rate_penalty": round(-rate_p,         4),
        "airspeed":     round(spd_r,           4),
    }
    total = float(np.clip(
        bd["roll_level"] + bd["pitch_level"] + bd["rate_penalty"] + bd["airspeed"],
        -2.0, 2.0))
    return total, bd


# ------------------------------------------------------------- ALTITUDE_HOLD --

_ALT_HOLD_TARGET_AIRSPEED = 14.0   # m/s


def _altitude_hold(state, target: TargetInfo) -> tuple:
    signed_err = state.pos[2] - target.altitude
    alt_err    = abs(signed_err)
    # Wider tolerance (30 m vs old 20 m) keeps gradient non-zero during the
    # climb/descent phase and prevents reward collapse on large initial errors.
    alt_r = math.exp(-alt_err / 30.0)
    # Mild vz penalty: don't punish necessary climb/descent corrections.
    vz_p  = abs(state.vel[2]) * 0.05

    spd_err = abs(state.airspeed - _ALT_HOLD_TARGET_AIRSPEED)
    spd_r   = math.exp(-spd_err / 3.0) * 0.3

    bd = {
        "altitude_reward": round(alt_r,       4),
        "altitude_error":  round(signed_err,  2),
        "vspd_penalty":    round(-vz_p,       4),
        "airspeed_reward": round(spd_r,       4),
    }
    total = float(np.clip(alt_r - vz_p + spd_r, -2.0, 2.0))
    return total, bd


# -------------------------------------------------------------- HEADING_HOLD --

def _heading_hold(state, target: TargetInfo) -> tuple:
    hdg_err = abs(_angle_diff(state.yaw, target.heading))
    hdg_r   = 1.0 - hdg_err / math.pi
    bank_p  = abs(state.roll) / (math.pi / 2) * 0.3

    bd = {
        "heading_reward": round(hdg_r,   4),
        "bank_penalty":   round(-bank_p, 4),
    }
    total = float(np.clip(hdg_r - bank_p, -2.0, 2.0))
    return total, bd


# ------------------------------------------------------------------ WAYPOINT --

def _waypoint(state, target: TargetInfo, prev_pos: np.ndarray,
              prev_dist: float = None, arrived_override: bool = None) -> tuple:
    tgt       = target.position
    curr_dist = float(np.linalg.norm(state.pos - tgt))
    # Use leg-local prev_dist if provided; fall back to prev_pos for legacy callers.
    prev_d    = prev_dist if prev_dist is not None else float(np.linalg.norm(prev_pos - tgt))
    progress  = (prev_d - curr_dist) * 0.3

    # arrived_override carries pre-mission-update arrival truth from the env;
    # without it, fall back to current distance (correct for single-WP training).
    arrived = arrived_override if arrived_override is not None else (curr_dist < 20.0)
    if arrived:
        progress = max(progress, 0.0)

    alt_err = abs(state.pos[2] - target.altitude)
    alt_p   = min(alt_err / 80.0, 0.5)
    bonus   = 50.0 if arrived else 0.0

    bd = {
        "progress":      round(progress,  4),
        "alt_penalty":   round(-alt_p,    4),
        "arrival_bonus": round(bonus,     4),
        "wp_prev_dist":  round(prev_d,    1),
        "wp_curr_dist":  round(curr_dist, 1),
    }
    total = float(np.clip(progress - alt_p + bonus, -3.0, 55.0))
    return total, bd


# -------------------------------------------------------------------- LOITER --

_LOITER_TARGET_SPEED = 15.0   # m/s — cruise at healthy energy, low AoA
_G = 9.81


def _loiter(state, target: TargetInfo) -> tuple:
    cx, cy = target.position[0], target.position[1]
    dx = state.pos[0] - cx
    dy = state.pos[1] - cy
    dist   = math.sqrt(dx * dx + dy * dy) + 1e-6
    radius = max(target.radius, 1.0)

    # Radius tracking (0.40 max)
    rad_err = abs(dist - radius)
    rad_r   = max(0.0, 1.0 - rad_err / radius) * 0.40

    # Tangential alignment (0.25 max)
    tx = -dy / dist
    ty =  dx / dist
    spd_h  = math.sqrt(state.vel[0] ** 2 + state.vel[1] ** 2) + 1e-6
    tang   = (state.vel[0] * tx + state.vel[1] * ty) / spd_h
    tang_r = max(0.0, tang) * 0.25

    # Altitude hold (0.10 max)
    alt_err = abs(state.pos[2] - target.altitude)
    alt_r   = max(0.0, 1.0 - alt_err / 20.0) * 0.10

    # Target loiter cruise speed — 15 m/s (0.10 max)
    v = max(state.airspeed, 1.0)
    spd_err = abs(state.airspeed - _LOITER_TARGET_SPEED)
    speed_r = math.exp(-spd_err / 3.0) * 0.10

    # Required bank for this orbit: tan(bank_target) = v² / (R·g)
    bank_target = math.atan(v ** 2 / (radius * _G))
    bank_err    = abs(abs(state.roll) - bank_target)

    # Coordinated turn reward: bank matches required orbit bank, weighted by tangential
    # alignment — no credit when not flying the orbit direction (0.15 max)
    coord_r = max(0.0, tang) * math.exp(-bank_err / math.radians(15)) * 0.15

    # Bank tracking reward: always active — drives policy to bank correctly
    # even during orbit acquisition, not just when tangential (0.15 max)
    bank_track_r = math.exp(-bank_err / math.radians(12)) * 0.15

    # Sideslip penalty: use exact g·tan(bank)/v formula.
    # After physics fix, yaw_rate converges to this value naturally;
    # penalty discourages any residual rudder-only steering.
    cr_safe      = max(math.cos(state.roll), 0.1)
    expected_yr  = (_G / v) * math.sin(state.roll) / cr_safe   # g·tan(bank)/v
    sideslip_err = abs(state.yaw_rate - expected_yr)
    sideslip_p   = min(sideslip_err * 0.5, 0.20) if abs(state.yaw_rate) > 0.02 else 0.0

    # Uncoordinated-turn penalty: penalize turning faster than the bank justifies.
    # Residual yaw_rate beyond the banked equilibrium = yaw-steering cheating.
    yr_excess = max(0.0, abs(state.yaw_rate) - abs(expected_yr))
    cheat_p   = min(yr_excess * 0.4, 0.20)

    bd = {
        "radius_reward":     round(rad_r,        4),
        "tangential_flight": round(tang_r,       4),
        "altitude_reward":   round(alt_r,        4),
        "speed_reward":      round(speed_r,      4),
        "coord_turn":        round(coord_r,       4),
        "bank_tracking":     round(bank_track_r,  4),
        "sideslip_pen":      round(-sideslip_p,   4),
        "cheat_turn_pen":    round(-cheat_p,      4),
    }
    total = float(np.clip(
        rad_r + tang_r + alt_r + speed_r + coord_r + bank_track_r
        - sideslip_p - cheat_p,
        -2.0, 2.0))
    return total, bd


# ------------------------------------------------------------------- APPROACH --

def _approach(state, target: TargetInfo, prev_pos: np.ndarray) -> tuple:
    rwy = target.position
    pos = state.pos

    prev_dist = float(np.linalg.norm(prev_pos - rwy))
    curr_dist = float(np.linalg.norm(pos       - rwy))
    progress  = (prev_dist - curr_dist) * 0.3

    hdg = target.heading
    cx  =  math.sin(hdg)
    cy  = -math.cos(hdg)
    rx  = rwy[0] - pos[0]
    ry  = rwy[1] - pos[1]
    lateral   = rx * cy - ry * cx
    lateral_p = min(abs(lateral) / 30.0, 1.0) * 0.4

    dx    = pos[0] - target.position[0]
    dy    = pos[1] - target.position[1]
    horiz = math.sqrt(dx * dx + dy * dy)
    ideal_alt = horiz * math.tan(math.radians(3.0))
    gs_err = pos[2] - ideal_alt
    if gs_err < -5:
        gs_r = gs_err / 10.0
    else:
        gs_r = max(0.0, 1.0 - abs(gs_err) / 30.0) * 0.3

    sink_p = max(0.0, -state.vel[2] - 3.0) * 0.1

    bd = {
        "progress":       round(progress,   4),
        "glide_slope":    round(gs_r,       4),
        "lateral_align":  round(-lateral_p, 4),
        "sink_rate":      round(-sink_p,    4),
    }
    total = float(np.clip(progress + gs_r - lateral_p - sink_p, -3.0, 3.0))
    return total, bd


# -------------------------------------------------------------------- LANDING --

def _landing(state, target: TargetInfo, landed: bool) -> tuple:
    if landed:
        pos   = state.pos
        rwy   = target.position
        x_err = abs(pos[0] - rwy[0])
        vz_err = abs(state.vel[2])
        vx_err = abs(state.vel[0])

        quality = (max(0.0, 1.0 - x_err  / 13.0) *
                   max(0.0, 1.0 - vz_err /  4.0) *
                   max(0.0, 1.0 - vx_err /  3.0))
        bd = {
            "touchdown_bonus": 50.0,
            "quality_bonus":   round(50.0 * quality, 4),
        }
        return float(50.0 + 50.0 * quality), bd

    rwy   = target.position
    lat_p = min(abs(state.pos[0] - rwy[0]) / 13.0, 1.0) * 0.3
    bd = {
        "step_penalty":   -0.01,
        "lateral_align":  round(-lat_p, 4),
    }
    total = float(np.clip(-0.01 - lat_p, -1.0, 0.0))
    return total, bd


# ------------------------------------------------------------------- RECOVERY --

def _recovery(state) -> tuple:
    stall_spd  = 6.0
    cruise_spd = 10.0

    spd   = state.airspeed
    spd_r = min(max(spd - stall_spd, 0.0) / cruise_spd, 1.0) * 0.5
    level_r = (1.0 - abs(state.roll)  / math.pi)        * 0.3
    pitch_r = max(0.0, 1.0 - abs(state.pitch) / (math.pi / 2)) * 0.2

    bd = {
        "airspeed_reward": round(spd_r,   4),
        "wings_level":     round(level_r, 4),
        "pitch_safe":      round(pitch_r, 4),
    }
    total = float(np.clip(spd_r + level_r + pitch_r, 0.0, 2.0))
    return total, bd
