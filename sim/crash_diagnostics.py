"""
Crash diagnostics: classify crash episodes and save telemetry to disk.

Classification hierarchy (checked in order, first match wins):
    STALL          – sustained stall (stall_steps >= 15) caused the crash
    CEILING_EXCEED – aircraft exceeded MAX_ALTITUDE
    SPIRAL         – high roll + high airspeed → spiral dive
    OVERBANK       – excessive roll into ground (low speed)
    ENERGY_LOSS    – insufficient airspeed at ground contact
    OSCILLATION    – control oscillations led to ground contact
    GROUND_IMPACT  – general ground contact in non-landing mode
    UNKNOWN        – none of the above matched
"""

import json
import math
import os
import time
from enum import Enum


class CrashType(str, Enum):
    STALL          = "STALL"
    CEILING_EXCEED = "CEILING_EXCEED"
    SPIRAL         = "SPIRAL"
    OVERBANK       = "OVERBANK"
    ENERGY_LOSS    = "ENERGY_LOSS"
    OSCILLATION    = "OSCILLATION"
    GROUND_IMPACT  = "GROUND_IMPACT"
    UNKNOWN        = "UNKNOWN"


_SPIRAL_ROLL   = math.radians(50)   # |roll| > 50° AND fast → SPIRAL
_SPIRAL_SPEED  = 12.0               # airspeed threshold for spiral
_OVERBANK_ROLL = math.radians(70)   # |roll| > 70° → OVERBANK / SPIRAL
_OSC_THRESHOLD = 0.12               # ctrl_osc_ep_mean threshold for OSCILLATION


def classify_crash(stall_steps: int, state, info: dict,
                   stall_speed: float = 6.0) -> CrashType:
    """
    Determine the primary cause of a crash from the final-step state.

    Parameters
    ----------
    stall_steps : consecutive stall steps at crash moment
    state       : AircraftState at crash step
    info        : step info dict from FixedWingEnv.step()
    stall_speed : minimum safe airspeed (m/s)
    """
    alt      = float(state.pos[2])
    airspeed = float(state.airspeed)
    abs_roll = abs(float(state.roll))

    if stall_steps >= 15:
        return CrashType.STALL

    if alt > 790.0:
        return CrashType.CEILING_EXCEED

    # Ground contact — classify by attitude / energy state
    if abs_roll > _OVERBANK_ROLL:
        return CrashType.SPIRAL if airspeed > _SPIRAL_SPEED else CrashType.OVERBANK

    if abs_roll > _SPIRAL_ROLL and airspeed > _SPIRAL_SPEED:
        return CrashType.SPIRAL

    if airspeed < stall_speed + 2.0:
        return CrashType.ENERGY_LOSS

    if float(info.get("ctrl_osc_ep_mean", 0.0)) > _OSC_THRESHOLD:
        return CrashType.OSCILLATION

    if alt <= 0.0:
        return CrashType.GROUND_IMPACT

    return CrashType.UNKNOWN


# ── File I/O ──────────────────────────────────────────────────────────────────

_SAVE_DIR   = "logs/crashes"
_SAVE_EVERY = 5     # save 1 of every N crashes to avoid flooding disk
_MAX_FILES  = 200   # rotate oldest files beyond this count

_crash_counter = 0


def save_crash_report(crash_type: CrashType,
                      telemetry_buf: list,
                      state,
                      info: dict,
                      mode_name: str = "UNKNOWN") -> "str | None":
    """
    Write a crash telemetry JSON to logs/crashes/.  Rate-limited to 1-in-N
    crashes.  Returns the file path on write, None if skipped or on error.
    """
    global _crash_counter
    _crash_counter += 1

    if _crash_counter % _SAVE_EVERY != 0:
        return None

    try:
        os.makedirs(_SAVE_DIR, exist_ok=True)
        _trim_oldest(_SAVE_DIR, _MAX_FILES)

        ts    = time.strftime("%Y%m%d_%H%M%S")
        fname = f"crash_{ts}_{_crash_counter:05d}_{crash_type.value}.json"
        fpath = os.path.join(_SAVE_DIR, fname)

        doc = {
            "crash_type": crash_type.value,
            "mode":       mode_name,
            "timestamp":  ts,
            "final_state": {
                "pos":              list(state.pos),
                "airspeed":         round(float(state.airspeed), 2),
                "roll_deg":         round(math.degrees(float(state.roll)),  1),
                "pitch_deg":        round(math.degrees(float(state.pitch)), 1),
                "yaw_deg":          round(math.degrees(float(state.yaw)),   1),
                "roll_rate_rad_s":  round(float(state.roll_rate),  3),
                "pitch_rate_rad_s": round(float(state.pitch_rate), 3),
            },
            "episode_metrics": {
                "ctrl_osc_ep_mean": round(float(info.get("ctrl_osc_ep_mean",   0)), 4),
                "stalled_this_ep":  bool( info.get("stalled_this_ep",  False)),
                "altitude_error":   round(float(info.get("altitude_error",     0)), 1),
            },
            "reward_breakdown": info.get("reward_breakdown", {}),
            "telemetry": telemetry_buf,
        }

        with open(fpath, "w") as f:
            json.dump(doc, f, indent=2)

        return fpath

    except Exception as e:
        print(f"[CRASH_DIAG] Save failed: {e}")
        return None


def _trim_oldest(directory: str, max_files: int):
    """Delete oldest crash JSON files if total count exceeds max_files."""
    files = sorted(
        (os.path.join(directory, fn) for fn in os.listdir(directory)
         if fn.endswith(".json")),
        key=os.path.getmtime,
    )
    for old in files[:-max_files]:
        try:
            os.remove(old)
        except OSError:
            pass
