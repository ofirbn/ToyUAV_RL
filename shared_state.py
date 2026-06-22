"""
Thread-safe shared state for live training visualization.

Training thread writes via update() / push_history() / push_event().
Pygame thread reads via read() every frame.
No env objects, no tensors — plain Python dicts only.
"""

import threading
import collections
import time

_lock = threading.Lock()

HISTORY_LEN = 500
TRAJ_LEN    = 400

_state: dict = {
    # ── aircraft state ───────────────────────────────────────────────────────
    "pos":             [0.0, 0.0, 50.0],
    "vel":             [0.0, 0.0, 0.0],
    "pitch":           0.0,
    "roll":            0.0,
    "yaw":             0.0,
    "pitch_rate":      0.0,
    "roll_rate":       0.0,
    "yaw_rate":        0.0,
    "throttle_pos":     0.3,
    "throttle_command": 0.3,    # raw PPO output after deadband
    "throttle_delta":   0.0,    # abs change in throttle_actual this step
    "target_airspeed":  0.0,    # 14.0 in ALT_HOLD, else 0
    "airspeed_error":   0.0,    # signed: airspeed - target_airspeed
    "airspeed":        10.0,
    "elevator":        0.0,
    "aileron":         0.0,
    "rudder":          0.0,
    "stall_warning":   False,

    # ── mission / target ─────────────────────────────────────────────────────
    "mode":            0,
    "target_position": [0.0, 0.0, 50.0],
    "target_heading":  0.0,
    "target_altitude": 50.0,
    "target_radius":   60.0,
    "mission_seg":     0,
    "mission_total":   0,

    # ── reward ───────────────────────────────────────────────────────────────
    "reward":           0.0,
    "reward_breakdown": {},

    # ── PPO training metrics ─────────────────────────────────────────────────
    "timesteps":       0,
    "episode_count":   0,
    "mean_ep_reward":  0.0,
    "best_reward":     -9999.0,
    "best_success_rate": 0.0,
    "policy_loss":     0.0,
    "value_loss":      0.0,
    "entropy":         0.0,
    "approx_kl":       0.0,
    "explained_var":   0.0,
    "learning_rate":   3e-4,
    "fps":             0.0,
    "training_iter":   0,
    "success_rate":    0.0,
    "crash_rate":      0.0,
    "landing_rate":    0.0,
    "stall_rate":      0.0,

    # ── curriculum ───────────────────────────────────────────────────────────
    "curriculum_phase": "mixed",

    # ── mode-specific success rates ──────────────────────────────────────────
    "stabilize_success_rate":    0.0,
    "altitude_hold_success_rate": 0.0,
    "heading_hold_success_rate": 0.0,
    "waypoint_success_rate":     0.0,
    "loiter_success_rate":       0.0,
    "approach_success_rate":     0.0,
    "landing_success_rate":      0.0,
    "recovery_success_rate":     0.0,

    # ── pipeline ─────────────────────────────────────────────────────────────
    "pipeline_phase_label": "",   # shown as banner in 3D view

    # ── session ───────────────────────────────────────────────────────────────
    "ready":           False,
    "training_done":   False,
    "camera_mode":     0,

    # ── events ────────────────────────────────────────────────────────────────
    "events": [],
}

_histories: dict = {
    "reward":       collections.deque(maxlen=HISTORY_LEN),
    "mean_reward":  collections.deque(maxlen=HISTORY_LEN),
    "crash":        collections.deque(maxlen=HISTORY_LEN),
    "success":      collections.deque(maxlen=HISTORY_LEN),
    "stall":        collections.deque(maxlen=HISTORY_LEN),
    "policy_loss":  collections.deque(maxlen=HISTORY_LEN),
    "value_loss":   collections.deque(maxlen=HISTORY_LEN),
    "trajectory":   collections.deque(maxlen=TRAJ_LEN),
}


# ── write helpers ─────────────────────────────────────────────────────────────

def update(patch: dict) -> None:
    with _lock:
        _state.update(patch)


def push_history(name: str, value) -> None:
    with _lock:
        if name in _histories:
            _histories[name].append(value)


def push_event(text: str, color=(255, 220, 0)) -> None:
    expire = time.monotonic() + 4.0
    with _lock:
        _state["events"].append({"text": text, "color": color, "expire": expire})
        if len(_state["events"]) > 6:
            _state["events"] = _state["events"][-6:]


# ── read ──────────────────────────────────────────────────────────────────────

def read() -> dict:
    now = time.monotonic()
    with _lock:
        d = dict(_state)
        active = [e for e in d["events"] if e["expire"] > now]
        _state["events"] = active
        d["events"] = list(active)
        d["_hist"] = {k: list(v) for k, v in _histories.items()}
        return d
