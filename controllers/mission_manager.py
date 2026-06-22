"""
Mission Manager — routes inference requests to the active expert policy
for the current flight mode.

M1 scope: exactly one expert (the default monolithic PPO policy) is
active for every FlightMode, reproducing today's single-model behavior
exactly.

M2 scope: if models/experts/<mode>.zip exists for a given FlightMode, it
is loaded into self._experts and used for that mode. Modes without a
matching file keep using the default policy via get_active_expert's
fallback. Routing logic itself (get_active_expert/predict) is unchanged
from M1 — populating self._experts is enough to activate routing.
M3 will build on this with smooth transitions between experts.
"""

import os

from stable_baselines3 import PPO

from sim.flight_modes import FlightMode


class Expert:
    """Wraps a loaded SB3 policy (or None) behind a uniform .predict()."""

    def __init__(self, model=None):
        self.model = model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, obs, deterministic=True):
        if self.model is None:
            return None, None
        return self.model.predict(obs, deterministic=deterministic)


class MissionManager:
    """
    Selects the active expert for the current flight mode and forwards
    predict() calls to it.
    """

    # Expected filename (without extension) under experts_dir for each mode.
    _MODE_FILENAMES = {
        FlightMode.STABILIZE:     "stabilize",
        FlightMode.RECOVERY:      "recovery",
        FlightMode.ALTITUDE_HOLD: "alt_hold",
        FlightMode.HEADING_HOLD:  "hdg_hold",
        FlightMode.WAYPOINT:      "waypoint",
        FlightMode.LOITER:        "loiter",
        FlightMode.APPROACH:      "approach",
        FlightMode.LANDING:       "landing",
    }

    def __init__(self, default_model_path: str = "models/latest.zip",
                 experts_dir: str = "models/experts",
                 fallback_paths=None):
        self.default_model_path = default_model_path
        self.experts_dir        = experts_dir
        self.fallback_paths     = fallback_paths or [
            "models/latest", "models/best_reward", "models/fixed_wing_multimode_ppo",
        ]
        self._default_expert = Expert()
        self._experts: dict[FlightMode, Expert] = {}

    def load(self) -> "MissionManager":
        """Load the default policy, trying fallback paths in order
        (mirrors the previous loading behavior in visualize.py), then
        load any per-mode experts found in experts_dir."""
        path = self.default_model_path
        if path.endswith(".zip"):
            path = path[:-4]
        candidates = [path] + [p for p in self.fallback_paths if p != path]

        for candidate in candidates:
            if os.path.exists(candidate + ".zip"):
                try:
                    print(f"[MISSION] Loading default policy from {candidate}.zip")
                    self._default_expert = Expert(PPO.load(candidate))
                    break
                except Exception as e:
                    print(f"[MISSION] Could not load {candidate}: {e}")
        else:
            print("[MISSION] No default policy loaded — falling back to random actions.")

        self._load_experts()
        return self

    def _load_experts(self):
        """Populate self._experts with any per-mode checkpoints found in
        experts_dir. Modes without a matching file are left unset, so
        get_active_expert() keeps falling back to the default policy."""
        if not os.path.isdir(self.experts_dir):
            return
        for mode, filename in self._MODE_FILENAMES.items():
            zip_path = os.path.join(self.experts_dir, filename + ".zip")
            if not os.path.exists(zip_path):
                continue
            try:
                print(f"[MISSION] Loading expert for {mode.name} from {zip_path}")
                self._experts[mode] = Expert(PPO.load(zip_path[:-4]))
            except Exception as e:
                print(f"[MISSION] Could not load expert {zip_path}: {e}")

    @property
    def loaded(self) -> bool:
        return self._default_expert.loaded

    def get_active_expert(self, flight_mode: FlightMode) -> Expert:
        """Return the expert responsible for `flight_mode`: the matching
        entry in self._experts if one was loaded, else the default policy."""
        return self._experts.get(flight_mode, self._default_expert)

    def predict(self, obs, flight_mode: FlightMode, deterministic: bool = True):
        return self.get_active_expert(flight_mode).predict(obs, deterministic=deterministic)
