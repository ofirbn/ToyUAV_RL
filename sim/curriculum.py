"""
Curriculum-based staged training with MASTERY GATES.

Progression (9 stages):
  1. stabilize     – Wings-level stability only
  2. cruise        – Add unusual-attitude recovery
  3. altitude_hold – Add altitude control
  4. heading_hold  – Add heading / turn control
  5. waypoint      – Add navigation to waypoints
  6. loiter        – Add orbit around a point
  7. approach      – Add ILS-style approach
  8. landing       – Add touchdown
  9. mixed         – All modes; full mission variety

Progression is MASTERY-BASED — ALL gate criteria must be met simultaneously
over a rolling eval window before advancing. Simple timestep or single-metric
thresholds alone are not sufficient.

Gate criteria (all must pass):
  - success_rate       > min_success_rate
  - crash_rate         < max_crash_rate
  - stall_rate         < max_stall_rate
  - mean_ep_length     > min_ep_length       (0 = disabled)
  - mean_roll_rate     < max_roll_rate rad/s (0 = disabled)
  - mean_pitch_rate    < max_pitch_rate rad/s (0 = disabled)
  - ctrl_oscillation   < max_ctrl_osc        (0 = disabled)
"""

import collections
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from sim.flight_modes import FlightMode


# ── Per-episode data passed from the environment ──────────────────────────────

@dataclass
class EpisodeData:
    """Metrics collected by FixedWingEnv over a single episode."""
    success: bool
    crashed: bool
    stalled: bool
    ep_length: int
    mean_roll_rate: float    # mean |roll_rate|  (rad/s) over episode
    mean_pitch_rate: float   # mean |pitch_rate| (rad/s) over episode
    ctrl_oscillation: float  # mean |delta_action| per step


# ── Per-stage mastery gate ────────────────────────────────────────────────────

@dataclass
class MasteryGate:
    """
    All criteria must be met (over eval_window recent episodes) before the
    curriculum advances to the next stage.  Set a threshold to 0 to disable it.
    """
    min_success_rate: float = 0.90   # fraction of episodes that succeed
    max_crash_rate:   float = 0.05   # fraction of episodes that crash
    max_stall_rate:   float = 0.05   # fraction with any stall event
    min_ep_length:    float = 0.0    # mean episode length (steps); 0 = off
    max_roll_rate:    float = 0.0    # mean |roll_rate|  (rad/s);   0 = off
    max_pitch_rate:   float = 0.0    # mean |pitch_rate| (rad/s);   0 = off
    max_ctrl_osc:     float = 0.0    # mean |delta_surface|;        0 = off
    max_phase_steps:  int   = 0      # force-advance after N env steps; 0 = off
    eval_window:      int   = 150    # minimum episodes in window before gate fires


# ── Per-stage environment randomization ranges ────────────────────────────────

@dataclass
class StageDifficulty:
    # Starting conditions
    alt_range:           tuple = (60, 300)     # m    start altitude
    spd_range:           tuple = (8, 13)       # m/s  start airspeed
    roll_perturb:        float = 1.0           # rad  max initial roll
    pitch_perturb:       float = 0.4           # rad  max initial pitch
    # ALTITUDE_HOLD: how far the target altitude is from start altitude
    alt_delta_range:     tuple = (20, 80)      # m
    # HEADING_HOLD: yaw turn magnitude
    hdg_delta_range:     tuple = (0.3, 1.5)   # rad
    # WAYPOINT: distance to target
    wp_dist_range:       tuple = (100, 400)    # m
    # LOITER: orbit radius
    loiter_rad_range:    tuple = (30, 70)      # m
    # APPROACH: start distance from runway threshold
    approach_dist_range: tuple = (250, 600)    # m
    # LANDING: start distance from touchdown zone
    landing_dist_range:  tuple = (80, 200)     # m


# ── Stage definitions ─────────────────────────────────────────────────────────

@dataclass
class StageConfig:
    name:              str
    modes:             List[FlightMode]
    difficulty:        StageDifficulty = field(default_factory=StageDifficulty)
    mastery_gate:      MasteryGate     = field(default_factory=MasteryGate)
    description:       str             = ""
    emergency_recovery_enabled: bool   = False


STAGES: List[StageConfig] = [

    # 1 ─ Stabilize ───────────────────────────────────────────────────────────
    # Must achieve smooth, stall-free, wings-level flight before anything else.
    StageConfig(
        name="stabilize",
        modes=[FlightMode.STABILIZE],
        difficulty=StageDifficulty(
            alt_range=(120, 200),
            spd_range=(10, 12),
            roll_perturb=0.12,    # ±7°
            pitch_perturb=0.06,   # ±3°
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.85,   # achievable target: ~11% crash rate is now the baseline
            max_crash_rate=0.10,     # 10% — allow progression without overtraining stabilize
            max_stall_rate=0.06,
            min_ep_length=300.0,     # rarely crashes mid-episode
            max_roll_rate=0.40,      # ~23 deg/s — smooth
            max_pitch_rate=0.30,     # ~17 deg/s — smooth
            max_ctrl_osc=0.05,       # actual surface delta L2; rate-limiter caps at 0.104
            max_phase_steps=2_000_000,  # force-advance safety net
            eval_window=200,
        ),
        description="Wings-level stability. Near-zero initial perturbations.",
    ),

    # 2 ─ Cruise ──────────────────────────────────────────────────────────────
    StageConfig(
        name="cruise",
        modes=[FlightMode.STABILIZE, FlightMode.RECOVERY],
        difficulty=StageDifficulty(
            alt_range=(80, 280),
            spd_range=(9, 13),
            roll_perturb=0.35,
            pitch_perturb=0.20,
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.88,
            max_crash_rate=0.05,
            max_stall_rate=0.05,
            min_ep_length=280.0,
            max_roll_rate=0.50,
            max_pitch_rate=0.38,
            max_ctrl_osc=0.32,
            eval_window=150,
        ),
        description="Cruise and unusual-attitude recovery.",
    ),

    # 3 ─ Altitude Hold ───────────────────────────────────────────────────────
    # Must track altitude stably before adding turning complexity.
    StageConfig(
        name="altitude_hold",
        modes=[FlightMode.STABILIZE, FlightMode.RECOVERY,
               FlightMode.ALTITUDE_HOLD],
        difficulty=StageDifficulty(
            alt_range=(60, 300),
            alt_delta_range=(15, 60),
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.85,
            max_crash_rate=0.05,
            max_stall_rate=0.06,
            min_ep_length=250.0,
            max_roll_rate=0.55,
            max_pitch_rate=0.42,
            max_ctrl_osc=0.36,
            eval_window=150,
        ),
        description="Altitude control: climb and descend.",
    ),

    # 4 ─ Heading Hold ────────────────────────────────────────────────────────
    # Must make smooth coordinated turns before introducing waypoints.
    StageConfig(
        name="heading_hold",
        modes=[FlightMode.STABILIZE, FlightMode.RECOVERY,
               FlightMode.ALTITUDE_HOLD, FlightMode.HEADING_HOLD],
        difficulty=StageDifficulty(
            alt_delta_range=(20, 70),
            hdg_delta_range=(0.3, 1.2),
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.82,
            max_crash_rate=0.07,
            max_stall_rate=0.08,
            min_ep_length=0.0,
            max_roll_rate=0.60,
            max_pitch_rate=0.45,
            max_ctrl_osc=0.42,
            eval_window=150,
        ),
        description="Heading control: turns.",
    ),

    # 5 ─ Waypoint ────────────────────────────────────────────────────────────
    StageConfig(
        name="waypoint",
        modes=[FlightMode.STABILIZE, FlightMode.RECOVERY,
               FlightMode.ALTITUDE_HOLD, FlightMode.HEADING_HOLD,
               FlightMode.WAYPOINT],
        difficulty=StageDifficulty(
            wp_dist_range=(80, 250),
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.78,
            max_crash_rate=0.10,
            max_stall_rate=0.10,
            min_ep_length=0.0,
            max_roll_rate=0.0,
            max_pitch_rate=0.0,
            max_ctrl_osc=0.0,
            eval_window=120,
        ),
        description="Navigate to waypoints.",
    ),

    # 6 ─ Loiter ──────────────────────────────────────────────────────────────
    StageConfig(
        name="loiter",
        modes=[FlightMode.STABILIZE, FlightMode.RECOVERY,
               FlightMode.ALTITUDE_HOLD, FlightMode.HEADING_HOLD,
               FlightMode.WAYPOINT, FlightMode.LOITER],
        difficulty=StageDifficulty(
            wp_dist_range=(100, 350),
            loiter_rad_range=(30, 70),
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.75,
            max_crash_rate=0.12,
            max_stall_rate=0.12,
            eval_window=120,
        ),
        description="Loiter / orbit around a point.",
    ),

    # 7 ─ Approach ────────────────────────────────────────────────────────────
    StageConfig(
        name="approach",
        modes=[FlightMode.STABILIZE, FlightMode.RECOVERY,
               FlightMode.ALTITUDE_HOLD, FlightMode.HEADING_HOLD,
               FlightMode.WAYPOINT, FlightMode.LOITER,
               FlightMode.APPROACH],
        difficulty=StageDifficulty(
            approach_dist_range=(200, 500),
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.72,
            max_crash_rate=0.15,
            max_stall_rate=0.15,
            eval_window=120,
        ),
        description="ILS-style approach on a 3° glide slope.",
    ),

    # 8 ─ Landing ─────────────────────────────────────────────────────────────
    StageConfig(
        name="landing",
        modes=[FlightMode.STABILIZE, FlightMode.RECOVERY,
               FlightMode.ALTITUDE_HOLD, FlightMode.HEADING_HOLD,
               FlightMode.WAYPOINT, FlightMode.LOITER,
               FlightMode.APPROACH, FlightMode.LANDING],
        difficulty=StageDifficulty(
            approach_dist_range=(250, 600),
            landing_dist_range=(80, 200),
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.65,
            max_crash_rate=0.20,
            max_stall_rate=0.20,
            eval_window=100,
        ),
        description="Full landing sequence.",
    ),

    # 9 ─ Mixed ───────────────────────────────────────────────────────────────
    StageConfig(
        name="mixed",
        modes=list(FlightMode),
        emergency_recovery_enabled=True,
        difficulty=StageDifficulty(
            alt_range=(30, 500),
            spd_range=(7, 14),
            roll_perturb=1.5,
            pitch_perturb=0.6,
            alt_delta_range=(20, 150),
            hdg_delta_range=(0.3, 1.5),
            wp_dist_range=(100, 500),
            loiter_rad_range=(25, 80),
            approach_dist_range=(200, 700),
            landing_dist_range=(60, 250),
        ),
        mastery_gate=MasteryGate(
            min_success_rate=0.0,   # final stage — no advancement
            eval_window=100,
        ),
        description="Mixed missions. All modes active.",
    ),
]

STAGE_BY_NAME = {s.name: i for i, s in enumerate(STAGES)}


# ── CurriculumManager ─────────────────────────────────────────────────────────

class CurriculumManager:
    """
    Shared across all parallel envs in DummyVecEnv (single-threaded).

    Each env calls record_episode(EpisodeData) when an episode ends.
    The manager advances only when ALL mastery gate criteria are simultaneously
    satisfied over the required eval window — not just success rate alone.
    """

    def __init__(self, start_stage: str = 'stabilize',
                 save_path: Optional[str] = None):
        self._save_path  = save_path
        self._stage_idx  = 0
        self._total_eps  = 0
        self._advances   = 0
        self._phase_steps = 0   # total env steps accumulated in the current stage

        # start_stage is AUTHORITATIVE — it always sets the initial stage.
        # The save file may restore advance history but must never override
        # the requested starting point.  Without this, a previous run that
        # auto-progressed to (e.g.) WAYPOINT would silently restart in
        # WAYPOINT even when config says curriculum_phase=stabilize.
        requested_idx = STAGE_BY_NAME.get(start_stage, 0)

        if save_path and os.path.exists(save_path):
            self._load()
            if self._stage_idx != requested_idx:
                print(f"[CURRICULUM] OVERRIDE: start_stage='{start_stage}' "
                      f"supersedes saved stage '{STAGES[self._stage_idx].name}'. "
                      f"Delete models/curriculum_state.json to suppress this message.")
                self._stage_idx = requested_idx
                self._advances  = 0
            else:
                print(f"[CURRICULUM] Resumed at stage '{self.stage_name}' "
                      f"(advances: {self._advances})")
        else:
            self._stage_idx = requested_idx
            print(f"[CURRICULUM] Starting at stage '{self.stage_name}'")

        self._init_deques()

    def _init_deques(self):
        w = self.stage.mastery_gate.eval_window
        self._ep_success  = collections.deque(maxlen=w)
        self._ep_crash    = collections.deque(maxlen=w)
        self._ep_stall    = collections.deque(maxlen=w)
        self._ep_length   = collections.deque(maxlen=w)
        self._ep_roll_r   = collections.deque(maxlen=w)
        self._ep_pitch_r  = collections.deque(maxlen=w)
        self._ep_ctrl_osc = collections.deque(maxlen=w)

    # ── read-only properties ──────────────────────────────────────────────────

    @property
    def stage(self) -> StageConfig:
        return STAGES[self._stage_idx]

    @property
    def stage_name(self) -> str:
        return self.stage.name

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def num_stages(self) -> int:
        return len(STAGES)

    @property
    def is_final(self) -> bool:
        return self._stage_idx >= len(STAGES) - 1

    @property
    def total_advances(self) -> int:
        return self._advances

    def success_rate(self) -> float:
        """Rolling success rate for HUD / logging compatibility."""
        if not self._ep_success:
            return 0.0
        return sum(self._ep_success) / len(self._ep_success)

    def get_active_modes(self) -> List[FlightMode]:
        return list(self.stage.modes)

    def get_difficulty(self) -> StageDifficulty:
        return self.stage.difficulty

    def get_emergency_recovery_enabled(self) -> bool:
        return self.stage.emergency_recovery_enabled

    # ── mastery evaluation ────────────────────────────────────────────────────

    @staticmethod
    def _mean(deq) -> float:
        return (sum(deq) / len(deq)) if deq else 0.0

    def locked_criteria(self) -> List[str]:
        """Return list of criterion labels that are currently failing."""
        if self.is_final:
            return []

        gate = self.stage.mastery_gate
        n    = len(self._ep_success)

        if n < 10:
            return [f"COLLECTING_DATA ({n}/{gate.eval_window})"]

        failing = []

        suc  = self._mean(self._ep_success)
        cra  = self._mean(self._ep_crash)
        sta  = self._mean(self._ep_stall)
        epln = self._mean(self._ep_length)
        rr   = self._mean(self._ep_roll_r)
        pr   = self._mean(self._ep_pitch_r)
        osc  = self._mean(self._ep_ctrl_osc)

        if suc < gate.min_success_rate:
            failing.append(
                f"SUCCESS_TOO_LOW ({suc:.0%} < {gate.min_success_rate:.0%})")
        if cra > gate.max_crash_rate:
            failing.append(
                f"CRASH_RATE_TOO_HIGH ({cra:.0%} > {gate.max_crash_rate:.0%})")
        if sta > gate.max_stall_rate:
            failing.append(
                f"STALL_RATE_TOO_HIGH ({sta:.0%} > {gate.max_stall_rate:.0%})")
        if gate.min_ep_length > 0 and epln < gate.min_ep_length:
            failing.append(
                f"EPISODE_TOO_SHORT ({epln:.0f} < {gate.min_ep_length:.0f})")
        if gate.max_roll_rate > 0 and rr > gate.max_roll_rate:
            failing.append(
                f"ROLL_STABILITY_NOT_MET ({rr:.3f} > {gate.max_roll_rate:.3f})")
        if gate.max_pitch_rate > 0 and pr > gate.max_pitch_rate:
            failing.append(
                f"PITCH_STABILITY_NOT_MET ({pr:.3f} > {gate.max_pitch_rate:.3f})")
        if gate.max_ctrl_osc > 0 and osc > gate.max_ctrl_osc:
            failing.append(
                f"CONTROL_OSCILLATION_TOO_HIGH ({osc:.3f} > {gate.max_ctrl_osc:.3f})")

        return failing

    def is_mastered(self) -> bool:
        """True when ALL mastery gate criteria are currently met."""
        if self.is_final:
            return True
        gate = self.stage.mastery_gate
        if len(self._ep_success) < gate.eval_window:
            return False
        return len(self.locked_criteria()) == 0

    def mastery_details(self) -> Dict:
        """Return current metric values vs gate thresholds for the HUD."""
        gate = self.stage.mastery_gate
        n    = len(self._ep_success)
        return {
            'n_episodes':      n,
            'required':        gate.eval_window,
            'phase_steps':     self._phase_steps,
            'max_phase_steps': gate.max_phase_steps,
            'metrics': {
                'success_rate':    self._mean(self._ep_success),
                'crash_rate':      self._mean(self._ep_crash),
                'stall_rate':      self._mean(self._ep_stall),
                'ep_length':       self._mean(self._ep_length),
                'roll_rate':       self._mean(self._ep_roll_r),
                'pitch_rate':      self._mean(self._ep_pitch_r),
                'ctrl_osc':        self._mean(self._ep_ctrl_osc),
            },
            'thresholds': {
                'success_rate':    gate.min_success_rate,
                'crash_rate':      gate.max_crash_rate,
                'stall_rate':      gate.max_stall_rate,
                'ep_length':       gate.min_ep_length,
                'roll_rate':       gate.max_roll_rate,
                'pitch_rate':      gate.max_pitch_rate,
                'ctrl_osc':        gate.max_ctrl_osc,
            },
            'failing': self.locked_criteria(),
            'mastered': self.is_mastered(),
        }

    # ── episode outcome ───────────────────────────────────────────────────────

    def record_episode(self, ep: EpisodeData) -> bool:
        """
        Record one episode's metrics.  Returns True if the stage just advanced.
        Not thread-safe; only call from the training thread (DummyVecEnv).
        """
        self._ep_success.append(1 if ep.success else 0)
        self._ep_crash.append(  1 if ep.crashed else 0)
        self._ep_stall.append(  1 if ep.stalled else 0)
        self._ep_length.append( ep.ep_length)
        self._ep_roll_r.append( ep.mean_roll_rate)
        self._ep_pitch_r.append(ep.mean_pitch_rate)
        self._ep_ctrl_osc.append(ep.ctrl_oscillation)
        self._total_eps   += 1
        self._phase_steps += ep.ep_length

        if self.is_final:
            return False

        gate = self.stage.mastery_gate
        force = (gate.max_phase_steps > 0 and
                 self._phase_steps >= gate.max_phase_steps)

        if self.is_mastered() or force:
            if force and not self.is_mastered():
                print(f"[CURRICULUM] FORCE-ADVANCE from '{self.stage_name}': "
                      f"{self._phase_steps:,} phase steps >= limit "
                      f"{gate.max_phase_steps:,}")
            self._advance()
            return True
        return False

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self):
        if self._save_path:
            self._save()

    def status_str(self) -> str:
        gate = self.stage.mastery_gate
        n    = len(self._ep_success)
        suc  = self._mean(self._ep_success)
        cra  = self._mean(self._ep_crash)
        sta  = self._mean(self._ep_stall)
        rr   = self._mean(self._ep_roll_r)
        osc  = self._mean(self._ep_ctrl_osc)
        failing  = self.locked_criteria()
        lock_str = "MASTERED" if (not failing and n >= gate.eval_window) else \
                   f"LOCKED[{len(failing)}]"
        osc_str  = f"osc={osc:.4f}/{gate.max_ctrl_osc:.3f}" \
                   if gate.max_ctrl_osc > 0 else f"osc={osc:.4f}"
        ps_str   = (f"  phase_steps={self._phase_steps:,}/{gate.max_phase_steps:,}"
                    if gate.max_phase_steps > 0
                    else f"  phase_steps={self._phase_steps:,}")
        return (f"stage={self.stage_name}({self._stage_idx+1}/{len(STAGES)})  "
                f"suc={suc:.0%}/{gate.min_success_rate:.0%}  "
                f"crash={cra:.0%}  stall={sta:.0%}  "
                f"rr={rr:.2f}  {osc_str}  "
                f"n={n}/{gate.eval_window}  {lock_str}{ps_str}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _advance(self):
        old_name          = self.stage.name
        self._stage_idx   = min(self._stage_idx + 1, len(STAGES) - 1)
        self._advances   += 1
        self._phase_steps = 0   # reset step counter for the new stage

        # Recreate deques for the new stage's eval window
        self._init_deques()

        print(f"\n{'='*60}")
        print(f"  CURRICULUM ADVANCE #{self._advances}")
        print(f"  {old_name!r:>15}  ->  {self.stage.name!r}")
        print(f"  {self.stage.description}")
        print(f"  Modes: {[m.name for m in self.stage.modes]}")
        print(f"{'='*60}\n")

        if self._save_path:
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self._save_path) or '.', exist_ok=True)
        with open(self._save_path, 'w') as f:
            json.dump({'stage_idx': self._stage_idx,
                       'advances':  self._advances}, f, indent=2)

    def _load(self):
        try:
            with open(self._save_path) as f:
                d = json.load(f)
            self._stage_idx = int(d.get('stage_idx', 0))
            self._advances  = int(d.get('advances',  0))
            print(f"[CURRICULUM] Resumed at stage '{self.stage_name}' "
                  f"(advance #{self._advances})")
        except Exception as e:
            print(f"[CURRICULUM] Could not load state ({e}) — starting fresh.")


# ── backward-compat function ──────────────────────────────────────────────────

_LEGACY: dict = {s.name: s.modes for s in STAGES}
_LEGACY.update({
    'cruise':    [FlightMode.STABILIZE, FlightMode.RECOVERY],
    'altitude':  [FlightMode.STABILIZE, FlightMode.RECOVERY,
                  FlightMode.ALTITUDE_HOLD],
    'heading':   [FlightMode.STABILIZE, FlightMode.RECOVERY,
                  FlightMode.ALTITUDE_HOLD, FlightMode.HEADING_HOLD],
    'recovery':  [FlightMode.RECOVERY],
})

# Per-phase default for emergency_recovery_enabled when using static phase
_EMERGENCY_RECOVERY_DEFAULTS: dict = {
    'stabilize':    False,
    'cruise':       False,
    'altitude_hold':False,
    'altitude':     False,
    'heading_hold': False,
    'heading':      False,
    'waypoint':     False,
    'loiter':       False,
    'approach':     False,
    'landing':      False,
    'recovery':     True,
    'mixed':        True,
}


def get_active_modes(phase: str) -> List[FlightMode]:
    """Return the mode list for a named phase string (legacy / static use)."""
    return list(_LEGACY.get(phase.lower(), list(FlightMode)))


def get_emergency_recovery_default(phase: str) -> bool:
    """Return the default emergency_recovery_enabled for a static phase name."""
    return _EMERGENCY_RECOVERY_DEFAULTS.get(phase.lower(), False)


# ── Curriculum hard-lock: single mode per isolated phase ─────────────────────
# Maps each named curriculum phase to the ONE FlightMode that must be used.
# 'mixed' and 'cruise' are absent — no single-mode lock for those phases.

_PHASE_SINGLE_MODE: dict = {
    'stabilize':     FlightMode.STABILIZE,
    'altitude':      FlightMode.ALTITUDE_HOLD,
    'altitude_hold': FlightMode.ALTITUDE_HOLD,
    'heading':       FlightMode.HEADING_HOLD,
    'heading_hold':  FlightMode.HEADING_HOLD,
    'waypoint':      FlightMode.WAYPOINT,
    'loiter':        FlightMode.LOITER,
    'approach':      FlightMode.APPROACH,
    'landing':       FlightMode.LANDING,
    'recovery':      FlightMode.RECOVERY,
}


def get_locked_mode(phase: str):
    """Return the single locked FlightMode for an isolated phase, or None for mixed/cruise."""
    return _PHASE_SINGLE_MODE.get(phase.lower(), None)


# Per-phase weighted mode lists used instead of a hard lock.
# altitude_hold: 70 % ALT_HOLD + 30 % STABILIZE — the stabilise episodes
# keep the base flight-quality sharp while the policy learns energy management.
_PHASE_WEIGHTED_MODES: dict = {
    'altitude_hold': (
        [FlightMode.ALTITUDE_HOLD, FlightMode.STABILIZE],
        [0.70,                     0.30],
    ),
    'altitude': (
        [FlightMode.ALTITUDE_HOLD, FlightMode.STABILIZE],
        [0.70,                     0.30],
    ),
}


def get_phase_weighted_modes(phase: str):
    """
    Return (modes, weights) for phases with non-uniform sampling, else None.
    When non-None, the caller should use np.random.choice with p=weights
    instead of the hard single-mode lock.
    """
    return _PHASE_WEIGHTED_MODES.get(phase.lower(), None)
