import json
import math
import numpy as np

from sim.flight_modes import FlightMode


class TargetInfo:
    """All information the env needs to compute observations and rewards."""

    __slots__ = ('mode', 'position', 'heading', 'altitude', 'radius')

    def __init__(self, mode: FlightMode, position, heading=0.0,
                 altitude=None, radius=0.0):
        self.mode     = mode
        self.position = np.array(position, dtype=float)
        self.heading  = float(heading)
        self.altitude = float(altitude) if altitude is not None else float(position[2])
        self.radius   = float(radius)


def _angle_diff(a, b):
    """Shortest signed angular difference a - b, in [-π, π]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class MissionManager:
    """
    Loads a mission JSON and advances through segments as the aircraft
    reaches each waypoint/target.

    Segment types (matching demo_mission.json):
        takeoff   — climb to altitude above spawn point
        waypoint  — fly to (x, y, z)
        loiter    — orbit (x, y, z) at given radius for N orbits
        approach  — line up for runway on 3-degree glide slope
        landing   — land on runway; episode ends on touchdown
    """

    # Proximity threshold (m) for advancing past a waypoint
    WAYPOINT_RADIUS = 20.0
    # Proximity to runway threshold (m) to advance approach → landing
    APPROACH_GATE   = 120.0
    # Default orbit count before advancing past loiter
    DEFAULT_ORBITS  = 1.5

    def __init__(self, mission_path: str = None):
        self._segments:   list  = []
        self._idx:        int   = 0
        self._orbit_accum: float = 0.0
        self._prev_angle:  float = None

        if mission_path:
            self.load(mission_path)

    # ------------------------------------------------------------------ load --

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self._segments = data['mission']
        self.reset()

    def reset(self):
        self._idx         = 0
        self._orbit_accum = 0.0
        self._prev_angle  = None

    def jump_to(self, idx: int):
        """Jump directly to segment idx (for training resets)."""
        self._idx         = max(0, min(idx, len(self._segments) - 1))
        self._orbit_accum = 0.0
        self._prev_angle  = None

    # --------------------------------------------------------- public queries --

    @property
    def is_done(self) -> bool:
        return self._idx >= len(self._segments)

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    @property
    def current_index(self) -> int:
        return self._idx

    @property
    def current_type(self) -> str:
        if self.is_done:
            return 'done'
        return self._segments[self._idx]['type']

    def get_target(self, aircraft_pos: np.ndarray, aircraft_yaw: float) -> TargetInfo:
        """Return the current target, or None if mission is done."""
        if self.is_done:
            return None
        seg = self._segments[self._idx]
        return self._seg_to_target(seg, aircraft_pos, aircraft_yaw)

    # ----------------------------------------------------------- step / update -

    def update(self, aircraft_pos: np.ndarray, aircraft_yaw: float) -> bool:
        """
        Call once per env step.  Returns True if mission just completed.
        Advances to the next segment when the aircraft satisfies the
        completion condition for the current one.
        """
        if self.is_done:
            return True
        seg = self._segments[self._idx]
        t   = seg['type']

        if t == 'takeoff':
            target_alt = float(seg.get('altitude', 50.0))
            if aircraft_pos[2] >= target_alt - 5.0:
                self._advance()

        elif t == 'waypoint':
            tgt = np.array([seg['x'], seg['y'], seg['z']], dtype=float)
            if np.linalg.norm(aircraft_pos - tgt) < self.WAYPOINT_RADIUS:
                self._advance()

        elif t == 'loiter':
            cx, cy = float(seg['x']), float(seg['y'])
            dx = aircraft_pos[0] - cx
            dy = aircraft_pos[1] - cy
            angle = math.atan2(dy, dx)
            if self._prev_angle is not None:
                delta = _angle_diff(angle, self._prev_angle)
                self._orbit_accum += abs(delta) / (2 * math.pi)
            self._prev_angle = angle
            if self._orbit_accum >= float(seg.get('orbits', self.DEFAULT_ORBITS)):
                self._advance()

        elif t == 'approach':
            rwy_x = float(seg.get('runway_x', 0.0))
            rwy_y = float(seg.get('runway_y', 0.0))
            dist  = math.sqrt((aircraft_pos[0] - rwy_x) ** 2 +
                              (aircraft_pos[1] - rwy_y) ** 2)
            if dist < self.APPROACH_GATE:
                self._advance()

        elif t == 'landing':
            if aircraft_pos[2] <= 0.5:
                self._advance()

        return self.is_done

    # ---------------------------------------------------------------- helpers --

    def _advance(self):
        self._idx        += 1
        self._orbit_accum = 0.0
        self._prev_angle  = None

    def _seg_to_target(self, seg: dict, aircraft_pos: np.ndarray,
                       aircraft_yaw: float) -> TargetInfo:
        t = seg['type']

        if t == 'takeoff':
            target_alt = float(seg.get('altitude', 50.0))
            return TargetInfo(
                mode     = FlightMode.WAYPOINT,
                position = [aircraft_pos[0], aircraft_pos[1], target_alt],
                heading  = aircraft_yaw,
                altitude = target_alt,
            )

        elif t == 'waypoint':
            x, y, z = float(seg['x']), float(seg['y']), float(seg['z'])
            dx, dy  = x - aircraft_pos[0], y - aircraft_pos[1]
            hdg     = math.atan2(dx, -dy)   # yaw convention: 0 = -y direction
            return TargetInfo(
                mode     = FlightMode.WAYPOINT,
                position = [x, y, z],
                heading  = hdg,
                altitude = z,
            )

        elif t == 'loiter':
            cx = float(seg['x']); cy = float(seg['y']); cz = float(seg['z'])
            r  = float(seg.get('radius', 40.0))
            # Heading = tangent to CCW orbit at aircraft's current angular position
            dx = aircraft_pos[0] - cx
            dy = aircraft_pos[1] - cy
            tangent_hdg = math.atan2(-dy, dx)   # CCW tangent
            return TargetInfo(
                mode     = FlightMode.LOITER,
                position = [cx, cy, cz],
                heading  = tangent_hdg,
                altitude = cz,
                radius   = r,
            )

        elif t == 'approach':
            rwy_x = float(seg.get('runway_x', 0.0))
            rwy_y = float(seg.get('runway_y', 0.0))
            hdg   = math.radians(float(seg.get('heading', 0.0)))
            # Aim for a point on glide slope: 400m before runway at 3° slope
            gs_dist = 400.0
            app_x   = rwy_x - math.sin(hdg) * gs_dist
            app_y   = rwy_y + math.cos(hdg) * gs_dist
            app_z   = gs_dist * math.tan(math.radians(3.0))
            return TargetInfo(
                mode     = FlightMode.APPROACH,
                position = [app_x, app_y, app_z],
                heading  = hdg,
                altitude = app_z,
            )

        elif t == 'landing':
            rwy_x = float(seg.get('runway_x', 0.0))
            rwy_y = float(seg.get('runway_y', 0.0))
            hdg   = math.radians(float(seg.get('heading', 0.0)))
            return TargetInfo(
                mode     = FlightMode.LANDING,
                position = [rwy_x, rwy_y, 0.0],
                heading  = hdg,
                altitude = 0.0,
            )

        # Fallback: hover in place
        return TargetInfo(
            mode     = FlightMode.STABILIZE,
            position = aircraft_pos.copy(),
            heading  = aircraft_yaw,
        )
