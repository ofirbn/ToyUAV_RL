from enum import IntEnum


class FlightMode(IntEnum):
    STABILIZE    = 0
    ALTITUDE_HOLD = 1
    HEADING_HOLD = 2
    WAYPOINT     = 3
    LOITER       = 4
    APPROACH     = 5
    LANDING      = 6
    RECOVERY     = 7


NUM_MODES = 8

MODE_NAMES = [
    "STABILIZE",
    "ALT HOLD",
    "HDG HOLD",
    "WAYPOINT",
    "LOITER",
    "APPROACH",
    "LANDING",
    "RECOVERY",
]
