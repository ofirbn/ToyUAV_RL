"""
Diagnostic: print which policy each FlightMode routes to — the default
checkpoint or a mode-specific expert — without running any simulation.

Run via:  python tools/print_routing.py [--model models/latest.zip]
                                         [--experts-dir models/experts]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.mission_manager import MissionManager
from sim.flight_modes import MODE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Print expert routing per FlightMode")
    parser.add_argument("--model", default="models/latest.zip", help="Default policy checkpoint")
    parser.add_argument("--experts-dir", default="models/experts", help="Per-mode expert checkpoints dir")
    args = parser.parse_args()

    mgr = MissionManager(default_model_path=args.model, experts_dir=args.experts_dir).load()
    routing = mgr.describe_routing()

    print(f"{'MODE':<12} {'SOURCE':<8} PATH")
    for mode, info in routing.items():
        path = info["path"] or "(none — random actions)"
        print(f"{MODE_NAMES[mode]:<12} {info['source']:<8} {path}")


if __name__ == "__main__":
    main()
