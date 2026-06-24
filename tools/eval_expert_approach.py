"""
M4A/M5A — evaluate the APPROACH expert.

As of M5A this is a thin backward-compatible wrapper around the generalized
evaluator tools/eval_expert_mode.py. It forwards every argument unchanged and
simply pins --mode approach, so the historical command line keeps working:

    python tools/eval_expert_approach.py --episodes 50
    python tools/eval_expert_approach.py --model models/experts/approach --episodes 50
    python tools/eval_expert_approach.py --episodes 50 \
        --model models/latest --model models/experts/approach

The generalized tool produces the same metrics for APPROACH (lateral / heading
/ glideslope error, success/crash rates, comparison table, best/median/worst
trajectories) plus stall-rate, computed from the env's own state/target/info.
There is no separate approach-only code path to maintain anymore.

Equivalent direct invocation:
    python tools/eval_expert_mode.py --mode approach ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_expert_mode import main as _main

# Kept for backward compatibility with anything importing this symbol.
MODEL_PATH = "models/experts/approach"


def main():
    # Forward the original argv (sans program name) with --mode approach pinned.
    _main(["--mode", "approach"] + sys.argv[1:])


if __name__ == "__main__":
    main()
