"""
ToyUAV RL — main entry point.

Edit config.txt, then run:
    python main.py

Supported modes:
    train            — headless PPO training
    train_visual     — PPO training with live pygame dashboard
    pipeline_visual  — record teacher → BC → PPO, all in one pygame window
    visualize        — run trained model in pygame window
    demo             — alias for visualize
"""

import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.txt")

DEFAULT_CONFIG = """\
mode=train_visual
timesteps=3000000
mission=missions/demo_mission.json
model=models/latest.zip
# curriculum=true  → auto-progression through all 9 stages
# curriculum=false → static phase set by curriculum_phase below
curriculum=true
curriculum_phase=stabilize
visualize_training=true
save_every=100000
action_smooth_weight=0.03
stall_speed=6.0
# emergency_recovery_enabled: allow RECOVERY mode to activate on stall/bad attitude.
# Defaults: false for stabilize/cruise/altitude/heading/waypoint/loiter/approach/landing
#           true  for recovery/mixed
# Uncomment to override the per-phase default:
# emergency_recovery_enabled=false
"""


def read_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            f.write(DEFAULT_CONFIG)
        print(f"[CONFIG] Created default config.txt at {CONFIG_PATH}")
    cfg = {}
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    return cfg


def main():
    from config_gui import run_gui
    if not run_gui():
        return  # user closed the launcher without starting

    cfg  = read_config()
    mode = cfg.get("mode", "train").lower()

    auto = cfg.get("curriculum", "false").lower() == "true"
    curr_info = (f"AUTO start={cfg.get('curriculum_phase','stabilize')}"
                 if auto else f"static phase={cfg.get('curriculum_phase','mixed')}")
    force_new  = cfg.get("force_new", "false").lower() == "true"
    model_info = "SCRATCH" if force_new else cfg.get("model", "models/latest.zip")
    print(f"[CONFIG] mode={mode}  timesteps={cfg.get('timesteps','?')}  "
          f"curriculum={curr_info}  "
          f"model={model_info}")

    if mode == "train":
        from train import train
        train(cfg)

    elif mode in ("train_visual", "train_live"):
        from train import train_visual
        train_visual(cfg)

    elif mode in ("pipeline_visual", "pipeline"):
        from train import train_pipeline_visual
        train_pipeline_visual(cfg)

    elif mode in ("visualize", "demo"):
        from visualize import run
        run(cfg=cfg)

    else:
        print(f"[CONFIG] Unknown mode '{mode}'.")
        print("  Set mode=train, train_visual, pipeline_visual, visualize, or demo in config.txt.")


if __name__ == "__main__":
    main()
