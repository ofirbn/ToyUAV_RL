"""
Visualisation — loads a trained PPO model and runs inference in a live
pygame window.  No training happens here.

Run via:  python main.py  (set mode=visualize or mode=demo in config.txt)

Keyboard controls:
    1-8    switch flight mode
    R      reset episode  (or reset debug angles when in debug mode)
    F1-F4  camera modes
    TAB    toggle renderer debug mode (freeze physics, manual attitude)
    Q/E    roll left/right    (debug mode only)
    W/S    pitch up/down      (debug mode only)
    A/D    yaw left/right     (debug mode only)
    ESC    quit  (Q also quits when NOT in debug mode)
"""

import os
import sys
import pygame
import numpy as np

from stable_baselines3 import PPO

from envs.fixed_wing_env import FixedWingEnv
from render.pygame_renderer import Renderer
from sim.flight_modes import FlightMode, MODE_NAMES


FPS = 60


def run(cfg: dict = None, mission_path: str = None, use_model: bool = True):
    if cfg is None:
        cfg = {}

    model_spec   = cfg.get("model", "models/latest.zip")
    model_path   = model_spec.replace(".zip", "")
    mission_path = mission_path or cfg.get("mission", "missions/demo_mission.json")

    # Try fallback paths if primary not found
    candidates = [model_path, "models/latest", "models/best_reward",
                  "models/fixed_wing_multimode_ppo"]

    pygame.init()
    W, H = 1600, 950
    pygame.display.set_mode((W, H))
    pygame.display.set_caption("ToyUAV RL — Visualising")

    model = None
    if use_model:
        for candidate in candidates:
            if os.path.exists(candidate + ".zip"):
                try:
                    print(f"[VIZ] Loading model from {candidate}.zip")
                    model = PPO.load(candidate)
                    break
                except Exception as e:
                    print(f"[VIZ] Could not load {candidate}: {e}")
        if model is None:
            print("[VIZ] No model loaded — using random actions.")

    mission_exists = os.path.exists(mission_path)
    env = FixedWingEnv(
        mission_path          = mission_path if mission_exists else None,
        training_mode         = False,
        curriculum_phase      = cfg.get("curriculum_phase", "mixed"),
        action_smooth_weight  = float(cfg.get("action_smooth_weight", 0.03)),
        stall_speed           = float(cfg.get("stall_speed", 6.0)),
    )

    renderer = Renderer(W, H)
    cam_mode = 0
    clock    = pygame.time.Clock()

    obs, _      = env.reset()
    last_reward = 0.0
    done        = False
    episode     = 0

    font = pygame.font.SysFont("Consolas", 18)

    print("[VIZ] Window open. Press 1-8 to switch modes, R to reset, ESC to quit.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            elif event.type == pygame.KEYDOWN:
                key = event.key

                # ESC always quits; Q quits only outside debug mode
                if key == pygame.K_ESCAPE or (key == pygame.K_q and not renderer.debug_mode):
                    pygame.quit()
                    sys.exit(0)

                elif key == pygame.K_TAB:
                    renderer.toggle_debug_mode()
                    state = "ON" if renderer.debug_mode else "OFF"
                    print(f"[VIZ] Renderer debug mode {state}")

                elif key == pygame.K_r:
                    if renderer.debug_mode:
                        renderer.reset_debug_angles()
                        print("[VIZ] Debug angles reset to zero")
                    else:
                        obs, _ = env.reset()
                        done   = False
                        episode += 1
                        print(f"[VIZ] Reset — episode {episode}")

                elif pygame.K_1 <= key <= pygame.K_8 and not renderer.debug_mode:
                    mode_idx = key - pygame.K_1
                    mode     = FlightMode(mode_idx)
                    env.force_mode(mode)
                    obs = env._build_obs()
                    print(f"[VIZ] Mode override → {MODE_NAMES[mode_idx]}")

                elif key == pygame.K_F1: cam_mode = 0
                elif key == pygame.K_F2: cam_mode = 1
                elif key == pygame.K_F3: cam_mode = 2
                elif key == pygame.K_F4: cam_mode = 3

        if not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, last_reward, done, truncated, info = env.step(action)

            if done:
                status = "LANDED" if info.get('landed') else "CRASHED"
                print(f"[VIZ] Episode {episode} ended — {status}  "
                      f"alt={info['altitude']:.1f}  spd={info['airspeed']:.1f}")

            if done or truncated:
                obs, _ = env.reset()
                done   = False
                episode += 1

        import shared_state as _ss
        _ss.update({"camera_mode": cam_mode})
        renderer.render(env, last_reward)
        clock.tick(FPS)
