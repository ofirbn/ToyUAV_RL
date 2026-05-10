import sys
import os
import io
import copy
import math
import threading
import numpy as np
import pygame
import torch

from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from aircraft import AircraftPhysics, AircraftState
from envs import PilotageEnv, LandingEnv


# ============================================================
# MODEL PATHS
# ============================================================

MODELS_DIR     = "models"
PILOTAGE_PATH  = os.path.join(MODELS_DIR, "pilotage")
LANDING_PATH   = os.path.join(MODELS_DIR, "landing")


def _models_exist():
    return (os.path.exists(PILOTAGE_PATH  + ".zip") and
            os.path.exists(LANDING_PATH   + ".zip"))


def _prompt_user():
    print("\n" + "="*55)
    print("  Saved models found.")
    print("  [L]  Load and fly  (no training)")
    print("  [C]  Load and continue training")
    print("  [S]  Train from scratch")
    print("="*55)
    while True:
        c = input("  Choice: ").strip().upper()
        if c in ("L", "C", "S"):
            return c
        print("  Please enter L, C or S.")


# ============================================================
# SHARED RENDER STATE
# ============================================================

_policy_lock    = threading.Lock()
_display_policy = None          # landing policy snapshot (inference)
_policy_ready   = threading.Event()

_stats_lock  = threading.Lock()
_train_stats = {
    'stage':     'pilotage',    # 'pilotage' | 'landing' | 'done'
    'scenario':  '',            # current pilotage scenario name
    'updates':   0,
    'timesteps': 0,
    'converged': False,
}

_stop = threading.Event()

# trained pilotage policy — set after Stage 1 so the display env can use it
_trained_pilot_lock = threading.Lock()
_trained_pilot      = None     # frozen AcrorPolicy set by training thread


# ============================================================
# TRAINING THREAD
# ============================================================

N_ENVS           = 4
N_STEPS          = 512   # was 256 — too few complete episodes per rollout
STEPS_PER_UPDATE = N_ENVS * N_STEPS


def _make_ppo(env_cls, env_kwargs=None):
    vec = make_vec_env(env_cls, n_envs=N_ENVS,
                       env_kwargs=env_kwargs or {})
    return PPO(
        "MlpPolicy", vec,
        n_steps       = N_STEPS,
        batch_size    = 128,
        ent_coef      = 0.05,   # was 0.01 — needs more exploration early on
        learning_rate = 3e-4,
        policy_kwargs = dict(net_arch=[256, 256]),  # was [128,128] — underpowered
        verbose       = 0,
    )


def _clone_policy(policy):
    buf = io.BytesIO()
    torch.save(policy, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def _sync_display_policy(model):
    """Copy model weights to the display snapshot (brief lock)."""
    sd = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    with _policy_lock:
        _display_policy.load_state_dict(sd)


def _training_worker(mode: str):
    """
    mode: 'scratch' | 'continue' | 'load_only'
    Stages:
        1. Pilotage training until convergence or timestep cap.
           Display uses random actions during this stage (pilotage obs ≠ landing obs).
        2. Landing training until convergence or timestep cap.
           Display switches to the landing policy once it is ready.
    """
    global _display_policy, _trained_pilot

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Stage 1: Pilotage (goal-conditioned, no curriculum) ----

    if mode == "scratch":
        pilotage_model = _make_ppo(PilotageEnv)
        print("[TRAIN] Pilotage: training from scratch (goal-conditioned)")
    else:
        pilotage_model = PPO.load(
            PILOTAGE_PATH,
            env=make_vec_env(PilotageEnv, n_envs=N_ENVS)
        )
        print("[TRAIN] Pilotage: loaded from", PILOTAGE_PATH)

    with _policy_lock:
        _display_policy = _clone_policy(pilotage_model.policy)
        _display_policy.set_training_mode(False)
    _policy_ready.set()

    with _stats_lock:
        _train_stats['stage'] = 'pilotage'

    if mode != "load_only":
        total_ts    = 0
        conv_window = deque(maxlen=PilotageEnv.CONVERGENCE_WINDOW)
        print(f"[PILOTAGE] Training until mean reward ≥ "
              f"{PilotageEnv.CONVERGENCE_REWARD} over "
              f"{PilotageEnv.CONVERGENCE_WINDOW} updates ...")

        while not _stop.is_set() and total_ts < PilotageEnv.MAX_TIMESTEPS:
            pilotage_model.learn(
                total_timesteps     = STEPS_PER_UPDATE,
                reset_num_timesteps = False,
            )
            total_ts += STEPS_PER_UPDATE

            if pilotage_model.ep_info_buffer:
                mean_r = float(np.mean(
                    [ep['r'] for ep in pilotage_model.ep_info_buffer]
                ))
                conv_window.append(mean_r)

            _sync_display_policy(pilotage_model)
            with _stats_lock:
                _train_stats['updates']   += 1
                _train_stats['timesteps'] += STEPS_PER_UPDATE
                _train_stats['scenario']   = PilotageEnv._last_scenario

            if (len(conv_window) == PilotageEnv.CONVERGENCE_WINDOW and
                    np.mean(conv_window) >= PilotageEnv.CONVERGENCE_REWARD):
                print(f"[PILOTAGE] Converged — mean reward "
                      f"{np.mean(conv_window):.1f} → advancing to landing")
                break

        if total_ts >= PilotageEnv.MAX_TIMESTEPS:
            print("[PILOTAGE] Timestep cap — advancing with best pilot so far")

        pilotage_model.save(PILOTAGE_PATH)
        print("[TRAIN] Pilotage model saved →", PILOTAGE_PATH)

    # ---- Stage 2: Landing ----

    # Freeze the trained pilotage policy and share it with the display env
    frozen_pilot = _clone_policy(pilotage_model.policy)
    frozen_pilot.set_training_mode(False)
    with _trained_pilot_lock:
        _trained_pilot = frozen_pilot   # main loop will swap _disp_env._pilot

    if mode == "scratch":
        landing_model = _make_ppo(LandingEnv,
                                  env_kwargs={"pilotage_model": frozen_pilot})
        print("[TRAIN] Landing: training from scratch")
    else:
        landing_model = PPO.load(
            LANDING_PATH,
            env=make_vec_env(LandingEnv, n_envs=N_ENVS,
                             env_kwargs={"pilotage_model": frozen_pilot})
        )
        print("[TRAIN] Landing: loaded from", LANDING_PATH)

    # Switch display to landing policy (7-D) and signal stage change.
    # Main loop will swap _disp_env from PilotageEnv → LandingEnv on seeing this.
    with _policy_lock:
        _display_policy = _clone_policy(landing_model.policy)
        _display_policy.set_training_mode(False)

    with _stats_lock:
        _train_stats['stage']   = 'landing'
        _train_stats['updates'] = 0

    if mode != "load_only":
        reward_window = deque(maxlen=LandingEnv.CONVERGENCE_WINDOW)
        total_ts      = 0

        while not _stop.is_set() and total_ts < LandingEnv.MAX_TIMESTEPS:
            landing_model.learn(
                total_timesteps     = STEPS_PER_UPDATE,
                reset_num_timesteps = False,
            )
            total_ts += STEPS_PER_UPDATE

            if landing_model.ep_info_buffer:
                mean_r = float(np.mean(
                    [ep['r'] for ep in landing_model.ep_info_buffer]
                ))
                reward_window.append(mean_r)

            _sync_display_policy(landing_model)

            with _stats_lock:
                _train_stats['updates']   += 1
                _train_stats['timesteps'] += STEPS_PER_UPDATE

            if (len(reward_window) == LandingEnv.CONVERGENCE_WINDOW and
                    np.mean(reward_window) >= LandingEnv.CONVERGENCE_REWARD):
                print(f"[TRAIN] Landing converged at {total_ts:,} timesteps "
                      f"(mean reward {np.mean(reward_window):.1f})")
                break

        landing_model.save(LANDING_PATH)
        print("[TRAIN] Landing model saved →", LANDING_PATH)

    with _stats_lock:
        _train_stats['stage']    = 'done'
        _train_stats['converged'] = True
    print("[TRAIN] All training complete.")


# ---- decide mode, start thread ----

if _models_exist():
    _mode = _prompt_user()
else:
    _mode = "scratch"
    print("[TRAIN] No saved models found — training from scratch.")

_train_thread = threading.Thread(
    target=_training_worker, args=(_mode,), daemon=True
)
_train_thread.start()


# ============================================================
# DISPLAY ENV  (uses frozen pilotage once available)
# ============================================================

# Stage 1 display: PilotageEnv  (11-D obs, no runway)
# Stage 2 display: LandingEnv   (7-D obs, with runway)
_pilot_disp_env = PilotageEnv()

_tmp_pilotage = PPO("MlpPolicy", PilotageEnv(), verbose=0).policy
_tmp_pilotage.set_training_mode(False)
_land_disp_env  = LandingEnv(_tmp_pilotage)

_disp_env      = _pilot_disp_env   # start with pilotage display
_disp_obs, _   = _disp_env.reset()
_disp_episodes = 0

_disp_state  = 'fly'      # 'fly' | 'roll' | 'crash'
_roll_vel    = np.zeros(3)
_crash_parts = []
_crash_frame = 0
_CRASH_DURATION = 90


def _spawn_crash(screen_pos):
    parts = []
    for _ in range(40):
        angle = np.random.uniform(0, 2 * math.pi)
        speed = np.random.uniform(1.5, 8.0)
        kind  = np.random.randint(4)
        col   = [(255,255,180),(255,210,40),(255,110,0),(180,40,0)][kind]
        parts.append({'x': float(screen_pos[0]), 'y': float(screen_pos[1]),
                       'vx': speed*math.cos(angle),
                       'vy': speed*math.sin(angle) - np.random.uniform(0.5,4),
                       'col': col, 'sz': int(np.random.randint(2,7)),
                       'life': int(np.random.randint(30,75))})
    for _ in range(15):
        angle = np.random.uniform(0, 2*math.pi)
        speed = np.random.uniform(0.5, 3.0)
        g     = int(np.random.randint(100,200))
        parts.append({'x': float(screen_pos[0]), 'y': float(screen_pos[1]),
                       'vx': speed*math.cos(angle),
                       'vy': speed*math.sin(angle) - np.random.uniform(0,2),
                       'col': (g,g,g), 'sz': int(np.random.randint(3,9)),
                       'life': int(np.random.randint(50,90))})
    return parts


# ============================================================
# PYGAME
# ============================================================

pygame.init()

WIDTH  = 1400
HEIGHT =  900

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Fixed Wing UAV — Hierarchical DRL")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("Consolas", 22)

CAMERA_POS = np.array([-250, -250, 180], dtype=float)
FOV        = 800

# Smoothed look-at point — updated every frame to track the plane.
# Exponential smoothing keeps movement gradual and jitter-free.
_cam_target = np.array([0.0, 300.0, 50.0], dtype=float)
_CAMERA_SMOOTH = 0.06   # fraction to close per frame — lower = smoother

# Camera basis vectors (recomputed each frame from _cam_target)
_cf = np.array([0.0, 0.0, 0.0], dtype=float)
_cr = np.array([0.0, 0.0, 0.0], dtype=float)
_cu = np.array([0.0, 0.0, 0.0], dtype=float)


def _update_camera(plane_pos):
    """Smoothly pivot camera to keep the plane in view."""
    # Move look-at point toward the plane
    _cam_target[:] += (plane_pos - _cam_target) * _CAMERA_SMOOTH

    fwd = _cam_target - CAMERA_POS
    fwd_len = np.linalg.norm(fwd)
    if fwd_len < 0.1:
        return
    fwd /= fwd_len

    # Build orthonormal basis; fall back if looking straight up/down
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    r_len = np.linalg.norm(right)
    if r_len < 0.01:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        r_len = np.linalg.norm(right)
    right /= r_len

    _cf[:] = fwd
    _cr[:] = right
    _cu[:] = np.cross(right, fwd)


# Initialise camera vectors pointing at the default scene centre
_update_camera(np.array([0.0, 300.0, 50.0]))


# ============================================================
# PROJECT
# ============================================================

def project(point):
    rel   = point - CAMERA_POS
    depth = float(np.dot(rel, _cf))
    if depth <= 0.5:
        return None
    sx = WIDTH  // 2 + int(float(np.dot(rel, _cr)) / depth * FOV)
    sy = HEIGHT // 2 - int(float(np.dot(rel, _cu)) / depth * FOV)
    return sx, sy


# ============================================================
# DRAW WORLD
# ============================================================

def draw_ground():
    for x in range(-320, 321, 20):
        p1 = project(np.array([x,-80,0]))
        p2 = project(np.array([x,800,0]))
        if p1 and p2:
            pygame.draw.line(screen, (40,40,55), p1, p2, 1)
    for y in range(-80, 801, 20):
        p1 = project(np.array([-320,y,0]))
        p2 = project(np.array([ 320,y,0]))
        if p1 and p2:
            pygame.draw.line(screen, (40,40,55), p1, p2, 1)


def _poly(corners, color, width=0):
    pts = [project(c) for c in corners]
    if all(pts):
        pygame.draw.polygon(screen, color, pts, width)


def draw_runway():
    _poly([np.array([-13,-25,0]), np.array([13,-25,0]),
           np.array([13, 95,0]), np.array([-13,95,0])], (70,70,70))
    for x0,x1 in [(-13,-10),(-9,-6),(-5,-2),(2,5),(6,9),(10,13)]:
        _poly([np.array([x0,87,0.01]), np.array([x1,87,0.01]),
               np.array([x1,95,0.01]), np.array([x0,95,0.01])], (240,240,240))
    CL = 0.55
    for y0 in range(5, 83, 15):
        _poly([np.array([-CL,y0,   0.01]), np.array([CL,y0,   0.01]),
               np.array([ CL,y0+9,0.01]), np.array([-CL,y0+9,0.01])], (230,230,180))
    _poly([np.array([-13,-25,0]), np.array([13,-25,0]),
           np.array([13, 95,0]), np.array([-13,95,0])], (255,255,255), 2)


# ============================================================
# DRAW PLANE
# ============================================================

def draw_plane(pos, vel, yaw=None, roll=0.0, pitch=0.0):
    # ground shadow
    shadow = project(np.array([pos[0], pos[1], 0.0]))
    if shadow:
        alt    = max(pos[2], 0.1)
        radius = max(2, int(220 / (alt + 10)))
        pygame.draw.circle(screen, (55,70,90), shadow, radius)
        c  = (90,110,135); r2 = max(2, radius+4)
        pygame.draw.line(screen, c, (shadow[0]-r2,shadow[1]), (shadow[0]+r2,shadow[1]), 1)
        pygame.draw.line(screen, c, (shadow[0],shadow[1]-r2), (shadow[0],shadow[1]+r2), 1)

    if yaw is None:
        yaw = math.atan2(vel[0], -vel[1])

    size = 9
    local = {
        "nose": np.array([0,        -size*2.5, 0        ]),
        "tail": np.array([0,         size*2,   0        ]),
        "lw":   np.array([-size*4.5, 0,        0        ]),
        "rw":   np.array([ size*4.5, 0,        0        ]),
        "lwt":  np.array([-size*3.5, size*0.6, 0        ]),
        "rwt":  np.array([ size*3.5, size*0.6, 0        ]),
        "fin":  np.array([0,         size*1.8,  size*2.0 ]),
    }

    cy, sy = math.cos(yaw),   math.sin(yaw)
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # Full 3-axis rotation: R_yaw @ R_roll @ R_pitch
    # Roll tilts the wings around the nose-tail axis;
    # Pitch raises/lowers the nose around the lateral axis.
    rot = np.array([
        [ cy*cr,  cy*sr*sp - sy*cp,  cy*sr*cp + sy*sp],
        [ sy*cr,  sy*sr*sp + cy*cp,  sy*sr*cp - cy*sp],
        [-sr,     cr*sp,              cr*cp            ],
    ])

    world = {k: pos + rot @ v for k, v in local.items()}
    proj  = {k: project(v)    for k, v in world.items()}

    if proj["nose"] and proj["tail"]:
        pygame.draw.line(screen, (0,80,120),   proj["nose"], proj["tail"], 14)
        pygame.draw.line(screen, (0,220,255),  proj["nose"], proj["tail"],  8)
    for tip,trail in [("lw","lwt"),("rw","rwt")]:
        if proj[tip] and proj[trail] and proj["nose"]:
            pts = [proj["nose"], proj[tip], proj[trail]]
            pygame.draw.polygon(screen, (40,60,80),    pts)
            pygame.draw.polygon(screen, (200,220,255), pts, 2)
    if proj["tail"] and proj["fin"]:
        pygame.draw.line(screen, (180,100,0),  proj["tail"], proj["fin"], 8)
        pygame.draw.line(screen, (255,180,80), proj["tail"], proj["fin"], 4)


# ============================================================
# SCENARIO VISUAL CUES
# ============================================================

small_font = pygame.font.SysFont("Consolas", 15)


def _lbl(pt, text, color):
    if pt:
        screen.blit(small_font.render(text, True, color), (pt[0]+6, pt[1]-8))


def _altitude_diamond(pos, target_alt, color):
    """Horizontal diamond at target altitude near the aircraft."""
    arm = 22
    corners = [
        np.array([pos[0],       pos[1]-arm, target_alt]),
        np.array([pos[0]+arm,   pos[1],     target_alt]),
        np.array([pos[0],       pos[1]+arm, target_alt]),
        np.array([pos[0]-arm,   pos[1],     target_alt]),
    ]
    pts = [project(c) for c in corners]
    if all(pts):
        pygame.draw.polygon(screen, color, pts, 2)
    tip = project(np.array([pos[0]+arm+4, pos[1], target_alt]))
    _lbl(tip, f"{target_alt:.0f} m", color)


def _vertical_guide(pos, target_alt, color):
    """Dashed vertical line from aircraft to target altitude."""
    z0, z1 = pos[2], target_alt
    steps = 10
    for i in range(steps):
        za = z0 + (z1-z0) * i       / steps
        zb = z0 + (z1-z0) * (i+0.4) / steps
        pa = project(np.array([pos[0], pos[1], za]))
        pb = project(np.array([pos[0], pos[1], zb]))
        if pa and pb:
            pygame.draw.line(screen, color, pa, pb, 1)


def _heading_arrow(pos, target_yaw, color):
    """Arrow on the ground pointing in the target heading."""
    length = 70
    ex = pos[0] + math.sin(target_yaw) * length
    ey = pos[1] - math.cos(target_yaw) * length
    p0 = project(np.array([pos[0], pos[1], 1.0]))
    p1 = project(np.array([ex,     ey,     1.0]))
    if p0 and p1:
        pygame.draw.line(screen, color, p0, p1, 3)
        dx = p1[0]-p0[0]; dy = p1[1]-p0[1]
        ln = max(math.sqrt(dx*dx+dy*dy), 1)
        nx, ny = dx/ln, dy/ln
        h1 = (int(p1[0]-nx*14-ny*8), int(p1[1]-ny*14+nx*8))
        h2 = (int(p1[0]-nx*14+ny*8), int(p1[1]-ny*14-nx*8))
        pygame.draw.polygon(screen, color, [p1, h1, h2])
    _lbl(p1, "TGT HDG", color)


def _speed_arrow(pos, yaw, target_spd, color):
    """Arrow ahead of aircraft; length encodes target speed."""
    length = target_spd * 5
    ex = pos[0] + math.sin(yaw) * length
    ey = pos[1] - math.cos(yaw) * length
    p0 = project(np.array([pos[0], pos[1], pos[2]]))
    p1 = project(np.array([ex, ey, pos[2]]))
    if p0 and p1:
        pygame.draw.line(screen, color, p0, p1, 3)
    _lbl(p1, f"TGT {target_spd:.1f} m/s", color)


def _wings_level_ref(pos, color):
    """Horizontal reference bar at aircraft altitude — target for recovery."""
    arm = 40
    lft = project(np.array([pos[0]-arm, pos[1], pos[2]]))
    rgt = project(np.array([pos[0]+arm, pos[1], pos[2]]))
    ctr = project(np.array([pos[0],     pos[1], pos[2]]))
    if lft and rgt:
        pygame.draw.line(screen, color, lft, rgt, 2)
    if lft:
        pygame.draw.line(screen, color, lft, (lft[0], lft[1]-8), 2)
    if rgt:
        pygame.draw.line(screen, color, rgt, (rgt[0], rgt[1]-8), 2)
    if ctr:
        pygame.draw.line(screen, color, ctr, (ctr[0], ctr[1]-14), 2)
    _lbl(rgt, "LEVEL REF", color)


_MODE_COLOR = {
    'level'   : (  0, 200, 255),
    'climb'   : (  0, 255,  80),
    'descent' : (255, 140,   0),
    'turn'    : (255, 220,   0),
    'speed'   : (200,  80, 255),
    'recovery': (255,  60,  60),
}


def draw_scenario_cues(pos, disp_env):
    if not hasattr(disp_env, '_scenario') or not disp_env._scenario:
        return
    mode = disp_env._scenario
    cmd  = disp_env._cmd if hasattr(disp_env, '_cmd') else None
    if cmd is None:
        return
    col = _MODE_COLOR.get(mode, (255, 255, 255))

    # cmd = [speed, alt, vz, yaw, roll]
    cmd_alt = float(cmd[1])
    cmd_yaw = float(cmd[3])
    cmd_spd = float(cmd[0])

    if mode in ('level', 'speed'):
        _altitude_diamond(pos, cmd_alt, col)
        yaw = disp_env._state.yaw if disp_env._state else 0.0
        _speed_arrow(pos, yaw, cmd_spd, col)

    elif mode in ('climb', 'descent'):
        _altitude_diamond(pos, cmd_alt, col)
        _vertical_guide(pos, cmd_alt, col)

    elif mode == 'turn':
        _heading_arrow(pos, cmd_yaw, col)
        _altitude_diamond(pos, cmd_alt, col)

    elif mode == 'recovery':
        _wings_level_ref(pos, col)


# ============================================================
# MAIN LOOP
# ============================================================

_GRADE_COLOR = {
    "PERFECT SCORE LANDING": (255,215,  0),
    "EXCELLENT LANDING":     (  0,255,140),
    "VERY GOOD LANDING":     (100,220, 80),
    "GOOD LANDING":          (180,220, 50),
    "ALMOST GOOD LANDING":   (255,210,  0),
    "FAIR LANDING":          (255,150,  0),
    "POOR LANDING":          (255, 80, 40),
    "BAD LANDING":           (220, 30, 30),
    "*CRASHED*":             (255,  0,  0),
}
WHITE = (255,255,255)
DIM   = (160,160,160)

while True:

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _stop.set()
            pygame.quit()
            sys.exit()

    # --------------------------------------------------------
    # STAGE SWITCH: pilotage → landing display env
    # --------------------------------------------------------

    with _stats_lock:
        current_stage = _train_stats['stage']

    if current_stage in ('landing', 'done') and _disp_env is _pilot_disp_env:
        # swap to landing display env and update its pilot policy
        with _trained_pilot_lock:
            tp = _trained_pilot
        if tp is not None:
            _land_disp_env._pilot = tp
        _disp_env    = _land_disp_env
        _disp_obs, _ = _disp_env.reset()
        _disp_state  = 'fly'

    # --------------------------------------------------------
    # STATE MACHINE: fly → roll / crash
    # --------------------------------------------------------

    if _disp_state == 'fly':
        if _policy_ready.is_set():
            with _policy_lock:
                action, _ = _display_policy.predict(_disp_obs, deterministic=True)
        else:
            action = _disp_env.action_space.sample()

        _disp_obs, _, done, truncated, _ = _disp_env.step(action)

        if done and _disp_env.pos[2] > 1000 or \
           done and abs(float(_disp_env.pos[0])) > 400 or \
           done and float(_disp_env.pos[1]) < -400 or \
           done and float(_disp_env.pos[1]) > 1200:
            _disp_state = 'crash'
            sp = project(np.array([float(_disp_env.pos[0]),
                                   float(_disp_env.pos[1]),
                                   float(_disp_env.pos[2])])) or (WIDTH//2, HEIGHT//3)
            _crash_parts[:] = _spawn_crash(sp)
            _crash_frame = 0
        elif done and _disp_env.pos[2] <= 0:
            on_x      = abs(float(_disp_env.pos[0])) < LandingEnv.RWY_X
            on_y      = LandingEnv.RWY_Y0 <= float(_disp_env.pos[1]) <= LandingEnv.RWY_Y1
            hard_slam = abs(float(_disp_env.vel[2])) > 5.0
            if on_x and on_y and not hard_slam:
                _disp_state = 'roll'
                _roll_vel   = _disp_env.vel.astype(float).copy()
                _roll_vel[2]= 0.0
            else:
                _disp_state = 'crash'
                sp = project(np.array([float(_disp_env.pos[0]),
                                       float(_disp_env.pos[1]), 0.0])) or (WIDTH//2, HEIGHT//2)
                _crash_parts[:] = _spawn_crash(sp)
                _crash_frame = 0
        elif done or truncated or float(np.linalg.norm(_disp_env.vel)) < 2.0:
            # also reset if the plane has stalled to near-zero speed — avoids
            # watching it crawl for hundreds of steps during early training
            _disp_obs, _ = _disp_env.reset()
            _disp_episodes += 1

    elif _disp_state == 'roll':
        _roll_vel[0] *= 0.90
        _roll_vel[1] *= 0.97
        _roll_vel[2]  = 0.0
        _disp_env._state.pos = (_disp_env._state.pos + _roll_vel * 0.1)
        _disp_env._state.pos[2] = 0.0
        _disp_env._state.vel[:] = _roll_vel
        if np.linalg.norm(_roll_vel[:2]) < 0.3:
            _disp_state = 'fly'
            _disp_obs, _ = _disp_env.reset()
            _disp_episodes += 1

    elif _disp_state == 'crash':
        _crash_frame += 1
        if _crash_frame >= _CRASH_DURATION or not _crash_parts:
            _disp_state = 'fly'
            _disp_obs, _ = _disp_env.reset()
            _disp_episodes += 1

    pos   = _disp_env.pos.astype(float)
    vel   = _disp_env.vel.astype(float)
    _s    = _disp_env._state if hasattr(_disp_env, '_state') else None
    yaw   = _s.yaw   if _s else None
    roll  = _s.roll  if _s else 0.0
    pitch = _s.pitch if _s else 0.0

    # Smooth-pivot camera to keep the plane in view
    _update_camera(pos)

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    screen.fill((5,8,22))
    pygame.draw.rect(screen, (15,25,65), (0,        0, WIDTH, HEIGHT//2))
    pygame.draw.rect(screen, (12,15,18), (0, HEIGHT//2, WIDTH, HEIGHT//2))

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    draw_ground()
    if _disp_env is _land_disp_env:
        draw_runway()

    if _disp_state != 'crash':
        if _disp_env is _pilot_disp_env:
            draw_scenario_cues(pos, _disp_env)

    if _disp_state == 'crash':
        for ring_r, ring_col in [(int(_crash_frame*2.2),(255,180,60)),
                                  (int(_crash_frame*1.4),(255,240,120)),
                                  (int(_crash_frame*0.7),(255,255,200))]:
            if 0 < ring_r < 300 and _crash_parts:
                fade = max(0, 255 - _crash_frame*5)
                col  = tuple(min(255,int(c*fade/255)) for c in ring_col)
                pygame.draw.circle(screen, col,
                                   (int(_crash_parts[0]['x']),
                                    int(_crash_parts[0]['y'])), ring_r, 2)
        survivors = []
        for p in _crash_parts:
            p['vy'] += 0.25; p['x'] += p['vx']; p['y'] += p['vy']; p['life'] -= 1
            if p['life'] > 0:
                alpha = p['life']/75.0
                col   = tuple(min(255,int(c*alpha)) for c in p['col'])
                pygame.draw.circle(screen, col, (int(p['x']),int(p['y'])), p['sz'])
                survivors.append(p)
        _crash_parts[:] = survivors
    else:
        draw_plane(pos, vel, yaw, roll, pitch)

    # --------------------------------------------------------
    # HUD
    # --------------------------------------------------------

    with _stats_lock:
        stage     = _train_stats['stage']
        scenario  = _train_stats['scenario']
        updates   = _train_stats['updates']
        timesteps = _train_stats['timesteps']
        converged = _train_stats['converged']

    grade     = _disp_env.last_grade
    grade_col = _GRADE_COLOR.get(grade, DIM)
    score_txt = f"SCORE: {_disp_env.last_score:.3f}" if grade else "SCORE: ---"

    stage_label = {
        'pilotage': "STAGE 1 — PILOTAGE",
        'landing':  "STAGE 2 — LANDING",
        'done':     "TRAINING COMPLETE",
    }.get(stage, stage.upper())

    is_pilotage  = (_disp_env is _pilot_disp_env)
    sc_name      = scenario.replace('_', ' ').upper() if scenario else '---'
    pitch_deg    = math.degrees(pitch)
    roll_deg     = math.degrees(roll)
    throttle_pct = int(_s.throttle_pos * 100) if _s else 0
    horiz_spd    = float(math.sqrt(vel[0]**2 + vel[1]**2))
    vert_spd     = float(vel[2])
    cmd          = _disp_env._cmd if hasattr(_disp_env, '_cmd') else np.zeros(5)

    if is_pilotage:
        tgt_spd = float(cmd[0]); tgt_alt = float(cmd[1]); tgt_vz = float(cmd[2])
        hud = [
            (f"RL  {stage_label}",                          (100,200,255)),
            (f"MODE:      {sc_name}",                       (180,180,255)),
            ("",                                             WHITE),
            (f"AIRSPEED:  {np.linalg.norm(vel):6.2f} m/s",  WHITE),
            (f"HORIZ SPD: {horiz_spd:6.2f} m/s",            (200,230,255)),
            (f"VERT SPD:  {vert_spd:+6.2f} m/s",            (255,180,100) if vert_spd < -0.5 else WHITE),
            (f"TGT SPEED: {tgt_spd:6.2f} m/s",              (180,255,180)),
            (f"TGT ALT:   {tgt_alt:6.1f} m  vz={tgt_vz:+.1f}", (180,255,180)),
            (f"THROTTLE:  {throttle_pct:5d} %",              (255,220,80)),
            (f"ALTITUDE:  {pos[2]:6.1f} m",                  WHITE),
            (f"PITCH:     {pitch_deg:+6.1f}°",               WHITE),
            (f"BANK:      {roll_deg:+6.1f}°",                WHITE),
            ("",                                             WHITE),
            (f"EPISODES:  {_disp_episodes}",                 WHITE),
            (f"UPDATES:   {updates}",                        WHITE),
            (f"TIMESTEPS: {timesteps:,}",                    WHITE),
        ]
    else:
        hud = [
            (f"RL  {stage_label}",                   (100,200,255) if not converged else (100,255,100)),
            (f"X ERR:     {pos[0]:+6.1f}",           WHITE),
            (f"RWY DIST:  {pos[1]:6.1f}",            WHITE),
            (f"ALTITUDE:  {pos[2]:6.1f}",            WHITE),
            (f"SPEED:     {np.linalg.norm(vel):6.2f}", WHITE),
            (f"EPISODES:  {_disp_episodes}",          WHITE),
            (f"UPDATES:   {updates}",                 WHITE),
            (f"TIMESTEPS: {timesteps:,}",             WHITE),
            ("",                                      WHITE),
            (f"LAST: {grade or '---'}",               grade_col),
            (score_txt,                               grade_col),
        ]

    for i, (line, color) in enumerate(hud):
        txt = font.render(line, True, color)
        screen.blit(txt, (20, 20 + i * 32))

    pygame.display.flip()
    clock.tick(60)
