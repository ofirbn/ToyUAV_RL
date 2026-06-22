"""
Full telemetry dashboard renderer for ToyUAV RL.

Layout (1600 × 950):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  LEFT 280px │        3D VIEW  1040×670       │  RIGHT 280px         │
  │  PPO stats  │                                │  Flight HUD          │
  │  episode    │     world-space 3-D scene      │  Reward breakdown    │
  │  rates      │     + trajectory trail         │  Attitude indicator  │
  │  objective  │     + mission overlays         │                      │
  ├─────────────┴────────────────────────────────┴──────────────────────┤
  │                   BOTTOM  1600×280  —  scrolling graphs             │
  └──────────────────────────────────────────────────────────────────────┘

Renderer is READ-ONLY: it never calls env.step() or model.predict().
Camera modes: F1=chase  F2=side  F3=top  F4=cockpit  (toggled in main loop)
"""

import collections
import math
import pygame
import numpy as np

from sim.flight_modes import FlightMode, MODE_NAMES


# ═══════════════════════════════════════════════════════════ layout ══════════

TOTAL_W  = 1600
TOTAL_H  = 950

LEFT_W   = 280
RIGHT_W  = 280
BOT_H    = 280
VIEW_W   = TOTAL_W - LEFT_W - RIGHT_W   # 1040
VIEW_H   = TOTAL_H - BOT_H              # 670
VIEW_X   = LEFT_W
VIEW_Y   = 0

# 3-D projection screen-centre (within the view panel)
PROJ_CX  = VIEW_X + VIEW_W // 2
PROJ_CY  = VIEW_Y + VIEW_H // 2
FOV      = 700


# ═══════════════════════════════════════════════════════════ palette ═════════

BG        = (  5,   8,  22)
PANEL_BG  = (  8,  12,  28)
PANEL_BDR = ( 30,  40,  70)
SKY_TOP   = (  5,   8,  22)
SKY_BOT   = ( 15,  25,  65)
GROUND    = ( 12,  15,  18)
GRID      = ( 35,  40,  55)
WHITE     = (255, 255, 255)
DIM       = (100, 110, 130)
GREEN     = (  0, 220,  80)
AMBER     = (255, 180,  40)
RED       = (255,  60,  60)
CYAN      = (  0, 200, 255)
TEAL      = ( 80, 200, 180)
YELLOW    = (255, 230,  30)
MAGENTA   = (220,  80, 255)
HUD_GREEN = ( 50, 220, 100)

MODE_COLORS = {
    FlightMode.STABILIZE:     (160, 200, 255),
    FlightMode.ALTITUDE_HOLD: (  0, 220,  80),
    FlightMode.HEADING_HOLD:  (255, 220,   0),
    FlightMode.WAYPOINT:      (  0, 200, 255),
    FlightMode.LOITER:        (200,  80, 255),
    FlightMode.APPROACH:      (255, 140,   0),
    FlightMode.LANDING:       (255,  60,  60),
    FlightMode.RECOVERY:      (255,  60, 200),
}

GRAPH_COLORS = {
    "reward":      (  0, 210, 100),
    "mean_reward": (180, 180, 255),
    "crash":       (255,  60,  60),
    "success":     (  0, 210, 100),
    "stall":       (255, 140,   0),
    "policy_loss": (255, 180,  40),
    "value_loss":  ( 80, 160, 255),
}

_CURRICULUM_PHASES = [
    "stabilize", "altitude_hold", "heading_hold",
    "waypoint", "loiter", "approach", "landing",
    "recovery", "mixed",
]

_CONFIG_PATH = "config.txt"


# ═══════════════════════════════════════════════════════════ helpers ═════════

class _StateProxy:
    __slots__ = ('pos', 'vel', 'pitch', 'roll', 'yaw',
                 'pitch_rate', 'roll_rate', 'yaw_rate',
                 'throttle_pos', 'airspeed')

    def __init__(self, d: dict):
        self.pos          = np.array(d["pos"],  dtype=float)
        self.vel          = np.array(d["vel"],  dtype=float)
        self.pitch        = float(d["pitch"])
        self.roll         = float(d["roll"])
        self.yaw          = float(d["yaw"])
        self.pitch_rate   = float(d["pitch_rate"])
        self.roll_rate    = float(d["roll_rate"])
        self.yaw_rate     = float(d["yaw_rate"])
        self.throttle_pos = float(d["throttle_pos"])
        self.airspeed     = float(d["airspeed"])


class _TargetProxy:
    __slots__ = ('position', 'heading', 'altitude', 'radius')

    def __init__(self, d: dict):
        self.position = np.array(d.get("target_position", [0., 0., 50.]), dtype=float)
        self.heading  = float(d.get("target_heading",  0.0))
        self.altitude = float(d.get("target_altitude", 50.0))
        self.radius   = float(d.get("target_radius",   60.0))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _read_wireframe_cfg():
    try:
        with open("config.txt") as f:
            for line in f:
                if "wireframe" in line.lower():
                    return line.split("=", 1)[1].strip().lower() == "true"
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════ renderer ════════

class Renderer:
    """
    Owns the pygame window and renders the full telemetry dashboard.
    Call render_state(state_dict) once per frame.
    """

    SMOOTH       = 0.04   # camera lag — lower = smoother chase
    VIS_SMOOTH   = 0.12   # attitude visual smoothing per frame

    def __init__(self, width: int = TOTAL_W, height: int = TOTAL_H):
        self.W, self.H = width, height

        # Camera
        self._cam_pos    = np.array([-250., -250., 180.])
        self._cam_target = np.array([   0.,  200.,  50.])
        self._cf = np.array([1., 0., 0.])
        self._cr = np.array([0., 1., 0.])
        self._cu = np.array([0., 0., 1.])
        self._chase_offset = np.array([0., 70., 25.])   # further back, slightly higher
        self._update_camera_basis()
        self._wireframe = _read_wireframe_cfg()

        # Smoothed visual attitude (lerped toward true state each frame)
        self._vis_roll  = 0.0
        self._vis_pitch = 0.0
        self._vis_yaw   = 0.0

        # Last known aircraft position (updated every frame)
        self._last_pos = np.array([0., 50., 50.])

        # Waypoint capture flash state — counts down after each capture event
        self._wp_capture_flash_timer = 0
        self._wp_capture_count       = 0

        # ── Renderer debug mode ──────────────────────────────────────────────
        # TAB to toggle; Q/E = roll, W/S = pitch, A/D = yaw
        self._debug_mode       = False
        self._dbg_roll         = 0.0
        self._dbg_pitch        = 0.0
        self._dbg_yaw          = 0.0
        self._dbg_pos          = np.array([0., 50., 60.])
        self._dbg_cam_pos      = np.array([0., 170., 85.])
        self._dbg_cam_target   = np.array([0.,  50., 60.])
        self._DBG_STEP         = math.radians(1.0)   # per frame while key held

        # Fonts
        self._font_lg  = pygame.font.SysFont("Consolas", 22)
        self._font_md  = pygame.font.SysFont("Consolas", 17)
        self._font_sm  = pygame.font.SysFont("Consolas", 13)
        self._font_hdr = pygame.font.SysFont("Consolas", 15, bold=True)

        self._screen = pygame.display.get_surface()
        if self._screen is None:
            self._screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("ToyUAV RL — Training Dashboard")

        # Clip rect for the 3-D viewport (used to scissor world-draw calls)
        self._view_rect = pygame.Rect(VIEW_X, VIEW_Y, VIEW_W, VIEW_H)

        # Off-screen surface for 3-D viewport (avoids bleed into panels)
        self._view_surf = pygame.Surface((VIEW_W, VIEW_H))

        # Pre-training config screen state
        self._cfg_draft    = {}
        self._cfg_widgets  = {}
        self._cfg_dragging = None

        # Cross-mode diagnostics: rolling vz buffer for vertical oscillation score
        self._diag_vz_buf: collections.deque = collections.deque(maxlen=60)

    # ═══════════════════════════════════════════════════ debug mode API ══════

    @property
    def debug_mode(self) -> bool:
        return self._debug_mode

    def toggle_debug_mode(self) -> None:
        self._debug_mode = not self._debug_mode
        if self._debug_mode:
            # Freeze aircraft at last known position and camera at current view
            self._dbg_pos        = self._last_pos.copy()
            self._dbg_pos[2]     = max(float(self._dbg_pos[2]), 20.0)
            self._dbg_cam_pos    = self._cam_pos.copy()
            self._dbg_cam_target = self._dbg_pos.copy()
            # Initialise debug angles to match current visual state
            self._dbg_roll  = self._vis_roll
            self._dbg_pitch = self._vis_pitch
            self._dbg_yaw   = self._vis_yaw

    def reset_debug_angles(self) -> None:
        self._dbg_roll = self._dbg_pitch = self._dbg_yaw = 0.0

    def _update_debug_keys(self) -> None:
        """Poll held keys to increment debug attitude angles."""
        keys = pygame.key.get_pressed()
        s = self._DBG_STEP
        if keys[pygame.K_e]: self._dbg_roll  += s
        if keys[pygame.K_q]: self._dbg_roll  -= s
        if keys[pygame.K_w]: self._dbg_pitch += s
        if keys[pygame.K_s]: self._dbg_pitch -= s
        if keys[pygame.K_d]: self._dbg_yaw   += s
        if keys[pygame.K_a]: self._dbg_yaw   -= s
        self._dbg_roll  = _clamp(self._dbg_roll,  -math.pi,     math.pi)
        self._dbg_pitch = _clamp(self._dbg_pitch, -math.pi/2,   math.pi/2)
        self._dbg_yaw   = math.fmod(self._dbg_yaw, 2 * math.pi)

    def _draw_cross_mode_diag(self, surf: pygame.Surface, d: dict,
                              mode: FlightMode, s) -> None:
        """Cross-mode behavior diagnostics — bottom-left of 3-D view."""
        W_D, H_D = 256, 182
        X_D      = 6
        Y_D      = VIEW_H - H_D - 6

        panel = pygame.Surface((W_D, H_D), pygame.SRCALPHA)
        panel.fill((6, 8, 20, 210))
        pygame.draw.rect(panel, (40, 60, 100, 220), (0, 0, W_D, H_D), 1)
        surf.blit(panel, (X_D, Y_D))

        fnt = self._font_sm

        def drow(label, value, ry, vcol=WHITE):
            ls = fnt.render(label, True, DIM)
            vs = fnt.render(value, True, vcol)
            surf.blit(ls, (X_D + 4,               Y_D + ry))
            surf.blit(vs, (X_D + W_D - vs.get_width() - 4, Y_D + ry))

        ry = 4
        # Header
        hdr = fnt.render("CROSS-MODE DIAGNOSTICS", True, CYAN)
        surf.blit(hdr, (X_D + (W_D - hdr.get_width()) // 2, Y_D + ry))
        ry += 13
        pygame.draw.line(surf, (40, 60, 100),
                         (X_D + 2, Y_D + ry), (X_D + W_D - 2, Y_D + ry), 1)
        ry += 4

        # Mode one-hot — 8 colored squares, active bit highlighted
        mode_int = int(d.get("mode", 0))
        sq = 11; gap = 2
        sx = X_D + 4
        for i in range(8):
            mc  = MODE_COLORS.get(FlightMode(i), DIM)
            bg  = mc if i == mode_int else (20, 25, 40)
            brd = mc if i == mode_int else (50, 60, 80)
            pygame.draw.rect(surf, bg,  (sx + i * (sq + gap), Y_D + ry, sq, sq))
            pygame.draw.rect(surf, brd, (sx + i * (sq + gap), Y_D + ry, sq, sq), 1)
        mn = fnt.render(MODE_NAMES[mode_int], True, MODE_COLORS.get(mode, WHITE))
        surf.blit(mn, (sx + 8 * (sq + gap) + 3, Y_D + ry - 1))
        ry += sq + 5

        # Target vector direction (horizontal bearing + elevation to target)
        tpos = np.array(d.get("target_position", [0., 0., 50.]), dtype=float)
        tdx, tdy, tdz = tpos - s.pos
        tgt_bearing = math.degrees(math.atan2(tdx, -tdy)) % 360
        tgt_dist    = float(np.linalg.norm([tdx, tdy, tdz]))
        tgt_elev    = math.degrees(math.atan2(tdz, max(math.sqrt(tdx**2+tdy**2), 0.1)))
        drow("TGT BRG", f"{tgt_bearing:5.1f}°", ry, DIM)
        ry += 13
        drow("TGT ELEV", f"{tgt_elev:+5.1f}°", ry, DIM)
        ry += 13

        # Heading error (yaw vs target heading)
        tgt_hdg    = float(d.get("target_heading", s.yaw))
        hdg_err    = math.degrees(math.atan2(math.sin(s.yaw - tgt_hdg),
                                             math.cos(s.yaw - tgt_hdg)))
        hdg_col = GREEN if abs(hdg_err) < 10 else (AMBER if abs(hdg_err) < 30 else RED)
        drow("HDG ERR", f"{hdg_err:+6.1f}°", ry, hdg_col)
        ry += 13

        # Altitude error
        alt_err = float(d.get("altitude_error", s.pos[2] - tpos[2]))
        alt_col = GREEN if abs(alt_err) < 10 else (AMBER if abs(alt_err) < 30 else RED)
        drow("ALT ERR", f"{alt_err:+6.1f} m", ry, alt_col)
        ry += 13

        pygame.draw.line(surf, (30, 40, 65),
                         (X_D + 2, Y_D + ry), (X_D + W_D - 2, Y_D + ry), 1)
        ry += 4

        # Turn rate
        turn_dps = math.degrees(abs(s.yaw_rate))
        turn_col = GREEN if turn_dps < 5 else (AMBER if turn_dps < 20 else RED)
        drow("TURN RATE", f"{turn_dps:5.1f}°/s", ry, turn_col)
        ry += 13

        # Trajectory curvature = |yaw_rate| / airspeed  (rad/m)
        curvature = abs(s.yaw_rate) / max(s.airspeed, 1.0)
        cur_col = GREEN if curvature < 0.05 else (AMBER if curvature < 0.15 else RED)
        drow("CURVATURE", f"{curvature:.4f}/m", ry, cur_col)
        ry += 13

        # Straight-line score  (1 = perfectly straight, 0 = tight turn)
        straight = math.exp(-abs(s.yaw_rate) * 3.0)
        str_col = GREEN if straight > 0.8 else (AMBER if straight > 0.5 else RED)
        drow("STRAIGHT", f"{straight:.3f}", ry, str_col)
        ry += 13

        # Orbit score (only meaningful in LOITER)
        if mode == FlightMode.LOITER:
            orb_err = float(d.get("loiter_radial_error", 0.0))
            orb_des = max(float(d.get("loiter_radius_desired", 1.0)), 1.0)
            orb_frac = abs(orb_err) / orb_des
            orb_col = GREEN if orb_frac < 0.1 else (AMBER if orb_frac < 0.3 else RED)
            drow("ORBIT SCR", f"{max(0.0, 1.0 - orb_frac):.3f}", ry, orb_col)
        else:
            drow("ORBIT SCR", "N/A", ry, DIM)
        ry += 13

        # Vertical oscillation score (std of last 60 vz samples)
        if len(self._diag_vz_buf) >= 5:
            vz_osc = float(np.std(list(self._diag_vz_buf)))
        else:
            vz_osc = abs(float(d.get("vertical_speed", 0.0)))
        vz_col = GREEN if vz_osc < 0.5 else (AMBER if vz_osc < 2.0 else RED)
        drow("V-OSC", f"{vz_osc:.3f}", ry, vz_col)

    def _draw_debug_overlay(self, surf: pygame.Surface) -> None:
        bx, bw, bh = 10, 310, 130
        by = VIEW_H - bh - 50
        overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        surf.blit(overlay, (bx, by))
        pygame.draw.rect(surf, (255, 100, 0), (bx, by, bw, bh), 2)

        def txt(text, ry, col):
            s = self._font_md.render(text, True, col)
            surf.blit(s, (bx + 8, by + ry))

        txt("■ RENDERER DEBUG MODE ■",  6,  (255,  80,  80))
        txt(f"Roll : {math.degrees(self._dbg_roll):+7.1f}°   [Q/E]", 26, (180, 180, 255))
        txt(f"Pitch: {math.degrees(self._dbg_pitch):+7.1f}°   [W/S]", 44, (180, 255, 180))
        txt(f"Yaw  : {math.degrees(self._dbg_yaw):+7.1f}°   [A/D]", 62, (255, 180, 180))
        pygame.draw.line(surf, (60, 60, 80), (bx+4, by+82), (bx+bw-4, by+82), 1)
        txt("TAB=exit  R=zero angles", 87, (80, 100, 120))
        txt("RED=fwd  GRN=right  BLU=up", 104, (80, 100, 120))

    # ═══════════════════════════════════════════════════════ entry point ═════

    def render_state(self, d: dict) -> None:
        """Full dashboard render from shared-state snapshot."""
        s      = _StateProxy(d)
        mode   = FlightMode(int(d["mode"]))
        target = _TargetProxy(d)
        cam    = int(d.get("camera_mode", 0))
        hist   = d.get("_hist", {})

        self._diag_vz_buf.append(float(d.get("vertical_speed", s.vel[2])))

        scr = self._screen
        scr.fill(BG)

        # ── 3-D view ─────────────────────────────────────────────────────────
        self._render_3d_view(d, s, mode, target, cam, hist)

        # ── side panels ──────────────────────────────────────────────────────
        self._draw_left_panel(scr, d, mode)
        self._draw_right_panel(scr, d, s, mode, target)

        # ── bottom graphs ────────────────────────────────────────────────────
        self._draw_bottom_graphs(scr, hist)

        # ── panel borders ────────────────────────────────────────────────────
        pygame.draw.line(scr, PANEL_BDR, (LEFT_W, 0),       (LEFT_W, VIEW_H),  1)
        pygame.draw.line(scr, PANEL_BDR, (LEFT_W+VIEW_W, 0),(LEFT_W+VIEW_W, VIEW_H), 1)
        pygame.draw.line(scr, PANEL_BDR, (0, VIEW_H),       (TOTAL_W, VIEW_H), 1)

        # ── pipeline phase banner (drawn last so it's always on top) ─────────
        pl = str(d.get("pipeline_phase_label", ""))
        if pl:
            _PCOLS = {"1": (255, 160, 0), "2": (0, 200, 255), "3": (0, 220, 100)}
            col = (200, 180, 255)
            for k, c in _PCOLS.items():
                if f"[{k}/3]" in pl:
                    col = c
                    break
            bh = 44
            pygame.draw.rect(scr, (8, 10, 25), (0, 0, TOTAL_W, bh))
            pygame.draw.line(scr, col, (0, bh - 1), (TOTAL_W, bh - 1), 2)
            bs = self._font_lg.render(pl, True, col)
            scr.blit(bs, (TOTAL_W // 2 - bs.get_width() // 2,
                          (bh - bs.get_height()) // 2 + 1))

        pygame.display.flip()

    # ═════════════════════════════════════════════════════ 3-D view ══════════

    def _render_3d_view(self, d, s, mode, target, cam, hist):
        vs = self._view_surf
        vs.fill(SKY_TOP)
        pygame.draw.rect(vs, SKY_BOT,  (0, 0,        VIEW_W, VIEW_H // 2))
        pygame.draw.rect(vs, GROUND,   (0, VIEW_H//2, VIEW_W, VIEW_H // 2))

        # Track last position (used when entering debug mode)
        self._last_pos = s.pos.copy()

        if self._debug_mode:
            # ── Debug mode: manual attitude, frozen position + camera ────────
            self._update_debug_keys()
            s.pos   = self._dbg_pos.copy()
            s.roll  = self._dbg_roll
            s.pitch = self._dbg_pitch
            s.yaw   = self._dbg_yaw
            self._vis_roll  = self._dbg_roll
            self._vis_pitch = self._dbg_pitch
            self._vis_yaw   = self._dbg_yaw
            # Restore frozen camera (don't follow aircraft)
            self._cam_pos[:]    = self._dbg_cam_pos
            self._cam_target[:] = self._dbg_cam_target
            self._update_camera_basis()
        else:
            # ── Normal mode: smooth visual attitude toward physics state ─────
            k = self.VIS_SMOOTH
            self._vis_roll  += (s.roll  - self._vis_roll)  * k
            self._vis_pitch += (s.pitch - self._vis_pitch) * k
            dyaw = math.atan2(math.sin(s.yaw - self._vis_yaw),
                              math.cos(s.yaw - self._vis_yaw))
            self._vis_yaw += dyaw * k * 2.0   # yaw follows faster
            self._set_camera(s, cam)

        # Horizon line
        hor = self._project(np.array([0., 1000., 0.]))
        if hor:
            pygame.draw.line(vs, (30, 50, 90), (0, hor[1]), (VIEW_W, hor[1]), 1)

        self._draw_ground_grid(vs)
        self._draw_runway(vs)

        # Trajectory trail
        traj = hist.get("trajectory", [])
        self._draw_trajectory(vs, traj)

        self._draw_target_overlay(vs, s.pos, mode, target)
        self._draw_velocity_vector(vs, s)
        self._draw_aircraft(vs, s)

        # Attitude indicator (top-right of view)
        self._draw_attitude(vs, s, VIEW_W - 115, 115)

        # Camera mode label (top-left of view)
        cam_labels = ["F1 CHASE", "F2 SIDE", "F3 TOP", "F4 COCKPIT"]
        lbl = self._font_sm.render(cam_labels[cam], True, (80, 120, 180))
        vs.blit(lbl, (6, 6))

        # Mission segment label
        seg    = int(d.get("mission_seg",   0))
        total  = int(d.get("mission_total", 0))
        done   = bool(d.get("training_done", False))
        if total > 0:
            seg_s  = self._font_sm.render(f"SEG {seg+1}/{total}", True, CYAN)
            vs.blit(seg_s, (6, 22))
        if done:
            done_s = self._font_md.render("TRAINING COMPLETE", True, GREEN)
            vs.blit(done_s, (VIEW_W//2 - done_s.get_width()//2, 8))

        # Pipeline phase banner (recording / BC / PPO label)
        pl = str(d.get("pipeline_phase_label", ""))
        if pl:
            pl_s = self._font_md.render(pl, True, AMBER)
            vs.blit(pl_s, (VIEW_W // 2 - pl_s.get_width() // 2, 52))

        # Waypoint distance + arrival overlay
        wp_dist        = float(d.get("wp_curr_dist", -1.0))
        wp_state       = str(d.get("wp_state", "approaching"))
        wp_capture_ev  = bool(d.get("wp_capture_event", False))
        wp_cap_count   = int(d.get("wp_captured_count", 0))
        if mode == FlightMode.WAYPOINT and wp_dist >= 0:
            dist_col = AMBER if wp_state == 'transitioning' else CYAN
            dist_s   = self._font_sm.render(f"WP {wp_dist:.0f}m", True, dist_col)
            vs.blit(dist_s, (6, 38))
        # Flash "WAYPOINT REACHED" for ~3 s on capture; auto-dismiss afterward.
        if wp_capture_ev or wp_cap_count > self._wp_capture_count:
            self._wp_capture_flash_timer = 90
            self._wp_capture_count       = wp_cap_count
        if self._wp_capture_flash_timer > 0:
            self._wp_capture_flash_timer -= 1
            arr_s = self._font_md.render(
                f"WAYPOINT REACHED  [#{self._wp_capture_count}]", True, GREEN)
            vs.blit(arr_s, (VIEW_W // 2 - arr_s.get_width() // 2, 30))

        # Event overlay (bottom of view)
        self._draw_events(vs, d.get("events", []))

        # Cross-mode diagnostics overlay (bottom-left of 3D view)
        self._draw_cross_mode_diag(vs, d, mode, s)

        # Debug mode overlay
        if self._debug_mode:
            self._draw_debug_overlay(vs)

        # Blit view surface onto main screen
        self._screen.blit(vs, (VIEW_X, VIEW_Y))

    # ═════════════════════════════════════════════════════ camera ════════════

    def _set_camera(self, state, cam_mode: int):
        pos = state.pos
        yaw = state.yaw

        if cam_mode == 0:    # Chase
            bx = -math.sin(yaw)
            by =  math.cos(yaw)
            off = self._chase_offset
            target_cam = pos + np.array([bx * off[1], by * off[1], off[2]])
            self._cam_pos    += (target_cam - self._cam_pos)    * self.SMOOTH
            look_fwd  = np.array([math.sin(yaw), -math.cos(yaw), 0.])
            target_at = pos + look_fwd * 20.
            self._cam_target += (target_at - self._cam_target) * self.SMOOTH

        elif cam_mode == 1:  # Side view
            target_cam = pos + np.array([150., 0., 40.])
            self._cam_pos    += (target_cam - self._cam_pos)    * self.SMOOTH
            self._cam_target += (pos        - self._cam_target) * self.SMOOTH

        elif cam_mode == 2:  # Top-down
            target_cam = pos + np.array([0., 0., 350.])
            self._cam_pos    += (target_cam - self._cam_pos)    * self.SMOOTH
            look_down = pos + np.array([0., 1., 0.])
            self._cam_target += (look_down  - self._cam_target) * self.SMOOTH

        elif cam_mode == 3:  # Cockpit
            fwd = np.array([math.sin(yaw), -math.cos(yaw),
                            math.sin(state.pitch)])
            fwd /= (np.linalg.norm(fwd) + 1e-9)
            self._cam_pos    = pos + np.array([0., 0., 1.5])
            self._cam_target = self._cam_pos + fwd * 50.

        self._update_camera_basis()

    def _update_camera_basis(self):
        fwd = self._cam_target - self._cam_pos
        ln  = float(np.linalg.norm(fwd))
        if ln < 0.1:
            return
        fwd /= ln
        up    = np.array([0., 0., 1.])
        right = np.cross(fwd, up)
        rlen  = float(np.linalg.norm(right))
        if rlen < 0.01:
            up    = np.array([0., 1., 0.])
            right = np.cross(fwd, up)
            rlen  = float(np.linalg.norm(right))
        right /= rlen
        self._cf[:] = fwd
        self._cr[:] = right
        self._cu[:] = np.cross(right, fwd)

    def _project(self, point: np.ndarray):
        """Project a world point to view-surface pixel coords. None = behind camera."""
        rel   = point - self._cam_pos
        depth = float(np.dot(rel, self._cf))
        if depth <= 0.5:
            return None
        sx = VIEW_W // 2 + int(float(np.dot(rel, self._cr)) / depth * FOV)
        sy = VIEW_H // 2 - int(float(np.dot(rel, self._cu)) / depth * FOV)
        return sx, sy

    # ═════════════════════════════════════════════════════ world geometry ════

    def _draw_ground_grid(self, surf):
        for x in range(-600, 601, 40):
            p1 = self._project(np.array([x, -300., 0.]))
            p2 = self._project(np.array([x,  900., 0.]))
            if p1 and p2:
                pygame.draw.line(surf, GRID, p1, p2, 1)
        for y in range(-300, 901, 40):
            p1 = self._project(np.array([-600., y, 0.]))
            p2 = self._project(np.array([ 600., y, 0.]))
            if p1 and p2:
                pygame.draw.line(surf, GRID, p1, p2, 1)

    def _draw_runway(self, surf):
        RWY_X = 13.0; RWY_Y0 = -25.0; RWY_Y1 = 95.0

        def poly(corners, color, width=0):
            pts = [self._project(np.array(c, dtype=float)) for c in corners]
            if all(pts):
                pygame.draw.polygon(surf, color, pts, width)

        poly([[-RWY_X, RWY_Y0, 0], [ RWY_X, RWY_Y0, 0],
              [ RWY_X, RWY_Y1, 0], [-RWY_X, RWY_Y1, 0]], (60, 60, 65))
        for x0, x1 in [(-RWY_X,-9),(-7,-4),(-2,1),(3,6),(8,RWY_X)]:
            poly([[x0,RWY_Y1-8,0.01],[x1,RWY_Y1-8,0.01],
                  [x1,RWY_Y1,0.01],  [x0,RWY_Y1,0.01]], (220,220,220))
        for y0 in range(int(RWY_Y0)+5, int(RWY_Y1)-10, 15):
            poly([[-0.6,y0,0.01],[0.6,y0,0.01],
                  [0.6,y0+8,0.01],[-0.6,y0+8,0.01]], (210,210,170))
        poly([[-RWY_X,RWY_Y0,0],[ RWY_X,RWY_Y0,0],
              [ RWY_X,RWY_Y1,0],[-RWY_X,RWY_Y1,0]], (200,200,200), 2)
        for i, x in enumerate([-8,-5,-2,1]):
            glidecol = (255, 80, 80) if i < 2 else (255, 255, 255)
            p = self._project(np.array([x, RWY_Y1-4, 0.05]))
            if p:
                pygame.draw.circle(surf, glidecol, p, 4)

    def _draw_trajectory(self, surf, traj):
        if len(traj) < 2:
            return
        pts = [self._project(np.array(p, dtype=float)) for p in traj]
        n = len(pts)
        for i in range(1, n):
            if pts[i-1] and pts[i]:
                age = (i / n)
                alpha = int(age * 200)
                col = (0, alpha, int(alpha * 0.5))
                pygame.draw.line(surf, col, pts[i-1], pts[i], 1)

    def _draw_velocity_vector(self, surf, s):
        """Draw a short velocity vector from the aircraft nose."""
        spd = float(np.linalg.norm(s.vel))
        if spd < 0.5:
            return
        tip = s.pos + s.vel * 3.0
        p0  = self._project(s.pos)
        p1  = self._project(tip)
        if p0 and p1:
            pygame.draw.line(surf, (80, 255, 200), p0, p1, 2)
            pygame.draw.circle(surf, (80, 255, 200), p1, 3)

    # ═════════════════════════════════════════════════ target overlays ═══════

    def _draw_target_overlay(self, surf, aircraft_pos, mode, target):
        col = MODE_COLORS.get(mode, WHITE)
        if mode in (FlightMode.WAYPOINT, FlightMode.ALTITUDE_HOLD):
            self._draw_waypoint_marker(surf, target.position, col)
            self._draw_altitude_guide(surf, aircraft_pos, target.altitude, col)
        elif mode == FlightMode.HEADING_HOLD:
            self._draw_heading_arrow(surf, aircraft_pos, target.heading, col)
            self._draw_altitude_guide(surf, aircraft_pos, target.altitude, col)
        elif mode == FlightMode.LOITER:
            self._draw_loiter_circle(surf, target.position, target.radius, col)
            self._draw_altitude_guide(surf, aircraft_pos, target.altitude, col)
        elif mode in (FlightMode.APPROACH, FlightMode.LANDING):
            self._draw_glide_path(surf, target)
            self._draw_runway_centerline(surf, target)
        elif mode == FlightMode.RECOVERY:
            self._draw_wings_level_ref(surf, aircraft_pos, col)

    def _draw_waypoint_marker(self, surf, pos, col):
        arm = 20
        pts = [np.array([pos[0],     pos[1]-arm, pos[2]]),
               np.array([pos[0]+arm, pos[1],     pos[2]]),
               np.array([pos[0],     pos[1]+arm, pos[2]]),
               np.array([pos[0]-arm, pos[1],     pos[2]])]
        projected = [self._project(p) for p in pts]
        if all(projected):
            pygame.draw.polygon(surf, col, projected, 2)
        top = self._project(pos + np.array([0, 0, 12]))
        bot = self._project(pos)
        if top and bot:
            pygame.draw.line(surf, col, top, bot, 2)

    def _draw_altitude_guide(self, surf, aircraft_pos, target_alt, col):
        z0, z1, steps = aircraft_pos[2], target_alt, 8
        for i in range(steps):
            za = z0 + (z1-z0) * i       / steps
            zb = z0 + (z1-z0) * (i+0.4) / steps
            pa = self._project(np.array([aircraft_pos[0], aircraft_pos[1], za]))
            pb = self._project(np.array([aircraft_pos[0], aircraft_pos[1], zb]))
            if pa and pb:
                pygame.draw.line(surf, col, pa, pb, 1)

    def _draw_heading_arrow(self, surf, pos, target_yaw, col):
        length = 80
        ex = pos[0] + math.sin(target_yaw) * length
        ey = pos[1] - math.cos(target_yaw) * length
        p0 = self._project(np.array([pos[0], pos[1], 1.]))
        p1 = self._project(np.array([ex, ey, 1.]))
        if p0 and p1:
            pygame.draw.line(surf, col, p0, p1, 3)
            dx, dy = p1[0]-p0[0], p1[1]-p0[1]
            ln = max(math.sqrt(dx*dx+dy*dy), 1)
            nx, ny = dx/ln, dy/ln
            h1 = (int(p1[0]-nx*14-ny*8), int(p1[1]-ny*14+nx*8))
            h2 = (int(p1[0]-nx*14+ny*8), int(p1[1]-ny*14-nx*8))
            pygame.draw.polygon(surf, col, [p1, h1, h2])

    def _draw_loiter_circle(self, surf, center, radius, col):
        cx, cy, cz = center
        steps = 36
        pts = []
        for i in range(steps + 1):
            angle = 2 * math.pi * i / steps
            p = self._project(np.array([cx + math.cos(angle)*radius,
                                        cy + math.sin(angle)*radius, cz]))
            if p:
                pts.append(p)
        for i in range(len(pts)-1):
            pygame.draw.line(surf, col, pts[i], pts[i+1], 2)

    def _draw_glide_path(self, surf, target):
        rwy = target.position; hdg = target.heading
        gs  = math.radians(3.0); col = (255, 160, 60); steps = 12; max_d = 500.
        for i in range(steps):
            d0 = max_d * i / steps; d1 = max_d * (i+0.7) / steps
            p0 = self._project(np.array([rwy[0]-math.sin(hdg)*d0,
                                          rwy[1]+math.cos(hdg)*d0, d0*math.tan(gs)]))
            p1 = self._project(np.array([rwy[0]-math.sin(hdg)*d1,
                                          rwy[1]+math.cos(hdg)*d1, d1*math.tan(gs)]))
            if p0 and p1:
                pygame.draw.line(surf, col, p0, p1, 1)

    def _draw_runway_centerline(self, surf, target):
        rwy = target.position; hdg = target.heading; col = (255, 255, 100)
        for d in range(0, 500, 20):
            p = self._project(np.array([rwy[0]-math.sin(hdg)*d,
                                        rwy[1]+math.cos(hdg)*d, 0.05]))
            if p:
                pygame.draw.circle(surf, col, p, 2)

    def _draw_wings_level_ref(self, surf, pos, col):
        arm = 50
        lft = self._project(np.array([pos[0]-arm, pos[1], pos[2]]))
        rgt = self._project(np.array([pos[0]+arm, pos[1], pos[2]]))
        if lft and rgt:
            pygame.draw.line(surf, col, lft, rgt, 2)
            pygame.draw.line(surf, col, lft, (lft[0], lft[1]-10), 2)
            pygame.draw.line(surf, col, rgt, (rgt[0], rgt[1]-10), 2)

    # ═══════════════════════════════════════════════════ aircraft model ═══════

    # Set to True to draw body-frame debug axes (X=red fwd, Y=green right, Z=blue up)
    DEBUG_AXES = True

    def _draw_aircraft(self, surf, state):
        pos  = state.pos
        # Use smoothed attitude for visual rendering — eliminates flutter
        yaw  = self._vis_yaw
        roll = self._vis_roll
        pit  = self._vis_pitch

        shadow = self._project(np.array([pos[0], pos[1], 0.]))
        if shadow:
            alt = max(float(pos[2]), 0.1)
            r   = max(2, int(200 / (alt + 10)))
            pygame.draw.circle(surf, (45, 60, 80), shadow, r)

        SZ = 10.0
        # Body frame: X = forward (nose), Y = right wing, Z = up
        NOSE_X  =  SZ * 2.8   # nose tip (forward)
        TAIL_X  = -SZ * 2.5   # tail (aft)
        # Main wing: leading edge slightly forward, trailing edge aft
        WLE_X   =  SZ * 0.3   # leading edge X (forward of CG)
        WTE_X   = -SZ * 0.6   # trailing edge X (aft of CG)
        WSPAN   =  SZ * 4.5   # half-wingspan along Y
        # Tailplane — smaller horizontal stabilizer at tail
        HLE_X   = -SZ * 1.8
        HTE_X   = -SZ * 2.5
        HSPAN   =  SZ * 1.7
        # Vertical fin — upright rectangle in X-Z plane at tail
        FLE_X   = -SZ * 1.6
        FTE_X   = -SZ * 2.5
        FHEIGHT =  SZ * 2.0

        local = {
            'nose': np.array([NOSE_X,   0.,     0.      ]),
            'tail': np.array([TAIL_X,   0.,     0.      ]),
            # Main wing: right-leading, left-leading, left-trailing, right-trailing
            'wrl':  np.array([WLE_X,  +WSPAN,   0.      ]),
            'wll':  np.array([WLE_X,  -WSPAN,   0.      ]),
            'wlt':  np.array([WTE_X,  -WSPAN,   0.      ]),
            'wrt':  np.array([WTE_X,  +WSPAN,   0.      ]),
            # Tailplane: same pattern
            'hrl':  np.array([HLE_X,  +HSPAN,   0.      ]),
            'hll':  np.array([HLE_X,  -HSPAN,   0.      ]),
            'hlt':  np.array([HTE_X,  -HSPAN,   0.      ]),
            'hrt':  np.array([HTE_X,  +HSPAN,   0.      ]),
            # Vertical fin: base-leading, top-leading, top-trailing, base-trailing
            'fbl':  np.array([FLE_X,   0.,       0.      ]),
            'ftl':  np.array([FLE_X,   0.,       FHEIGHT ]),
            'ftt':  np.array([FTE_X,   0.,       FHEIGHT ]),
            'fbt':  np.array([FTE_X,   0.,       0.      ]),
        }

        cy, sy = math.cos(yaw),  math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pit),  math.sin(pit)

        # Body → world rotation matrix (ZYX aerospace convention).
        #
        # World frame: X=right, Y=south (behind runway), Z=up (right-handed).
        # yaw=0 → aircraft faces -Y (north/toward runway).
        #
        # Zero-attitude body axes in world:
        #   body-X (fwd)   → world [0, -1, 0]   (north)
        #   body-Y (right) → world [+1,  0, 0]  (east)
        #   body-Z (up)    → world [ 0,  0, 1]  (up)
        #
        # Columns of rot = where each body axis goes in world:
        #   col 0 (fwd)  : [sy·cp,  -cy·cp,  sp]              (yaw+pitch only)
        #   col 1 (right): [cy·cr+sy·sp·sr,  sy·cr-cy·sp·sr,  -cp·sr]
        #   col 2 (up)   : [cy·sr-sy·sp·cr,  sy·sr+cy·sp·cr,   cp·cr]
        #
        # Verification (roll=pitch=0):
        #   yaw=0  → fwd=[0,-1,0] ✓  right=[1,0,0] ✓  up=[0,0,1] ✓
        #   yaw=π/2→ fwd=[1,0,0]  ✓  right=[0,1,0] ✓  up=[0,0,1] ✓
        # Verification (yaw=pitch=0, roll=90°):
        #   right → [0,0,-1] (right-wing tips down) ✓
        #   up    → [1,0,0]  (top tilts east)        ✓
        rot = np.array([
            [ sy*cp,   cy*cr + sy*sp*sr,   cy*sr - sy*sp*cr],
            [-cy*cp,   sy*cr - cy*sp*sr,   sy*sr + cy*sp*cr],
            [ sp,     -cp*sr,              cp*cr            ],
        ])

        world = {k: pos + rot @ v for k, v in local.items()}
        prj   = {k: self._project(world[k]) for k in world}

        def line(a, b, col, w=2):
            if prj[a] and prj[b]:
                pygame.draw.line(surf, col, prj[a], prj[b], w)

        def quad(keys, fill_col, border_col, bw=1):
            pts = [prj[k] for k in keys]
            if all(pts):
                if not self._wireframe:
                    pygame.draw.polygon(surf, fill_col, pts)
                pygame.draw.polygon(surf, border_col, pts, bw)

        # Fuselage — thick glowing center line
        line('nose', 'tail', (0,  80, 120), 14)
        line('nose', 'tail', (0, 210, 255),  8)

        # Main wings — filled rectangle perpendicular to fuselage
        quad(['wll', 'wrl', 'wrt', 'wlt'], (40, 65, 95),  (160, 200, 255))

        # Tailplane — smaller filled rectangle
        quad(['hll', 'hrl', 'hrt', 'hlt'], (35, 55, 80),  (140, 175, 220))

        # Vertical fin — upright filled rectangle
        quad(['fbl', 'ftl', 'ftt', 'fbt'], (80, 60,   0), (240, 170,  50))

        # Debug body-frame axes: X=red(fwd), Y=green(right), Z=blue(up)
        if self.DEBUG_AXES:
            ax_len = SZ * 3.5
            p0 = self._project(pos)
            if p0:
                for axis_vec, col in [
                    (rot[:, 0] * ax_len, (255,  60,  60)),   # X forward — red
                    (rot[:, 1] * ax_len, ( 60, 220,  60)),   # Y right   — green
                    (rot[:, 2] * ax_len, ( 60, 100, 255)),   # Z up      — blue
                ]:
                    p1 = self._project(pos + axis_vec)
                    if p1:
                        pygame.draw.line(surf, col, p0, p1, 2)
                        pygame.draw.circle(surf, col, p1, 3)

    # ═══════════════════════════════════════════════════ attitude indicator ═══

    def _draw_attitude(self, surf, state, cx, cy, r=75):
        pygame.draw.circle(surf, (20, 30, 50), (cx, cy), r)

        # Use smoothed visual attitude so AI doesn't flutter
        roll  = self._vis_roll
        pitch = self._vis_pitch
        # px,py: horizon extends in roll direction; cos/sin correct for screen coords
        # (roll=0 → horizontal line: px=r, py=0)
        px = int(math.cos(roll) * r * 0.95)
        py = int(math.sin(roll) * r * 0.95)
        pitch_offset = int(pitch / (math.pi / 2) * r * 0.8)
        # ox,oy: perpendicular to horizon; positive pitch moves horizon down (screen +y)
        ox = int(-math.sin(roll) * pitch_offset)
        oy = int( math.cos(roll) * pitch_offset)

        # Sky fill (above horizon)
        pts_sky = []
        for a in range(181):
            ax = cx + int(math.cos(math.radians(a)) * r)
            ay = cy - int(math.sin(math.radians(a)) * r)
            pts_sky.append((ax, ay))
        pts_sky.extend([(cx - px + ox, cy - py + oy),
                        (cx + px + ox, cy + py + oy)])
        if len(pts_sky) >= 3:
            pygame.draw.polygon(surf, (20, 50, 100), pts_sky)

        # Horizon line
        pygame.draw.line(surf, HUD_GREEN,
                         (cx - px + ox, cy - py + oy),
                         (cx + px + ox, cy + py + oy), 2)
        pygame.draw.circle(surf, (0, 0, 0), (cx, cy), r, 2)  # border

        # Fixed aircraft symbol
        pygame.draw.line(surf, WHITE, (cx-22, cy), (cx+22, cy), 2)
        pygame.draw.line(surf, WHITE, (cx, cy), (cx, cy-9), 2)
        pygame.draw.circle(surf, AMBER, (cx, cy), 3)

        # Roll tick marks
        for a in range(-90, 91, 30):
            ax = cx + int(math.sin(math.radians(a)) * (r-6))
            ay = cy - int(math.cos(math.radians(a)) * (r-6))
            pygame.draw.circle(surf, DIM, (ax, ay), 2)

        rax = cx + int(math.sin(roll) * (r-10))
        ray = cy - int(math.cos(roll) * (r-10))
        pygame.draw.circle(surf, AMBER, (rax, ray), 4)

    # ═══════════════════════════════════════════════════════ event overlay ════

    def _draw_events(self, surf, events):
        if not events:
            return
        y = VIEW_H - 30
        for ev in reversed(events[-5:]):
            text = ev["text"]
            col  = ev.get("color", YELLOW)
            s = self._font_md.render(f"▶ {text}", True, col)
            surf.blit(s, (VIEW_W // 2 - s.get_width() // 2, y))
            y -= 22

    # ═══════════════════════════════════════════════════ LEFT PANEL: PPO ═════

    def _draw_left_panel(self, scr, d, mode):
        x, y, w = 0, 0, LEFT_W
        pygame.draw.rect(scr, PANEL_BG, (x, y, w, VIEW_H))

        def header(text, ry, col=CYAN):
            s = self._font_hdr.render(text, True, col)
            scr.blit(s, (x + 8, y + ry))
            pygame.draw.line(scr, PANEL_BDR, (x+4, y+ry+14), (x+w-4, y+ry+14), 1)

        def row(label, value, ry, vcol=WHITE, lbl_col=DIM):
            ls = self._font_sm.render(label, True, lbl_col)
            vs = self._font_sm.render(value, True, vcol)
            scr.blit(ls, (x + 8,      y + ry))
            scr.blit(vs, (x + w - vs.get_width() - 8, y + ry))

        def bar_row(label, frac, ry, col):
            """Horizontal bar indicator (0..1)."""
            ls = self._font_sm.render(label, True, DIM)
            scr.blit(ls, (x + 8, y + ry))
            bx = x + 70; bw = w - 78; bh = 8; by = y + ry + 2
            pygame.draw.rect(scr, (30, 35, 50), (bx, by, bw, bh))
            fill = int(_clamp(frac, 0, 1) * bw)
            if fill > 0:
                pygame.draw.rect(scr, col, (bx, by, fill, bh))

        ry = 8

        # ── Training status ──────────────────────────────────────────────────
        header("TRAINING STATUS", ry)
        ry += 20

        ts   = int(d.get("timesteps", 0))
        ep   = int(d.get("episode_count", 0))
        itr  = int(d.get("training_iter", 0))
        done = bool(d.get("training_done", False))
        ts_col = (100, 255, 100) if done else (180, 180, 255)

        row("TIMESTEPS", f"{ts:,}", ry, ts_col); ry += 16
        row("EPISODES",  f"{ep:,}", ry);         ry += 16
        row("ITERATION", f"{itr}",  ry);         ry += 20

        # ── Episode reward ────────────────────────────────────────────────────
        header("EPISODE REWARD", ry)
        ry += 20

        ep_r   = float(d.get("reward",         0.0))
        mean_r = float(d.get("mean_ep_reward",  0.0))
        best_r = float(d.get("best_reward",     0.0))
        lr     = float(d.get("learning_rate",   3e-4))

        row("CURRENT",  f"{ep_r:+.2f}", ry,
            (100,255,100) if ep_r > 0 else RED); ry += 16
        row("MEAN",     f"{mean_r:+.2f}", ry,
            (180,180,255) if mean_r >= 0 else (255,140,140)); ry += 16
        row("BEST",     f"{best_r:+.2f}", ry, AMBER);          ry += 16
        row("LR",       f"{lr:.2e}",      ry, DIM);             ry += 20

        # ── PPO metrics ───────────────────────────────────────────────────────
        header("PPO METRICS", ry)
        ry += 20

        pol_l = float(d.get("policy_loss",  0.0))
        val_l = float(d.get("value_loss",   0.0))
        entr  = float(d.get("entropy",      0.0))
        kl    = float(d.get("approx_kl",    0.0))
        ev    = float(d.get("explained_var",0.0))
        fps   = float(d.get("fps",          0.0))

        def kl_col(v):
            if v < 0.01: return GREEN
            if v < 0.02: return AMBER
            return RED

        row("POL LOSS", f"{pol_l:.4f}",  ry, AMBER);         ry += 16
        row("VAL LOSS", f"{val_l:.3f}",  ry, (80,160,255));  ry += 16
        row("ENTROPY",  f"{entr:.4f}",   ry, (160,100,255)); ry += 16
        row("APPROX KL",f"{kl:.4f}",     ry, kl_col(kl));    ry += 16
        row("EXPL VAR", f"{ev:.3f}",     ry,
            GREEN if ev > 0.5 else (AMBER if ev > 0 else RED)); ry += 16
        row("FPS",      f"{fps:.0f}",    ry, DIM);            ry += 20

        # ── Explained variance bar ────────────────────────────────────────────
        ev_clamped = _clamp((ev + 1) / 2, 0, 1)   # map [-1,1] → [0,1]
        bar_row("EV", ev_clamped, ry,
                GREEN if ev > 0.5 else (AMBER if ev > 0 else RED))
        ry += 18

        # ── Episode statistics ────────────────────────────────────────────────
        header("EPISODE STATS", ry)
        ry += 20

        suc   = float(d.get("success_rate",    0.0))
        cra   = float(d.get("crash_rate",       0.0))
        land  = float(d.get("landing_rate",     0.0))
        stall = float(d.get("stall_rate",       0.0))
        bsuc  = float(d.get("best_success_rate",0.0))

        row("SUCCESS",      f"{suc:.0%}",  ry, GREEN);          ry += 14
        bar_row("", suc,    ry, GREEN);                           ry += 14
        row("BEST SUC",     f"{bsuc:.0%}", ry, (80,255,160));   ry += 14
        row("CRASH",        f"{cra:.0%}",  ry, RED);             ry += 14
        bar_row("", cra,    ry, RED);                             ry += 14
        row("STALL",        f"{stall:.0%}",ry, (255,140,0));    ry += 14
        bar_row("", stall,  ry, (255,140,0));                    ry += 14
        row("LANDING",      f"{land:.0%}", ry, (0,190,255));    ry += 14
        bar_row("", land,   ry, (0,190,255));                    ry += 16

        # ── Curriculum Phase + Active Mode ───────────────────────────────────
        header("CURRICULUM", ry)
        ry += 18

        phase_str = str(d.get("curriculum_phase", "mixed")).upper()
        row("Phase", phase_str, ry, CYAN)
        ry += 16

        active_mode_int  = int(d.get("active_mode", d.get("mode", 0)))
        active_mode_enum = FlightMode(active_mode_int)
        active_mode_name = MODE_NAMES[active_mode_int]
        active_mode_col  = MODE_COLORS.get(active_mode_enum, WHITE)
        row("Mode", active_mode_name, ry, active_mode_col)
        ry += 16

        auto_sw = d.get("autonomous_switching_enabled", True)
        auto_sw_str = "ON" if auto_sw else "OFF (LOCKED)"
        auto_sw_col = (150, 150, 150) if auto_sw else (255, 200, 0)
        row("AutoSwitch", auto_sw_str, ry, auto_sw_col)
        ry += 16

        # ── Mastery gate status ───────────────────────────────────────────────
        mastery_locked  = bool(d.get("mastery_locked", False))
        mastery_failing = d.get("mastery_failing", [])
        mastery_details = d.get("mastery_details", {})
        n_ep   = int(mastery_details.get("n_episodes", 0))
        req_ep = int(mastery_details.get("required",   0))

        if mastery_locked:
            # "NEXT PHASE LOCKED" banner
            banner_text = "NEXT PHASE LOCKED"
            bs = self._font_sm.render(banner_text, True, RED)
            bx_off = (w - bs.get_width()) // 2
            pygame.draw.rect(scr, (40, 12, 12), (x+2, y+ry, w-4, 14))
            pygame.draw.rect(scr, (100, 20, 20), (x+2, y+ry, w-4, 14), 1)
            scr.blit(bs, (x + bx_off, y + ry))
            ry += 16

            # Episode progress bar
            if req_ep > 0:
                frac = min(n_ep / req_ep, 1.0)
                row("Samples", f"{n_ep}/{req_ep}", ry, DIM)
                ry += 13
                bx2 = x + 6; bw2 = w - 12; bh2 = 6; by2 = y + ry
                pygame.draw.rect(scr, (25, 30, 48), (bx2, by2, bw2, bh2))
                fill = int(frac * bw2)
                if fill > 0:
                    col_p = AMBER if frac < 1.0 else GREEN
                    pygame.draw.rect(scr, col_p, (bx2, by2, fill, bh2))
                ry += 10

            # Failing criteria (up to 4 lines, truncated to fit)
            for crit in mastery_failing[:4]:
                # Strip values in parentheses for compact display
                short = crit.split(" (")[0] if " (" in crit else crit
                cs = self._font_sm.render(short, True, (255, 120, 60))
                scr.blit(cs, (x + 4, y + ry))
                ry += 13
        else:
            # "MASTERED" green badge
            ms_s = self._font_sm.render("MASTERED", True, GREEN)
            pygame.draw.rect(scr, (10, 35, 18), (x+2, y+ry, w-4, 14))
            pygame.draw.rect(scr, (20, 80, 35), (x+2, y+ry, w-4, 14), 1)
            scr.blit(ms_s, (x + (w - ms_s.get_width())//2, y + ry))
            ry += 16

        # Camera help
        ry = VIEW_H - 60
        help_lines = ["F1 Chase  F2 Side", "F3 Top   F4 Cockpit", "[ESC] Quit"]
        for hl in help_lines:
            hs = self._font_sm.render(hl, True, (50, 60, 80))
            scr.blit(hs, (x + (w - hs.get_width())//2, y + ry))
            ry += 14

    # ═══════════════════════════════════════════════ RIGHT PANEL: FLIGHT HUD ═

    def _draw_right_panel(self, scr, d, s, mode, target):
        x = LEFT_W + VIEW_W
        y = 0
        w = RIGHT_W
        pygame.draw.rect(scr, PANEL_BG, (x, y, w, VIEW_H))

        def header(text, ry, col=CYAN):
            hs = self._font_hdr.render(text, True, col)
            scr.blit(hs, (x + 8, y + ry))
            pygame.draw.line(scr, PANEL_BDR, (x+4, y+ry+14), (x+w-4, y+ry+14), 1)

        def row(label, value, ry, vcol=WHITE):
            ls = self._font_sm.render(label, True, DIM)
            vs = self._font_sm.render(value, True, vcol)
            scr.blit(ls, (x + 8,      y + ry))
            scr.blit(vs, (x + w - vs.get_width() - 8, y + ry))

        def surface_bar(label, val, ry, bipolar=True):
            ls = self._font_sm.render(label, True, DIM)
            scr.blit(ls, (x + 8, y + ry))
            bx = x + 60; bw = w - 70; bh = 8; by = y + ry + 2
            pygame.draw.rect(scr, (25, 30, 48), (bx, by, bw, bh))
            if bipolar:
                mid = bx + bw // 2
                pygame.draw.line(scr, (60,60,80), (mid, by), (mid, by+bh), 1)
                fill_w = int(_clamp(abs(val), 0, 1) * (bw // 2))
                if val >= 0:
                    pygame.draw.rect(scr, HUD_GREEN, (mid, by, fill_w, bh))
                else:
                    pygame.draw.rect(scr, (255,120,50), (mid-fill_w, by, fill_w, bh))
            else:
                fill_w = int(_clamp(val, 0, 1) * bw)
                pygame.draw.rect(scr, HUD_GREEN, (bx, by, fill_w, bh))
            vl = self._font_sm.render(f"{val:+.2f}", True, WHITE)
            scr.blit(vl, (x + w - vl.get_width() - 4, y + ry))

        ry = 6

        # ── Mode banner ───────────────────────────────────────────────────────
        mode_name = MODE_NAMES[int(d.get("mode", 0))]
        mode_col  = MODE_COLORS.get(mode, WHITE)
        pygame.draw.rect(scr, (15, 20, 40), (x+2, y+ry, w-4, 22))
        pygame.draw.rect(scr, mode_col,     (x+2, y+ry, w-4, 22), 2)
        ms = self._font_md.render(mode_name, True, mode_col)
        scr.blit(ms, (x + (w - ms.get_width())//2, y + ry + 3))
        ry += 30

        # ── Flight state ──────────────────────────────────────────────────────
        header("FLIGHT STATE", ry); ry += 18

        alt   = float(s.pos[2])
        spd   = float(s.airspeed)
        hdg   = math.degrees(s.yaw) % 360
        pitch = math.degrees(s.pitch)
        roll  = math.degrees(s.roll)
        thr_act = s.throttle_pos * 100
        thr_cmd = float(d.get("throttle_command", s.throttle_pos)) * 100
        thr_dlt = float(d.get("throttle_delta", 0.0))
        vz      = float(s.vel[2])

        # G-load approximation (1/cos(bank) for level turns)
        g_load = 1.0 / max(math.cos(abs(s.roll)), 0.1)

        # AOA approximation
        spd_h = math.sqrt(s.vel[0]**2 + s.vel[1]**2) + 1e-9
        aoa   = math.degrees(math.atan2(-s.vel[2], spd_h) + s.pitch)

        # Stall check (use shared state flag if available, else compute)
        stall = bool(d.get("stall_warning", s.airspeed < 6.5))

        # Altitude color
        alt_col = GREEN if alt > 20 else (AMBER if alt > 5 else RED)

        row("ALT",    f"{alt:7.1f} m",     ry, alt_col);      ry += 15
        row("SPEED",  f"{spd:7.2f} m/s",   ry,
            RED if stall else (AMBER if spd < 8 else WHITE));  ry += 15
        row("HDG",    f"{hdg:7.1f}\xb0",   ry);               ry += 15
        row("PITCH",  f"{pitch:+6.1f}\xb0",ry, DIM);          ry += 15
        row("ROLL",   f"{roll:+6.1f}\xb0", ry, DIM);          ry += 15
        row("VSPD",    f"{vz:+6.2f} m/s",    ry,
            RED if vz < -5 else (AMBER if vz < -2 else WHITE)); ry += 14
        row("THR_CMD", f"{thr_cmd:4.0f}%",  ry, DIM);          ry += 14
        row("THR_ACT", f"{thr_act:4.0f}%",  ry, DIM);          ry += 14
        dlt_col = RED if thr_dlt > 0.003 else (AMBER if thr_dlt > 0.001 else DIM)
        row("THR_Δ",  f"{thr_dlt:.4f}", ry, dlt_col);     ry += 14
        row("G-LOAD",  f"{g_load:4.2f} g",  ry,
            RED if g_load > 3 else (AMBER if g_load > 2 else WHITE)); ry += 15
        row("AOA",    f"{aoa:+5.1f}\xb0",   ry,
            RED if abs(aoa) > 20 else DIM);                    ry += 18

        # Stall warning
        if stall:
            sw = self._font_hdr.render("! STALL WARNING !", True, RED)
            pygame.draw.rect(scr, (60,10,10), (x+4, y+ry-2, w-8, 18))
            scr.blit(sw, (x + (w-sw.get_width())//2, y+ry))
        ry += 18

        # ── Control surfaces ─────────────────────────────────────────────────
        header("CTRL SURFACES", ry); ry += 18

        elev = float(d.get("elevator",    0.0))
        ail  = float(d.get("aileron",     0.0))
        rud  = float(d.get("rudder",      0.0))
        surface_bar("ELEV", elev, ry); ry += 18
        surface_bar("AIL",  ail,  ry); ry += 18
        surface_bar("RDR",  rud,  ry); ry += 20

        # ── Navigation ────────────────────────────────────────────────────────
        header("NAVIGATION", ry); ry += 18

        tpos  = target.position
        dist  = float(np.linalg.norm(tpos - s.pos))
        row("TGT DIST", f"{dist:6.1f} m", ry, mode_col); ry += 15

        # Glide slope error (approach/landing modes)
        if mode in (FlightMode.APPROACH, FlightMode.LANDING):
            dx    = float(s.pos[0] - tpos[0])
            dy    = float(s.pos[1] - tpos[1])
            horiz = math.sqrt(dx*dx + dy*dy)
            ideal = horiz * math.tan(math.radians(3.0))
            gs_err = float(s.pos[2]) - ideal
            gs_col = RED if gs_err < -5 else (AMBER if abs(gs_err) > 10 else GREEN)
            row("GS ERR",  f"{gs_err:+6.1f} m", ry, gs_col);  ry += 15

            # Lateral alignment
            hdg_t = target.heading
            cx_t  =  math.sin(hdg_t); cy_t = -math.cos(hdg_t)
            rx    = tpos[0] - float(s.pos[0]); ry_v = tpos[1] - float(s.pos[1])
            lat   = rx * cy_t - ry_v * cx_t
            lat_col = RED if abs(lat) > 30 else (AMBER if abs(lat) > 10 else GREEN)
            row("LTL ERR", f"{lat:+6.1f} m", ry, lat_col);    ry += 15

        ry += 4

        # ── WAYPOINT telemetry ─────────────────────────────────────────────────
        if mode == FlightMode.WAYPOINT:
            header("WAYPOINT INFO", ry, MODE_COLORS.get(FlightMode.WAYPOINT, CYAN)); ry += 16
            wp_st      = str(d.get("wp_state", "approaching")).upper()
            wp_idx     = int(d.get("wp_mission_idx", -1))
            wp_tmr     = int(d.get("wp_transition_timer", 0))
            wp_cnt     = int(d.get("wp_captured_count", 0))
            wp_dst     = float(d.get("wp_curr_dist", 0.0))
            wp_strt    = float(d.get("wp_leg_start_dist", 0.0))
            wp_thresh  = float(d.get("wp_capture_threshold", 20.0))
            wp_elig    = bool(d.get("wp_capture_eligible", True))
            wp_cool    = bool(d.get("wp_cooldown_active", False))
            st_col     = AMBER if wp_st == 'TRANSITIONING' else CYAN
            elig_col   = GREEN if wp_elig else AMBER
            row("STATE",    wp_st,   ry, st_col);  ry += 14
            row("WP IDX",   str(wp_idx) if wp_idx >= 0 else "N/A", ry, CYAN);  ry += 14
            row("CAPTURED", str(wp_cnt), ry, GREEN if wp_cnt > 0 else DIM);    ry += 14
            row("COOLDOWN", f"{wp_tmr}",     ry, AMBER if wp_cool else DIM);   ry += 14
            row("ELIGIBLE", "YES" if wp_elig else "NO", ry, elig_col);         ry += 14
            row("THRESHOLD",f"{wp_thresh:.0f} m", ry, DIM);                    ry += 14
            row("LEG DIST", f"{wp_dst:.0f} m", ry);  ry += 14
            if wp_strt > 0:
                pct = max(0.0, 1.0 - wp_dst / wp_strt) * 100
                row("LEG PROG", f"{pct:.0f}%", ry, CYAN);  ry += 14
            ry += 2

        # ── LOITER coordination telemetry ─────────────────────────────────────
        if mode == FlightMode.LOITER:
            header("LOITER INFO", ry, MODE_COLORS.get(FlightMode.LOITER, CYAN)); ry += 16
            v        = max(s.airspeed, 1.0)
            R        = max(target.radius, 1.0)
            # After physics fix yaw_rate = g·tan(bank)/v is the actual total turn rate
            turn_dps = math.degrees(s.yaw_rate)
            bank_deg = math.degrees(s.roll)
            req_bank = math.degrees(math.atan(v ** 2 / (R * 9.81)))
            # Expected turn rate from current bank: g·tan(bank)/v (exact formula)
            cr_safe  = max(math.cos(s.roll), 0.1)
            coord_yr = (9.81 / v) * math.sin(s.roll) / cr_safe
            coord_dps = math.degrees(coord_yr)
            sideslip = s.yaw_rate - coord_yr   # deviation from coordinated rate
            lat_g    = (v * abs(s.yaw_rate)) / 9.81

            bank_col = (WHITE if abs(bank_deg) > 8 else
                        AMBER if abs(bank_deg) > 3 else RED)
            sd_col   = (GREEN if abs(sideslip) < 0.05 else
                        AMBER if abs(sideslip) < 0.15 else RED)

            row("TURN RATE",  f"{turn_dps:+5.1f} \xb0/s",  ry);              ry += 14
            row("COORD RATE", f"{coord_dps:+5.1f} \xb0/s",  ry, DIM);        ry += 14
            row("BANK",       f"{bank_deg:+5.1f}\xb0",      ry, bank_col);   ry += 14
            row("REQ BANK",   f"{req_bank:4.1f}\xb0",       ry, DIM);        ry += 14
            row("LAT-G",      f"{lat_g:4.2f} g",            ry,
                RED if lat_g > 2 else (AMBER if lat_g > 1 else WHITE));      ry += 14
            row("COORD ERR",  f"{sideslip:+.3f}",           ry, sd_col);     ry += 14

            # Orbit quality diagnostics
            pygame.draw.line(scr, PANEL_BDR, (x+4, y+ry+1), (x+w-4, y+ry+1), 1)
            ry += 6
            orb_cur  = float(d.get("loiter_radius_current", 0.0))
            orb_des  = float(d.get("loiter_radius_desired",  0.0))
            orb_rerr = float(d.get("loiter_radial_error",    0.0))
            orb_vrad = float(d.get("loiter_radial_vel",      0.0))
            orb_std  = float(d.get("loiter_radius_std",      0.0))
            orb_tang = float(d.get("loiter_tang_ratio",      0.0))
            rerr_col = GREEN if abs(orb_rerr) < 10 else (AMBER if abs(orb_rerr) < 25 else RED)
            vrad_col = GREEN if abs(orb_vrad) < 1.0 else (AMBER if abs(orb_vrad) < 3.0 else RED)
            std_col  = GREEN if orb_std < 5 else (AMBER if orb_std < 15 else RED)
            tang_col = GREEN if orb_tang > 0.9 else (AMBER if orb_tang > 0.7 else RED)
            row("CUR RAD",   f"{orb_cur:.1f} m",      ry, DIM);       ry += 14
            row("DES RAD",   f"{orb_des:.1f} m",      ry, DIM);       ry += 14
            row("RAD ERR",   f"{orb_rerr:+.1f} m",    ry, rerr_col); ry += 14
            row("RAD VEL",   f"{orb_vrad:+.2f} m/s",  ry, vrad_col); ry += 14
            row("RAD STD",   f"{orb_std:.1f} m",       ry, std_col);  ry += 14
            row("TANG FRAC", f"{orb_tang:+.3f}",       ry, tang_col); ry += 10

        # ── ALT HOLD info (only in ALTITUDE_HOLD mode) ────────────────────────
        if mode == FlightMode.ALTITUDE_HOLD:
            header("ALT HOLD", ry, MODE_COLORS.get(FlightMode.ALTITUDE_HOLD, CYAN)); ry += 16
            tgt_alt = float(d.get("target_altitude", 0.0))
            alt_err = float(d.get("altitude_error",  0.0))
            tgt_spd = float(d.get("target_airspeed", 14.0))
            spd_err = float(d.get("airspeed_error",  0.0))
            alt_col = RED if abs(alt_err) > 20 else (AMBER if abs(alt_err) > 10 else GREEN)
            spd_col = RED if abs(spd_err) > 4  else (AMBER if abs(spd_err) > 2  else GREEN)
            vz_col  = RED if abs(vz)      > 3  else (AMBER if abs(vz)      > 1.5 else GREEN)
            row("TGT_ALT", f"{tgt_alt:.1f} m",    ry, CYAN);    ry += 14
            row("ALT_ERR", f"{alt_err:+.1f} m",   ry, alt_col); ry += 14
            row("TGT_SPD", f"{tgt_spd:.1f} m/s",  ry, CYAN);    ry += 14
            row("SPD_ERR", f"{spd_err:+.2f} m/s", ry, spd_col); ry += 14
            row("VRT_SPD", f"{vz:+.2f} m/s",      ry, vz_col);  ry += 14
            ry += 2

        # ── Reward breakdown ─────────────────────────────────────────────────
        header("REWARD BREAKDOWN", ry); ry += 18

        breakdown = d.get("reward_breakdown", {})
        total_r = float(d.get("reward", 0.0))

        if breakdown:
            max_abs = max(abs(v) for v in breakdown.values()) + 1e-9
            for term, val in sorted(breakdown.items(), key=lambda x: -abs(x[1])):
                lbl = (term[:12] + ":").ljust(13)
                sign_col = (100,255,100) if val >= 0 else RED
                vs = self._font_sm.render(f"{val:+.3f}", True, sign_col)
                ls = self._font_sm.render(lbl, True, DIM)
                scr.blit(ls, (x + 8, y + ry))
                # Mini bar
                bx = x + 95; bw_max = 80
                bw = int(abs(val) / max_abs * bw_max)
                bh = 8
                by = y + ry + 2
                pygame.draw.rect(scr, (30,35,50), (bx, by, bw_max, bh))
                if bw > 0:
                    col = (30,160,60) if val >= 0 else (180,40,40)
                    pygame.draw.rect(scr, col, (bx, by, bw, bh))
                scr.blit(vs, (x + w - vs.get_width() - 4, y + ry))
                ry += 14
                if ry > VIEW_H - 30:
                    break

        # Total
        pygame.draw.line(scr, PANEL_BDR, (x+4, y+ry), (x+w-4, y+ry), 1); ry += 4
        total_col = (100,255,100) if total_r >= 0 else RED
        ts_label = self._font_md.render("TOTAL:", True, DIM)
        ts_val   = self._font_md.render(f"{total_r:+.3f}", True, total_col)
        scr.blit(ts_label, (x + 8, y + ry))
        scr.blit(ts_val,   (x + w - ts_val.get_width() - 8, y + ry))

    # ═══════════════════════════════════════════════ BOTTOM: graphs ══════════

    def _draw_bottom_graphs(self, scr, hist):
        y = VIEW_H
        h = BOT_H

        pygame.draw.rect(scr, PANEL_BG, (0, y, TOTAL_W, h))

        gw = TOTAL_W // 4 - 10
        gh = h - 40
        gy = y + 35

        graphs = [
            ("REWARD",          hist.get("reward",      []), "mean_reward",
             hist.get("mean_reward", []),
             GRAPH_COLORS["reward"],   GRAPH_COLORS["mean_reward"]),
            ("SUCCESS / CRASH", hist.get("success",     []), "crash",
             hist.get("crash",        []),
             GRAPH_COLORS["success"],  GRAPH_COLORS["crash"]),
            ("STALL RATE",      hist.get("stall",       []), None,
             None,
             GRAPH_COLORS["stall"],    None),
            ("POLICY LOSS",     hist.get("policy_loss", []), "value_loss",
             hist.get("value_loss",   []),
             GRAPH_COLORS["policy_loss"], GRAPH_COLORS["value_loss"]),
        ]

        for i, (title, data1, _name2, data2, col1, col2) in enumerate(graphs):
            gx = i * (gw + 10) + 5
            self._draw_mini_graph(scr, gx, gy, gw, gh, title, data1, col1, data2, col2)

    def _draw_mini_graph(self, scr, x, y, w, h, title,
                         data1, col1, data2=None, col2=None):
        # Background
        pygame.draw.rect(scr, (10, 12, 22), (x, y-22, w, h+22))
        pygame.draw.rect(scr, PANEL_BDR, (x, y-22, w, h+22), 1)

        # Title
        ts = self._font_sm.render(title, True, (100, 120, 160))
        scr.blit(ts, (x + 4, y - 19))

        all_data = list(data1 or []) + list(data2 or [])
        if not all_data:
            nd = self._font_sm.render("no data", True, (40,50,70))
            scr.blit(nd, (x + w//2 - nd.get_width()//2, y + h//2))
            return

        dmin = min(all_data)
        dmax = max(all_data)
        if dmax == dmin:
            dmax = dmin + 1.0

        def to_px(v):
            return y + h - int((v - dmin) / (dmax - dmin) * (h - 4)) - 2

        # Zero line
        if dmin < 0 < dmax:
            zy = to_px(0)
            pygame.draw.line(scr, (50, 55, 75), (x+2, zy), (x+w-2, zy), 1)

        # Draw datasets
        for data, col in [(data1, col1), (data2, col2)]:
            if not data or col is None:
                continue
            n = len(data)
            pts = []
            for j, v in enumerate(data):
                px = x + 2 + int(j / max(n-1, 1) * (w-4))
                py = to_px(v)
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(scr, col, False, pts, 1)
            # Latest value
            lv = self._font_sm.render(f"{data[-1]:.3f}", True, col)
            scr.blit(lv, (x + w - lv.get_width() - 3, y - 19))

        # Min/max labels
        mx = self._font_sm.render(f"{dmax:.2f}", True, (50,60,80))
        mn = self._font_sm.render(f"{dmin:.2f}", True, (50,60,80))
        scr.blit(mx, (x+2, y+1))
        scr.blit(mn, (x+2, y+h-12))

    # ══════════════════════════════════════ pre-training config screen ══════════

    def init_config_screen(self, cfg: dict) -> None:
        """Populate _cfg_draft from cfg dict (values are strings from config.txt)."""
        def _bool(key, default=False):
            return cfg.get(key, str(default)).lower() in ("true", "1", "yes")
        def _float(key, default=0.0):
            try:   return float(cfg.get(key, default))
            except: return float(default)
        def _str(key, default=""):
            return str(cfg.get(key, default))

        self._cfg_draft = {
            "curriculum":           _bool("curriculum",            True),
            "curriculum_phase":     _str("curriculum_phase",       "stabilize"),
            "force_new":            _bool("force_new",             False),
            "stall_speed":          _float("stall_speed",          6.0),
            "action_smooth_weight": _float("action_smooth_weight", 0.03),
            "wireframe":            _bool("wireframe",             False),
        }
        self._cfg_widgets  = {}
        self._cfg_dragging = None

    def get_config_draft(self) -> dict:
        """Return _cfg_draft as a string dict suitable for merging into cfg."""
        d = self._cfg_draft
        return {
            "curriculum":           str(d['curriculum']).lower(),
            "curriculum_phase":     d['curriculum_phase'],
            "force_new":            str(d['force_new']).lower(),
            "stall_speed":          f"{d['stall_speed']:.1f}",
            "action_smooth_weight": f"{d['action_smooth_weight']:.3f}",
            "wireframe":            str(d['wireframe']).lower(),
        }

    def save_config_to_file(self, cfg_draft: dict) -> None:
        """Merge cfg_draft into config.txt, preserving all other keys."""
        lines = []
        try:
            with open(_CONFIG_PATH) as f:
                lines = f.readlines()
        except Exception:
            pass
        written = set()
        out = []
        for line in lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                k = stripped.split("=", 1)[0].strip()
                if k in cfg_draft:
                    out.append(f"{k}={cfg_draft[k]}\n")
                    written.add(k)
                    continue
            out.append(line if line.endswith("\n") else line + "\n")
        for k, v in cfg_draft.items():
            if k not in written:
                out.append(f"{k}={v}\n")
        try:
            with open(_CONFIG_PATH, "w") as f:
                f.writelines(out)
            print(f"[CFG] Saved {_CONFIG_PATH}")
        except Exception as e:
            print(f"[CFG] Save failed: {e}")

    def render_config_screen(self) -> None:
        """Draw the pre-training configuration screen and flip display."""
        scr = self._screen
        scr.fill(BG)

        PW, PH = 740, 510
        px = (self.W - PW) // 2
        py = (self.H - PH) // 2

        self._cfg_widgets = {}
        d   = self._cfg_draft
        lx  = px + 24
        rw  = PW - 48
        ry  = py + 50

        # Card
        pygame.draw.rect(scr, (8, 12, 28), (px, py, PW, PH))
        pygame.draw.rect(scr, CYAN,        (px, py, PW, PH), 1)

        # Title bar
        pygame.draw.rect(scr, (10, 20, 50), (px, py, PW, 36))
        title = self._font_lg.render("ToyUAV RL  —  Launch Configuration", True, CYAN)
        scr.blit(title, (px + PW // 2 - title.get_width() // 2, py + 7))

        def divider():
            nonlocal ry
            pygame.draw.line(scr, PANEL_BDR, (lx, ry), (lx + rw, ry), 1)
            ry += 6

        def section(text):
            nonlocal ry
            divider()
            s = self._font_hdr.render(text, True, CYAN)
            scr.blit(s, (lx, ry))
            ry += 20

        def toggle_widget(key, label):
            nonlocal ry
            val    = d[key]
            col    = GREEN if val else (50, 55, 75)
            bdr    = GREEN if val else PANEL_BDR
            tw, th = 60, 22
            pygame.draw.rect(scr, col, (lx, ry, tw, th), border_radius=4)
            pygame.draw.rect(scr, bdr, (lx, ry, tw, th), 1, border_radius=4)
            ts = self._font_sm.render("ON" if val else "OFF", True, WHITE)
            scr.blit(ts, (lx + (tw - ts.get_width()) // 2,
                          ry + (th - ts.get_height()) // 2))
            self._cfg_widgets[f"{key}_toggle"] = pygame.Rect(lx, ry, tw, th)
            ls = self._font_sm.render(label, True, DIM)
            scr.blit(ls, (lx + tw + 12, ry + (th - ls.get_height()) // 2))
            ry += th + 10

        def slider_widget(key, label, lo, hi, fmt):
            nonlocal ry
            val = d[key]
            scr.blit(self._font_sm.render(label, True, DIM), (lx, ry))
            ry += 16
            sw = rw - 90; sh = 22; sx = lx + 8
            pygame.draw.rect(scr, (30, 35, 55), (sx, ry + sh // 2 - 3, sw, 6),
                             border_radius=3)
            frac = _clamp((val - lo) / (hi - lo), 0.0, 1.0)
            fw   = int(frac * sw)
            if fw > 0:
                pygame.draw.rect(scr, AMBER, (sx, ry + sh // 2 - 3, fw, 6),
                                 border_radius=3)
            tx = sx + fw
            pygame.draw.circle(scr, WHITE, (tx, ry + sh // 2), 9)
            pygame.draw.circle(scr, AMBER, (tx, ry + sh // 2), 7)
            vs = self._font_md.render(fmt(val), True, AMBER)
            scr.blit(vs, (sx + sw + 12, ry + sh // 2 - vs.get_height() // 2))
            self._cfg_widgets[f"{key}_slider"] = pygame.Rect(sx - 9, ry, sw + 18, sh)
            ry += sh + 10

        def cycle_widget(key, options, label):
            nonlocal ry
            curr = d[key]
            bw, bh = 28, 22
            # Prev
            pygame.draw.rect(scr, (35, 45, 65), (lx, ry, bw, bh), border_radius=3)
            ps = self._font_sm.render("◄", True, CYAN)
            scr.blit(ps, (lx + (bw - ps.get_width()) // 2,
                          ry + (bh - ps.get_height()) // 2))
            self._cfg_widgets[f"{key}_prev"] = pygame.Rect(lx, ry, bw, bh)
            # Value
            vs = self._font_md.render(curr.upper().replace("_", " "), True, WHITE)
            vx = lx + bw + 10
            scr.blit(vs, (vx, ry + (bh - vs.get_height()) // 2))
            # Next
            nx = vx + vs.get_width() + 10
            pygame.draw.rect(scr, (35, 45, 65), (nx, ry, bw, bh), border_radius=3)
            ns = self._font_sm.render("►", True, CYAN)
            scr.blit(ns, (nx + (bw - ns.get_width()) // 2,
                          ry + (bh - ns.get_height()) // 2))
            self._cfg_widgets[f"{key}_next"] = pygame.Rect(nx, ry, bw, bh)
            # Right-aligned label
            ls = self._font_sm.render(label, True, DIM)
            scr.blit(ls, (lx + rw - ls.get_width(), ry + (bh - ls.get_height()) // 2))
            ry += bh + 10

        def step_widget(key, label, fmt):
            nonlocal ry
            val    = d[key]
            bw, bh = 28, 22
            # Minus
            pygame.draw.rect(scr, (35, 45, 65), (lx, ry, bw, bh), border_radius=3)
            ms = self._font_sm.render("−", True, CYAN)
            scr.blit(ms, (lx + (bw - ms.get_width()) // 2,
                          ry + (bh - ms.get_height()) // 2))
            self._cfg_widgets[f"{key}_minus"] = pygame.Rect(lx, ry, bw, bh)
            # Value
            vs = self._font_sm.render(fmt(val), True, WHITE)
            vx = lx + bw + 10
            scr.blit(vs, (vx, ry + (bh - vs.get_height()) // 2))
            # Plus
            nx = vx + max(vs.get_width(), 70) + 10
            pygame.draw.rect(scr, (35, 45, 65), (nx, ry, bw, bh), border_radius=3)
            pl = self._font_sm.render("+", True, CYAN)
            scr.blit(pl, (nx + (bw - pl.get_width()) // 2,
                          ry + (bh - pl.get_height()) // 2))
            self._cfg_widgets[f"{key}_plus"] = pygame.Rect(nx, ry, bw, bh)
            # Right-aligned label
            ls = self._font_sm.render(label, True, DIM)
            scr.blit(ls, (lx + rw - ls.get_width(), ry + (bh - ls.get_height()) // 2))
            ry += bh + 10

        # ── CURRICULUM ─────────────────────────────────────────────────────────
        section("CURRICULUM")
        toggle_widget("curriculum",  "Enable auto-curriculum progression")
        cycle_widget("curriculum_phase", _CURRICULUM_PHASES, "Starting Phase")
        toggle_widget("force_new",   "Force new model  (discard saved weights)")

        # ── ENVIRONMENT ────────────────────────────────────────────────────────
        section("ENVIRONMENT")
        step_widget("stall_speed",          "Stall Speed (m/s)",    lambda v: f"{v:.1f} m/s")
        step_widget("action_smooth_weight", "Action Smoothing",     lambda v: f"{v:.3f}")
        toggle_widget("wireframe",          "Wireframe rendering")

        # ── BUTTONS ────────────────────────────────────────────────────────────
        ry += 4
        divider()
        ry += 10
        bw2 = 230; bh2 = 36; gap = 24
        bx1 = px + (PW - bw2 * 2 - gap) // 2
        bx2 = bx1 + bw2 + gap

        pygame.draw.rect(scr, (14, 35, 20), (bx1, ry, bw2, bh2), border_radius=5)
        pygame.draw.rect(scr, (40, 120, 60), (bx1, ry, bw2, bh2), 1, border_radius=5)
        sv = self._font_md.render("SAVE CONFIG.TXT", True, (80, 200, 100))
        scr.blit(sv, (bx1 + (bw2 - sv.get_width()) // 2,
                      ry  + (bh2 - sv.get_height()) // 2))
        self._cfg_widgets["save_cfg"] = pygame.Rect(bx1, ry, bw2, bh2)

        pygame.draw.rect(scr, (10, 50, 80), (bx2, ry, bw2, bh2), border_radius=5)
        pygame.draw.rect(scr, CYAN,         (bx2, ry, bw2, bh2), 2, border_radius=5)
        st = self._font_md.render("▶  START TRAINING", True, CYAN)
        scr.blit(st, (bx2 + (bw2 - st.get_width()) // 2,
                      ry  + (bh2 - st.get_height()) // 2))
        self._cfg_widgets["start_btn"] = pygame.Rect(bx2, ry, bw2, bh2)

        # Hint
        ry += bh2 + 12
        hint = self._font_sm.render("ENTER = start   ESC = quit", True, (50, 60, 80))
        scr.blit(hint, (px + PW // 2 - hint.get_width() // 2, ry))

        pygame.display.flip()

    def handle_config_event(self, event):
        """Process a pygame event for the config screen.
        Returns 'start', 'quit', or None."""
        widgets = self._cfg_widgets
        d       = self._cfg_draft

        if event.type == pygame.QUIT:
            return "quit"

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "quit"
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return "start"

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Slider drag start
            for name, rect in widgets.items():
                if name.endswith("_slider") and rect.collidepoint(mx, my):
                    self._cfg_dragging = name
                    self._update_cfg_slider(name, mx)
                    return None

            # Toggle buttons
            for name, rect in widgets.items():
                if name.endswith("_toggle") and rect.collidepoint(mx, my):
                    key = name[:-7]
                    d[key] = not d[key]
                    return None

            # Cycle prev / next
            for name, rect in widgets.items():
                if name.endswith("_prev") and rect.collidepoint(mx, my):
                    key    = name[:-5]
                    phases = _CURRICULUM_PHASES
                    idx    = phases.index(d[key]) if d[key] in phases else 0
                    d[key] = phases[(idx - 1) % len(phases)]
                    return None
                if name.endswith("_next") and rect.collidepoint(mx, my):
                    key    = name[:-5]
                    phases = _CURRICULUM_PHASES
                    idx    = phases.index(d[key]) if d[key] in phases else 0
                    d[key] = phases[(idx + 1) % len(phases)]
                    return None

            # Step buttons
            _step_map = {
                "stall_speed_minus":          ("stall_speed",          -0.5,   3.0, 15.0,  False),
                "stall_speed_plus":           ("stall_speed",          +0.5,   3.0, 15.0,  False),
                "action_smooth_weight_minus": ("action_smooth_weight", -0.005, 0.0,  0.2,  False),
                "action_smooth_weight_plus":  ("action_smooth_weight", +0.005, 0.0,  0.2,  False),
            }
            for name, (key, delta, lo, hi, is_int) in _step_map.items():
                if name in widgets and widgets[name].collidepoint(mx, my):
                    v      = d.get(key, 0)
                    result = _clamp(v + delta, lo, hi)
                    d[key] = int(result) if is_int else round(result, 3)
                    return None

            # Action buttons
            if "save_cfg" in widgets and widgets["save_cfg"].collidepoint(mx, my):
                self.save_config_to_file(self.get_config_draft())
                return None
            if "start_btn" in widgets and widgets["start_btn"].collidepoint(mx, my):
                return "start"

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._cfg_dragging = None

        elif event.type == pygame.MOUSEMOTION:
            if self._cfg_dragging:
                self._update_cfg_slider(self._cfg_dragging, event.pos[0])

        return None

    def _update_cfg_slider(self, name: str, mx: int) -> None:
        rect = self._cfg_widgets.get(name)
        if not rect:
            return
        key  = name[:-7]   # strip "_slider"
        frac = _clamp((mx - rect.x) / rect.w, 0.0, 1.0)
        lo_hi = {}
        if key in lo_hi:
            lo, hi = lo_hi[key]
            self._cfg_draft[key] = round(lo + frac * (hi - lo), 2)

    # ═══════════════════════════════════════════════════════ legacy API ═══════

    def render(self, env, last_reward: float = 0.0):
        """Legacy direct-env render (used by visualize.py)."""
        import shared_state as _ss
        s = env._state; mode = env.mode; t = env.target

        patch = {
            "pos":          s.pos.tolist(),
            "vel":          s.vel.tolist(),
            "pitch":        float(s.pitch),
            "roll":         float(s.roll),
            "yaw":          float(s.yaw),
            "pitch_rate":   float(s.pitch_rate),
            "roll_rate":    float(s.roll_rate),
            "yaw_rate":     float(s.yaw_rate),
            "throttle_pos": float(s.throttle_pos),
            "airspeed":     float(s.airspeed),
            "elevator":     float(env._prev_action[0]) if hasattr(env, '_prev_action') else 0.0,
            "aileron":      float(env._prev_action[1]) if hasattr(env, '_prev_action') else 0.0,
            "rudder":       float(env._prev_action[2]) if hasattr(env, '_prev_action') else 0.0,
            "mode":         int(mode),
            "reward":       float(last_reward),
            "ready":        True,
        }
        if t is not None:
            patch["target_position"] = t.position.tolist()
            patch["target_heading"]  = float(t.heading)
            patch["target_altitude"] = float(t.altitude)
            patch["target_radius"]   = float(t.radius)
        if hasattr(env, 'mission') and env.mission is not None:
            patch["mission_seg"]   = env.mission.current_index
            patch["mission_total"] = env.mission.num_segments
        if int(env.mode) == int(FlightMode.WAYPOINT) and t is not None:
            patch["wp_curr_dist"] = float(np.linalg.norm(env._state.pos - t.position))
        patch["wp_arrived"]          = bool(getattr(env, '_waypoint_reached', False))
        patch["wp_arrival_steps"]    = int(getattr(env, '_wp_arrival_steps', 0))
        patch["wp_capture_event"]    = bool(getattr(env, '_wp_capture_event', False))
        patch["wp_captured_count"]   = int(getattr(env, '_wp_captured_count', 0))
        patch["wp_state"]            = str(getattr(env, '_wp_state', 'approaching'))
        patch["wp_transition_timer"] = int(getattr(env, '_wp_transition_timer', 0))

        _ss.update(patch)
        self.render_state(_ss.read())
