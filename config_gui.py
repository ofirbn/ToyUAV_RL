"""
ToyUAV RL — Configuration Launcher

Run with:  python config_gui.py

Loads config.txt, lets the user edit all training settings, then
saves and launches main.py.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.txt")

CURRICULUM_STAGES = [
    "stabilize", "cruise", "altitude_hold", "heading_hold",
    "waypoint", "loiter", "approach", "landing", "mixed",
]
MODES = ["train_visual", "pipeline_visual", "train", "train_expert",
         "visualize", "demo"]

# Experts trainable via mode=train_expert (saved to models/experts/<mode>.zip).
# "all" trains every expert sequentially.
EXPERT_MODES = [
    "all",
    "stabilize", "recovery", "altitude_hold", "heading_hold",
    "waypoint", "loiter", "approach", "landing",
]

_SCRATCH     = "(train from scratch)"
_RECORD_NEW  = "(record new demos)"


# ── config I/O ────────────────────────────────────────────────────────────────

def _read_config() -> dict:
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    cfg[key.strip()] = val.strip()
    return cfg


def _write_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        for key, val in cfg.items():
            f.write(f"{key}={val}\n")


def _scan(subdir: str, ext: str) -> list[str]:
    d = os.path.join(BASE_DIR, subdir)
    if not os.path.isdir(d):
        return []
    return sorted(f"{subdir}/{fn}" for fn in os.listdir(d) if fn.endswith(ext))


# ── GUI ───────────────────────────────────────────────────────────────────────

class ConfigGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ToyUAV RL — Training Launcher")
        root.resizable(False, False)

        s = ttk.Style()
        for theme in ("vista", "winnative", "clam", "default"):
            try:
                s.theme_use(theme)
                break
            except tk.TclError:
                pass

        cfg = _read_config()

        outer = ttk.Frame(root, padding=20)
        outer.grid(row=0, column=0)

        r = 0

        # ── header ────────────────────────────────────────────────────────────
        ttk.Label(outer, text="ToyUAV RL — Training Launcher",
                  font=("Segoe UI", 13, "bold")).grid(
            row=r, column=0, columnspan=2, pady=(0, 4), sticky="w")
        r += 1
        ttk.Separator(outer, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        r += 1

        # ── TRAINING ──────────────────────────────────────────────────────────
        r = self._section(outer, "TRAINING", r)

        self.mode_var = tk.StringVar(value=cfg.get("mode", "train_visual"))
        r = self._combo(outer, r, "Mode", self.mode_var, MODES)

        self.timesteps_var = tk.StringVar(value=cfg.get("timesteps", "3000000"))
        r = self._entry(outer, r, "Timesteps", self.timesteps_var)

        self.save_every_var = tk.StringVar(value=cfg.get("save_every", "100000"))
        r = self._entry(outer, r, "Save every (steps)", self.save_every_var)

        # Expert mode — used only when Mode = train_expert. Trains a single-mode
        # PPO expert and saves it to models/experts/<mode>.zip.
        self.expert_mode_var = tk.StringVar(
            value=cfg.get("expert_mode", "approach"))
        self._expert_lbl, self._expert_cb, r = self._combo(
            outer, r, "Expert mode  (train_expert)", self.expert_mode_var,
            EXPERT_MODES, return_widgets=True)

        # Visualize expert training in the live pygame dashboard (single expert;
        # expert_mode=all stays headless).
        self.expert_visual_var = tk.BooleanVar(
            value=cfg.get("expert_visual", "true").lower() == "true")
        self._expert_vis_chk = ttk.Checkbutton(
            outer, text="Visualize expert training (live dashboard)",
            variable=self.expert_visual_var)
        self._expert_vis_chk.grid(row=r, column=0, columnspan=2,
                                  sticky="w", pady=3)
        r += 1

        # Seed a *fresh* expert from an existing model (e.g. models/latest.zip).
        # "(auto)" = resume the expert if present, else BC / scratch. Ignored
        # when the expert already has a checkpoint (it resumes instead).
        _AUTO = "(auto: resume / BC / scratch)"
        init_opts = [_AUTO] + (_scan("models", ".zip") or [])
        init_default = cfg.get("expert_init_from", "") or _AUTO
        if init_default not in init_opts:
            init_opts.insert(1, init_default)
        self.expert_init_var = tk.StringVar(value=init_default)
        self._expert_init_lbl, self._expert_init_cb, r = self._combo(
            outer, r, "Init new expert from", self.expert_init_var,
            init_opts, editable=True, pin_top=True, return_widgets=True)

        self.mode_var.trace_add("write", lambda *_: self._sync_expert())
        self._sync_expert()

        # ── PIPELINE (pipeline_visual mode) ───────────────────────────────────
        r = self._section(outer, "PIPELINE  (pipeline_visual mode only)", r)

        demos = [_RECORD_NEW] + (_scan("results/demos", ".npz") or [])
        demo_default = cfg.get("demo_path", _RECORD_NEW)
        if demo_default not in demos:
            demo_default = _RECORD_NEW
        self.demo_var = tk.StringVar(value=demo_default)
        r = self._combo(outer, r, "Demo recordings (step 1)", self.demo_var,
                        demos, editable=False, pin_top=True)

        _phase_opts = ["all"] + CURRICULUM_STAGES[:-1]  # all + individual (exclude mixed)
        self.record_phases_var = tk.StringVar(value=cfg.get("record_phases", "all"))
        r = self._combo(outer, r, "  Record phases", self.record_phases_var,
                        _phase_opts, editable=False)

        self.record_ep_var = tk.StringVar(value=cfg.get("record_episodes", "100"))
        r = self._entry(outer, r, "  Episodes per phase", self.record_ep_var)

        bc_models = ["(run BC training)"] + (_scan("models/bc", ".zip") or [])
        bc_default = cfg.get("bc_model_path", "(run BC training)")
        if bc_default not in bc_models:
            bc_default = "(run BC training)"
        self.bc_model_var = tk.StringVar(value=bc_default)
        r = self._combo(outer, r, "BC model (step 2)", self.bc_model_var,
                        bc_models, editable=False, pin_top=True)

        self.bc_epochs_var = tk.StringVar(value=cfg.get("bc_epochs", "60"))
        r = self._entry(outer, r, "  BC epochs", self.bc_epochs_var)

        # ── CURRICULUM ────────────────────────────────────────────────────────
        r = self._section(outer, "CURRICULUM", r)

        self.curriculum_var = tk.BooleanVar(
            value=cfg.get("curriculum", "true").lower() == "true")
        r = self._check(outer, r, "Auto-curriculum enabled",
                        self.curriculum_var, self._sync_curriculum)

        self.phase_var = tk.StringVar(value=cfg.get("curriculum_phase", "stabilize"))
        self._phase_lbl, self._phase_cb, r = self._combo(
            outer, r, "Start phase", self.phase_var,
            CURRICULUM_STAGES, return_widgets=True)

        self._sync_curriculum()

        # ── MODEL & MISSION ───────────────────────────────────────────────────
        r = self._section(outer, "MODEL & MISSION", r)

        models = [_SCRATCH] + (_scan("models", ".zip") or ["models/latest.zip"])
        model_default = (_SCRATCH if cfg.get("force_new", "false").lower() == "true"
                         else cfg.get("model", "models/latest.zip"))
        self.model_var = tk.StringVar(value=model_default)
        r = self._combo(outer, r, "Model file", self.model_var, models, editable=True, pin_top=True)

        missions = _scan("missions", ".json") or ["missions/demo_mission.json"]
        self.mission_var = tk.StringVar(
            value=cfg.get("mission", "missions/demo_mission.json"))
        r = self._combo(outer, r, "Mission file", self.mission_var, missions, editable=True)

        # ── ENVIRONMENT ───────────────────────────────────────────────────────
        r = self._section(outer, "ENVIRONMENT", r)

        self.stall_var = tk.StringVar(value=cfg.get("stall_speed", "6.0"))
        r = self._entry(outer, r, "Stall speed (m/s)", self.stall_var)

        self.smooth_var = tk.StringVar(value=cfg.get("action_smooth_weight", "0.03"))
        r = self._entry(outer, r, "Action smooth weight", self.smooth_var)

        self.wireframe_var = tk.BooleanVar(
            value=cfg.get("wireframe", "false").lower() == "true")
        r = self._check(outer, r, "Wireframe rendering", self.wireframe_var)

        # ── buttons ───────────────────────────────────────────────────────────
        ttk.Separator(outer, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(18, 12))
        r += 1

        btn_frame = ttk.Frame(outer)
        btn_frame.grid(row=r, column=0, columnspan=2)
        ttk.Button(btn_frame, text="Save Config", width=18,
                   command=self._save).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Save & Start Training", width=24,
                   command=self._save_and_start).pack(side="left", padx=6)
        r += 1

        # ── status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value=f"Config: {CONFIG_PATH}")
        ttk.Label(outer, textvariable=self._status_var,
                  foreground="gray", font=("Consolas", 8)).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # centre on screen
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    # ── widget helpers ────────────────────────────────────────────────────────

    def _section(self, p, text: str, r: int) -> int:
        f = ttk.Frame(p)
        f.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text=text, font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, padx=(0, 8))
        ttk.Separator(f, orient="horizontal").grid(row=0, column=1, sticky="ew")
        return r + 1

    def _entry(self, p, r: int, label: str, var: tk.StringVar) -> int:
        ttk.Label(p, text=label).grid(
            row=r, column=0, sticky="e", padx=(0, 12), pady=3)
        ttk.Entry(p, textvariable=var, width=26).grid(
            row=r, column=1, sticky="w", pady=3)
        return r + 1

    def _combo(self, p, r: int, label: str, var: tk.StringVar,
               values: list, editable: bool = False,
               return_widgets: bool = False, pin_top: bool = False):
        lbl = ttk.Label(p, text=label)
        lbl.grid(row=r, column=0, sticky="e", padx=(0, 12), pady=3)
        cb = ttk.Combobox(p, textvariable=var, values=values,
                           state="normal" if editable else "readonly", width=24)
        cb.grid(row=r, column=1, sticky="w", pady=3)
        if pin_top:
            cb['postcommand'] = lambda: cb.after(
                10, lambda: cb.tk.eval(f'{str(cb)}.popdown.f.l see 0'))
        if return_widgets:
            return lbl, cb, r + 1
        return r + 1

    def _check(self, p, r: int, label: str, var: tk.BooleanVar,
               command=None) -> int:
        ttk.Checkbutton(p, text=label, variable=var, command=command).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=3)
        return r + 1

    def _sync_curriculum(self):
        on = self.curriculum_var.get()
        self._phase_cb.config(state="readonly" if on else "disabled")
        self._phase_lbl.config(foreground="black" if on else "gray")

    def _sync_expert(self):
        on = self.mode_var.get() == "train_expert"
        self._expert_cb.config(state="readonly" if on else "disabled")
        self._expert_lbl.config(foreground="black" if on else "gray")
        self._expert_vis_chk.config(state="normal" if on else "disabled")
        self._expert_init_cb.config(state="normal" if on else "disabled")
        self._expert_init_lbl.config(foreground="black" if on else "gray")

    # ── config logic ──────────────────────────────────────────────────────────

    def _collect(self) -> dict:
        scratch = self.model_var.get() == _SCRATCH
        return {
            "mode":                 self.mode_var.get(),
            "timesteps":            self.timesteps_var.get(),
            "expert_mode":          self.expert_mode_var.get(),
            "expert_visual":        "true" if self.expert_visual_var.get() else "false",
            "expert_init_from":     ("" if self.expert_init_var.get().startswith("(auto")
                                     else self.expert_init_var.get()),
            "mission":              self.mission_var.get(),
            "model":                ("models/latest.zip" if scratch
                                     else self.model_var.get()),
            "force_new":            "true" if scratch else "false",
            "curriculum":           "true" if self.curriculum_var.get() else "false",
            "curriculum_phase":     self.phase_var.get(),
            "visualize_training":   "true",
            "save_every":           self.save_every_var.get(),
            "action_smooth_weight": self.smooth_var.get(),
            "stall_speed":          self.stall_var.get(),
            "wireframe":            "true" if self.wireframe_var.get() else "false",
            "demo_path":            ("" if self.demo_var.get() == _RECORD_NEW
                                     else self.demo_var.get()),
            "record_phases":        self.record_phases_var.get(),
            "record_episodes":      self.record_ep_var.get(),
            "bc_model_path":        ("" if self.bc_model_var.get() == "(run BC training)"
                                     else self.bc_model_var.get()),
            "bc_epochs":            self.bc_epochs_var.get(),
        }

    def _validate(self) -> bool:
        try:
            ts = int(self.timesteps_var.get())
            se = int(self.save_every_var.get())
            re = int(self.record_ep_var.get())
            bc = int(self.bc_epochs_var.get())
            float(self.stall_var.get())
            float(self.smooth_var.get())
            if ts <= 0 or se <= 0 or re <= 0 or bc <= 0:
                raise ValueError("timesteps, save_every, record_episodes, bc_epochs must be > 0")
            return True
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return False

    def _save(self):
        if not self._validate():
            return
        try:
            _write_config(self._collect())
            self._status_var.set("Saved.")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _save_and_start(self):
        if not self._validate():
            return
        self._save()
        self.start_requested = True
        self.root.destroy()


# ── entry point ───────────────────────────────────────────────────────────────

def run_gui() -> bool:
    """
    Show the config GUI.  Returns True if the user clicked
    'Save & Start Training', False if they closed the window.
    """
    root = tk.Tk()
    gui = ConfigGUI(root)
    root.mainloop()
    return getattr(gui, "start_requested", False)


if __name__ == "__main__":
    # When run directly, just show the GUI (no training follows).
    run_gui()
