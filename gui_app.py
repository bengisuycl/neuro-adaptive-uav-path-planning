import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from wcci_conference_project.gui_runner import run_benchmark_from_config


SCENARIOS = ["S1_Base", "S2_Dense", "S3_Long", "S4_DynamicThreat", "S5_RLPilotDemo"]
PLANNERS = [
    "A-Star",
    "Dijkstra",
    "RRT-Star",
    "PSO",
    "K-GNP",
    "T-GnP",
    "RL-Pilot",
    "Neuro-Adaptive",
    "Neuro-Adaptive-NoDNN",
    "ALL ALGORITHMS",
]


class BenchmarkGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Academic Benchmark Control Panel")
        self.geometry("1180x820")
        self.minsize(900, 620)

        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.last_output_dir = None

        self._build_vars()
        self._build_theme()
        self._build_layout()
        self._refresh_visibility()
        self.after(150, self._poll_log_queue)

    def _build_vars(self):
        self.scenario_var = tk.StringVar(value="S1_Base")
        self.planner_var = tk.StringVar(value="RL-Pilot")
        self.enable_dnn_var = tk.BooleanVar(value=False)
        self.compare_var = tk.BooleanVar(value=True)
        self.enable_vis_var = tk.BooleanVar(value=True)
        self.benchmark_only_var = tk.BooleanVar(value=False)
        self.runs_var = tk.StringVar(value="30")
        self.single_runs_var = tk.StringVar(value="12")
        self.time_budget_var = tk.StringVar(value="2.0")
        self.output_tag_var = tk.StringVar(value="")

        self.common_weight_var = tk.StringVar(value="120.0")
        self.common_samples_var = tk.StringVar(value="3")
        self.common_width_var = tk.StringVar(value="1800.0")

        self.rl_weight_var = tk.StringVar(value="0.18")
        self.rl_samples_var = tk.StringVar(value="3")
        self.rl_width_var = tk.StringVar(value="1800.0")

        self.tgnp_weight_var = tk.StringVar(value="300.0")
        self.tgnp_samples_var = tk.StringVar(value="3")
        self.tgnp_width_var = tk.StringVar(value="1800.0")
        self.tgnp_speed_weight_var = tk.StringVar(value="0.35")
        self.tgnp_stability_weight_var = tk.StringVar(value="0.55")

        self.summary_var = tk.StringVar()

    def _build_theme(self):
        self.palette = {
            "bg": "#edf3fb",
            "panel": "#ffffff",
            "panel_alt": "#f7fbff",
            "border": "#c9d8ec",
            "navy": "#163c78",
            "blue": "#2f6cb3",
            "blue_soft": "#dfeeff",
            "text": "#17324d",
            "muted": "#5e7693",
            "green": "#2e8b57",
        }

        self.configure(bg=self.palette["bg"])
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", font=("Segoe UI", 10), foreground=self.palette["text"])
        style.configure("App.TFrame", background=self.palette["bg"])
        style.configure("Panel.TFrame", background=self.palette["panel"], relief="flat")
        style.configure("Hero.TFrame", background=self.palette["panel"])
        style.configure(
            "HeroTitle.TLabel",
            background=self.palette["panel"],
            foreground=self.palette["navy"],
            font=("Segoe UI Semibold", 27),
        )
        style.configure(
            "HeroSub.TLabel",
            background=self.palette["panel"],
            foreground=self.palette["muted"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "SectionTitle.TLabel",
            background=self.palette["panel"],
            foreground=self.palette["navy"],
            font=("Segoe UI Semibold", 13),
        )
        style.configure(
            "Summary.TLabel",
            background=self.palette["blue_soft"],
            foreground=self.palette["navy"],
            font=("Segoe UI", 10),
            padding=10,
        )
        style.configure(
            "TLabelframe",
            background=self.palette["panel"],
            bordercolor=self.palette["border"],
            relief="solid",
            borderwidth=1,
            padding=8,
        )
        style.configure(
            "TLabelframe.Label",
            background=self.palette["panel"],
            foreground=self.palette["navy"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Field.TLabel",
            background=self.palette["panel"],
            foreground=self.palette["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Info.TLabel",
            background=self.palette["panel"],
            foreground=self.palette["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Primary.TButton",
            background=self.palette["navy"],
            foreground="white",
            padding=(14, 10),
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.palette["blue"]), ("disabled", "#9bb3d0")],
            foreground=[("disabled", "#f2f5fa")],
        )
        style.configure(
            "Secondary.TButton",
            background=self.palette["panel_alt"],
            foreground=self.palette["navy"],
            padding=(12, 10),
            font=("Segoe UI", 10),
            bordercolor=self.palette["border"],
            borderwidth=1,
        )
        style.map("Secondary.TButton", background=[("active", self.palette["blue_soft"])])
        style.configure(
            "TCheckbutton",
            background=self.palette["panel"],
            foreground=self.palette["text"],
            font=("Segoe UI", 10),
        )
        style.map(
            "TCheckbutton",
            background=[("active", self.palette["panel"])],
            indicatorcolor=[("selected", self.palette["blue"])],
        )
        style.configure(
            "TEntry",
            fieldbackground="white",
            bordercolor=self.palette["border"],
            lightcolor=self.palette["border"],
            darkcolor=self.palette["border"],
            padding=6,
        )
        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            bordercolor=self.palette["border"],
            lightcolor=self.palette["border"],
            darkcolor=self.palette["border"],
            padding=4,
        )

    def _build_layout(self):
        outer = ttk.Frame(self, style="App.TFrame")
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            outer,
            background=self.palette["bg"],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vscroll = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=vscroll.set)

        root = ttk.Frame(self.canvas, style="App.TFrame", padding=16)
        self.canvas_window = self.canvas.create_window((0, 0), window=root, anchor="nw")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)

        root.bind("<Configure>", self._on_root_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind_all("<MouseWheel>", self._on_mousewheel)

        hero = ttk.Frame(root, style="Hero.TFrame", padding=(18, 16))
        hero.grid(row=0, column=0, sticky="ew")
        hero.columnconfigure(0, weight=1)
        ttk.Label(hero, text="Adaptive Tactical Benchmark GUI", style="HeroTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            hero,
            text=(
                "Academic control panel for scenario selection, planner comparison, "
                "and optional DNN-TRE integration across the thesis benchmark suite."
            ),
            style="HeroSub.TLabel",
            wraplength=1100,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        summary = ttk.Label(root, textvariable=self.summary_var, style="Summary.TLabel", anchor="w", justify="left")
        summary.grid(row=1, column=0, sticky="ew", pady=(12, 14))

        content = ttk.Frame(root, style="App.TFrame")
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        left = ttk.Frame(content, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(content, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.columnconfigure(0, weight=1)

        mission = ttk.LabelFrame(left, text="Mission Setup", padding=12)
        mission.grid(row=0, column=0, sticky="ew")
        mission.columnconfigure(1, weight=1)

        self._add_labeled_combo(mission, "Scenario", self.scenario_var, SCENARIOS, 0)
        self._add_labeled_combo(mission, "Planner", self.planner_var, PLANNERS, 1)
        self.planner_var.trace_add("write", lambda *_: self._refresh_visibility())
        self.scenario_var.trace_add("write", lambda *_: self._refresh_visibility())

        toggles = ttk.LabelFrame(left, text="Execution Options", padding=12)
        toggles.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        toggles.columnconfigure(0, weight=1)
        ttk.Checkbutton(toggles, text="Enable DNN-TRE", variable=self.enable_dnn_var, command=self._refresh_visibility).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(
            toggles,
            text="Compare baseline vs. DNN-enabled variant",
            variable=self.compare_var,
            command=self._refresh_visibility,
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Checkbutton(toggles, text="Export visualization figures", variable=self.enable_vis_var).grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(toggles, text="Benchmark-only comparative outputs", variable=self.benchmark_only_var).grid(
            row=3, column=0, sticky="w", pady=2
        )

        runtime = ttk.LabelFrame(left, text="Run Budget and Output", padding=12)
        runtime.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        runtime.columnconfigure(1, weight=1)
        self._add_labeled_entry(runtime, "Monte Carlo runs", self.runs_var, 0)
        self._add_labeled_entry(runtime, "Single-planner runs", self.single_runs_var, 1)
        self._add_labeled_entry(runtime, "Base time budget (s)", self.time_budget_var, 2)
        self._add_labeled_entry(runtime, "Output tag", self.output_tag_var, 3)
        ttk.Label(
            runtime,
            text="Use an output tag when you want a reproducible folder name for a slide figure or ablation run.",
            style="Info.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        dnn = ttk.LabelFrame(right, text="DNN-TRE Configuration", padding=12)
        dnn.grid(row=0, column=0, sticky="ew")
        dnn.columnconfigure(0, weight=1)
        ttk.Label(
            dnn,
            text=(
                "The common feasibility engine can inject DNN-TRE risk for classical planners and K-GNP. "
                "RL-Pilot and T-GnP also expose planner-specific corridor-risk settings."
            ),
            style="Info.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.common_frame = ttk.LabelFrame(dnn, text="Common DNN Feasibility Settings", padding=10)
        self.common_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.common_frame.columnconfigure(1, weight=1)
        self._add_labeled_entry(self.common_frame, "Weight", self.common_weight_var, 0)
        self._add_labeled_entry(self.common_frame, "Samples", self.common_samples_var, 1)
        self._add_labeled_entry(self.common_frame, "Corridor half width (m)", self.common_width_var, 2)

        self.rl_frame = ttk.LabelFrame(dnn, text="RL-Pilot Specific DNN Settings", padding=10)
        self.rl_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.rl_frame.columnconfigure(1, weight=1)
        self._add_labeled_entry(self.rl_frame, "Weight", self.rl_weight_var, 0)
        self._add_labeled_entry(self.rl_frame, "Samples", self.rl_samples_var, 1)
        self._add_labeled_entry(self.rl_frame, "Corridor half width (m)", self.rl_width_var, 2)

        self.tgnp_frame = ttk.LabelFrame(dnn, text="T-GnP Specific DNN Settings", padding=10)
        self.tgnp_frame.grid(row=3, column=0, sticky="ew")
        self.tgnp_frame.columnconfigure(1, weight=1)
        self._add_labeled_entry(self.tgnp_frame, "Neural weight", self.tgnp_weight_var, 0)
        self._add_labeled_entry(self.tgnp_frame, "Samples", self.tgnp_samples_var, 1)
        self._add_labeled_entry(self.tgnp_frame, "Corridor half width (m)", self.tgnp_width_var, 2)
        self._add_labeled_entry(self.tgnp_frame, "Speed weight", self.tgnp_speed_weight_var, 3)
        self._add_labeled_entry(self.tgnp_frame, "Stability weight", self.tgnp_stability_weight_var, 4)

        actions = ttk.Frame(root, style="App.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(16, 12))
        self.run_button = ttk.Button(actions, text="Run Benchmark", style="Primary.TButton", command=self._start_run)
        self.run_button.pack(side="left")
        ttk.Button(actions, text="Open Output Folder", style="Secondary.TButton", command=self._open_output_folder).pack(
            side="left", padx=8
        )
        ttk.Button(actions, text="Clear Log", style="Secondary.TButton", command=self._clear_log).pack(side="left")

        log_frame = ttk.LabelFrame(root, text="Execution Log", padding=10)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            font=("Consolas", 10),
            bg="#fbfdff",
            fg=self.palette["text"],
            insertbackground=self.palette["navy"],
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._update_summary()

    def _on_root_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _add_labeled_entry(self, parent, label, variable, row):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _add_labeled_combo(self, parent, label, variable, values, row):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=24)
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _refresh_visibility(self):
        planner = self.planner_var.get()
        dnn_on = self.enable_dnn_var.get()
        compare_ok = planner not in ("ALL ALGORITHMS",)
        if not compare_ok:
            self.compare_var.set(False)

        rl_visible = planner in ("RL-Pilot", "ALL ALGORITHMS")
        tgnp_visible = planner in ("T-GnP", "ALL ALGORITHMS")
        common_visible = planner not in ("RL-Pilot", "T-GnP", "Neuro-Adaptive")
        if planner == "ALL ALGORITHMS":
            common_visible = True
            rl_visible = True
            tgnp_visible = True

        self._toggle_frame(self.common_frame, dnn_on and common_visible)
        self._toggle_frame(self.rl_frame, dnn_on and rl_visible)
        self._toggle_frame(self.tgnp_frame, dnn_on and tgnp_visible)
        self._update_summary()

    def _update_summary(self):
        planner = self.planner_var.get()
        scenario = self.scenario_var.get()
        compare_text = "Comparison mode enabled" if self.compare_var.get() else "Single configuration"
        dnn_text = "DNN-TRE active" if self.enable_dnn_var.get() else "DNN-TRE disabled"
        self.summary_var.set(
            f"Scenario: {scenario}   |   Planner: {planner}   |   {compare_text}   |   {dnn_text}\n"
            "Focus: benchmark-ready outputs with shared feasibility settings and optional planner-specific DNN corridor-risk tuning."
        )

    def _toggle_frame(self, frame, visible):
        if visible:
            frame.grid()
        else:
            frame.grid_remove()

    def _append_log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self.run_button.configure(state="normal")
                    self.last_output_dir = payload
                    self._append_log(f"\nCompleted. Output folder: {payload}")
                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self._append_log(f"\nERROR: {payload}")
                    messagebox.showerror("Benchmark Error", payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

    def _make_config(self):
        time_budget_base = max(0.5, float(self.time_budget_var.get()))
        return {
            "scenario_id": self.scenario_var.get(),
            "planner_name": self.planner_var.get(),
            "enable_dnn": self.enable_dnn_var.get(),
            "compare_baseline_vs_dnn": self.compare_var.get(),
            "enable_visualization": self.enable_vis_var.get(),
            "benchmark_only": self.benchmark_only_var.get(),
            "runs_per_alg": int(self.runs_var.get()),
            "single_alg_runs": int(self.single_runs_var.get()),
            "time_budget_base": time_budget_base,
            "output_tag": self.output_tag_var.get().strip(),
            "common_dnn_weight": float(self.common_weight_var.get()),
            "common_dnn_samples": int(self.common_samples_var.get()),
            "common_dnn_width": float(self.common_width_var.get()),
            "rl_dnn_weight": float(self.rl_weight_var.get()),
            "rl_dnn_samples": int(self.rl_samples_var.get()),
            "rl_dnn_width": float(self.rl_width_var.get()),
            "tgnp_dnn_weight": float(self.tgnp_weight_var.get()),
            "tgnp_dnn_samples": int(self.tgnp_samples_var.get()),
            "tgnp_dnn_width": float(self.tgnp_width_var.get()),
            "tgnp_speed_weight": float(self.tgnp_speed_weight_var.get()),
            "tgnp_stability_weight": float(self.tgnp_stability_weight_var.get()),
        }

    def _start_run(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Benchmark Running", "A benchmark is already running.")
            return

        try:
            config = self._make_config()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", f"Please check numeric fields.\n\n{exc}")
            return

        self.run_button.configure(state="disabled")
        self._append_log("Starting benchmark run...")
        self._append_log(f"Scenario: {config['scenario_id']} | Planner: {config['planner_name']}")

        def worker():
            try:
                result = run_benchmark_from_config(config, log_fn=lambda msg: self.log_queue.put(("log", msg)))
                self.log_queue.put(("done", result["output_dir"]))
            except Exception as exc:
                self.log_queue.put(("error", str(exc)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _open_output_folder(self):
        if not self.last_output_dir or not os.path.isdir(self.last_output_dir):
            messagebox.showinfo("No Output Folder", "Run a benchmark first so the output folder becomes available.")
            return
        try:
            os.startfile(self.last_output_dir)
        except Exception as exc:
            messagebox.showerror("Open Folder Error", str(exc))


if __name__ == "__main__":
    app = BenchmarkGUI()
    app.mainloop()
