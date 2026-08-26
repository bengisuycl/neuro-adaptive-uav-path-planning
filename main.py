# wcci_conference_project/main.py

import copy
import os
import sys
import time
from datetime import datetime

import numpy as np

from wcci_conference_project.aircraft.jet_model import Aircraft
from wcci_conference_project.analysis.risk_map_comparison import run_comparative_study
from wcci_conference_project.ai_modules.generate_real_dataset import resolve_dem_path
from wcci_conference_project.environment.terrain import Terrain
from wcci_conference_project.evaluation.metrics import (
    calculate_min_terrain_clearance,
    calculate_risk_exposure,
    compute_guidance_metrics,
    path_length,
)
from wcci_conference_project.planners.astar import AstarPlanner
from wcci_conference_project.planners.dijkstra import DijkstraPlanner
from wcci_conference_project.planners.f16_dijkstra import F16DijkstraPlanner
from wcci_conference_project.planners.f16_dijkstra_robust import F16DijkstraPlannerRobust
from wcci_conference_project.planners.neuro_adaptive_planner import NeuroAdaptivePlanner
from wcci_conference_project.planners.pso_planner import PSOPlanner
from wcci_conference_project.planners.rl_planner import RLPlanner
from wcci_conference_project.planners.rrt_star import RRTStarPlanner
from wcci_conference_project.scenarios.definitions import get_all_scenarios
from wcci_conference_project.simulator.simulation import Simulation
from wcci_conference_project.utils.trajectory_processor import process_path_for_flight
from wcci_conference_project.visualization.animate_flight import animate_tactical_flight
from wcci_conference_project.visualization.benchmark_outputs import (
    plot_benchmark_summary,
    plot_stochastic_run_outcomes,
    save_benchmark_csv,
    save_benchmark_latex,
    save_stochastic_run_csv,
    summarize_reports,
)
from wcci_conference_project.visualization.plot2d import plot_flight_analysis_2d
from wcci_conference_project.visualization.plot3d import plot_dynamic_trajectory_3d, plot_unified_tactical_3d
from wcci_conference_project.visualization.plot_riskmap import plot_professional_riskmap

ENABLE_VISUALIZATION = True
STOCHASTIC_RUNS = 30
SINGLE_ALG_RUNS = 12
TIME_BUDGET_BASE = 2.0
VISUALIZATION_MODE_3D = "BOTH"  # "UNIFIED" | "DYNAMIC" | "BOTH"
VIDEO_MODE = "ALL"  # "ALL" | "NONE" | <algorithm-name>
VIDEO_FORMAT = "GIF"
T_GNP_USE_NEURAL_RISK = False
T_GNP_NEURAL_RISK_WEIGHT = 300.0
T_GNP_NEURAL_RISK_SAMPLES = 3
T_GNP_NEURAL_CORRIDOR_HALF_WIDTH = 1800.0
T_GNP_SPEED_WEIGHT = 0.35
T_GNP_STABILITY_WEIGHT = 0.55
RL_PILOT_USE_NEURAL_RISK = False
RL_PILOT_NEURAL_RISK_WEIGHT = 0.18
RL_PILOT_NEURAL_RISK_SAMPLES = 3
RL_PILOT_NEURAL_CORRIDOR_HALF_WIDTH = 1800.0
DEM_PATH = resolve_dem_path(os.environ.get("WCCI_DEM_PATH"))

BASE_OUTPUT_DIR = os.environ.get(
    "WCCI_OUTPUT_DIR",
    r"C:\Users\DELL\PycharmProjects\thesis_project\wcci_conference_project\outputs\THESIS_FINAL_RESULTS",
)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, TODAY_STR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALG_COLORS = {
    "A-Star": "#2050ff",
    "Dijkstra": "#ffd400",
    "RRT-Star": "#7c1dc9",
    "PSO": "#ff2dfb",
    "K-GNP": "#00b8ff",
    "T-GnP": "#12ad2b",
    "T-GnP+DNN": "#0a7d1b",
    "RL-Pilot": "#111111",
    "RL-Pilot+DNN": "#8b0000",
    "Neuro-Adaptive": "#d62f2f",
    "Neuro-Adaptive-NoDNN": "#ff7a7a",
}


def analyze_flight_result(traj, threats, goal_pos, goal_radius):
    if traj is None or len(traj) == 0:
        return False, "NO_TRAJ", 0

    violation_steps = sum(
        1 for state in traj for threat in threats if np.hypot(state[0] - threat.x, state[1] - threat.y) <= threat.radius
    )
    if np.linalg.norm(traj[-1, :2] - goal_pos) <= goal_radius:
        if violation_steps > 0:
            return False, "FAIL_THREAT", violation_steps
        return True, "SUCCESS", 0
    return False, "FAIL_DYNAMICS", 0


def _print_benchmark_table(stochastic_reports, scenario_name):
    print("\n" + "=" * 105)
    print(f" MSc THESIS BENCHMARK RESULTS - {scenario_name} ".center(105))
    print("=" * 105)
    print(
        f"| {'ALGORITHM':<15} | {'SUCCESS':<8} | {'PATH(km)':<10} | {'TIME(s)':<8} | {'RISK(M)':<8} | {'AGL(m)':<6} | {'VIOL':<5} | {'TRACK(m)':<8} | {'CALC(s)':<8} |"
    )
    print("-" * 105)
    for alg, rows in stochastic_reports.items():
        succ_rows = [r for r in rows if r.get("sim_status") == "SUCCESS"]
        succ_rate = f"{len(succ_rows)}/{len(rows)}"
        if succ_rows:
            vals = {k: np.mean([r[k] for r in succ_rows]) for k in ["Path", "Time", "Risk", "AGL", "ThrFail", "Track", "Calc"]}
            print(
                f"| {alg:<15} | {succ_rate:<8} | {vals['Path']:<10.2f} | {vals['Time']:<8.2f} | {vals['Risk']:<8.2f} | {vals['AGL']:<6.0f} | {int(vals['ThrFail']):<5} | {vals['Track']:<8.2f} | {vals['Calc']:<8.2f} |"
            )
        else:
            print(f"| {alg:<15} | {succ_rate:<8} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<6} | {'N/A':<5} | {'N/A':<8} | {'N/A':<8} |")
    print("=" * 105 + "\n")


def _should_make_video(alg_name):
    if VIDEO_MODE == "NONE":
        return False
    if VIDEO_MODE == "ALL":
        return True
    return alg_name == VIDEO_MODE


def _save_demo_assets(scen, alg_name, terrain, threats, best_run, target_pos, goal_radius, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    metric_path = best_run.get("_metric_path")
    traj = best_run.get("_traj")
    history = best_run.get("_history")

    filename_2d = os.path.join(output_dir, f"ThesisFinal_{scen['id']}_ALG-{alg_name}_FIG-Flight2D_{TODAY_STR}.png")
    plot_flight_analysis_2d(
        terrain,
        threats,
        traj,
        title=f"MSc Thesis: {alg_name} Flight Analysis (Risk: {best_run['Risk']:.2f}M)",
        save_path=filename_2d,
        goal_center=target_pos,
        goal_radius=goal_radius,
    )

    if VISUALIZATION_MODE_3D in ("UNIFIED", "BOTH"):
        filename_3d = os.path.join(output_dir, f"ThesisFinal_{scen['id']}_ALG-{alg_name}_FIG-Tactical3D_{TODAY_STR}.png")
        plot_unified_tactical_3d(
            terrain,
            threats,
            path=metric_path,
            traj=traj,
            title=f"MSc Thesis: {alg_name} Tactical 3D View",
            save_path=filename_3d,
            goal_center=target_pos,
            goal_radius=goal_radius,
        ) 

    if VISUALIZATION_MODE_3D in ("DYNAMIC", "BOTH"):
        filename_dyn = os.path.join(output_dir, f"ThesisFinal_{scen['id']}_ALG-{alg_name}_FIG-Dynamic3D_{TODAY_STR}.png")
        plot_dynamic_trajectory_3d(
            terrain,
            threats,
            path=metric_path,
            traj=traj,
            title=f"MSc Thesis: {alg_name} Dynamic 3D View",
            save_path=filename_dyn,
            goal_center=target_pos,
            goal_radius=goal_radius,
        )

    if _should_make_video(alg_name):
        ext = ".gif" if VIDEO_FORMAT.upper() == "GIF" else ".mp4"
        filename_anim = os.path.join(output_dir, f"ThesisFinal_{scen['id']}_ALG-{alg_name}_VID-Flight_{TODAY_STR}{ext}")
        animate_tactical_flight(
            terrain,
            threats,
            traj,
            history,
            filename_anim,
            title=f"{alg_name} Demo Flight",
            goal_center=target_pos,
            goal_radius=goal_radius,
            planned_path=metric_path,
        )


def _prepare_metric_path(raw_path, alg_name, threats):
    if not raw_path:
        return None

    # K-GNP already plans with heading and turn-feasibility constraints, so
    # aggressive circular post-smoothing can introduce artificial loops in
    # dense scenarios. Keep it on a resample-only pipeline.
    apply_fillet = alg_name != "K-GNP"
    _, metric_path = process_path_for_flight(
        raw_path,
        r_min=1500.0,
        validation_threats=threats,
        apply_fillet=apply_fillet,
    )
    return metric_path


def main():
    print("Initializing MSc Thesis Benchmark System...")
    if DEM_PATH is None:
        raise FileNotFoundError(
            "DEM file could not be resolved. Set WCCI_DEM_PATH or place dem_utm.tif in the project workspace."
        )
    print(f"DEM Path Check: {DEM_PATH}")
    terrain = Terrain(DEM_PATH)
    aircraft = Aircraft(V_cruise=220.0)
    safe_alt = max(np.nanmax(terrain.data) + 200.0, 600.0)
    scenarios = get_all_scenarios(safe_alt)

    print("\n--- Scenario Selection ---")
    for idx, scen in enumerate(scenarios):
        print(f"[{idx + 1}] {scen['name']}")
    sel = int(input(f"\nSelect Scenario (1-{len(scenarios)}): ")) - 1
    scen = scenarios[sel]

    current_time_budget = max(0.5, TIME_BUDGET_BASE * {0: 1.0, 1: 1.5, 2: 2.5, 3: 3.0, 4: 1.0}.get(sel, 1.0))
    terrain.threats = scen["threats"]
    planner_threats = copy.deepcopy(scen["threats"])
    for threat in planner_threats:
        threat.radius += 200.0

    target_pos = np.array(scen["goal"][:2])
    goal_radius = 3000.0
    common_cfg = {"check_resolution": 50.0, "threat_weight": 20.0, "safe_alt": safe_alt, "goal_tolerance": goal_radius}

    planner_factories = {
        "A-Star": lambda: AstarPlanner(terrain, planner_threats, step=100, heuristic_weight=1.5, **common_cfg),
        "Dijkstra": lambda: DijkstraPlanner(terrain, planner_threats, step=100, max_iter=2000000, **common_cfg),
        "RRT-Star": lambda: RRTStarPlanner(terrain, planner_threats, max_iter=25000, step_len=2500.0, goal_sample_rate=0.15, search_radius=3000.0, **common_cfg),
        "PSO": lambda: PSOPlanner(terrain, planner_threats, num_particles=150, max_iter=60000, waypoint_spread_m=6000.0, num_waypoints=8, **common_cfg),
        "K-GNP": lambda: F16DijkstraPlanner(terrain, planner_threats, aircraft=aircraft, xy_resolution=200.0, **common_cfg),
        "T-GnP": lambda: F16DijkstraPlannerRobust(
            terrain,
            planner_threats,
            aircraft=aircraft,
            xy_resolution=200.0,
            steps_per_primitive=5,
            use_neural_risk=T_GNP_USE_NEURAL_RISK,
            neural_risk_weight=T_GNP_NEURAL_RISK_WEIGHT,
            neural_risk_samples=T_GNP_NEURAL_RISK_SAMPLES,
            neural_corridor_half_width=T_GNP_NEURAL_CORRIDOR_HALF_WIDTH,
            speed_weight=T_GNP_SPEED_WEIGHT,
            stability_weight=T_GNP_STABILITY_WEIGHT,
            **common_cfg,
        ),
        "RL-Pilot": lambda: RLPlanner(
            terrain,
            planner_threats,
            aircraft,
            use_neural_risk=RL_PILOT_USE_NEURAL_RISK,
            neural_risk_weight=RL_PILOT_NEURAL_RISK_WEIGHT,
            neural_risk_samples=RL_PILOT_NEURAL_RISK_SAMPLES,
            neural_corridor_half_width=RL_PILOT_NEURAL_CORRIDOR_HALF_WIDTH,
            **common_cfg,
        ),
    }
    planner_factories["Neuro-Adaptive"] = lambda: NeuroAdaptivePlanner(
        terrain,
        planner_threats,
        planner_factories,
        safe_alt=safe_alt,
        use_dnn_tre=True,
    )
    planner_factories["Neuro-Adaptive-NoDNN"] = lambda: NeuroAdaptivePlanner(
        terrain,
        planner_threats,
        planner_factories,
        safe_alt=safe_alt,
        use_dnn_tre=False,
    )
    ablation_factories = {
        "RL-Pilot": planner_factories["RL-Pilot"],
        "RL-Pilot+DNN": lambda: RLPlanner(
            terrain,
            planner_threats,
            aircraft,
            use_neural_risk=True,
            neural_risk_weight=RL_PILOT_NEURAL_RISK_WEIGHT,
            neural_risk_samples=RL_PILOT_NEURAL_RISK_SAMPLES,
            neural_corridor_half_width=RL_PILOT_NEURAL_CORRIDOR_HALF_WIDTH,
            **common_cfg,
        ),
        "T-GnP": planner_factories["T-GnP"],
        "T-GnP+DNN": lambda: F16DijkstraPlannerRobust(
            terrain,
            planner_threats,
            aircraft=aircraft,
            xy_resolution=200.0,
            steps_per_primitive=5,
            use_neural_risk=True,
            neural_risk_weight=T_GNP_NEURAL_RISK_WEIGHT,
            neural_risk_samples=T_GNP_NEURAL_RISK_SAMPLES,
            neural_corridor_half_width=T_GNP_NEURAL_CORRIDOR_HALF_WIDTH,
            speed_weight=T_GNP_SPEED_WEIGHT,
            stability_weight=T_GNP_STABILITY_WEIGHT,
            **common_cfg,
        ),
    }

    print("\n--- Algorithm Selection ---")
    alg_keys = list(planner_factories.keys())
    for idx, key in enumerate(alg_keys):
        print(f"[{idx + 1}] {key}")
    print(f"[{len(alg_keys) + 1}] ALL ALGORITHMS (Thesis Benchmark)")
    print(f"[{len(alg_keys) + 2}] ALL ALGORITHMS (Benchmark Only / Comparative Outputs)")
    print(f"[{len(alg_keys) + 3}] DNN-TRE COMPARATIVE STUDY (Risk Maps)")
    print(f"[{len(alg_keys) + 4}] RL-PILOT ABLATION (Baseline vs DNN-TRE Corridor)")
    print(f"[{len(alg_keys) + 5}] RL/T-GNP DNN-TRE ABLATION (Baseline vs DNN-TRE Corridor)")
    alg_sel = int(input("\nSelect Algorithm: "))

    if alg_sel == len(alg_keys) + 3:
        run_comparative_study(terrain, planner_threats, scenario_id=scen["id"])
        sys.exit(0)

    benchmark_only = alg_sel == len(alg_keys) + 2
    planner_source = planner_factories
    run_output_dir = OUTPUT_DIR

    if alg_sel == len(alg_keys) + 4:
        selected_algs = ["RL-Pilot", "RL-Pilot+DNN"]
        planner_source = ablation_factories
        run_output_dir = os.path.join(OUTPUT_DIR, f"{scen['id']}_rl_dnntre_ablation")
    elif alg_sel == len(alg_keys) + 5:
        selected_algs = ["RL-Pilot", "RL-Pilot+DNN", "T-GnP", "T-GnP+DNN"]
        planner_source = ablation_factories
        run_output_dir = os.path.join(OUTPUT_DIR, f"{scen['id']}_rl_tgnp_dnntre_ablation")
    else:
        selected_algs = alg_keys if alg_sel in (len(alg_keys) + 1, len(alg_keys) + 2) else [alg_keys[alg_sel - 1]]

    os.makedirs(run_output_dir, exist_ok=True)
    runs_per_alg = STOCHASTIC_RUNS if len(selected_algs) > 1 else SINGLE_ALG_RUNS
    sim = Simulation(aircraft, terrain, lookahead_dist=1000.0, V_cruise=220.0)
    rng = np.random.default_rng()
    stochastic_reports = {}
    best_paths_for_plot = {}
    best_trajs_for_plot = {}
    best_runs_by_alg = {}

    for alg_name in selected_algs:
        print(f"\nProcessing: {alg_name} (Monte Carlo Analysis)")
        run_rows = []
        is_stochastic = alg_name in ["RRT-Star", "PSO", "RL-Pilot", "RL-Pilot+DNN"]
        fixed_metric_path = None

        if not is_stochastic:
            try:
                fixed_path = planner_source[alg_name]().plan(scen["start"], scen["goal"])
                if fixed_path:
                    fixed_metric_path = _prepare_metric_path(fixed_path, alg_name, scen["threats"])
            except Exception:
                fixed_metric_path = None

        for _ in range(runs_per_alg):
            if is_stochastic:
                try:
                    planner = planner_source[alg_name]()
                    t0 = time.time()
                    raw_path = planner.plan(scen["start"], scen["goal"], time_budget_s=current_time_budget)
                    t_plan = time.time() - t0
                    if raw_path is None and alg_name.startswith("RL-Pilot"):
                        rl_debug = getattr(planner, "last_plan_debug", {})
                        print(
                            "   > RL debug: "
                            f"weights_loaded={rl_debug.get('weights_loaded')} | "
                            f"guide_available={rl_debug.get('guide_available')} | "
                            f"steps_taken={rl_debug.get('steps_taken')} | "
                            f"fail_reason={rl_debug.get('fail_reason')} | "
                            f"weight_error={rl_debug.get('weight_error')}"
                        )
                    metric_path = _prepare_metric_path(raw_path, alg_name, scen["threats"])
                    if raw_path is not None and metric_path is None and alg_name.startswith("RL-Pilot"):
                        print(f"   > RL debug: raw path length={len(raw_path)} but metric path preparation returned None")
                except Exception:
                    metric_path, t_plan = None, 0.0
            else:
                metric_path, t_plan = fixed_metric_path, 0.0

            if metric_path:
                for threat in terrain.threats:
                    if hasattr(threat, "reset"):
                        threat.reset()
                    else:
                        threat.x, threat.y = threat.x_center, threat.y_center

                disturb = {"wind_xy_mps": rng.normal(0, 5, 2), "phi_noise_std_deg": 0.5, "gamma_noise_std_deg": 0.5}
                aircraft.V = aircraft.V_cruise
                traj, history = sim.run(metric_path, goal_radius=goal_radius, goal_center=scen["goal"][:2], disturbance=disturb, rng=rng)

                gm = compute_guidance_metrics(traj, metric_path, history=history, phi_max_deg=60.0)
                _, status, viols = analyze_flight_result(traj, scen["threats"], target_pos, goal_radius)
                run_rows.append({
                    "sim_status": status,
                    "Path": path_length(metric_path) / 1000.0,
                    "Time": len(traj) * 0.25,
                    "Risk": calculate_risk_exposure(traj, scen["threats"]) / 1e6,
                    "AGL": calculate_min_terrain_clearance(traj, terrain),
                    "ThrFail": viols,
                    "Track": gm.track_mean_m,
                    "TrackMax": gm.track_max_m,
                    "Calc": t_plan,
                    "SatRatio": gm.sat_ratio,
                    "PhiMax": gm.phi_max_deg_observed,
                    "VMin": gm.v_min_observed,
                    "_traj": traj,
                    "_history": history,
                    "_metric_path": metric_path,
                })
            else:
                run_rows.append({"sim_status": "NO_PATH"})

        stochastic_reports[alg_name] = run_rows
        succ_runs = [r for r in run_rows if r.get("sim_status") == "SUCCESS"]
        print(f"   > Success Rate: {len(succ_runs)}/{runs_per_alg}")

        if succ_runs:
            best_run = sorted(succ_runs, key=lambda row: (row.get("Risk", 1e9), row.get("Track", 1e9)))[0]
            best_runs_by_alg[alg_name] = best_run
            best_paths_for_plot[alg_name] = best_run["_metric_path"]
            best_trajs_for_plot[alg_name] = best_run["_traj"]

            if ENABLE_VISUALIZATION and not benchmark_only:
                _save_demo_assets(scen, alg_name, terrain, scen["threats"], best_run, target_pos, goal_radius, output_dir=run_output_dir)

    _print_benchmark_table(stochastic_reports, scen["name"])

    summary_rows = summarize_reports(stochastic_reports)
    csv_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-BenchmarkSummary_TABLE_{TODAY_STR}.csv")
    tex_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-BenchmarkSummary_TABLE_{TODAY_STR}.tex")
    fig_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-BenchmarkSummary_GRAPH_{TODAY_STR}.png")
    save_benchmark_csv(summary_rows, csv_path)
    save_benchmark_latex(
        summary_rows,
        tex_path,
        caption=f"Benchmark summary for {scen['name']}. Values are averaged over successful runs.",
        label=f"tab:{scen['id'].lower()}_benchmark_summary",
    )
    plot_benchmark_summary(summary_rows, scen["id"], fig_path)
    stochastic_csv_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-StochasticRunOutcomes_TABLE_{TODAY_STR}.csv")
    stochastic_fig_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-StochasticRunOutcomes_GRAPH_{TODAY_STR}.png")
    save_stochastic_run_csv(stochastic_reports, stochastic_csv_path)
    plot_stochastic_run_outcomes(stochastic_reports, scen["id"], stochastic_fig_path)

    if ENABLE_VISUALIZATION and len(best_trajs_for_plot) > 0:
        save_pth = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-ComparativeTrajectories_FIG_{TODAY_STR}.png")
        plot_professional_riskmap(
            terrain,
            scen["threats"],
            paths_dict=best_paths_for_plot,
            traj_dict=best_trajs_for_plot,
            title=f"Comparative Benchmark Analysis ({scen['id']})",
            save_path=save_pth,
            show_plot=False,
            goal_center=target_pos,
            goal_radius=goal_radius,
        )

    print(f"Results and charts saved to: {run_output_dir}")


if __name__ == "__main__":
    main()
