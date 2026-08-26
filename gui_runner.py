import copy
import os
import time
from datetime import datetime

import numpy as np

from wcci_conference_project.aircraft.jet_model import Aircraft
from wcci_conference_project.ai_modules.generate_real_dataset import resolve_dem_path
from wcci_conference_project.environment.terrain import Terrain
from wcci_conference_project.evaluation.metrics import (
    calculate_min_terrain_clearance,
    calculate_risk_exposure,
    compute_guidance_metrics,
    path_length,
)
from wcci_conference_project.main import (
    _prepare_metric_path,
    _print_benchmark_table,
    _save_demo_assets,
    analyze_flight_result,
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
from wcci_conference_project.visualization.benchmark_outputs import (
    plot_benchmark_summary,
    plot_stochastic_run_outcomes,
    save_benchmark_csv,
    save_benchmark_latex,
    save_stochastic_run_csv,
    summarize_reports,
)
from wcci_conference_project.visualization.plot_riskmap import plot_professional_riskmap


DEFAULT_BASE_OUTPUT_DIR = os.environ.get(
    "WCCI_OUTPUT_DIR",
    r"C:\Users\DELL\PycharmProjects\thesis_project\wcci_conference_project\outputs\THESIS_FINAL_RESULTS",
)


def _log(log_fn, message):
    if log_fn is not None:
        log_fn(message)


def _scenario_index_by_id(scenarios, scenario_id):
    for idx, scen in enumerate(scenarios):
        if scen["id"] == scenario_id:
            return idx
    return 0


def _build_common_cfg(safe_alt):
    return {
        "check_resolution": 50.0,
        "threat_weight": 20.0,
        "safe_alt": safe_alt,
        "goal_tolerance": 3000.0,
    }


def _make_planner_factory(alg_name, terrain, planner_threats, aircraft, safe_alt, config):
    dnn_on = bool(config.get("enable_dnn", False))
    common_cfg = _build_common_cfg(safe_alt)

    if alg_name == "A-Star":
        return lambda: AstarPlanner(
            terrain,
            planner_threats,
            step=100,
            heuristic_weight=1.5,
            use_neural_risk=dnn_on,
            neural_risk_weight=float(config.get("common_dnn_weight", 120.0)),
            neural_risk_samples=int(config.get("common_dnn_samples", 3)),
            neural_corridor_half_width=float(config.get("common_dnn_width", 1800.0)),
            **common_cfg,
        )
    if alg_name == "Dijkstra":
        return lambda: DijkstraPlanner(
            terrain,
            planner_threats,
            step=100,
            max_iter=2000000,
            use_neural_risk=dnn_on,
            neural_risk_weight=float(config.get("common_dnn_weight", 120.0)),
            neural_risk_samples=int(config.get("common_dnn_samples", 3)),
            neural_corridor_half_width=float(config.get("common_dnn_width", 1800.0)),
            **common_cfg,
        )
    if alg_name == "RRT-Star":
        return lambda: RRTStarPlanner(
            terrain,
            planner_threats,
            max_iter=25000,
            step_len=2500.0,
            goal_sample_rate=0.15,
            search_radius=3000.0,
            use_neural_risk=dnn_on,
            neural_risk_weight=float(config.get("common_dnn_weight", 120.0)),
            neural_risk_samples=int(config.get("common_dnn_samples", 3)),
            neural_corridor_half_width=float(config.get("common_dnn_width", 1800.0)),
            **common_cfg,
        )
    if alg_name == "PSO":
        return lambda: PSOPlanner(
            terrain,
            planner_threats,
            num_particles=150,
            max_iter=60000,
            waypoint_spread_m=6000.0,
            num_waypoints=8,
            use_neural_risk=dnn_on,
            neural_risk_weight=float(config.get("common_dnn_weight", 120.0)),
            neural_risk_samples=int(config.get("common_dnn_samples", 3)),
            neural_corridor_half_width=float(config.get("common_dnn_width", 1800.0)),
            **common_cfg,
        )
    if alg_name == "K-GNP":
        return lambda: F16DijkstraPlanner(
            terrain,
            planner_threats,
            aircraft=aircraft,
            xy_resolution=200.0,
            use_neural_risk=dnn_on,
            neural_risk_weight=float(config.get("common_dnn_weight", 120.0)),
            neural_risk_samples=int(config.get("common_dnn_samples", 3)),
            neural_corridor_half_width=float(config.get("common_dnn_width", 1800.0)),
            **common_cfg,
        )
    if alg_name == "T-GnP":
        return lambda: F16DijkstraPlannerRobust(
            terrain,
            planner_threats,
            aircraft=aircraft,
            xy_resolution=200.0,
            steps_per_primitive=5,
            use_neural_risk=dnn_on,
            neural_risk_weight=float(config.get("tgnp_dnn_weight", 300.0)),
            neural_risk_samples=int(config.get("tgnp_dnn_samples", 3)),
            neural_corridor_half_width=float(config.get("tgnp_dnn_width", 1800.0)),
            speed_weight=float(config.get("tgnp_speed_weight", 0.35)),
            stability_weight=float(config.get("tgnp_stability_weight", 0.55)),
            **common_cfg,
        )
    if alg_name == "RL-Pilot":
        return lambda: RLPlanner(
            terrain,
            planner_threats,
            aircraft,
            use_neural_risk=dnn_on,
            neural_risk_weight=float(config.get("rl_dnn_weight", 0.18)),
            neural_risk_samples=int(config.get("rl_dnn_samples", 3)),
            neural_corridor_half_width=float(config.get("rl_dnn_width", 1800.0)),
            **common_cfg,
        )
    if alg_name == "Neuro-Adaptive":
        return lambda: NeuroAdaptivePlanner(
            terrain,
            planner_threats,
            planner_factories={},
            safe_alt=safe_alt,
            use_dnn_tre=True,
        )
    if alg_name == "Neuro-Adaptive-NoDNN":
        return lambda: NeuroAdaptivePlanner(
            terrain,
            planner_threats,
            planner_factories={},
            safe_alt=safe_alt,
            use_dnn_tre=False,
        )
    raise KeyError(f"Unsupported planner: {alg_name}")


def _make_variant_factory(base_alg, terrain, planner_threats, aircraft, safe_alt, config, force_dnn):
    cfg = dict(config)
    cfg["enable_dnn"] = bool(force_dnn)
    return _make_planner_factory(base_alg, terrain, planner_threats, aircraft, safe_alt, cfg)


def _get_selected_algorithms(config):
    planner_name = config.get("planner_name", "RL-Pilot")
    compare = bool(config.get("compare_baseline_vs_dnn", False))
    all_planners = ["A-Star", "Dijkstra", "RRT-Star", "PSO", "K-GNP", "T-GnP", "RL-Pilot", "Neuro-Adaptive", "Neuro-Adaptive-NoDNN"]

    if planner_name == "ALL ALGORITHMS":
        return all_planners

    if compare and planner_name == "Neuro-Adaptive":
        return ["Neuro-Adaptive-NoDNN", "Neuro-Adaptive"]

    if compare and planner_name == "Neuro-Adaptive-NoDNN":
        return ["Neuro-Adaptive-NoDNN", "Neuro-Adaptive"]

    if compare and planner_name != "Neuro-Adaptive-NoDNN":
        return [planner_name, f"{planner_name}+DNN"]

    return [planner_name]


def run_benchmark_from_config(config, log_fn=None):
    os.makedirs(DEFAULT_BASE_OUTPUT_DIR, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    date_output_dir = os.path.join(DEFAULT_BASE_OUTPUT_DIR, today_str)
    os.makedirs(date_output_dir, exist_ok=True)

    scenario_id = config.get("scenario_id", "S1_Base")
    planner_name = config.get("planner_name", "RL-Pilot")
    compare = bool(config.get("compare_baseline_vs_dnn", False))
    benchmark_only = bool(config.get("benchmark_only", False))
    enable_visualization = bool(config.get("enable_visualization", True))
    runs_override = int(config.get("runs_per_alg", 30))
    time_budget_base = max(0.5, float(config.get("time_budget_base", 2.0)))

    output_tag = config.get("output_tag", "").strip()
    if output_tag:
        run_output_dir = os.path.join(date_output_dir, output_tag)
    elif compare:
        safe_name = planner_name.replace("*", "Star").replace(" ", "_").replace("+", "plus")
        run_output_dir = os.path.join(date_output_dir, f"{scenario_id}_{safe_name}_dnn_compare")
    else:
        safe_name = planner_name.replace("*", "Star").replace(" ", "_").replace("+", "plus")
        run_output_dir = os.path.join(date_output_dir, f"{scenario_id}_{safe_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    dem_path = resolve_dem_path(os.environ.get("WCCI_DEM_PATH"))
    if dem_path is None:
        raise FileNotFoundError("DEM file could not be resolved. Set WCCI_DEM_PATH or place dem_utm.tif in the workspace.")

    _log(log_fn, "Initializing benchmark environment...")
    _log(log_fn, f"DEM Path: {dem_path}")
    terrain = Terrain(dem_path)
    aircraft = Aircraft(V_cruise=220.0)
    safe_alt = max(np.nanmax(terrain.data) + 200.0, 600.0)
    scenarios = get_all_scenarios(safe_alt)

    scen_idx = _scenario_index_by_id(scenarios, scenario_id)
    scen = scenarios[scen_idx]
    current_time_budget = max(0.5, time_budget_base * {0: 1.0, 1: 1.5, 2: 2.5, 3: 3.0, 4: 1.0}.get(scen_idx, 1.0))

    terrain.threats = scen["threats"]
    planner_threats = copy.deepcopy(scen["threats"])
    for threat in planner_threats:
        threat.radius += 200.0

    target_pos = np.array(scen["goal"][:2])
    goal_radius = 3000.0

    selected_algs = _get_selected_algorithms(config)
    runs_per_alg = runs_override if len(selected_algs) > 1 else int(config.get("single_alg_runs", 12))
    sim = Simulation(aircraft, terrain, lookahead_dist=1000.0, V_cruise=220.0)
    rng = np.random.default_rng()
    stochastic_reports = {}
    best_paths_for_plot = {}
    best_trajs_for_plot = {}

    base_factories = {
        "A-Star": _make_planner_factory("A-Star", terrain, planner_threats, aircraft, safe_alt, config),
        "Dijkstra": _make_planner_factory("Dijkstra", terrain, planner_threats, aircraft, safe_alt, config),
        "RRT-Star": _make_planner_factory("RRT-Star", terrain, planner_threats, aircraft, safe_alt, config),
        "PSO": _make_planner_factory("PSO", terrain, planner_threats, aircraft, safe_alt, config),
        "K-GNP": _make_planner_factory("K-GNP", terrain, planner_threats, aircraft, safe_alt, config),
        "T-GnP": _make_planner_factory("T-GnP", terrain, planner_threats, aircraft, safe_alt, config),
        "RL-Pilot": _make_planner_factory("RL-Pilot", terrain, planner_threats, aircraft, safe_alt, config),
    }
    base_factories["Neuro-Adaptive"] = lambda: NeuroAdaptivePlanner(
        terrain,
        planner_threats,
        base_factories,
        safe_alt=safe_alt,
        use_dnn_tre=True,
    )
    base_factories["Neuro-Adaptive-NoDNN"] = lambda: NeuroAdaptivePlanner(
        terrain,
        planner_threats,
        base_factories,
        safe_alt=safe_alt,
        use_dnn_tre=False,
    )

    planner_factories = dict(base_factories)
    if compare and planner_name == "Neuro-Adaptive":
        planner_factories["Neuro-Adaptive"] = lambda: NeuroAdaptivePlanner(
            terrain,
            planner_threats,
            base_factories,
            safe_alt=safe_alt,
            use_dnn_tre=True,
        )
        planner_factories["Neuro-Adaptive-NoDNN"] = lambda: NeuroAdaptivePlanner(
            terrain,
            planner_threats,
            base_factories,
            safe_alt=safe_alt,
            use_dnn_tre=False,
        )
    elif compare and planner_name != "Neuro-Adaptive-NoDNN":
        planner_factories[f"{planner_name}+DNN"] = _make_variant_factory(
            planner_name,
            terrain,
            planner_threats,
            aircraft,
            safe_alt,
            config,
            force_dnn=True,
        )
        planner_factories[planner_name] = _make_variant_factory(
            planner_name,
            terrain,
            planner_threats,
            aircraft,
            safe_alt,
            config,
            force_dnn=False,
        )

    _log(log_fn, f"Scenario: {scen['name']}")
    _log(log_fn, f"Selected planners: {', '.join(selected_algs)}")
    _log(log_fn, f"Time budget base: {time_budget_base:.2f}s | scenario-adjusted: {current_time_budget:.2f}s")
    _log(log_fn, f"Output directory: {run_output_dir}")

    for alg_name in selected_algs:
        _log(log_fn, f"\nProcessing: {alg_name} (Monte Carlo Analysis)")
        run_rows = []
        is_stochastic = alg_name in ["RRT-Star", "PSO", "RL-Pilot", "RL-Pilot+DNN"]
        fixed_metric_path = None

        if not is_stochastic:
            try:
                fixed_path = planner_factories[alg_name]().plan(scen["start"], scen["goal"])
                if fixed_path:
                    fixed_metric_path = _prepare_metric_path(fixed_path, alg_name, scen["threats"])
                else:
                    _log(log_fn, f"   > Planner returned no path during fixed planning: {alg_name}")
            except Exception as exc:
                _log(log_fn, f"   > Fixed planning error in {alg_name}: {exc}")
                fixed_metric_path = None

        for _ in range(runs_per_alg):
            if is_stochastic:
                try:
                    planner = planner_factories[alg_name]()
                    t0 = time.time()
                    raw_path = planner.plan(scen["start"], scen["goal"], time_budget_s=current_time_budget)
                    t_plan = time.time() - t0
                    if raw_path is None and alg_name.startswith("RL-Pilot"):
                        rl_debug = getattr(planner, "last_plan_debug", {})
                        _log(
                            log_fn,
                            "   > RL debug: "
                            f"weights_loaded={rl_debug.get('weights_loaded')} | "
                            f"guide_available={rl_debug.get('guide_available')} | "
                            f"steps_taken={rl_debug.get('steps_taken')} | "
                            f"fail_reason={rl_debug.get('fail_reason')} | "
                            f"weight_error={rl_debug.get('weight_error')}",
                        )
                    metric_path = _prepare_metric_path(raw_path, alg_name, scen["threats"])
                    if raw_path is not None and metric_path is None and alg_name.startswith("RL-Pilot"):
                        _log(log_fn, f"   > RL debug: raw path length={len(raw_path)} but metric path preparation returned None")
                except Exception as exc:
                    _log(log_fn, f"   > Stochastic planning error in {alg_name}: {exc}")
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
        _log(log_fn, f"   > Success Rate: {len(succ_runs)}/{runs_per_alg}")

        if succ_runs:
            best_run = sorted(succ_runs, key=lambda row: (row.get("Risk", 1e9), row.get("Track", 1e9)))[0]
            best_paths_for_plot[alg_name] = best_run["_metric_path"]
            best_trajs_for_plot[alg_name] = best_run["_traj"]

            if enable_visualization and not benchmark_only:
                _save_demo_assets(
                    scen,
                    alg_name,
                    terrain,
                    scen["threats"],
                    best_run,
                    target_pos,
                    goal_radius,
                    output_dir=run_output_dir,
                )

    _print_benchmark_table(stochastic_reports, scen["name"])

    summary_rows = summarize_reports(stochastic_reports)
    csv_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-BenchmarkSummary_TABLE_{today_str}.csv")
    tex_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-BenchmarkSummary_TABLE_{today_str}.tex")
    fig_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-BenchmarkSummary_GRAPH_{today_str}.png")
    save_benchmark_csv(summary_rows, csv_path)
    save_benchmark_latex(
        summary_rows,
        tex_path,
        caption=f"Benchmark summary for {scen['name']}. Values are averaged over successful runs.",
        label=f"tab:{scen['id'].lower()}_benchmark_summary",
    )
    plot_benchmark_summary(summary_rows, scen["id"], fig_path)

    stochastic_csv_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-StochasticRunOutcomes_TABLE_{today_str}.csv")
    stochastic_fig_path = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-StochasticRunOutcomes_GRAPH_{today_str}.png")
    save_stochastic_run_csv(stochastic_reports, stochastic_csv_path)
    plot_stochastic_run_outcomes(stochastic_reports, scen["id"], stochastic_fig_path)

    if enable_visualization and len(best_trajs_for_plot) > 0:
        save_pth = os.path.join(run_output_dir, f"ThesisFinal_{scen['id']}_OPT-ComparativeTrajectories_FIG_{today_str}.png")
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

    _log(log_fn, f"\nResults and charts saved to: {run_output_dir}")
    return {"output_dir": run_output_dir, "summary_rows": summary_rows}
