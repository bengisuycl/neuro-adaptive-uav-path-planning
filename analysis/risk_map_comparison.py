import os
import time
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from wcci_conference_project.ai_modules.neural_risk_map import NeuralRiskMapper
from wcci_conference_project.ai_modules.generate_real_dataset import calculate_glos_risk

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "figure.dpi": 300,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
})


def method_a_geometric(terrain, threats, downsample=5, flight_agl_m=800.0):
    t0 = time.time()
    dem_data = np.asarray(terrain.data, dtype=float)
    height, width = dem_data.shape
    h_ds, w_ds = height // downsample, width // downsample
    risk_map = np.zeros((h_ds, w_ds), dtype=float)
    t_transform = terrain.transform

    for r in range(h_ds):
        for c in range(w_ds):
            px = t_transform.c + (c * downsample) * t_transform.a + (r * downsample) * t_transform.b
            py = t_transform.f + (c * downsample) * t_transform.d + (r * downsample) * t_transform.e
            terr_h = terrain.get_height(px, py)
            if np.isnan(terr_h):
                continue
            pz = terr_h + float(flight_agl_m)
            risk_map[r, c] = calculate_glos_risk(terrain, px, py, pz, threats)

    return risk_map, time.time() - t0


def _print_map_stats(name, arr):
    arr = np.asarray(arr, dtype=float)
    print(
        f"      > {name} stats: "
        f"min={float(np.min(arr)):.4f}, "
        f"max={float(np.max(arr)):.4f}, "
        f"mean={float(np.mean(arr)):.4f}"
    )


def _compute_map_metrics(map_a, map_b):
    a = np.asarray(map_a, dtype=float).ravel()
    b = np.asarray(map_b, dtype=float).ravel()
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    max_a = float(np.max(a))
    max_b = float(np.max(b))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "baseline_mean": mean_a,
        "dnntre_mean": mean_b,
        "baseline_max": max_a,
        "dnntre_max": max_b,
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
    }


def _save_risk_comparison_outputs(out_dir, scenario_id, metrics, time_a, time_b, speedup):
    prefix = f"ThesisFinal_{scenario_id}_OPT-DNN-TRE-Comparison_TABLE"
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    tex_path = os.path.join(out_dir, f"{prefix}.tex")

    fieldnames = [
        "BaselineTime_s",
        "DNNTRETime_s",
        "Speedup_x",
        "BaselineMean",
        "DNNTREMean",
        "BaselineMax",
        "DNNTREMax",
        "MAE",
        "RMSE",
        "Correlation",
    ]
    row = {
        "BaselineTime_s": time_a,
        "DNNTRETime_s": time_b,
        "Speedup_x": speedup,
        "BaselineMean": metrics["baseline_mean"],
        "DNNTREMean": metrics["dnntre_mean"],
        "BaselineMax": metrics["baseline_max"],
        "DNNTREMax": metrics["dnntre_max"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "Correlation": metrics["corr"],
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{DNN-TRE tactical risk mapping comparison against the geometric baseline.}",
        r"\label{tab:dnn_tre_risk_comparison}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Metric & Baseline / Value & DNN-TRE / Value \\",
        r"\hline",
        f"Processing Time (s) & {time_a:.2f} & {time_b:.2f} \\\\",
        f"Mean Risk & {metrics['baseline_mean']:.4f} & {metrics['dnntre_mean']:.4f} \\\\",
        f"Max Risk & {metrics['baseline_max']:.4f} & {metrics['dnntre_max']:.4f} \\\\",
        f"Speedup (x) & \\multicolumn{{2}}{{c}}{{{speedup:.2f}}} \\\\",
        f"MAE & \\multicolumn{{2}}{{c}}{{{metrics['mae']:.4f}}} \\\\",
        f"RMSE & \\multicolumn{{2}}{{c}}{{{metrics['rmse']:.4f}}} \\\\",
        f"Correlation & \\multicolumn{{2}}{{c}}{{{metrics['corr']:.4f}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_comparative_study(terrain, threats, scenario_id="Sx"):
    print("\n" + "=" * 85)
    print(" Tactical Risk Mapping Computation Benchmark ".center(85, "="))
    print("=" * 85)

    downsample_rate = 5
    from datetime import datetime

    today_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "THESIS_FINAL_RESULTS",
        today_str,
    )
    os.makedirs(out_dir, exist_ok=True)

    print("[1/2] Running Method A: Conventional Geometric Baseline (GLOS)...")
    map_a, time_a = method_a_geometric(terrain, threats, downsample=downsample_rate)
    print(f"      > Processing Time : {time_a:.4f} seconds")
    _print_map_stats("Method A", map_a)

    print("\n[2/2] Running Method B: Proposed DNN-Based Tactical Risk Estimator (DNN-TRE)...")
    neural_net = NeuralRiskMapper()
    t0 = time.time()

    dem_data = np.asarray(terrain.data, dtype=float)
    height, width = dem_data.shape
    h_ds, w_ds = height // downsample_rate, width // downsample_rate
    t_transform = terrain.transform

    map_b = np.zeros((h_ds, w_ds), dtype=float)
    if neural_net.ready:
        cols = np.arange(w_ds) * downsample_rate
        rows = np.arange(h_ds) * downsample_rate
        c_grid, r_grid = np.meshgrid(cols, rows)
        px = t_transform.c + c_grid * t_transform.a + r_grid * t_transform.b
        py = t_transform.f + c_grid * t_transform.d + r_grid * t_transform.e
        points = np.column_stack((px.ravel(), py.ravel()))

        if len(points) > 0:
            preds = neural_net.get_risk_at_points(points, terrain, threats)
            map_b = np.asarray(preds, dtype=float).reshape((h_ds, w_ds))

    time_b = time.time() - t0
    print(f"      > Processing Time : {time_b:.4f} seconds")
    _print_map_stats("Method B", map_b)

    if float(np.std(map_b)) < 1e-3:
        print("      > WARNING: Method B predictions are nearly constant; model/data mismatch is likely.")

    speedup = time_a / max(1e-9, time_b)
    metrics = _compute_map_metrics(map_a, map_b)
    print("-" * 85)
    print(f"    * Baseline Time : {time_a:.4f} s")
    print(f"    * DNN-TRE Time  : {time_b:.4f} s")
    print(f"    * Speedup Ratio : {speedup:.1f}x FASTER")
    print(f"    * MAE           : {metrics['mae']:.4f}")
    print(f"    * RMSE          : {metrics['rmse']:.4f}")
    print(f"    * Correlation   : {metrics['corr']:.4f}")
    print("=" * 85 + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    extent = [t_transform.c, t_transform.c + width * t_transform.a, t_transform.f + height * t_transform.e, t_transform.f]
    vmax_val = max(float(np.max(map_a)), float(np.max(map_b)), 1.0)

    im0 = axes[0].imshow(map_a, cmap="inferno", origin="upper", extent=extent, alpha=0.95, vmin=0, vmax=vmax_val)
    axes[0].set_title(f"Method A: Geometric Baseline\nCalc Time: {time_a:.2f} s", weight="bold", pad=12)
    axes[0].set_xlabel("Easting (UTM-m)")
    axes[0].set_ylabel("Northing (UTM-m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Risk Exposure Score")

    im1 = axes[1].imshow(map_b, cmap="inferno", origin="upper", extent=extent, alpha=0.95, vmin=0, vmax=vmax_val)
    axes[1].set_title(
        f"Method B: Proposed DNN-TRE\nCalc Time: {time_b:.2f} s (Speedup: {speedup:.1f}x)",
        weight="bold",
        pad=12,
    )
    axes[1].set_xlabel("Easting (UTM-m)")
    axes[1].set_ylabel("Northing (UTM-m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Predicted Risk Score")

    for ax in axes:
        for th in threats:
            circle = Circle((th.x, th.y), getattr(th, "radius", 2000.0), edgecolor="cyan", facecolor="none", linestyle="--", linewidth=2.0, alpha=0.9)
            ax.add_patch(circle)
            ax.plot(th.x, th.y, "c+", markersize=12, markeredgewidth=2.0)

    plt.suptitle("MSc Thesis: Real-Time Tactical Risk Mapping Comparison", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"ThesisFinal_{scenario_id}_OPT-DNN-TRE-Comparison_FIG_{today_str}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"    > Enhanced comparative map saved to: {save_path}")
    plt.close(fig)

    _save_risk_comparison_outputs(out_dir, scenario_id, metrics, time_a, time_b, speedup)

    return map_a, map_b, time_a, time_b
