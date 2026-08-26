from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def summarize_reports(stochastic_reports: Dict[str, List[dict]]) -> List[dict]:
    rows_out = []
    metric_keys = [
        "Path",
        "Time",
        "Risk",
        "AGL",
        "ThrFail",
        "Track",
        "TrackMax",
        "Calc",
        "SatRatio",
        "PhiMax",
        "VMin",
    ]

    for alg, rows in stochastic_reports.items():
        succ_rows = [r for r in rows if r.get("sim_status") == "SUCCESS"]
        total_runs = len(rows)
        succ_count = len(succ_rows)
        row = {
            "Algorithm": alg,
            "SuccessCount": succ_count,
            "RunCount": total_runs,
            "SuccessRatePct": (100.0 * succ_count / total_runs) if total_runs else 0.0,
        }
        if succ_rows:
            for key in metric_keys:
                vals = [r[key] for r in succ_rows if key in r and r[key] is not None]
                row[key] = float(np.mean(vals)) if vals else float("nan")
        else:
            for key in metric_keys:
                row[key] = float("nan")
        rows_out.append(row)

    return rows_out


def save_benchmark_csv(summary_rows: List[dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = [
        "Algorithm",
        "SuccessCount",
        "RunCount",
        "SuccessRatePct",
        "Path",
        "Time",
        "Risk",
        "AGL",
        "ThrFail",
        "Track",
        "TrackMax",
        "Calc",
        "SatRatio",
        "PhiMax",
        "VMin",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def save_benchmark_latex(summary_rows: List[dict], save_path: str, caption: str, label: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\hline",
        r"Algorithm & Success & Path (km) & Time (s) & Risk (M) & Min AGL (m) & Track Mean (m) & Track Max (m) & Calc. (s) \\",
        r"\hline",
    ]

    for row in summary_rows:
        success = f"{int(row['SuccessCount'])}/{int(row['RunCount'])}"
        if np.isnan(row["Path"]):
            lines.append(f"{row['Algorithm']} & {success} & N/A & N/A & N/A & N/A & N/A & N/A & N/A \\\\")
        else:
            lines.append(
                f"{row['Algorithm']} & {success} & "
                f"{row['Path']:.2f} & {row['Time']:.2f} & {row['Risk']:.2f} & "
                f"{row['AGL']:.0f} & {row['Track']:.2f} & {row['TrackMax']:.2f} & {row['Calc']:.2f} \\\\"
            )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_benchmark_summary(summary_rows: List[dict], scenario_id: str, save_path: str) -> None:
    if not summary_rows:
        return

    algs = [row["Algorithm"] for row in summary_rows]
    success = [row["SuccessRatePct"] for row in summary_rows]
    path = [row["Path"] if not np.isnan(row["Path"]) else 0.0 for row in summary_rows]
    track = [row["Track"] if not np.isnan(row["Track"]) else 0.0 for row in summary_rows]
    track_max = [row["TrackMax"] if not np.isnan(row["TrackMax"]) else 0.0 for row in summary_rows]
    phi_max = [row["PhiMax"] if not np.isnan(row["PhiMax"]) else 0.0 for row in summary_rows]
    sat = [100.0 * row["SatRatio"] if not np.isnan(row["SatRatio"]) else 0.0 for row in summary_rows]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    if len(algs) == 1:
        title = f"MSc Thesis Benchmark Summary ({scenario_id}) - {algs[0]}"
    elif len(algs) <= 3:
        title = f"MSc Thesis Benchmark Summary ({scenario_id}) - " + ", ".join(algs)
    else:
        title = f"MSc Thesis Benchmark Summary ({scenario_id}) - Multiple Algorithms"
    fig.suptitle(title, weight="bold", y=0.98)

    panels = [
        ("Success Rate (%)", success, "#2f9e44"),
        ("Path Length (km)", path, "#1c7ed6"),
        ("Track Mean (m)", track, "#c2255c"),
        ("Track Max (m)", track_max, "#e03131"),
        ("Max Bank Angle (deg)", phi_max, "#f08c00"),
        ("Sat. Ratio (%)", sat, "#495057"),
    ]

    for ax, (title, values, color) in zip(axes.flat, panels):
        bars = ax.bar(algs, values, color=color, alpha=0.9)
        ax.set_title(title, weight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=11, pad=2)
        ax.tick_params(axis="y", labelsize=9)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_stochastic_run_csv(stochastic_reports: Dict[str, List[dict]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = ["Algorithm", "RunIndex", "Status"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alg, rows in stochastic_reports.items():
            for idx, row in enumerate(rows, start=1):
                writer.writerow({
                    "Algorithm": alg,
                    "RunIndex": idx,
                    "Status": row.get("sim_status", "UNKNOWN"),
                })


def plot_stochastic_run_outcomes(stochastic_reports: Dict[str, List[dict]], scenario_id: str, save_path: str) -> None:
    stochastic_algs = []
    for alg, rows in stochastic_reports.items():
        statuses = [row.get("sim_status", "UNKNOWN") for row in rows]
        if len(rows) > 1 and len(set(statuses)) >= 1:
            stochastic_algs.append((alg, statuses))

    stochastic_algs = [
        item
        for item in stochastic_algs
        if item[0] in {"RRT-Star", "PSO", "RL-Pilot", "RL-Pilot+DNN"}
    ]
    if not stochastic_algs:
        return

    status_map = {"SUCCESS": 1.0, "FAIL_DYNAMICS": 0.0, "FAIL_THREAT": -1.0, "NO_PATH": -2.0, "UNKNOWN": -3.0}
    colors = {
        "SUCCESS": "#2f9e44",
        "FAIL_DYNAMICS": "#f08c00",
        "FAIL_THREAT": "#d00000",
        "NO_PATH": "#6c757d",
        "UNKNOWN": "#adb5bd",
    }

    fig, axes = plt.subplots(len(stochastic_algs), 1, figsize=(14, 2.5 * len(stochastic_algs)), squeeze=False)
    fig.suptitle(f"Stochastic Run Outcomes ({scenario_id})", weight="bold", y=0.98)

    for ax, (alg, statuses) in zip(axes.flat, stochastic_algs):
        run_ids = np.arange(1, len(statuses) + 1)
        yvals = np.ones_like(run_ids)
        bar_colors = [colors.get(s, colors["UNKNOWN"]) for s in statuses]
        ax.bar(run_ids, yvals, color=bar_colors, edgecolor="black", linewidth=0.5)
        succ_flags = np.array([1 if s == "SUCCESS" else 0 for s in statuses], dtype=int)
        cum_succ = np.cumsum(succ_flags)
        ax2 = ax.twinx()
        ax2.plot(run_ids, cum_succ, color="black", marker="o", linewidth=1.8, markersize=3.5, label="Cumulative Success")
        ax2.set_ylabel("Cum. Success", fontsize=9)
        ax2.tick_params(axis="y", labelsize=8)

        ax.set_title(alg, loc="left", weight="bold")
        ax.set_xlim(0.3, len(statuses) + 0.7)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([])
        ax.set_xlabel("Run Index")
        ax.grid(axis="x", linestyle=":", alpha=0.2)

        for x, status in zip(run_ids, statuses):
            label = "S" if status == "SUCCESS" else ("D" if status == "FAIL_DYNAMICS" else ("T" if status == "FAIL_THREAT" else "N"))
            ax.text(x, 0.5, label, ha="center", va="center", color="white", fontsize=8, weight="bold")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["SUCCESS"], ec="black", lw=0.5, label="Success"),
        plt.Rectangle((0, 0), 1, 1, color=colors["FAIL_DYNAMICS"], ec="black", lw=0.5, label="Fail Dynamics"),
        plt.Rectangle((0, 0), 1, 1, color=colors["FAIL_THREAT"], ec="black", lw=0.5, label="Fail Threat"),
        plt.Rectangle((0, 0), 1, 1, color=colors["NO_PATH"], ec="black", lw=0.5, label="No Path"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.93))
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
