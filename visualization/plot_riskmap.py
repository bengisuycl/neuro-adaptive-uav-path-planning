from __future__ import annotations

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.patches import Circle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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


def _resolve_alg_color(name):
    if name in ALG_COLORS:
        return ALG_COLORS[name]
    if "+DNN" in str(name):
        base = str(name).replace("+DNN", "")
        return ALG_COLORS.get(base, "white")
    return "white"


def _resample_xy_polyline(arr, n_samples=120):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr[:, :2] if arr.ndim == 2 else np.empty((0, 2), dtype=float)

    xy = arr[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return np.repeat(xy[:1], n_samples, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n_samples)
    out = []
    j = 0
    for d in targets:
        while j < len(seg) - 1 and cum[j + 1] < d:
            j += 1
        local_len = max(cum[j + 1] - cum[j], 1e-6)
        t = (d - cum[j]) / local_len
        pt = xy[j] + t * (xy[j + 1] - xy[j])
        out.append(pt)
    return np.asarray(out, dtype=float)


def _mean_path_distance(arr_a, arr_b):
    a = _resample_xy_polyline(arr_a)
    b = _resample_xy_polyline(arr_b)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _max_path_distance(arr_a, arr_b):
    a = _resample_xy_polyline(arr_a)
    b = _resample_xy_polyline(arr_b)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def _path_length_xy(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    xy = arr[:, :2]
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def _start_goal_error(arr_a, arr_b):
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or len(a) < 2 or len(b) < 2:
        return float("inf")
    start_err = np.linalg.norm(a[0, :2] - b[0, :2])
    goal_err = np.linalg.norm(a[-1, :2] - b[-1, :2])
    return float(max(start_err, goal_err))


def _is_duplicate_route(candidate_arr, ref_arr, *, mean_tol_m=220.0, max_tol_m=420.0, endpoint_tol_m=260.0, length_ratio_tol=0.08):
    if candidate_arr is None or ref_arr is None:
        return False

    mean_dist = _mean_path_distance(candidate_arr, ref_arr)
    max_dist = _max_path_distance(candidate_arr, ref_arr)
    endpoint_err = _start_goal_error(candidate_arr, ref_arr)

    len_a = _path_length_xy(candidate_arr)
    len_b = _path_length_xy(ref_arr)
    if min(len_a, len_b) < 1e-6:
        return False
    length_ratio = abs(len_a - len_b) / max(min(len_a, len_b), 1e-6)

    return (
        mean_dist <= mean_tol_m
        and max_dist <= max_tol_m
        and endpoint_err <= endpoint_tol_m
        and length_ratio <= length_ratio_tol
    )


def _deduplicate_neuro_adaptive(paths_dict=None, traj_dict=None):
    if not traj_dict or "Neuro-Adaptive" not in traj_dict:
        return paths_dict, traj_dict, None

    na_traj = traj_dict.get("Neuro-Adaptive")
    if na_traj is None or len(na_traj) < 2:
        return paths_dict, traj_dict, None

    duplicate_of = None
    for ref_name in ["T-GnP", "T-GnP+DNN", "K-GNP", "RL-Pilot", "RL-Pilot+DNN", "Dijkstra", "A-Star", "RRT-Star", "PSO"]:
        ref_traj = traj_dict.get(ref_name)
        if ref_traj is None or len(ref_traj) < 2:
            continue
        ref_path = paths_dict.get(ref_name) if paths_dict else None
        na_path = paths_dict.get("Neuro-Adaptive") if paths_dict else None

        same_flight_corridor = _is_duplicate_route(na_traj, ref_traj)
        same_planned_corridor = na_path is not None and ref_path is not None and _is_duplicate_route(
            na_path,
            ref_path,
            mean_tol_m=250.0,
            max_tol_m=450.0,
            endpoint_tol_m=260.0,
            length_ratio_tol=0.10,
        )

        if same_flight_corridor or same_planned_corridor:
            duplicate_of = ref_name
            break

    if duplicate_of is None:
        return paths_dict, traj_dict, None

    new_paths = dict(paths_dict) if paths_dict else {}
    new_trajs = dict(traj_dict)
    new_paths.pop("Neuro-Adaptive", None)
    new_trajs.pop("Neuro-Adaptive", None)
    return new_paths, new_trajs, duplicate_of


def generate_risk_field(terrain, threats, resolution=400):
    bounds = terrain.dataset.bounds
    xs = np.linspace(bounds.left, bounds.right, resolution)
    ys = np.linspace(bounds.bottom, bounds.top, resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)
    risk_map = np.zeros_like(x_grid, dtype=float)

    for threat in threats:
        dist = np.hypot(x_grid - threat.x, y_grid - threat.y)
        hard_mask = dist <= threat.radius
        risk_map[hard_mask] += 1e5 * getattr(threat, "level", 1.0)

        soft_r = getattr(threat, "soft_radius", threat.radius)
        soft_mask = (dist > threat.radius) & (dist <= soft_r)
        if np.any(soft_mask):
            d_norm = (dist[soft_mask] - threat.radius) / max(1e-9, (soft_r - threat.radius))
            factor = np.clip(1.0 - d_norm, 0.0, 1.0)
            risk_map[soft_mask] += 5000.0 * (factor ** 2) * getattr(threat, "level", 1.0)

    return x_grid, y_grid, risk_map


def _infer_start_goal(paths_dict, traj_dict, goal_center):
    start_xy = None
    goal_xy = None

    if paths_dict:
        first_path = next((v for v in paths_dict.values() if v is not None and len(v) > 0), None)
        if first_path is not None:
            start_xy = (float(first_path[0][0]), float(first_path[0][1]))
            goal_xy = (float(first_path[-1][0]), float(first_path[-1][1]))

    if start_xy is None and traj_dict:
        first_traj = next((v for v in traj_dict.values() if v is not None and len(v) > 0), None)
        if first_traj is not None:
            arr = np.asarray(first_traj, dtype=float)
            start_xy = (float(arr[0, 0]), float(arr[0, 1]))
            goal_xy = (float(arr[-1, 0]), float(arr[-1, 1]))

    if goal_center is not None:
        goal_xy = (float(goal_center[0]), float(goal_center[1]))

    return start_xy, goal_xy


def plot_professional_riskmap(
    terrain,
    threats,
    paths_dict=None,
    traj_dict=None,
    title="Comparative Benchmark Analysis",
    resolution=500,
    save_path=None,
    show_plot=True,
    *,
    goal_center=None,
    goal_radius: float | None = None,
):
    paths_dict, traj_dict, na_duplicate_of = _deduplicate_neuro_adaptive(paths_dict, traj_dict)
    fig, ax = plt.subplots(figsize=(14, 10))

    dem_data = np.asarray(terrain.data, dtype=float)
    dem_data = np.ma.masked_less_equal(dem_data, -9999)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(dem_data, cmap=plt.cm.terrain, vert_exag=0.2, blend_mode="soft")

    ax.imshow(
        rgb,
        origin="upper",
        extent=[terrain.dataset.bounds.left, terrain.dataset.bounds.right, terrain.dataset.bounds.bottom, terrain.dataset.bounds.top],
        interpolation="bilinear",
        alpha=1.0,
        zorder=0,
    )

    x_grid, y_grid, risk_map = generate_risk_field(terrain, threats, resolution=resolution)
    risk_masked = np.ma.masked_where(risk_map < 10.0, risk_map)
    mesh = ax.pcolormesh(x_grid, y_grid, risk_masked, cmap=plt.cm.Reds, alpha=0.38, shading="auto", zorder=2)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("Threat Risk Intensity", weight="bold")

    for threat in threats:
        hard = Circle((threat.x, threat.y), float(threat.radius), color="darkred", fill=False, lw=2.0, ls="-", zorder=3)
        ax.add_patch(hard)
        label = getattr(threat, "name", getattr(threat, "type", "Threat"))
        txt = ax.text(
            float(threat.x),
            float(threat.y),
            str(label),
            color="white",
            ha="center",
            va="center",
            weight="bold",
            fontsize=9,
            zorder=4,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground="black")])

    if paths_dict:
        for name, path in paths_dict.items():
            if path is None or len(path) == 0:
                continue
            arr = np.asarray(path, dtype=float)
            color = _resolve_alg_color(name)
            ax.plot(arr[:, 0], arr[:, 1], color="black", lw=2.1, alpha=0.25, zorder=5)
            ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.35, ls="--", alpha=0.95, zorder=6)

    if traj_dict:
        for name, traj in traj_dict.items():
            if traj is None or len(traj) == 0:
                continue
            arr = np.asarray(traj, dtype=float)
            color = _resolve_alg_color(name)
            line, = ax.plot(arr[:, 0], arr[:, 1], color=color, lw=3.0, ls="-", label=f"{name} (Fly)", zorder=7)
            line.set_path_effects([path_effects.withStroke(linewidth=4.6, foreground="black", alpha=0.55)])

    start_xy, goal_xy = _infer_start_goal(paths_dict, traj_dict, goal_center)

    if goal_xy is not None and goal_radius is not None:
        goal_circle = Circle(
            (goal_xy[0], goal_xy[1]),
            float(goal_radius),
            edgecolor="black",
            linestyle="--",
            linewidth=2.0,
            fill=False,
            alpha=0.55,
            zorder=8,
            label="Goal Acceptance Region",
        )
        ax.add_patch(goal_circle)

    if start_xy is not None:
        ax.plot(start_xy[0], start_xy[1], "v", color="lime", markersize=15, markeredgecolor="black", zorder=9, label="Start")
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1], "X", color="black", markersize=15, markeredgecolor="white", mew=2, zorder=9, label="Goal")

    ax.set_title(title, weight="bold", pad=16)
    if na_duplicate_of is not None:
        ax.text(
            0.015,
            0.02,
            f"Neuro-Adaptive selected {na_duplicate_of}; duplicate route hidden.",
            transform=ax.transAxes,
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.35", alpha=0.92),
            zorder=20,
        )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            uniq_h.append(handle)
            uniq_l.append(label)
            seen.add(label)
    ax.legend(uniq_h, uniq_l, loc="upper right", framealpha=0.94, shadow=True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Benchmark Map Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
