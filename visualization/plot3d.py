from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _terrain_mesh(terrain, downsample=10):
    dem = np.asarray(terrain.data, dtype=float)
    dem_ds = dem[::downsample, ::downsample]
    height, width = dem_ds.shape
    transform = terrain.transform
    cols = np.arange(0, width) * downsample
    rows = np.arange(0, height) * downsample
    c_grid, r_grid = np.meshgrid(cols, rows)
    x = transform.c + c_grid * transform.a + r_grid * transform.b
    y = transform.f + c_grid * transform.d + r_grid * transform.e
    return x, y, dem_ds


def _height_at(terrain, x, y, z_fallback):
    try:
        z = terrain.get_height(float(x), float(y))
        return float(np.nan_to_num(z, nan=z_fallback))
    except Exception:
        return float(z_fallback)


def _lifted_line_xyz(terrain, arr, visibility_offset=60.0):
    arr = np.asarray(arr, dtype=float)
    z_fallback = float(np.nanmin(terrain.data))
    terrain_h = np.asarray([_height_at(terrain, x, y, z_fallback) for x, y in arr[:, :2]], dtype=float)
    lifted_z = np.maximum(arr[:, 2], terrain_h + visibility_offset)
    return arr[:, 0], arr[:, 1], lifted_z


def _mission_envelope(*arrays, pad_xy=4000.0):
    valid = [np.asarray(arr, dtype=float) for arr in arrays if arr is not None and len(arr) > 0]
    if not valid:
        return None

    all_xy = np.vstack([arr[:, :2] for arr in valid])
    xmin = float(np.min(all_xy[:, 0]) - pad_xy)
    xmax = float(np.max(all_xy[:, 0]) + pad_xy)
    ymin = float(np.min(all_xy[:, 1]) - pad_xy)
    ymax = float(np.max(all_xy[:, 1]) + pad_xy)
    return xmin, xmax, ymin, ymax


def _apply_common_scene(ax, terrain, threats, path, traj, goal_center, goal_radius, downsample, elev, azim, title):
    x_surf, y_surf, z_surf = _terrain_mesh(terrain, downsample=downsample)
    env = _mission_envelope(path, traj)
    if env is not None:
        xmin, xmax, ymin, ymax = env
        mask = (x_surf >= xmin) & (x_surf <= xmax) & (y_surf >= ymin) & (y_surf <= ymax)
        if np.any(mask):
            rows = np.where(np.any(mask, axis=1))[0]
            cols = np.where(np.any(mask, axis=0))[0]
            x_surf = x_surf[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
            y_surf = y_surf[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
            z_surf = z_surf[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    ax.plot_surface(x_surf, y_surf, z_surf, cmap="gist_earth", alpha=0.78, linewidth=0, antialiased=True, zorder=1)
    zmin = float(np.nanmin(z_surf))

    if threats:
        theta = np.linspace(0, 2 * np.pi, 80)
        for threat in threats:
            xc = threat.x + threat.radius * np.cos(theta)
            yc = threat.y + threat.radius * np.sin(theta)
            zg = np.asarray([_height_at(terrain, xi, yi, zmin) + 10.0 for xi, yi in zip(xc, yc)], dtype=float)
            ax.plot(xc, yc, zg, color="#d62728", linewidth=2.2, ls="--", zorder=3)

    if goal_center is not None and goal_radius is not None:
        gx, gy = float(goal_center[0]), float(goal_center[1])
        theta = np.linspace(0, 2 * np.pi, 120)
        xc = gx + float(goal_radius) * np.cos(theta)
        yc = gy + float(goal_radius) * np.sin(theta)
        zc = np.asarray([_height_at(terrain, xi, yi, zmin) + 12.0 for xi, yi in zip(xc, yc)], dtype=float)
        ax.plot(xc, yc, zc, color="#12ad2b", linewidth=3.0, ls="--", zorder=4, label="Goal Area")
        ax.scatter([gx], [gy], [_height_at(terrain, gx, gy, zmin) + 15.0], marker="x", s=90, color="#12ad2b", zorder=5)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(title, weight="bold")
    ax.view_init(elev=elev, azim=azim)

    if env is not None:
        xmin, xmax, ymin, ymax = env
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    z_candidates = [float(np.nanmin(z_surf)), float(np.nanmax(z_surf))]
    for arr in (path, traj):
        if arr is not None and len(arr) > 0:
            z_candidates.append(float(np.min(np.asarray(arr)[:, 2])))
            z_candidates.append(float(np.max(np.asarray(arr)[:, 2])))
    ax.set_zlim(min(z_candidates) - 50.0, max(z_candidates) + 200.0)
    try:
        ax.set_box_aspect((1.2, 1.0, 0.45))
    except Exception:
        pass
    return zmin


def plot_unified_tactical_3d(
    terrain,
    threats=None,
    path=None,
    traj=None,
    downsample=12,
    elev=38,
    azim=-128,
    title="3D Tactical Flight View",
    save_path=None,
    *,
    goal_center=None,
    goal_radius: float | None = None,
    show_plot: bool = False,
):
    if traj is None or len(traj) == 0:
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    _apply_common_scene(ax, terrain, threats, path, traj, goal_center, goal_radius, downsample, elev, azim, title)

    if path is not None and len(path) > 0:
        px, py, pz = _lifted_line_xyz(terrain, np.asarray(path, dtype=float), visibility_offset=75.0)
        ax.plot(px, py, pz, color="#ffe000", ls="--", lw=2.8, zorder=8, label="Planned")

    tx, ty, tz = _lifted_line_xyz(terrain, np.asarray(traj, dtype=float), visibility_offset=90.0)
    ax.plot(tx, ty, tz, color="#111111", lw=3.4, zorder=9, label="Flown")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper left")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"3D Tactical Plot saved to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_dynamic_trajectory_3d(
    terrain,
    threats=None,
    path=None,
    traj=None,
    downsample=12,
    elev=38,
    azim=-128,
    title="3D Dynamic Flight Simulation",
    save_path=None,
    *,
    goal_center=None,
    goal_radius: float | None = None,
    show_plot: bool = False,
):
    if traj is None or len(traj) == 0:
        print("Cannot plot 3D dynamics: No trajectory data available.")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    _apply_common_scene(ax, terrain, threats, path, traj, goal_center, goal_radius, downsample, elev, azim, title)

    if path is not None and len(path) > 0:
        px, py, pz = _lifted_line_xyz(terrain, np.asarray(path, dtype=float), visibility_offset=70.0)
        ax.plot(px, py, pz, color="#f4f4f4", ls="--", lw=2.2, zorder=7, label="Planned")

    arr = np.asarray(traj, dtype=float)
    x, y, z = _lifted_line_xyz(terrain, arr, visibility_offset=90.0)
    if arr.ndim != 2 or arr.shape[1] < 4:
        ax.plot(x, y, z, color="black", lw=3.0, zorder=9, label="Flown")
    else:
        psi = arr[:, 3]
        dt = 0.25
        vx = np.gradient(arr[:, 0], dt)
        vy = np.gradient(arr[:, 1], dt)
        v_h = np.sqrt(vx ** 2 + vy ** 2)
        turn_rate = np.gradient(np.unwrap(psi), dt)
        g_load = np.sqrt(1.0 + (v_h * turn_rate / 9.81) ** 2)
        norm = colors.Normalize(vmin=1.0, vmax=max(2.5, float(np.max(g_load))))
        cmap = cm.get_cmap("turbo")

        for i in range(len(x) - 1):
            ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=cmap(norm(g_load[i])), linewidth=3.8, zorder=9)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.08)
        cbar.set_label("Calculated G-Force", weight="bold")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper left")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"3D Dynamic Plot saved to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)
