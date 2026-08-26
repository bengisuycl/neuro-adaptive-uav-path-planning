import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource


def animate_tactical_flight(
    terrain,
    threats: list,
    traj: np.ndarray,
    history: list,
    save_path: str,
    title: str = "Flight Animation",
    *,
    goal_center=None,
    goal_radius=2000.0,
    planned_path=None,
    fps: int = 16,
    dpi: int = 120,
):
    if traj is None or len(traj) == 0:
        return

    xs_traj, ys_traj = traj[:, 0], traj[:, 1]
    if planned_path is not None and len(planned_path) > 0:
        pp = np.asarray(planned_path, dtype=float)
        xs_all = np.concatenate([xs_traj, pp[:, 0]])
        ys_all = np.concatenate([ys_traj, pp[:, 1]])
    else:
        pp = None
        xs_all, ys_all = xs_traj, ys_traj

    pad = 3500.0
    xmin, xmax = float(np.min(xs_all) - pad), float(np.max(xs_all) + pad)
    ymin, ymax = float(np.min(ys_all) - pad), float(np.max(ys_all) + pad)

    xmin = max(xmin, terrain.dataset.bounds.left)
    xmax = min(xmax, terrain.dataset.bounds.right)
    ymin = max(ymin, terrain.dataset.bounds.bottom)
    ymax = min(ymax, terrain.dataset.bounds.top)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal", adjustable="box")

    data = np.asarray(terrain.data, dtype=float)
    data = np.ma.masked_less_equal(data, -9999)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(data, cmap=plt.cm.terrain, vert_exag=0.2, blend_mode="soft")
    ax.imshow(
        rgb,
        extent=[terrain.dataset.bounds.left, terrain.dataset.bounds.right, terrain.dataset.bounds.bottom, terrain.dataset.bounds.top],
        origin="upper",
        interpolation="bilinear",
        alpha=0.95,
        zorder=0,
    )

    if goal_center is not None and goal_radius is not None:
        goal_circle = plt.Circle((goal_center[0], goal_center[1]), goal_radius, fill=False, edgecolor="black", ls="--", lw=2.0, alpha=0.6, zorder=2)
        ax.add_patch(goal_circle)
        ax.plot(goal_center[0], goal_center[1], "X", color="black", markersize=12, markeredgecolor="white", zorder=3)

    if pp is not None:
        ax.plot(pp[:, 0], pp[:, 1], color="white", lw=2.0, ls="--", alpha=0.95, label="Planned", zorder=4)

    trail_line, = ax.plot([], [], color="black", lw=2.8, alpha=0.95, label="Flown", zorder=6)
    aircraft_dot = ax.scatter([], [], s=140, c="#1f77ff", edgecolors="white", zorder=7)

    threat_patches = []
    threat_texts = []
    for threat in threats:
        circ = plt.Circle((threat.x, threat.y), threat.radius, fill=False, edgecolor="red", lw=2, ls="-", zorder=5)
        ax.add_patch(circ)
        threat_patches.append(circ)
        txt = ax.text(threat.x, threat.y, getattr(threat, "type", "T"), color="white", fontsize=8, ha="center", weight="bold", zorder=6)
        threat_texts.append(txt)

    hud_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="0.3"),
        zorder=10,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    total_frames = len(traj)
    step = max(1, int(total_frames / max(1, fps * 12)))
    frame_indices = np.arange(0, total_frames, step)

    def init():
        trail_line.set_data([], [])
        aircraft_dot.set_offsets(np.zeros((0, 2)))
        hud_text.set_text("")
        return trail_line, aircraft_dot, hud_text, *threat_patches, *threat_texts

    def update(frame_idx):
        idx = min(int(frame_idx), len(traj) - 1)
        x, y, z = traj[idx, 0], traj[idx, 1], traj[idx, 2]
        trail_line.set_data(traj[:idx + 1, 0], traj[:idx + 1, 1])
        aircraft_dot.set_offsets(np.asarray([[x, y]]))

        if history and isinstance(history[idx], dict) and "threat_states" in history[idx]:
            for i, (tx, ty, tr) in enumerate(history[idx]["threat_states"]):
                if i < len(threat_patches):
                    threat_patches[i].center = (tx, ty)
                    threat_patches[i].radius = tr
                    threat_texts[i].set_position((tx, ty))
        else:
            for i, threat in enumerate(threats):
                threat_patches[i].center = (threat.x, threat.y)
                threat_patches[i].radius = threat.radius
                threat_texts[i].set_position((threat.x, threat.y))

        speed = 0.0
        speed_target = 0.0
        if history and isinstance(history[idx], dict):
            speed = float(history[idx].get("speed", 0.0))
            speed_target = float(history[idx].get("speed_target", 0.0))
        ground_h = terrain.get_height(float(x), float(y))
        ground_h = 0.0 if ground_h is None or np.isnan(ground_h) else float(ground_h)
        agl = z - ground_h
        hud_text.set_text(
            f"Time: {idx * 0.25:.1f} s\n"
            f"Alt(MSL): {z:.0f} m\n"
            f"AGL: {agl:.0f} m\n"
            f"Spd: {speed:.0f} m/s\n"
            f"V_target: {speed_target:.0f} m/s"
        )
        return trail_line, aircraft_dot, hud_text, *threat_patches, *threat_texts

    ani = FuncAnimation(fig, update, frames=frame_indices, init_func=init, blit=False)

    ext = os.path.splitext(save_path)[1].lower()
    try:
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
        ani.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved: {save_path}")
    except Exception as exc:
        print(f"Animation save failed: {exc}")

    plt.close(fig)
