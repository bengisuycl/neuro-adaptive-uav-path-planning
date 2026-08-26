# visualization/plot2d.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import os  # <--- HATA VEREN EKSİK SATIR BUYDU

# IEEE/Thesis Standartları
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def plot_flight_analysis_2d(terrain, threats, trajectory, title="Flight Analysis", save_path=None, goal_center=None, goal_radius=3000.0):
    fig, ax = plt.subplots(figsize=(10, 8))

    dem = np.asarray(terrain.data, dtype=float)
    t = terrain.transform
    extent = [t.c, t.c + dem.shape[1] * t.a, t.f + dem.shape[0] * t.e, t.f]
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(dem, cmap=plt.cm.terrain, vert_exag=0.08, blend_mode='soft')
    ax.imshow(rgb, extent=extent, origin="upper", alpha=0.9)

    for th in threats:
        lvl = getattr(th, 'level', 1.0)
        c = '#8b0000' if lvl >= 4.0 else ('#ff0000' if lvl >= 2.0 else '#ff8c00')
        circle = plt.Circle((th.x, th.y), th.radius, facecolor=c, alpha=0.35, edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(th.x, th.y + 1300, f"Lvl {lvl}", color='white', fontsize=8, weight='bold', ha='center', bbox=dict(facecolor=c, alpha=0.8, edgecolor='none', pad=0.3))

    if goal_center is not None:
        gx, gy = float(goal_center[0]), float(goal_center[1])
        g_circle = plt.Circle((gx, gy), goal_radius, facecolor='#00ff00', alpha=0.15, linestyle='--', edgecolor='green')
        ax.add_patch(g_circle)
        ax.scatter(gx, gy, c="black", s=180, marker="X", edgecolors="white", linewidths=1.8, label="Goal Area", zorder=15)

    if trajectory is not None and len(trajectory) > 0:
        t_arr = np.asarray(trajectory)
        ax.plot(t_arr[:, 0], t_arr[:, 1], color='black', linewidth=4.5, zorder=10)
        ax.plot(t_arr[:, 0], t_arr[:, 1], color='#00ff00', linewidth=3.0, label="Flown Trajectory", zorder=11)
        ax.scatter(t_arr[0, 0], t_arr[0, 1], c="blue", s=180, marker="^", edgecolors="white", label="Start", zorder=16)

    ax.legend(loc="upper right", framealpha=0.95, edgecolor="black")
    ax.set_xlabel("Easting (UTM-m)")
    ax.set_ylabel("Northing (UTM-m)")
    ax.set_title(title, weight="bold", pad=15)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"      > Saved 2D Chart: {os.path.basename(save_path)}")
    plt.close(fig)

def plot_multiple_paths(terrain, threats, paths_dict, colors_dict, title="Comparative Trajectories", save_path=None, goal_center=None, goal_radius=3000.0):
    fig, ax = plt.subplots(figsize=(12, 9))
    dem = np.asarray(terrain.data, dtype=float)
    t = terrain.transform
    extent = [t.c, t.c + dem.shape[1] * t.a, t.f + dem.shape[0] * t.e, t.f]
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(dem, cmap=plt.cm.terrain, vert_exag=0.08, blend_mode='soft')
    ax.imshow(rgb, extent=extent, origin="upper", alpha=0.8)

    for th in threats:
        lvl = getattr(th, 'level', 1.0)
        c = '#8b0000' if lvl >= 4.0 else ('#ff0000' if lvl >= 2.0 else '#ff8c00')
        ax.add_patch(plt.Circle((th.x, th.y), th.radius, facecolor=c, alpha=0.3))

    for alg_name, traj in paths_dict.items():
        if traj is not None and len(traj) > 0:
            t_arr = np.asarray(traj)
            color = colors_dict.get(alg_name, 'black')
            lw = 3.5 if alg_name in ["K-GNP", "T-GnP", "Neuro-Adaptive"] else 2.0
            ax.plot(t_arr[:, 0], t_arr[:, 1], color='black', linewidth=lw+1.5, zorder=9)
            ax.plot(t_arr[:, 0], t_arr[:, 1], color=color, linewidth=lw, label=alg_name, zorder=10)

    if goal_center is not None:
        ax.scatter(goal_center[0], goal_center[1], c="black", s=200, marker="X", edgecolors="white", label="Target", zorder=15)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.9, edgecolor='black')
    ax.set_title(title, weight="bold", fontsize=15, pad=15)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"      > Saved Comparative Chart: {os.path.basename(save_path)}")
    plt.close(fig)