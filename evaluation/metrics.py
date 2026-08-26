# evaluation/metrics.py
# UPDATED: IEEE/CODIT-Compliant Metrics (History-Based Saturation & Max CTE)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from wcci_conference_project.utils.cost_model import compute_threat_cost

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


@dataclass
class GuidanceMetrics:
    # Tracking Performance
    track_mean_m: float = float("nan")  # Ortalama sapma
    track_max_m: float = float("nan")  # Maksimum sapma (overshoot)
    track_norm: float = float("nan")  # Normalize sapma (CTE / Path Length)

    # Control Saturation (Dynamic Feasibility)
    sat_count: int = 0  # Limit aşım sayısı (phi > phi_max)
    sat_ratio: float = 0.0  # Aşım oranı
    phi_max_deg_observed: float = 0.0  # Görülen en yüksek yatış açısı
    v_min_observed: float = float("nan")  # Görülen en düşük hız


def path_length(path_xyz: Sequence[Sequence[float]]) -> float:
    if not path_xyz or len(path_xyz) < 2:
        return 0.0
    arr = np.asarray(path_xyz, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)))


def calculate_risk_metrics(traj: Optional[np.ndarray], threats: Sequence[Any]) -> Dict[str, float]:
    if traj is None or len(traj) == 0:
        return {"total": 0.0, "avg": 0.0, "max": 0.0}

    total = 0.0
    maxv = 0.0
    for k in range(len(traj)):
        x, y = float(traj[k, 0]), float(traj[k, 1])
        c, _ = compute_threat_cost(x, y, threats)
        total += float(c)
        if c > maxv: maxv = float(c)

    # dt = 0.25 (Simulation step) varsayılan risk exposure integral
    return {"total": total * 0.25, "avg": total / len(traj), "max": maxv}


def calculate_risk_exposure(traj, threats, dt=0.25):
    # Wrapper for main.py compatibility
    res = calculate_risk_metrics(traj, threats)
    return res["total"]


def calculate_min_terrain_clearance(traj, terrain):
    if traj is None or len(traj) == 0: return 0.0
    min_agl = float('inf')
    for state in traj:
        x, y, h = state[0], state[1], state[2]
        terr_h = terrain.get_height(x, y)
        if np.isnan(terr_h): terr_h = 0.0
        agl = h - terr_h
        if agl < min_agl: min_agl = agl
    return max(0.0, min_agl)


def _min_dist_point_to_polyline(p: np.ndarray, poly_xy: np.ndarray) -> float:
    min_dist = float("inf")
    for i in range(len(poly_xy) - 1):
        p1 = poly_xy[i]
        p2 = poly_xy[i + 1]
        seg = p2 - p1
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 <= 1e-12:
            d = float(np.linalg.norm(p - p1))
        else:
            t = float(np.clip(np.dot(p - p1, seg) / seg_len2, 0.0, 1.0))
            proj = p1 + t * seg
            d = float(np.linalg.norm(p - proj))
        if d < min_dist:
            min_dist = d
    return float(min_dist)


def calculate_tracking_stats(traj: np.ndarray, ref_path: ArrayLike) -> Tuple[float, float]:
    """Returns (mean_cte, max_cte)."""
    if traj is None or len(traj) == 0:
        return float("nan"), float("nan")
    ref = np.asarray(ref_path, dtype=float)
    if ref.ndim != 2 or ref.shape[0] < 2:
        return float("nan"), float("nan")

    # Long Monte Carlo runs can make exact CTE computation dominate runtime.
    # Downsample both trajectory and reference while preserving endpoints.
    traj_stride = max(1, len(traj) // 600)
    ref_stride = max(1, len(ref) // 300)
    traj_xy = np.asarray(traj[::traj_stride, :2], dtype=float)
    ref_xy = np.asarray(ref[::ref_stride, :2], dtype=float)
    if len(ref_xy) < 2:
        ref_xy = ref[:, :2]
    if not np.array_equal(traj_xy[-1], traj[-1, :2]):
        traj_xy = np.vstack([traj_xy, traj[-1, :2]])
    if not np.array_equal(ref_xy[-1], ref[-1, :2]):
        ref_xy = np.vstack([ref_xy, ref[-1, :2]])

    errs = []
    for p in traj_xy:
        errs.append(_min_dist_point_to_polyline(p, ref_xy))

    errs = np.array(errs)
    return float(np.mean(errs)), float(np.max(errs))


def calculate_tracking_error(traj, path):
    # Legacy wrapper for main.py compatibility (returns mean only)
    m, _ = calculate_tracking_stats(traj, path)
    return m


def compute_guidance_metrics(
        traj: Optional[np.ndarray],
        ref_path: ArrayLike,
        *,
        phi_max_deg: float = 60.0,
        history: Optional[List[Dict[str, Any]]] = None
) -> GuidanceMetrics:
    """
    Hesaplar:
    1. Tracking Error (Mean & Max)
    2. Saturation (D.SAT): Simülasyon history'sinden gerçek komutlara bakar.
    """
    gm = GuidanceMetrics()

    # 1. Tracking Metrics
    gm.track_mean_m, gm.track_max_m = calculate_tracking_stats(traj, ref_path)

    path_len = path_length(ref_path)
    if path_len > 1.0:
        gm.track_norm = gm.track_mean_m / path_len

    # 2. Saturation Metrics (History-Based if available)
    if history and len(history) > 0:
        # History varsa gerçek komutları kullan (EN DOĞRUSU)
        phis = []
        vs = []
        for rec in history:
            # Farklı isimlendirmeleri yakala
            val = rec.get("phi_cmd_deg", rec.get("phi_deg", None))
            if val is not None: phis.append(float(val))

            v_val = rec.get("speed", rec.get("V", None))
            if v_val is not None: vs.append(float(v_val))

        if phis:
            phis_arr = np.abs(np.array(phis))
            gm.sat_count = int(np.sum(phis_arr > (phi_max_deg - 0.1)))  # Toleranslı kontrol
            gm.sat_ratio = gm.sat_count / len(phis)
            gm.phi_max_deg_observed = float(np.max(phis_arr))

        if vs:
            gm.v_min_observed = float(np.min(vs))

    elif traj is not None and len(traj) > 1:
        # History yoksa trajectory türevinden tahmin et (FALLBACK)
        # psi_dot = (g * tan(phi)) / V => phi = atan(psi_dot * V / g)
        psi = traj[:, 3]
        dt = 0.25
        psi_unwrap = np.unwrap(psi)
        psi_dot = np.gradient(psi_unwrap, dt)
        V_nominal = 220.0  # Varsayım
        phi_est = np.degrees(np.arctan(psi_dot * V_nominal / 9.81))
        gm.sat_count = int(np.sum(np.abs(phi_est) > phi_max_deg))
        gm.phi_max_deg_observed = float(np.max(np.abs(phi_est)))

    return gm


# Legacy compatibility
def count_dynamic_violations(traj):
    gm = compute_guidance_metrics(traj, [[0, 0], [1, 1]])  # Dummy path
    return gm.sat_count
