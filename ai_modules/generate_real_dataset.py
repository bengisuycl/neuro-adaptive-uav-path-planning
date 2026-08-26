import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from wcci_conference_project.environment.terrain import Terrain
from wcci_conference_project.scenarios.definitions import get_all_scenarios

DEFAULT_DEM_CANDIDATES = [
    os.environ.get("WCCI_DEM_PATH", ""),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "dem_utm.tif"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "wcci_conference_project", "dem_utm.tif"),
    "/content/wcci_workspace/wcci_conference_project/dem_utm.tif",
    "/content/wcci_workspace/dem_utm.tif",
    r"C:\Users\DELL\PycharmProjects\thesis_project\wcci_conference_project\dem_utm.tif",
    r"C:\Users\DELL\OneDrive - student.cankaya.edu.tr\hacettepe_master\MSc\MSc Thesis\DEM\dem_utm.tif",
]


def resolve_dem_path(explicit_path=None):
    candidates = [explicit_path] if explicit_path else []
    candidates.extend(DEFAULT_DEM_CANDIDATES)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def calculate_glos_risk(terrain, px, py, pz, threats, max_range=20000.0):
    total_risk = 0.0

    for th in threats:
        dist = np.hypot(px - th.x, py - th.y)
        if dist > max_range:
            continue

        d_norm = dist / max_range
        d_factor = np.exp(-3.0 * d_norm)

        num_checks = 15
        is_occluded = False
        th_z = terrain.get_height(th.x, th.y) + 10.0

        for i in range(1, num_checks):
            t = i / num_checks
            ix = th.x + (px - th.x) * t
            iy = th.y + (py - th.y) * t
            iz = th_z + (pz - th_z) * t

            terr_h = terrain.get_height(ix, iy)
            if terr_h > iz:
                is_occluded = True
                break

        if not is_occluded:
            total_risk += d_factor * getattr(th, "level", 1.0)

    return total_risk


def extract_risk_features(terrain, px, py, agl, threats):
    terr_z = terrain.get_height(px, py)
    closest_th = min(threats, key=lambda t: np.hypot(px - t.x, py - t.y))
    dist_th = np.hypot(px - closest_th.x, py - closest_th.y)
    h_dx = terrain.get_height(px + 50.0, py)
    slope = abs(h_dx - terr_z) / 50.0

    return {
        "dist_norm": float(dist_th / 20000.0),
        "rel_alt_norm": float(agl / 2000.0),
        "slope": float(np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)),
        "th_level": float(getattr(closest_th, "level", 1.0)),
    }


def _sample_point(bounds, threats, focus_prob=0.65):
    if threats and np.random.rand() < focus_prob:
        threat = threats[np.random.randint(0, len(threats))]
        mode = np.random.rand()
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        hard_r = float(threat.radius)
        soft_r = float(max(getattr(threat, "soft_radius", hard_r), hard_r + 500.0))

        if mode < 0.35:
            dist = np.random.uniform(0.0, hard_r * 1.1)
        elif mode < 0.75:
            dist = np.random.uniform(hard_r * 0.8, soft_r * 1.05)
        else:
            dist = np.random.uniform(soft_r * 0.9, soft_r * 1.8)

        px = threat.x + dist * np.cos(angle)
        py = threat.y + dist * np.sin(angle)
    else:
        px = np.random.uniform(bounds.left, bounds.right)
        py = np.random.uniform(bounds.bottom, bounds.top)
    return px, py


def generate_dataset(num_samples=9000, scenario_ids=None, save_stem="risk_dataset_glos", dem_path=None, focus_prob=0.82):
    print("Loading Terrain & Scenarios...")
    resolved_dem_path = resolve_dem_path(dem_path)
    if resolved_dem_path is None:
        print("DEM dosyasi bulunamadi. WCCI_DEM_PATH ayarlayin veya dem_utm.tif dosyasini proje kokune koyun.")
        return

    try:
        terrain = Terrain(resolved_dem_path)
    except Exception:
        print(f"DEM Dosyasi yuklenemedi: {resolved_dem_path}")
        return

    safe_alt = max(float(np.nanmax(terrain.data)) + 200.0, 600.0)
    scenarios = get_all_scenarios(safe_alt)
    if scenario_ids is None:
        selected_scenarios = scenarios
    else:
        selected_scenarios = [s for s in scenarios if s["id"] in set(scenario_ids)]

    if not selected_scenarios:
        print("Secilen senaryolar icin veri olusturulamadi.")
        return

    print(f"Generating {num_samples} Ground Truth samples across {len(selected_scenarios)} scenarios...")
    data = []
    bounds = terrain.dataset.bounds
    samples_per_scenario = max(1, int(num_samples / len(selected_scenarios)))

    for scen in selected_scenarios:
        threats = scen["threats"]
        print(f"  Scenario: {scen['id']}")
        for i in range(samples_per_scenario):
            if i % 500 == 0:
                print(f"    Processed {i}/{samples_per_scenario}...")

            px, py = _sample_point(bounds, threats, focus_prob=focus_prob)
            if not terrain.is_inside(px, py):
                continue

            terr_z = terrain.get_height(px, py)
            agl = np.random.uniform(50.0, 1500.0)
            pz = terr_z + agl

            risk_gt = calculate_glos_risk(terrain, px, py, pz, threats)
            features = extract_risk_features(terrain, px, py, agl, threats)
            features["risk_label"] = float(risk_gt)
            features["scenario_id"] = scen["id"]
            data.append(features)

    df = pd.DataFrame(data)
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, f"{save_stem}.csv")
    meta_path = os.path.join(base_dir, f"{save_stem}_meta.json")

    df.to_csv(csv_path, index=False)
    metadata = {
        "dataset_path": os.path.basename(csv_path),
        "scenario_ids": [s["id"] for s in selected_scenarios],
        "num_samples": int(len(df)),
        "feature_columns": ["dist_norm", "rel_alt_norm", "slope", "th_level"],
        "label_column": "risk_label",
        "dem_path": resolved_dem_path,
        "focus_prob": focus_prob,
    }
    with open(meta_path, "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset Saved: {csv_path}")
    print(f"Metadata Saved: {meta_path}")


if __name__ == "__main__":
    generate_dataset()
