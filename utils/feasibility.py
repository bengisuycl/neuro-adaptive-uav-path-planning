# utils/feasibility.py
import math
import numpy as np
from wcci_conference_project.ai_modules.neural_risk_map import NeuralRiskMapper
from wcci_conference_project.environment.collision import check_terrain_collision
from wcci_conference_project.utils.cost_model import compute_threat_cost


class FeasibilityEngine:
    """
    Shared feasibility + cost engine.

    IMPORTANT for benchmarking:
    - check_count counts the number of check_segment() calls.
      This is a useful proxy for compute budget comparisons and reporting.
    """
    def __init__(
        self,
        terrain,
        threats,
        check_resolution=50.0,
        w_risk=20.0,
        use_neural_risk=False,
        neural_risk_weight=0.0,
        neural_risk_samples=3,
        neural_corridor_half_width=1800.0,
    ):
        self.terrain = terrain
        self.threats = threats
        self.check_res = float(check_resolution)
        self.w_risk = float(w_risk)
        self.use_neural_risk = bool(use_neural_risk)
        self.neural_risk_weight = float(neural_risk_weight)
        self.neural_risk_samples = int(neural_risk_samples)
        self.neural_corridor_half_width = float(neural_corridor_half_width)
        self.neural_mapper = NeuralRiskMapper(verbose=False) if self.use_neural_risk else None

        # --- METRIC: segment check counter ---
        self.check_count = 0

    def reset_counter(self):
        self.check_count = 0

    def _estimate_neural_corridor_risk(self, p1, p2, altitude):
        if not self.use_neural_risk or self.neural_mapper is None or not getattr(self.neural_mapper, "ready", False):
            return 0.0

        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        seg = np.array([x2 - x1, y2 - y1], dtype=float)
        seg_norm = float(np.linalg.norm(seg))
        tangent = np.array([1.0, 0.0], dtype=float) if seg_norm < 1e-9 else seg / seg_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        query_points = []
        offsets = [0.0]
        if self.neural_corridor_half_width > 1.0:
            offsets.extend([-self.neural_corridor_half_width, self.neural_corridor_half_width])

        samples = max(1, self.neural_risk_samples)
        for i in range(samples):
            t = (i + 1) / float(samples + 1)
            base_xy = np.array([x1, y1], dtype=float) + t * seg
            for off in offsets:
                pt = base_xy + off * normal
                query_points.append((float(pt[0]), float(pt[1])))

        if not query_points:
            return 0.0

        terrain_h = self.terrain.get_height(x1, y1)
        terrain_h = 0.0 if terrain_h is None or math.isnan(terrain_h) else float(terrain_h)
        agl_m = max(50.0, float(altitude) - terrain_h)
        preds = self.neural_mapper.get_risk_at_points(query_points, self.terrain, self.threats, agl_m=agl_m)
        return float(np.mean(preds)) if preds else 0.0

    def check_segment(self, p1, p2, altitude):
        self.check_count += 1  # count calls (not per-sample)

        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dist = math.hypot(x2 - x1, y2 - y1)

        # Sampling
        num_checks = max(2, int(dist / self.check_res))
        total_risk = 0.0

        for i in range(1, num_checks + 1):
            t = i / float(num_checks)
            ix = x1 + (x2 - x1) * t
            iy = y1 + (y2 - y1) * t

            # 1) Boundary
            if not self.terrain.is_inside(ix, iy):
                return False, float("inf"), "boundary"

            # 2) Terrain clearance
            if check_terrain_collision((ix, iy, altitude), self.terrain):
                return False, float("inf"), "terrain"

            # 3) Hard threat
            risk, is_hard = compute_threat_cost(ix, iy, self.threats)
            if is_hard:
                return False, float("inf"), "hard_threat"

            total_risk += float(risk)

        avg_risk = total_risk / float(num_checks)
        cost = float(dist) + (avg_risk * self.w_risk)
        if self.use_neural_risk and self.neural_risk_weight > 0.0:
            neural_risk = self._estimate_neural_corridor_risk(p1, p2, altitude)
            cost += self.neural_risk_weight * neural_risk * float(dist) / max(self.check_res, 1.0)
        return True, cost, None
