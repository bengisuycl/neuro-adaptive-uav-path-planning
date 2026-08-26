# planners/pso_planner.py
# CODIT2026: Time-budgeted PSO with corridor sampling (start-goal local bounds) + ordered waypoints.
# Key goals:
#   1) Never freeze: hard deadline checks inside segment evaluation.
#   2) Produce geometrically sensible planned paths: waypoints are ordered along progress (t_k) and sampled
#      as offsets around the nominal start->goal line (prevents star-shaped self-crossing paths).
#   3) Keep fair benchmarking: similar to RRT* local sampling bounds (informed / operational area sampling).

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from wcci_conference_project.utils.feasibility import FeasibilityEngine

@dataclass
class PSOStats:
    t_first: Optional[float] = None
    t_total: Optional[float] = None
    status: str = "FAIL"
    best_cost: float = float("inf")


class PSOPlanner:
    def __init__(
        self,
        terrain,
        threats,
        safe_alt: float = 800.0,
        goal_tolerance: float = 3000.0,
        check_resolution: float = 50.0,
        threat_weight: float = 20.0,
        num_particles: int = 35,
        max_iter: int = 250,
        num_waypoints: int = 12,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        # corridor sampling
        corridor_margin_m: float = 10000.0,
        waypoint_spread_m: float = 8000.0,
        # safety/perf guards
        max_segment_len_m: float = 30000.0,
        # geometric regularization (keeps paths realistic)
        turn_weight: float = 0.15,
        **kwargs,
    ):
        self.terrain = terrain
        self.threats = threats
        self.safe_alt = float(safe_alt)
        self.goal_tolerance = float(goal_tolerance)

        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.num_nodes = int(num_waypoints)

        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)

        self.corridor_margin_m = float(corridor_margin_m)
        self.waypoint_spread_m = float(waypoint_spread_m)

        self.max_segment_len_m = float(max_segment_len_m)
        self.turn_weight = float(turn_weight)

        self.feasibility = FeasibilityEngine(
            terrain,
            threats,
            check_resolution=check_resolution,
            w_risk=threat_weight,
            use_neural_risk=kwargs.get("use_neural_risk", False),
            neural_risk_weight=kwargs.get("neural_risk_weight", 0.0),
            neural_risk_samples=kwargs.get("neural_risk_samples", 3),
            neural_corridor_half_width=kwargs.get("neural_corridor_half_width", 1800.0),
        )
        self.bounds = terrain.dataset.bounds
        self.stats = PSOStats()

    # ---------------------------
    # Geometry helpers
    # ---------------------------
    @staticmethod
    def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            return 0.0
        c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        return float(math.acos(c))

    def _nominal_waypoints(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        # Ordered waypoints along the line: t_k = (k+1)/(N+1)
        wps = np.zeros((self.num_nodes, 2), dtype=float)
        for k in range(self.num_nodes):
            t = (k + 1) / (self.num_nodes + 1)
            wps[k] = start_xy + (goal_xy - start_xy) * t
        return wps

    def _compute_local_bounds(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        m = self.corridor_margin_m
        min_x = min(start_xy[0], goal_xy[0]) - m
        max_x = max(start_xy[0], goal_xy[0]) + m
        min_y = min(start_xy[1], goal_xy[1]) - m
        max_y = max(start_xy[1], goal_xy[1]) + m

        # Clip to DEM bounds
        x0 = max(self.bounds.left, min_x)
        x1 = min(self.bounds.right, max_x)
        y0 = max(self.bounds.bottom, min_y)
        y1 = min(self.bounds.top, max_y)

        # If DEM bounds are tight or margin collapses, ensure a non-degenerate box
        if x1 - x0 < 1.0:
            cx = 0.5 * (x0 + x1)
            x0, x1 = cx - 0.5, cx + 0.5
        if y1 - y0 < 1.0:
            cy = 0.5 * (y0 + y1)
            y0, y1 = cy - 0.5, cy + 0.5

        return (x0, x1), (y0, y1)

    def _decode_offsets_to_path(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        nominal: np.ndarray,
        offsets_flat: np.ndarray,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> List[Tuple[float, float, float]]:
        wp = offsets_flat.reshape(self.num_nodes, 2) + nominal
        wp[:, 0] = np.clip(wp[:, 0], xlim[0], xlim[1])
        wp[:, 1] = np.clip(wp[:, 1], ylim[0], ylim[1])

        pts: List[Tuple[float, float, float]] = [(float(start[0]), float(start[1]), self.safe_alt)]
        for k in range(self.num_nodes):
            pts.append((float(wp[k, 0]), float(wp[k, 1]), self.safe_alt))
        pts.append((float(goal[0]), float(goal[1]), self.safe_alt))
        return pts

    # ---------------------------
    # Cost (time-budget safe)
    # ---------------------------
    def _path_cost(self, path: List[Tuple[float, float, float]], deadline: float) -> float:
        """
        Returns total cost (lower is better). Uses feasibility.check_segment cost (which already includes risk weighting).
        Adds a light turn smoothness penalty to discourage self-crossings / sharp zig-zags.
        Hard stops:
          - if deadline exceeded
          - if any segment is infeasible
          - if segment is absurdly long (performance guard)
        """
        total = 0.0
        turn_pen = 0.0

        prev_vec = None
        for i in range(len(path) - 1):
            if time.time() > deadline:
                return float("inf")

            x1, y1 = path[i][0], path[i][1]
            x2, y2 = path[i + 1][0], path[i + 1][1]
            seg = np.array([x2 - x1, y2 - y1], dtype=float)
            seg_len = float(np.hypot(seg[0], seg[1]))

            if seg_len > self.max_segment_len_m:
                return float("inf")

            ok, c, _ = self.feasibility.check_segment((x1, y1), (x2, y2), self.safe_alt)
            if not ok:
                return float("inf")

            total += float(c)

            if prev_vec is not None:
                ang = self._angle_between(prev_vec, seg)
                # penalty grows with angle; scale by segment length to keep units stable
                turn_pen += (ang * ang) * max(1.0, seg_len / 1000.0)
            prev_vec = seg

        return total + self.turn_weight * turn_pen

    # ---------------------------
    # Main PSO
    # ---------------------------
    def plan(self, start, goal, time_budget_s: Optional[float] = None, verbose: bool = False, **kwargs):
        t0 = time.time()
        budget = float(time_budget_s) if time_budget_s is not None else 1e9
        deadline = t0 + budget

        # Reset counters for reporting
        if hasattr(self.feasibility, "reset_counter"):
            self.feasibility.reset_counter()

        self.stats = PSOStats()

        start_xy = np.array(start[:2], dtype=float)
        goal_xy = np.array(goal[:2], dtype=float)

        xlim, ylim = self._compute_local_bounds(start_xy, goal_xy)
        nominal = self._nominal_waypoints(start_xy, goal_xy)

        # Representation: offsets around nominal, shape (num_nodes,2), flattened
        dim = self.num_nodes * 2

        # Init swarm
        X = np.zeros((self.num_particles, dim), dtype=float)
        V = np.zeros_like(X)

        # Initialize all particles around nominal with bounded random offsets (ordered by construction).
        spread = self.waypoint_spread_m
        X[:] = np.random.uniform(-spread, spread, size=(self.num_particles, dim))

        pbest = X.copy()
        pbest_val = np.full((self.num_particles,), float("inf"), dtype=float)
        gbest = None
        gbest_val = float("inf")

        # Initial evaluation
        for i in range(self.num_particles):
            if time.time() > deadline:
                break
            path = self._decode_offsets_to_path(start, goal, nominal, X[i], xlim, ylim)
            val = self._path_cost(path, deadline)
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest[i] = X[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest = X[i].copy()
                if self.stats.t_first is None and val < float("inf"):
                    self.stats.t_first = time.time() - t0

        # Main PSO loop
        for it in range(self.max_iter):
            if time.time() > deadline:
                break

            for i in range(self.num_particles):
                if time.time() > deadline:
                    break

                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                g = gbest if gbest is not None else pbest[i]
                V[i] = self.w * V[i] + self.c1 * r1 * (pbest[i] - X[i]) + self.c2 * r2 * (g - X[i])

                # Limit step to avoid wild oscillations
                V[i] = np.clip(V[i], -5000.0, 5000.0)
                X[i] = X[i] + V[i]

                # Keep offsets bounded so waypoints remain inside corridor around nominal
                X[i] = np.clip(X[i], -spread, spread)

                path = self._decode_offsets_to_path(start, goal, nominal, X[i], xlim, ylim)
                val = self._path_cost(path, deadline)

                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = X[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest = X[i].copy()
                    if self.stats.t_first is None and val < float("inf"):
                        self.stats.t_first = time.time() - t0

        self.stats.t_total = time.time() - t0
        self.stats.best_cost = float(gbest_val)

        if gbest is None or not np.isfinite(gbest_val):
            self.stats.status = "FAIL"
            return None

        self.stats.status = "SUCCESS"
        return self._decode_offsets_to_path(start, goal, nominal, gbest, xlim, ylim)
