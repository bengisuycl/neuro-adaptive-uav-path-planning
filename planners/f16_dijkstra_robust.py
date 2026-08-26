"""
Heading-aware tactical planner used as the T-GnP implementation.

The planner follows a discretized state-space search in which position and heading
are jointly considered. Conceptually, this is closer to a state-lattice /
differentially constrained search than to a purely geometric grid shortest path.

The implementation remains pragmatic for thesis use:
- heading-aware node expansion
- bank-limited turn feasibility
- adaptive segment speed for maneuverability
- optional neural risk augmentation as an exploratory study
"""

import heapq
import math

import numpy as np

from wcci_conference_project.ai_modules.neural_risk_map import NeuralRiskMapper
from wcci_conference_project.utils.feasibility import FeasibilityEngine


class F16DijkstraPlannerRobust:
    def __init__(
        self,
        terrain,
        threats,
        aircraft,
        safe_alt=800.0,
        xy_resolution=200.0,
        goal_tolerance=3000.0,
        steps_per_primitive=5,
        turn_penalty_factor=1.5,
        check_resolution=50.0,
        threat_weight=20.0,
        **kwargs,
    ):
        self.terrain = terrain
        self.safe_alt = safe_alt
        self.aircraft = aircraft
        self.xy_res = xy_resolution
        self.goal_tolerance = goal_tolerance
        self.turn_penalty_factor = turn_penalty_factor
        self.steps_per_primitive = steps_per_primitive
        self.psi_res_rad = math.radians(10.0)
        self.max_iter = 1_000_000
        self.phi_max_deg = float(kwargs.get("phi_max_deg", 60.0))
        self.speed_weight = float(kwargs.get("speed_weight", 0.35))
        self.stability_weight = float(kwargs.get("stability_weight", 0.55))

        self.enable_terminal_guidance = kwargs.get("enable_terminal_guidance", True)
        self.terminal_max_angle_deg = kwargs.get("terminal_max_angle_deg", 90.0)
        self.use_neural_risk = kwargs.get("use_neural_risk", True)
        self.neural_risk_weight = float(kwargs.get("neural_risk_weight", 1500.0))
        self.neural_risk_samples = int(kwargs.get("neural_risk_samples", 3))

        self.feasibility = FeasibilityEngine(
            terrain,
            threats,
            check_resolution=check_resolution,
            w_risk=threat_weight,
        )
        self.neural_mapper = NeuralRiskMapper(verbose=False) if self.use_neural_risk else None
        self.neural_corridor_half_width = float(kwargs.get("neural_corridor_half_width", 1800.0))

        self.motions = []
        self.R_min = 0.0
        self.scale_factor = 1

    def _recompute_kinematics(self):
        current_v = self.aircraft.V
        phi_rad = math.radians(self.phi_max_deg)
        self.R_min = (current_v ** 2) / (9.81 * math.tan(phi_rad)) if phi_rad > 0 else 500.0
        self.scale_factor = max(1, int(math.ceil((self.R_min * math.pi / 4.0) / self.xy_res)))
        s = self.scale_factor
        self.motions = [
            (s, 0, s),
            (0, s, s),
            (-s, 0, s),
            (0, -s, s),
            (s, s, s * 1.414),
            (s, -s, s * 1.414),
            (-s, s, s * 1.414),
            (-s, -s, s * 1.414),
        ]

    def _discretize(self, x, y, psi):
        return (
            int(round(x / self.xy_res)),
            int(round(y / self.xy_res)),
            int(round(psi / self.psi_res_rad)),
        )

    def _get_angle_diff(self, a1, a2):
        return (a2 - a1 + math.pi) % (2 * math.pi) - math.pi

    def _estimate_neural_risk_along_segment(self, p1, p2):
        if not self.use_neural_risk or self.neural_mapper is None or not getattr(self.neural_mapper, "ready", False):
            return 0.0

        samples = max(1, self.neural_risk_samples)
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        seg = p2 - p1
        seg_norm = float(np.linalg.norm(seg))
        tangent = np.array([1.0, 0.0], dtype=float) if seg_norm < 1e-6 else seg / seg_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        offsets = [0.0]
        if self.neural_corridor_half_width > 1.0:
            offsets.extend([-self.neural_corridor_half_width, self.neural_corridor_half_width])

        query_points = []
        for i in range(samples):
            t = (i + 1) / (samples + 1)
            base_xy = p1 + t * seg
            for off in offsets:
                pt = base_xy + off * normal
                query_points.append((float(pt[0]), float(pt[1])))

        terrain_h = self.terrain.get_height(float(p1[0]), float(p1[1]))
        terrain_h = 0.0 if terrain_h is None or math.isnan(terrain_h) else float(terrain_h)
        agl_m = max(50.0, float(self.safe_alt) - terrain_h)
        preds = self.neural_mapper.get_risk_at_points(query_points, self.terrain, self.feasibility.threats, agl_m=agl_m)
        if not preds:
            return 0.0
        return float(sum(preds) / len(preds))

    def _local_turn_radius(self, speed_mps):
        phi_rad = math.radians(self.phi_max_deg)
        if phi_rad <= 0.0:
            return 500.0
        return (speed_mps ** 2) / (9.81 * math.tan(phi_rad))

    def _adaptive_segment_speed(self, diff_rad, dist_to_goal, neural_risk):
        """
        Select a local planning speed in [V_min, V_max].

        Logic:
        - straighter segments and long-range progress -> faster
        - larger heading change / higher neural risk -> slower
        - keep a floor above stall for stability

        This is a planning-side speed heuristic, not a replacement for the
        aircraft dynamics. The actual speed evolution is still handled in the
        simulation model through thrust/drag dynamics.
        """
        v_min = float(getattr(self.aircraft, "V_min", 100.0))
        v_max = float(getattr(self.aircraft, "V_max", 320.0))
        v_cruise = float(getattr(self.aircraft, "V_cruise", 220.0))

        turn_severity = np.clip(diff_rad / math.radians(50.0), 0.0, 1.0)
        risk_severity = neural_risk / (neural_risk + 0.8) if neural_risk > 0.0 else 0.0
        goal_drive = np.clip(dist_to_goal / 90000.0, 0.0, 1.0)

        # High turn and high risk should slow the aircraft down.
        caution = 0.55 * turn_severity + 0.45 * risk_severity
        speed_frac = 0.30 + 0.55 * goal_drive + 0.20 * (1.0 - caution)
        speed_frac = float(np.clip(speed_frac, 0.18, 0.98))

        target_v = v_min + speed_frac * (v_max - v_min)
        if caution > 0.55:
            target_v = min(target_v, v_cruise)
        return float(np.clip(target_v, v_min, v_max))

    def _densify_path(self, coarse_path, density_res=100.0):
        if not coarse_path or len(coarse_path) < 2:
            return coarse_path

        dense = [coarse_path[0]]
        for i in range(len(coarse_path) - 1):
            p1, p2 = coarse_path[i], coarse_path[i + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps = int(dist / density_res)
            for k in range(1, steps + 1):
                t = k / (steps + 1)
                dense.append((
                    p1[0] + (p2[0] - p1[0]) * t,
                    p1[1] + (p2[1] - p1[1]) * t,
                    self.safe_alt,
                ))
            dense.append(p2)
        return dense

    def reconstruct_path(self, came_from, current_key, current_state, start_state):
        path = []
        curr_k = current_key
        curr_s = current_state
        while curr_k in came_from:
            path.append((curr_s[0], curr_s[1], self.safe_alt))
            curr_k, curr_s = came_from[curr_k]
        path.append((start_state[0], start_state[1], self.safe_alt))
        path.reverse()
        return path

    def _try_terminal_guidance(self, cx, cy, cpsi, goal2d):
        if not self.enable_terminal_guidance:
            return False

        is_feasible_to_goal, _, _ = self.feasibility.check_segment((cx, cy), goal2d, self.safe_alt)
        if not is_feasible_to_goal:
            return False

        angle_to_goal = math.atan2(goal2d[1] - cy, goal2d[0] - cx)
        angle_diff = abs(self._get_angle_diff(cpsi, angle_to_goal))
        max_ang = math.radians(float(self.terminal_max_angle_deg))
        return angle_diff < max_ang

    def plan(self, start, goal):
        self._recompute_kinematics()
        planner_mode = "Adaptive Speed"
        if self.use_neural_risk and self.neural_mapper is not None and getattr(self.neural_mapper, "ready", False):
            planner_mode += " + Neural Risk"
        print(f"   F-16 Tactical Planner Started ({planner_mode})")
        stats = {"expanded": 0, "accepted": 0}

        start2d = (float(start[0]), float(start[1]))
        goal2d = (float(goal[0]), float(goal[1]))
        psi0 = math.atan2(goal2d[1] - start2d[1], goal2d[0] - start2d[0])
        start_node = (start2d[0], start2d[1], psi0)
        start_key = self._discretize(*start_node)

        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, start_node))
        cost_so_far = {start_key: 0.0}
        came_from = {}
        best_dist = float("inf")
        best_node = None

        while open_set:
            stats["expanded"] += 1
            if stats["expanded"] > self.max_iter:
                break

            _, current_g, current_state = heapq.heappop(open_set)
            cx, cy, cpsi = current_state
            ckey = self._discretize(cx, cy, cpsi)

            if current_g > cost_so_far.get(ckey, float("inf")):
                continue

            dist_to_goal = math.hypot(cx - goal2d[0], cy - goal2d[1])
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_node = current_state

            if dist_to_goal < self.goal_tolerance:
                raw = self.reconstruct_path(came_from, ckey, current_state, start_node)
                if self._try_terminal_guidance(cx, cy, cpsi, goal2d):
                    raw.append((goal2d[0], goal2d[1], self.safe_alt))
                return self._densify_path(raw, self.xy_res)

            for dx, dy, d_mult in self.motions:
                nx = cx + dx * self.xy_res
                ny = cy + dy * self.xy_res
                angle_to = math.atan2(ny - cy, nx - cx)
                diff = abs(self._get_angle_diff(cpsi, angle_to))
                step_dist = self.xy_res * d_mult
                if diff > (step_dist / self.R_min) * 1.5:
                    continue

                is_feasible, segment_base_cost, _ = self.feasibility.check_segment((cx, cy), (nx, ny), self.safe_alt)
                if not is_feasible:
                    continue

                dist_after = math.hypot(nx - goal2d[0], ny - goal2d[1])
                neural_risk = self._estimate_neural_risk_along_segment((cx, cy), (nx, ny))
                seg_speed = self._adaptive_segment_speed(diff, dist_after, neural_risk)
                local_r_min = self._local_turn_radius(seg_speed)
                local_turn_limit = (step_dist / max(local_r_min, 1.0)) * 1.6
                if diff > local_turn_limit:
                    continue

                turn_penalty = (diff / math.pi) * self.turn_penalty_factor * step_dist
                neural_penalty = self.neural_risk_weight * neural_risk * step_dist / max(1.0, self.xy_res)
                time_cost = self.speed_weight * (step_dist / max(seg_speed, 1.0))
                stability_penalty = self.stability_weight * (1.0 - seg_speed / max(float(getattr(self.aircraft, "V_max", 320.0)), 1.0)) * step_dist / max(1.0, self.xy_res)
                progress_bonus = 0.08 * max(0.0, dist_to_goal - dist_after)
                new_g = current_g + segment_base_cost + turn_penalty + neural_penalty + time_cost + stability_penalty - progress_bonus

                nkey = self._discretize(nx, ny, angle_to)
                if new_g < cost_so_far.get(nkey, float("inf")):
                    cost_so_far[nkey] = new_g
                    priority = new_g
                    heapq.heappush(open_set, (priority, new_g, (nx, ny, angle_to)))
                    came_from[nkey] = (ckey, current_state)
                    stats["accepted"] += 1

        if best_node:
            raw = self.reconstruct_path(came_from, self._discretize(*best_node), best_node, start_node)
            bx, by, bpsi = best_node
            if math.hypot(bx - goal2d[0], by - goal2d[1]) < self.goal_tolerance:
                if self._try_terminal_guidance(bx, by, bpsi, goal2d):
                    raw.append((goal2d[0], goal2d[1], self.safe_alt))
            return self._densify_path(raw, self.xy_res)

        return None
