# planners/f16_dijkstra.py
import heapq
import math
from wcci_conference_project.utils.feasibility import FeasibilityEngine


class F16DijkstraPlanner:
    def __init__(self, terrain, threats, aircraft,
                 safe_alt=800.0,
                 xy_resolution=200.0,
                 psi_resolution_deg=10.0,
                 goal_tolerance=3000.0,
                 check_resolution=50.0,  # Ortak
                 threat_weight=20.0,  # Ortak
                 **kwargs):

        self.terrain = terrain
        self.safe_alt = safe_alt
        self.aircraft = aircraft
        self.xy_res = xy_resolution
        self.psi_res_rad = math.radians(psi_resolution_deg)
        self.goal_tolerance = goal_tolerance

        # Terminal guidance parametreleri (IEEE fairness)
        self.enable_terminal_guidance = kwargs.get("enable_terminal_guidance", True)
        self.terminal_max_angle_deg = kwargs.get("terminal_max_angle_deg", 90.0)

        self.V = aircraft.V
        phi_rad = math.radians(getattr(aircraft, 'phi_max', 60.0))
        self.R_min = (self.V ** 2) / (9.81 * math.tan(phi_rad)) if phi_rad > 0 else 1000.0
        s = max(1, int(math.ceil((self.R_min * math.pi / 4.0) / self.xy_res)))
        self.motions = [
            (s, 0, s), (0, s, s), (-s, 0, s), (0, -s, s),
            (s, s, s * 1.414), (s, -s, s * 1.414),
            (-s, s, s * 1.414), (-s, -s, s * 1.414)
        ]
        self.max_turn_rad = (self.xy_res * s) / self.R_min * 1.5

        # ORTAK MOTOR
        self.feasibility = FeasibilityEngine(
            terrain, threats,
            check_resolution=check_resolution,
            w_risk=threat_weight,
            use_neural_risk=kwargs.get("use_neural_risk", False),
            neural_risk_weight=kwargs.get("neural_risk_weight", 0.0),
            neural_risk_samples=kwargs.get("neural_risk_samples", 3),
            neural_corridor_half_width=kwargs.get("neural_corridor_half_width", 1800.0),
        )

    def _discretize(self, x, y, psi):
        return (
            int(round(x / self.xy_res)),
            int(round(y / self.xy_res)),
            int(round(psi / self.psi_res_rad))
        )

    def _get_angle_diff(self, a1, a2):
        return (a2 - a1 + math.pi) % (2 * math.pi) - math.pi

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
        """
        Goal acceptance region içine girildiğinde, apple-to-apple benchmarking için
        (uygunsa) son noktadan goal merkezine LOS terminal segment ekler.
        """
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
        print("   🔄 F-16 Basic Planner Started (Standardized)")
        start2d = (float(start[0]), float(start[1]))
        goal2d = (float(goal[0]), float(goal[1]))

        psi0 = math.atan2(goal2d[1] - start2d[1], goal2d[0] - start2d[0])
        start_node = (start2d[0], start2d[1], psi0)
        start_key = self._discretize(*start_node)

        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, start_node))
        cost_so_far = {start_key: 0.0}
        came_from = {}

        while open_set:
            _, current_g, current_state = heapq.heappop(open_set)
            cx, cy, cpsi = current_state
            ckey = self._discretize(cx, cy, cpsi)

            # Goal acceptance check
            if math.hypot(cx - goal2d[0], cy - goal2d[1]) < self.goal_tolerance:
                path = self.reconstruct_path(came_from, ckey, current_state, start_node)

                # --- Terminal Guidance (LOS + heading gate) ---
                if self._try_terminal_guidance(cx, cy, cpsi, goal2d):
                    path.append((goal2d[0], goal2d[1], self.safe_alt))

                return path

            for dx, dy, d_mult in self.motions:
                nx = cx + dx * self.xy_res
                ny = cy + dy * self.xy_res

                # 1) Kinematik (heading change bound)
                angle_to = math.atan2(ny - cy, nx - cx)
                if abs(self._get_angle_diff(cpsi, angle_to)) > self.max_turn_rad:
                    continue

                # 2) Ortak Feasibility
                is_feasible, segment_cost, _ = self.feasibility.check_segment((cx, cy), (nx, ny), self.safe_alt)
                if not is_feasible:
                    continue

                new_g = current_g + segment_cost
                nkey = self._discretize(nx, ny, angle_to)

                if new_g < cost_so_far.get(nkey, float('inf')):
                    cost_so_far[nkey] = new_g
                    heapq.heappush(open_set, (new_g, new_g, (nx, ny, angle_to)))
                    came_from[nkey] = (ckey, current_state)

        return None
