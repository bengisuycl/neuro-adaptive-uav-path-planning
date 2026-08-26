# planners/astar.py
import heapq
import math
import time
from wcci_conference_project.utils.feasibility import FeasibilityEngine


class AstarPlanner:
    def __init__(self, terrain, threats, safe_alt=800.0, step=100, heuristic_weight=1.5, **kwargs):
        self.terrain = terrain
        self.safe_alt = safe_alt
        self.step = step
        self.heuristic_weight = heuristic_weight
        self.goal_tolerance = kwargs.get('goal_tolerance', 2000.0)
        self.feasibility = FeasibilityEngine(terrain, threats, kwargs.get('check_resolution', 50),
                                             kwargs.get('threat_weight', 20),
                                             use_neural_risk=kwargs.get('use_neural_risk', False),
                                             neural_risk_weight=kwargs.get('neural_risk_weight', 0.0),
                                             neural_risk_samples=kwargs.get('neural_risk_samples', 3),
                                             neural_corridor_half_width=kwargs.get('neural_corridor_half_width', 1800.0))
        # İstatistik
        self.stats = {"t_first": None, "n_nodes": 0}

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [(p[0], p[1], self.safe_alt) for p in path]

    def plan(self, start, goal):
        t0 = time.time()
        print("   🔄 A* Planner Started")
        self.stats["n_nodes"] = 0
        self.stats["t_first"] = None

        start2d = (float(start[0]), float(start[1]))
        goal2d = (float(goal[0]), float(goal[1]))

        open_set = []
        heapq.heappush(open_set, (0.0, start2d))
        came_from = {}
        g_score = {start2d: 0.0}
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_set:
            self.stats["n_nodes"] += 1
            current_f, current = heapq.heappop(open_set)

            if self.heuristic(current, goal2d) < self.goal_tolerance:
                path = self.reconstruct_path(came_from, current)
                is_feasible, _, _ = self.feasibility.check_segment(current, goal2d, self.safe_alt)
                if is_feasible: path.append((goal2d[0], goal2d[1], self.safe_alt))

                self.stats["t_first"] = time.time() - t0  # A* için T_first = T_total
                return path

            for dx, dy in motions:
                neighbor = (current[0] + dx * self.step, current[1] + dy * self.step)
                is_feasible, segment_cost, _ = self.feasibility.check_segment(current, neighbor, self.safe_alt)
                if not is_feasible: continue

                tentative_g = g_score[current] + segment_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal2d) * self.heuristic_weight
                    heapq.heappush(open_set, (f_score, neighbor))
        return None
