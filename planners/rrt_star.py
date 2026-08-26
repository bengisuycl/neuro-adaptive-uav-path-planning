
# planners/rrt_star.py
# FINAL VERSION: Time-Budget Compliance + Parameter Reset Fix

import math
import numpy as np
import random
import sys
import time
from wcci_conference_project.utils.feasibility import FeasibilityEngine
class RRTStarPlanner:
    def __init__(self, terrain, threats, safe_alt=800.0, max_iter=10000, **kwargs):
        self.terrain = terrain
        self.safe_alt = safe_alt
        self.max_iter = max_iter
        self.step_len = kwargs.get('step_len', 2000.0)
        self.goal_sample_rate = kwargs.get('goal_sample_rate', 0.2)
        self.search_radius = kwargs.get('search_radius', 5000.0)
        self.goal_tolerance = kwargs.get('goal_tolerance', 3000.0)
        self.feasibility = FeasibilityEngine(terrain, threats, kwargs.get('check_resolution', 50),
                                             kwargs.get('threat_weight', 20),
                                             use_neural_risk=kwargs.get('use_neural_risk', False),
                                             neural_risk_weight=kwargs.get('neural_risk_weight', 0.0),
                                             neural_risk_samples=kwargs.get('neural_risk_samples', 3),
                                             neural_corridor_half_width=kwargs.get('neural_corridor_half_width', 1800.0))
        self.bounds = terrain.dataset.bounds
        self.stats = {"t_first": None, "n_nodes": 0}

    class Node:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.parent = None
            self.cost = 0.0

    def _set_sampling_bounds(self, start, goal, margin=15000.0):
        min_x = min(start[0], goal[0]) - margin
        max_x = max(start[0], goal[0]) + margin
        min_y = min(start[1], goal[1]) - margin
        max_y = max(start[1], goal[1]) + margin
        self.x_min = max(self.bounds.left, min_x)
        self.x_max = min(self.bounds.right, max_x)
        self.y_min = max(self.bounds.bottom, min_y)
        self.y_max = min(self.bounds.top, max_y)

    def _get_random_node(self, goal_node):
        if random.random() > self.goal_sample_rate:
            return self.Node(random.uniform(self.x_min, self.x_max), random.uniform(self.y_min, self.y_max))
        return goal_node

    def _get_nearest_node_index(self, node_list, rnd):
        dlist = [(node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2 for node in node_list]
        return dlist.index(min(dlist))

    def _steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y), math.atan2(to_node.y - from_node.y,
                                                                                            to_node.x - from_node.x)
        if extend_length > d: extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.parent = from_node
        return new_node

    def _find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = min(self.search_radius * math.sqrt((math.log(nnode) / nnode)), self.search_radius)
        return [i for i, n in enumerate(self.node_list) if (n.x - new_node.x) ** 2 + (n.y - new_node.y) ** 2 <= r ** 2]

    def _choose_parent(self, new_node, near_inds, step_cost, default_parent):
        if not near_inds:
            new_node.parent = default_parent
            new_node.cost = default_parent.cost + step_cost
            return new_node
        costs = []
        valid_inds = []
        for i in near_inds:
            near_node = self.node_list[i]
            ok, c, _ = self.feasibility.check_segment((near_node.x, near_node.y), (new_node.x, new_node.y),
                                                      self.safe_alt)
            if ok:
                costs.append(near_node.cost + c)
                valid_inds.append(i)
        if not costs:
            new_node.parent = default_parent
            new_node.cost = default_parent.cost + step_cost
            return new_node
        min_ind = valid_inds[costs.index(min(costs))]
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min(costs)
        return new_node

    def _rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            ok, cost, _ = self.feasibility.check_segment((new_node.x, new_node.y), (near_node.x, near_node.y),
                                                         self.safe_alt)
            if ok and new_node.cost + cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_node.cost + cost

    def _generate_final_course(self, goal_node):
        path = [[goal_node.x, goal_node.y, self.safe_alt]]
        node = goal_node
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y, self.safe_alt])
        path.reverse()
        return path

    def plan(self, start, goal, time_budget_s=None, min_iter_before_exit=0, **kwargs):
        """
        RRT* planner with Time-Budget Enforcement.
        """
        t0 = time.time()

        # --- BUG FIX: Parametreleri sakla ve resetle ---
        original_sample_rate = self.goal_sample_rate

        try:
            # Feasibility counter reset
            if hasattr(self, "feasibility") and hasattr(self.feasibility, "check_count"):
                self.feasibility.check_count = 0

            self._set_sampling_bounds(start, goal, margin=10000.0)

            # Parametre adaptasyonu (bütçenin sonuna doğru agresifleşebilir)
            # Şu anlık basit tutuyoruz.

            self.stats = {"t_first": None, "n_nodes": 0, "checks": 0}

            start_node = self.Node(start[0], start[1])
            goal_node = self.Node(goal[0], goal[1])
            self.node_list = [start_node]

            best_goal_node = None
            min_goal_cost = float('inf')

            closest_node = None
            min_dist_to_goal = float('inf')

            for i in range(self.max_iter):
                # TIME BUDGET CHECK
                if time_budget_s is not None:
                    if (time.time() - t0) > float(time_budget_s):
                        break

                rnd = self._get_random_node(goal_node)
                nearest_ind = self._get_nearest_node_index(self.node_list, rnd)
                nearest_node = self.node_list[nearest_ind]
                new_node = self._steer(nearest_node, rnd, self.step_len)

                is_feasible, cost, _ = self.feasibility.check_segment(
                    (nearest_node.x, nearest_node.y), (new_node.x, new_node.y), self.safe_alt
                )

                if is_feasible:
                    near_inds = self._find_near_nodes(new_node)
                    new_node = self._choose_parent(new_node, near_inds, cost, default_parent=nearest_node)

                    if new_node:
                        self.node_list.append(new_node)
                        self._rewire(new_node, near_inds)

                        # Partial path takibi
                        d_to_goal = math.hypot(new_node.x - goal_node.x, new_node.y - goal_node.y)
                        if d_to_goal < min_dist_to_goal:
                            min_dist_to_goal = d_to_goal
                            closest_node = new_node

                        # Hedefe bağlantı
                        if d_to_goal <= self.goal_tolerance + self.step_len:
                            ok_f, c_f, _ = self.feasibility.check_segment(
                                (new_node.x, new_node.y), (goal_node.x, goal_node.y), self.safe_alt
                            )
                            if ok_f:
                                total = new_node.cost + c_f
                                if self.stats["t_first"] is None:
                                    self.stats["t_first"] = time.time() - t0

                                if total < min_goal_cost:
                                    min_goal_cost = total
                                    best_goal_node = self.Node(goal_node.x, goal_node.y)
                                    best_goal_node.parent = new_node
                                    best_goal_node.cost = total

            self.stats["n_nodes"] = len(self.node_list)
            self.stats["checks"] = getattr(self.feasibility, "check_count", 0)

            # Return best available solution
            if best_goal_node:
                return self._generate_final_course(best_goal_node)

            # Return partial path if no full path found
            if closest_node is not None:
                return self._generate_final_course(closest_node)

            return None

        finally:
            # BUG FIX: Sample rate'i bir sonraki run için eski haline getir
            self.goal_sample_rate = original_sample_rate

