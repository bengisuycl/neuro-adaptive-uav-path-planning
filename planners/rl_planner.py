import inspect
import io
import math
import os
import time
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn

from wcci_conference_project.ai_modules.neural_risk_map import NeuralRiskMapper
from wcci_conference_project.planners.astar import AstarPlanner
from wcci_conference_project.utils.feasibility import FeasibilityEngine
import wcci_conference_project.ai_modules.dqn_pilot as dqn_pilot_module


DQNClass = None
AgentClass = None
for name, obj in inspect.getmembers(dqn_pilot_module):
    if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
        DQNClass = obj
    if inspect.isclass(obj) and name == "F16Agent":
        AgentClass = obj

if DQNClass is None:
    raise ImportError("HATA: dqn_pilot.py icinde PyTorch tabanli bir sinir agi sinifi bulunamadi!")


class RLPlanner:
    """
    RL-based pilot with hybrid tactical scoring.

    The DQN remains the primary policy prior, but action selection is stabilized by
    additional mission-aware terms:
    - progress toward the goal
    - heading alignment
    - local segment risk / feasibility
    - revisit avoidance to reduce oscillation and loops

    This keeps the planner RL-driven while making it far more usable in long-range
    scenarios such as S3_Long.
    """

    _cached_state_dicts = {}
    _cached_guides = {}

    def __init__(self, terrain, threats, aircraft=None, model_path="f16_dqn_weights.pth", safe_alt=800.0, **kwargs):
        self.terrain = terrain
        self.threats = threats
        self.aircraft = aircraft
        self.safe_alt = safe_alt
        self.verbose = bool(kwargs.get("verbose", False))

        self.feasibility = kwargs.get("feasibility")
        if self.feasibility is None:
            self.feasibility = FeasibilityEngine(
                terrain,
                threats,
                check_resolution=kwargs.get("check_resolution", 50.0),
                w_risk=kwargs.get("threat_weight", 20.0),
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = AgentClass(state_dim=8, action_dim=5) if AgentClass is not None else None
        self.dqn = self.agent.policy_net if self.agent is not None else DQNClass(state_dim=8, action_dim=5).to(self.device)
        default_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "f16_dqn_weights.pth")
        self.model_path = model_path if os.path.isabs(model_path) else default_model_path

        self.weights_loaded = False
        self.weight_error = None
        try:
            state_dict = self._cached_state_dicts.get(self.model_path)
            if state_dict is None:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self._cached_state_dicts[self.model_path] = state_dict
            if self.agent is not None:
                self.agent.policy_net.load_state_dict(state_dict)
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                self.agent.policy_net.eval()
                self.agent.target_net.eval()
            else:
                self.dqn.load_state_dict(state_dict)
                self.dqn.eval()
            self.weights_loaded = True
            if self.verbose:
                print(f"RL Pilot Weights Loaded: {self.model_path}")
        except Exception as e:
            self.weight_error = str(e)
            if self.verbose:
                print(f"RL Pilot Weight Error: {e}")

        # Keep action space compatible with the saved 5-action DQN.
        self.actions = [-30.0, -15.0, 0.0, 15.0, 30.0]
        self.stats = {"t_total": 0.0, "status": "FAIL"}
        self.visit_resolution = float(kwargs.get("visit_resolution", 1200.0))
        self.guide_step = float(kwargs.get("guide_step", 500.0))
        self.guide_lookahead = float(kwargs.get("guide_lookahead", 2800.0))
        self.use_global_guide = bool(kwargs.get("use_global_guide", True))
        self.guide_refresh_steps = int(kwargs.get("guide_refresh_steps", 10))
        self.use_neural_risk = bool(kwargs.get("use_neural_risk", False))
        self.neural_risk_weight = float(kwargs.get("neural_risk_weight", 0.18))
        self.neural_risk_samples = int(kwargs.get("neural_risk_samples", 3))
        self.neural_corridor_half_width = float(kwargs.get("neural_corridor_half_width", 1800.0))
        self.neural_mapper = NeuralRiskMapper(verbose=False) if self.use_neural_risk else None
        self.last_plan_debug = {}

    def _normalize_heading_error(self, heading_deg, gx, gy, x, y):
        target_h = math.degrees(math.atan2(gy - y, gx - x))
        h_err = (target_h - heading_deg + 180.0) % 360.0 - 180.0
        return h_err / 180.0

    def _get_state(self, *args):
        """
        Supports both:
        - _get_state(x, y, heading_deg, gx, gy)
        - _get_state(pos_xy, heading_rad_or_deg, goal_xy)
        """
        if len(args) == 5:
            x, y, heading_deg, gx, gy = args
        elif len(args) == 3:
            pos_xy, heading_value, goal_xy = args
            x, y = float(pos_xy[0]), float(pos_xy[1])
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            heading_deg = math.degrees(float(heading_value)) if abs(float(heading_value)) <= 2.0 * math.pi else float(heading_value)
        else:
            raise ValueError("Unexpected _get_state signature")

        dx = gx - x
        dy = gy - y
        dist_norm = min(math.hypot(dx, dy) / 90000.0, 1.5)
        h_err = self._normalize_heading_error(heading_deg, gx, gy, x, y)

        sense_dist = float(np.clip(1800.0 + 0.04 * math.hypot(dx, dy), 1800.0, 4200.0))
        r_ll = self._sense_risk(x, y, heading_deg - 60.0, sense_dist)
        r_l = self._sense_risk(x, y, heading_deg - 30.0, sense_dist)
        r_c = self._sense_risk(x, y, heading_deg, sense_dist)
        r_r = self._sense_risk(x, y, heading_deg + 30.0, sense_dist)
        r_rr = self._sense_risk(x, y, heading_deg + 60.0, sense_dist)
        alt_err = 0.0

        return torch.FloatTensor([dist_norm, h_err, alt_err, r_ll, r_l, r_c, r_r, r_rr])

    def _sense_risk(self, x, y, heading_deg, dist_m):
        nx = x + dist_m * math.cos(math.radians(heading_deg))
        ny = y + dist_m * math.sin(math.radians(heading_deg))
        ok, cost, _ = self.feasibility.check_segment((x, y, self.safe_alt), (nx, ny, self.safe_alt), self.safe_alt)
        if not ok:
            return 1.0
        return float(np.clip(cost / max(dist_m, 1.0) / 25.0, 0.0, 1.0))

    def _estimate_neural_corridor_risk(self, p1, p2):
        if not self.use_neural_risk or self.neural_mapper is None or not getattr(self.neural_mapper, "ready", False):
            return 0.0

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        seg = p2 - p1
        seg_norm = float(np.linalg.norm(seg))
        tangent = np.array([1.0, 0.0], dtype=float) if seg_norm < 1e-6 else seg / seg_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        samples = max(1, self.neural_risk_samples)
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
        return float(np.mean(preds)) if preds else 0.0

    def _step_kinematics(self, x, y, heading_deg, turn_angle_deg, step_size):
        new_h = (heading_deg + turn_angle_deg) % 360.0
        rad = math.radians(new_h)
        nx = x + step_size * math.cos(rad)
        ny = y + step_size * math.sin(rad)
        return nx, ny, new_h

    def _visit_key(self, x, y):
        return int(round(x / self.visit_resolution)), int(round(y / self.visit_resolution))

    def _goal_direct_reachable(self, x, y, heading_deg, gx, gy, dist_to_goal):
        if dist_to_goal > 4500.0:
            return False
        ok, _, _ = self.feasibility.check_segment((x, y, self.safe_alt), (gx, gy, self.safe_alt), self.safe_alt)
        if not ok:
            return False
        h_err = abs(self._normalize_heading_error(heading_deg, gx, gy, x, y))
        return h_err <= 0.55

    def _compute_guide_path(self, start, goal):
        if not self.use_global_guide:
            return None
        cache_key = (
            round(float(start[0]), 1),
            round(float(start[1]), 1),
            round(float(goal[0]), 1),
            round(float(goal[1]), 1),
            tuple(
                (
                    round(float(getattr(t, "x_center", getattr(t, "x", 0.0))), 1),
                    round(float(getattr(t, "y_center", getattr(t, "y", 0.0))), 1),
                    round(float(getattr(t, "radius", 0.0)), 1),
                    str(getattr(t, "type", "")),
                    int(bool(getattr(t, "dynamic", False))),
                )
                for t in self.threats
            ),
        )
        if cache_key in self._cached_guides:
            return self._cached_guides[cache_key]
        try:
            guide = AstarPlanner(
                self.terrain,
                self.threats,
                safe_alt=self.safe_alt,
                step=self.guide_step,
                heuristic_weight=1.2,
                goal_tolerance=2500.0,
                check_resolution=50.0,
                threat_weight=20.0,
            )
            with redirect_stdout(io.StringIO()):
                path = guide.plan(start, goal)
            self._cached_guides[cache_key] = path
            return path
        except Exception:
            return None

    def _guide_target(self, guide_path, curr_x, curr_y, goal_x, goal_y):
        if not guide_path or len(guide_path) < 2:
            return goal_x, goal_y

        guide_xy = np.asarray(guide_path, dtype=float)[:, :2]
        pos = np.array([curr_x, curr_y], dtype=float)
        dists = np.linalg.norm(guide_xy - pos, axis=1)
        closest_idx = int(np.argmin(dists))

        accum = 0.0
        for i in range(closest_idx, len(guide_xy) - 1):
            seg = float(np.linalg.norm(guide_xy[i + 1] - guide_xy[i]))
            accum += seg
            if accum >= self.guide_lookahead:
                return float(guide_xy[i + 1, 0]), float(guide_xy[i + 1, 1])

        return goal_x, goal_y

    def _choose_action(self, curr_x, curr_y, curr_h, nav_x, nav_y, goal_x, goal_y, q_values, visited, last_action_idx, step_size):
        curr_dist = math.hypot(goal_x - curr_x, goal_y - curr_y)
        curr_nav_dist = math.hypot(nav_x - curr_x, nav_y - curr_y)
        valid = []
        emergency = []

        q_vals = np.asarray(q_values, dtype=float).reshape(-1)
        q_invalid = (q_vals.size != len(self.actions)) or (not np.all(np.isfinite(q_vals)))
        if q_invalid:
            # Fall back to a neutral prior when the network output is invalid.
            q_vals = np.nan_to_num(q_vals, nan=0.0, posinf=0.0, neginf=0.0)
            if q_vals.size < len(self.actions):
                q_vals = np.pad(q_vals, (0, len(self.actions) - q_vals.size), mode="constant")
            elif q_vals.size > len(self.actions):
                q_vals = q_vals[: len(self.actions)]

        for a_idx, turn_cmd in enumerate(self.actions):
            nx, ny, nh = self._step_kinematics(curr_x, curr_y, curr_h, turn_cmd, step_size)
            ok, seg_cost, _ = self.feasibility.check_segment((curr_x, curr_y, self.safe_alt), (nx, ny, self.safe_alt), self.safe_alt)
            next_dist = math.hypot(goal_x - nx, goal_y - ny)
            next_nav_dist = math.hypot(nav_x - nx, nav_y - ny)
            progress = curr_dist - next_dist
            nav_progress = curr_nav_dist - next_nav_dist
            h_err = abs(self._normalize_heading_error(nh, nav_x, nav_y, nx, ny))
            revisit = visited.get(self._visit_key(nx, ny), 0)
            risk_norm = 1.0 if not ok else float(np.clip((seg_cost - step_size) / max(step_size, 1.0) / 25.0, 0.0, 1.0))
            neural_risk = self._estimate_neural_corridor_risk((curr_x, curr_y), (nx, ny))
            neural_risk_norm = neural_risk / (neural_risk + 0.8) if neural_risk > 0.0 else 0.0

            data = {
                "idx": a_idx,
                "ok": ok,
                "progress": progress,
                "nav_progress": nav_progress,
                "risk": risk_norm,
                "neural_risk": neural_risk_norm,
                "heading_err": h_err,
                "revisit": revisit,
                "turn_mag": abs(turn_cmd) / 30.0,
                "next_dist": next_dist,
            }
            emergency.append(data)
            if ok:
                valid.append(data)

        candidates = valid if valid else emergency
        q_min = float(np.min(q_vals))
        q_rng = float(np.max(q_vals) - q_min)

        best = None
        best_score = -1e18
        best_greedy = None
        best_greedy_score = -1e18
        for data in candidates:
            q_norm = (q_vals[data["idx"]] - q_min) / max(q_rng, 1e-6)
            progress_norm = np.clip(data["progress"] / max(step_size, 1.0), -1.0, 1.0)
            nav_progress_norm = np.clip(data["nav_progress"] / max(step_size, 1.0), -1.0, 1.0)
            stick_bonus = 0.06 if last_action_idx is not None and data["idx"] == last_action_idx else 0.0

            score = (
                0.12 * q_norm
                + 0.18 * progress_norm
                + 0.34 * nav_progress_norm
                - 0.18 * data["heading_err"]
                - 0.28 * data["risk"]
                - self.neural_risk_weight * data["neural_risk"]
                - 0.10 * min(data["revisit"], 3)
                - 0.05 * data["turn_mag"]
                + stick_bonus
            )
            if data["next_dist"] < 4000.0:
                score += 0.08 * progress_norm

            if score > best_score:
                best_score = score
                best = data["idx"]

            greedy_score = (
                0.40 * nav_progress_norm
                + 0.25 * progress_norm
                - 0.18 * data["heading_err"]
                - 0.22 * data["risk"]
                - self.neural_risk_weight * data["neural_risk"]
                - 0.10 * min(data["revisit"], 3)
            )
            if greedy_score > best_greedy_score:
                best_greedy_score = greedy_score
                best_greedy = data["idx"]

        # If the DQN distribution is nearly flat, rely on the safer guide-aligned fallback.
        if q_invalid or q_rng < 0.10:
            return best_greedy if best_greedy is not None else best

        # If the selected action barely improves guide progress, prefer the greedy fallback.
        if best is not None:
            best_data = next((d for d in candidates if d["idx"] == best), None)
            if best_data is not None:
                weak_progress = best_data["nav_progress"] < 0.10 * step_size
                wrong_direction = best_data["nav_progress"] < 0.0
                high_risk = best_data["risk"] > 0.40 or best_data["neural_risk"] > 0.45
                if (weak_progress or high_risk) and best_greedy is not None:
                    return best_greedy
                if wrong_direction and best_greedy is not None:
                    return best_greedy

        return best

    def plan(self, start, goal, time_budget_s=5.0, **kwargs):
        t0 = time.time()
        curr_x, curr_y, _ = start
        goal_x, goal_y, _ = goal
        curr_h = math.degrees(math.atan2(goal_y - curr_y, goal_x - curr_x))
        guide_path = self._compute_guide_path(start, goal)
        self.last_plan_debug = {
            "weights_loaded": self.weights_loaded,
            "weight_error": self.weight_error,
            "guide_available": bool(guide_path),
            "steps_taken": 0,
            "fail_reason": "",
        }

        path = [(curr_x, curr_y, self.safe_alt)]
        max_steps = 420
        visited = {self._visit_key(curr_x, curr_y): 1}
        last_action_idx = None
        no_progress_count = 0
        best_dist = math.hypot(goal_x - curr_x, goal_y - curr_y)
        has_dynamic_threats = any(getattr(th, "dynamic", False) for th in self.threats)
        step_idx = 0

        for _ in range(max_steps):
            if time.time() - t0 > time_budget_s:
                break

            step_idx += 1
            self.last_plan_debug["steps_taken"] = step_idx

            dist_to_goal = math.hypot(goal_x - curr_x, goal_y - curr_y)
            if self._goal_direct_reachable(curr_x, curr_y, curr_h, goal_x, goal_y, dist_to_goal):
                path.append((goal_x, goal_y, self.safe_alt))
                self.stats["status"] = "SUCCESS"
                self.last_plan_debug["fail_reason"] = "goal_direct"
                break

            if dist_to_goal < 1800.0:
                path.append((goal_x, goal_y, self.safe_alt))
                self.stats["status"] = "SUCCESS"
                self.last_plan_debug["fail_reason"] = "goal_radius"
                break

            if self.use_global_guide and has_dynamic_threats and (
                step_idx % self.guide_refresh_steps == 0 or no_progress_count >= 5
            ):
                refreshed = self._compute_guide_path((curr_x, curr_y, self.safe_alt), goal)
                if refreshed:
                    guide_path = refreshed

            nav_x, nav_y = self._guide_target(guide_path, curr_x, curr_y, goal_x, goal_y)
            step_size = float(np.clip(900.0 + 0.045 * dist_to_goal, 1000.0, 2300.0))
            if no_progress_count >= 4:
                step_size = max(900.0, step_size * 0.75)

            state_tensor = self._get_state(curr_x, curr_y, curr_h, nav_x, nav_y).unsqueeze(0).to(self.device)
            q_values = self.dqn(state_tensor).squeeze().detach().cpu().numpy()
            chosen_action = self._choose_action(
                curr_x,
                curr_y,
                curr_h,
                nav_x,
                nav_y,
                goal_x,
                goal_y,
                q_values,
                visited,
                last_action_idx,
                step_size,
            )

            if chosen_action is None:
                self.last_plan_debug["fail_reason"] = "no_action"
                break

            curr_x, curr_y, curr_h = self._step_kinematics(curr_x, curr_y, curr_h, self.actions[chosen_action], step_size)
            path.append((curr_x, curr_y, self.safe_alt))
            last_action_idx = chosen_action

            vk = self._visit_key(curr_x, curr_y)
            visited[vk] = visited.get(vk, 0) + 1

            new_dist = math.hypot(goal_x - curr_x, goal_y - curr_y)
            if new_dist < best_dist - 120.0:
                best_dist = new_dist
                no_progress_count = 0
            else:
                no_progress_count += 1

        self.stats["t_total"] = time.time() - t0
        if len(path) <= 1 and not self.last_plan_debug.get("fail_reason"):
            self.last_plan_debug["fail_reason"] = "empty_path"
        return path if len(path) > 1 else None
