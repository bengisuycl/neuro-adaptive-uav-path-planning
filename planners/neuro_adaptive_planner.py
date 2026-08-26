import time

import numpy as np

from wcci_conference_project.ai_modules.neural_risk_map import NeuralRiskMapper
from wcci_conference_project.ai_modules.rl_advisor import RLPilotAdvisor
from wcci_conference_project.utils.feasibility import FeasibilityEngine


class NeuroAdaptivePlanner:
    """
    Proposed framework:
    1. Perception: DNN-TRE risk summary
    2. Decision: RL Advisor
    3. Action: RL-Pilot / K-GNP / T-GnP
    """

    def __init__(self, terrain, threats, planner_factories, **kwargs):
        self.terrain = terrain
        self.threats = threats
        self.factories = planner_factories
        self.safe_alt = kwargs.get("safe_alt", 800.0)
        self.use_dnn_tre = bool(kwargs.get("use_dnn_tre", True))

        self.feasibility = kwargs.get("feasibility")
        if self.feasibility is None:
            self.feasibility = FeasibilityEngine(self.terrain, self.threats)

        self.advisor = RLPilotAdvisor(
            terrain=self.terrain,
            threats=self.threats,
            feasibility=self.feasibility,
            safe_alt=self.safe_alt,
        )
        self.risk_oracle = NeuralRiskMapper(verbose=False) if self.use_dnn_tre else None
        self.stats = {}

    def _summarize_risk_field_geometric(self, start, goal):
        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])
        seg = np.array([gx - sx, gy - sy], dtype=float)
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm < 1e-6:
            line_pts = [(sx, sy)]
            corridor_pts = [(sx, sy)]
        else:
            tangent = seg / seg_norm
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            line_pts = []
            corridor_pts = []
            lateral_offsets = [0.0, -2500.0, 2500.0]
            for t in np.linspace(0.0, 1.0, 29):
                base_xy = np.array([sx, sy], dtype=float) + t * seg
                base_pt = (float(base_xy[0]), float(base_xy[1]))
                line_pts.append(base_pt)
                for off in lateral_offsets:
                    pt = base_xy + off * normal
                    corridor_pts.append((float(pt[0]), float(pt[1])))

        def point_risk(px, py):
            risk = 0.0
            for threat in self.threats:
                val = threat.get_risk_cost(px, py, cost_scale=1.0)
                if val == float("inf"):
                    return 2.0
                risk += float(val)
            return float(risk)

        preds = np.asarray([point_risk(px, py) for px, py in corridor_pts], dtype=float)
        corridor_preds = np.asarray([point_risk(px, py) for px, py in line_pts], dtype=float)
        if preds.size == 0:
            return {
                "peak_risk": 0.0,
                "mean_risk": 0.0,
                "corridor_risk": 0.0,
                "high_risk_fraction": 0.0,
            }

        peak_risk = float(np.percentile(preds, 92))
        mean_risk = float(np.mean(preds))
        corridor_risk = float(np.mean(corridor_preds)) if corridor_preds.size else 0.0
        high_thresh = max(0.18, 0.60 * peak_risk)
        return {
            "peak_risk": peak_risk,
            "mean_risk": mean_risk,
            "corridor_risk": corridor_risk,
            "high_risk_fraction": float(np.mean(preds >= high_thresh)),
        }

    def _summarize_risk_field(self, start, goal):
        if not self.use_dnn_tre or self.risk_oracle is None or not getattr(self.risk_oracle, "ready", False):
            return self._summarize_risk_field_geometric(start, goal)

        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])
        seg = np.array([gx - sx, gy - sy], dtype=float)
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm < 1e-6:
            line_pts = [(sx, sy)]
            corridor_pts = [(sx, sy)]
        else:
            tangent = seg / seg_norm
            normal = np.array([-tangent[1], tangent[0]], dtype=float)

            line_pts = []
            corridor_pts = []
            # Sample the direct corridor and two lateral envelopes around it. This
            # reflects mission-relevant exposure much better than probing threat
            # centers directly, which can make even benign scenarios look severe.
            lateral_offsets = [0.0, -2500.0, 2500.0]
            for t in np.linspace(0.0, 1.0, 29):
                base_xy = np.array([sx, sy], dtype=float) + t * seg
                line_pts.append((float(base_xy[0]), float(base_xy[1])))
                for off in lateral_offsets:
                    pt = base_xy + off * normal
                    corridor_pts.append((float(pt[0]), float(pt[1])))

        preds = np.asarray(
            self.risk_oracle.get_risk_at_points(corridor_pts, self.terrain, self.threats, agl_m=self.safe_alt),
            dtype=float,
        )
        corridor_preds = np.asarray(
            self.risk_oracle.get_risk_at_points(line_pts, self.terrain, self.threats, agl_m=self.safe_alt),
            dtype=float,
        )

        if preds.size == 0:
            return {
                "peak_risk": 0.0,
                "mean_risk": 0.0,
                "corridor_risk": 0.0,
                "high_risk_fraction": 0.0,
            }

        # Use robust mission-relevant statistics rather than the absolute maximum,
        # which is overly sensitive to tiny local peaks.
        peak_risk = float(np.percentile(preds, 92))
        mean_risk = float(np.mean(preds))
        corridor_risk = float(np.mean(corridor_preds)) if corridor_preds.size else 0.0
        high_thresh = max(0.18, 0.60 * peak_risk)
        return {
            "peak_risk": peak_risk,
            "mean_risk": mean_risk,
            "corridor_risk": corridor_risk,
            "high_risk_fraction": float(np.mean(preds >= high_thresh)),
        }

    def plan(self, start, goal, time_budget_s=5.0, **kwargs):
        t0 = time.time()

        risk_summary = self._summarize_risk_field(start, goal)
        mission_state = self.advisor.analyze_mission(
            self.terrain,
            self.threats,
            start,
            goal,
            time_budget_s,
            risk_summary=risk_summary,
        )

        strategy = self.advisor.decide_strategy(mission_state)
        alg_name = strategy["alg"]
        score_info = strategy.get("scores", {})
        if alg_name not in self.factories:
            alg_name = "K-GNP"

        print(
            f"   [Neuro-Adaptive{'-NoDNN' if not self.use_dnn_tre else ''}] Risk Summary: "
            f"peak={mission_state['peak_risk']:.2f}, "
            f"mean={mission_state['mean_risk']:.2f}, "
            f"corridor={mission_state['corridor_risk']:.2f}, "
            f"high_frac={mission_state['high_risk_fraction']:.2f}"
        )
        if score_info:
            print(
                f"   [Neuro-Adaptive{'-NoDNN' if not self.use_dnn_tre else ''}] Scores: "
                f"cap={score_info.get('capability', 0.0):.2f}, "
                f"opp={score_info.get('opportunity', 0.0):.2f}, "
                f"press={score_info.get('pressure', 0.0):.2f}, "
                f"fused={score_info.get('fused', 0.0):.2f}"
            )
        print(f"   [Neuro-Adaptive{'-NoDNN' if not self.use_dnn_tre else ''}] Decision: {alg_name} ({strategy['desc']})")

        self.stats["risk_summary"] = risk_summary
        self.stats["selected_strategy"] = strategy["desc"]
        self.stats["selected_alg"] = alg_name
        self.stats["used_dnn_tre"] = self.use_dnn_tre

        planner = self.factories[alg_name]()
        path = planner.plan(start, goal)

        self.stats["t_total"] = time.time() - t0
        return path
