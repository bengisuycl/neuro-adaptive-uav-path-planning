import inspect
import math

import numpy as np
import torch
import torch.nn as nn

import wcci_conference_project.ai_modules.dqn_pilot as dqn_pilot_module


DQNClass = None
for name, obj in inspect.getmembers(dqn_pilot_module):
    if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
        DQNClass = obj
        break


class RLPilotAdvisor:
    """
    Meta-level planner selector for the Neuro-Adaptive planner.

    Current logic is intentionally simple and explainable:
    - Perception: summarized DNN-TRE risk map statistics
    - Decision: choose among RL-Pilot, K-GNP, and T-GnP

    Although the module name contains "RL", the current selection policy is not a
    learned end-to-end reinforcement-learning selector. It is an MCDA-like fusion
    layer that combines interpretable tactical descriptors into a single planner
    recommendation. This wording is important for thesis consistency.
    """

    def __init__(self, terrain=None, threats=None, feasibility=None, model_path="f16_dqn_weights.pth", safe_alt=800.0, **kwargs):
        self.terrain = terrain
        self.threats = threats
        self.feasibility = feasibility
        self.safe_alt = safe_alt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = None

        if DQNClass is not None:
            self.dqn = DQNClass(state_dim=8, action_dim=5).to(self.device)
            try:
                self.dqn.load_state_dict(torch.load(model_path, map_location=self.device))
                self.dqn.eval()
            except Exception:
                self.dqn = None

    def analyze_mission(self, terrain, threats, start, goal, time_budget_s, risk_summary=None):
        """Build an explainable mission descriptor from geometry and DNN-TRE risk statistics."""
        dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        urgency = "HIGH" if time_budget_s <= 3.0 else "NORMAL"

        risk_summary = risk_summary or {}
        peak_risk = float(risk_summary.get("peak_risk", 0.0))
        mean_risk = float(risk_summary.get("mean_risk", 0.0))
        corridor_risk = float(risk_summary.get("corridor_risk", 0.0))
        high_risk_fraction = float(risk_summary.get("high_risk_fraction", 0.0))

        threat_score = sum(float(getattr(t, "level", 1.0)) for t in threats)
        dynamic_count = sum(1 for t in threats if getattr(t, "dynamic", False))
        threat_count = len(threats)

        return {
            "distance": dist,
            "threat_score": threat_score,
            "threat_count": threat_count,
            "dynamic_count": dynamic_count,
            "urgency": urgency,
            "time_budget": time_budget_s,
            "peak_risk": peak_risk,
            "mean_risk": mean_risk,
            "corridor_risk": corridor_risk,
            "high_risk_fraction": high_risk_fraction,
        }

    def decide_strategy(self, mission_state):
        """
        Choose a planner from RL-Pilot / K-GNP / T-GnP based on a fused risk index.

        Design rationale:
        - capability proxy   -> peak_risk
        - opportunity proxy  -> corridor_risk + high_risk_fraction
        - dynamic pressure   -> dynamic_count + urgency + mission distance

        This keeps the logic explainable while following the literature idea of
        multi-attribute threat assessment and adaptive risk weighting. In thesis
        language, this is best described as an MCDA-inspired fused tactical score.
        """
        urgency = mission_state["urgency"]
        peak_risk = mission_state["peak_risk"]
        mean_risk = mission_state["mean_risk"]
        corridor_risk = mission_state["corridor_risk"]
        high_risk_fraction = mission_state["high_risk_fraction"]
        dynamic_count = mission_state["dynamic_count"]
        threat_count = mission_state["threat_count"]
        distance = mission_state["distance"]
        # Normalize absolute DNN-TRE values into bounded decision descriptors.
        peak_norm = peak_risk / (peak_risk + 1.1) if peak_risk > 0.0 else 0.0
        mean_norm = mean_risk / (mean_risk + 0.35) if mean_risk > 0.0 else 0.0
        corridor_ratio = corridor_risk / max(peak_risk, 1e-6)
        corridor_norm = corridor_risk / (corridor_risk + 0.28) if corridor_risk > 0.0 else 0.0
        dynamic_norm = min(dynamic_count / 3.0, 1.0)
        distance_norm = min(distance / 120000.0, 1.0)
        threat_density = min(threat_count / 5.0, 1.0)
        urgency_norm = 1.0 if urgency == "HIGH" else 0.0

        # Multi-attribute fusion:
        # - capability: local peak severity
        # - opportunity: direct corridor exposure + dense risk occupancy
        # - operational pressure: dynamics, urgency, mission scale
        capability_score = 0.58 * peak_norm + 0.28 * mean_norm + 0.14 * threat_density
        opportunity_score = 0.50 * corridor_ratio + 0.30 * high_risk_fraction + 0.20 * corridor_norm
        pressure_score = 0.38 * dynamic_norm + 0.22 * urgency_norm + 0.25 * distance_norm + 0.15 * threat_density

        fused_risk = (
            0.34 * capability_score
            + 0.41 * opportunity_score
            + 0.25 * pressure_score
        )

        # Hard tactical trigger: dynamic/high-exposure corridor should bias the
        # tactical kinodynamic planner even if the fused score is only moderate.
        if dynamic_count >= 2 and (corridor_ratio >= 0.38 or corridor_norm >= 0.42):
            return {
                "alg": "T-GnP",
                "desc": "Dynamic high-exposure corridor. Tactical kinodynamic planner selected.",
                "scores": {
                    "capability": capability_score,
                    "opportunity": opportunity_score,
                    "pressure": pressure_score,
                    "fused": fused_risk,
                },
            }

        if (
            dynamic_count == 0
            and threat_count <= 3
            and distance < 45000.0
            and fused_risk < 0.27
            and corridor_norm < 0.22
            and high_risk_fraction < 0.10
        ):
            return {
                "alg": "RL-Pilot",
                "desc": "Low fused tactical risk. Fast reactive policy selected.",
                "scores": {
                    "capability": capability_score,
                    "opportunity": opportunity_score,
                    "pressure": pressure_score,
                    "fused": fused_risk,
                },
            }

        if (
            fused_risk >= 0.56
            or corridor_ratio >= 0.50
            or corridor_norm >= 0.46
            or high_risk_fraction >= 0.28
            or (dynamic_count >= 1 and corridor_norm >= 0.34)
        ):
            return {
                "alg": "T-GnP",
                "desc": "High fused tactical risk. Tactical kinodynamic planner selected.",
                "scores": {
                    "capability": capability_score,
                    "opportunity": opportunity_score,
                    "pressure": pressure_score,
                    "fused": fused_risk,
                },
            }

        return {
            "alg": "K-GNP",
            "desc": "Moderate fused tactical risk. Kinematic guidance planner selected.",
            "scores": {
                "capability": capability_score,
                "opportunity": opportunity_score,
                "pressure": pressure_score,
                "fused": fused_risk,
            },
        }
