"""
Reduced-order fixed-wing aircraft model used throughout the thesis benchmark.

This implementation is intentionally lightweight: it is not a full high-fidelity
F-16 aerodynamic model. Instead, it follows a point-mass / coordinated-turn
interpretation that is consistent with classical aircraft simulation practice.

Modeling ideas reflected here:
- coordinated-turn heading dynamics
- bounded bank-rate and flight-path-angle-rate commands
- thrust/drag driven longitudinal speed update
- load-factor-dependent induced drag increase

This lets the planning and guidance layers reason about flyability without the
cost of a full 6-DoF rigid-body simulation.
"""

import math
import numpy as np


class Aircraft:
    def __init__(self, V_cruise=220.0, V_min=120.0, V_max=280.0):
        self.mass = 12000.0
        self.g = 9.81
        self.ax_max = 9.0
        self.cx_drag = 0.0025
        self.k_induced_drag = 0.10
        self.max_roll_rate_deg_s = 60.0
        self.max_gamma_rate_deg_s = 6.0
        self.min_thrust_factor = 0.50
        self.thrust_altitude_scale_m = 18000.0

        self.V = V_cruise
        self.V_cruise = V_cruise
        self.V_min = V_min
        self.V_max = V_max

        self.x = 0.0
        self.y = 0.0
        self.h = 1000.0
        self.psi = 0.0
        self.phi_deg = 0.0
        self.gamma_deg = 0.0

    def reset_attitude(self):
        self.phi_deg = 0.0
        self.gamma_deg = 0.0

    def step(self, state, control, dt, terrain=None, agl_margin=100.0):
        x, y, h, psi = state
        throttle, phi_cmd_deg, gamma_cmd_deg = control

        # Apply first-order-like command tracking with explicit rate limits so the
        # vehicle follows bounded bank/gamma commands instead of instantaneous jumps.
        max_dphi = self.max_roll_rate_deg_s * dt
        max_dgamma = self.max_gamma_rate_deg_s * dt
        self.phi_deg += float(np.clip(phi_cmd_deg - self.phi_deg, -max_dphi, max_dphi))
        self.gamma_deg += float(np.clip(gamma_cmd_deg - self.gamma_deg, -max_dgamma, max_dgamma))

        phi_rad = math.radians(self.phi_deg)
        gamma_rad = math.radians(self.gamma_deg)

        thrust_factor = max(self.min_thrust_factor, math.exp(-max(h, 0.0) / self.thrust_altitude_scale_m))
        n_load = min(1.0 / max(math.cos(phi_rad), 1e-3), 9.0)
        induced_drag_factor = 1.0 + self.k_induced_drag * ((n_load ** 2) - 1.0)

        T_max = 2.0 * self.mass * self.g
        thrust = throttle * T_max * thrust_factor
        drag = self.cx_drag * induced_drag_factor * (self.V ** 2)
        accel = (thrust - drag) / self.mass - self.g * math.sin(gamma_rad)
        accel = np.clip(accel, -5.0, self.ax_max)

        self.V += accel * dt
        self.V = np.clip(self.V, self.V_min, self.V_max)

        # Coordinated-turn yaw kinematics: psi_dot = g * tan(phi) / V
        turn_rate = (self.g * math.tan(phi_rad)) / max(self.V, 1.0)
        psi += turn_rate * dt

        h_dot = self.V * math.sin(gamma_rad)
        h += h_dot * dt

        V_ground = self.V * math.cos(gamma_rad)
        x += V_ground * math.cos(psi) * dt
        y += V_ground * math.sin(psi) * dt

        if terrain is not None:
            ground_h = terrain.get_height(x, y)
            if np.isnan(ground_h):
                ground_h = 0.0
            if h < ground_h + agl_margin:
                h = ground_h + agl_margin

        self.x, self.y, self.h, self.psi = x, y, h, psi
        return np.array([x, y, h, psi])
