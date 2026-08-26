# project_v1/simulator/simulation.py
# VERSION: Strict Dynamics + Curvature-Based Speed Command (Fair vs waypoint density)

import numpy as np
import math


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class Simulation:
    """
    Key idea:
    - Keep the aircraft physics in Aircraft.step() (throttle/bank/gamma).
    - Add an adaptive SPEED COMMAND that depends on *path curvature ahead*.
    - Compute curvature using distance-based lookahead (NOT index-based), so it is robust to
      different waypoint densities (A*/Dijkstra vs PSO/RRT*).

    Guidance note:
    - The lateral channel uses an L1-style fixed-wing path-following law rather than a
      pure heading-proportional turn command. This tends to reduce unrealistic circular
      orbiting and better matches coordinated-turn path-following behavior for aircraft.
    """

    def __init__(
        self,
        aircraft,
        terrain,
        k_psi: float = 1.5,
        k_h: float = 0.03,
        lookahead_dist: float = 1000.0,
        agl_margin: float = 100.0,
        max_bank_deg: float = 60.0,
        V_cruise: float | None = None,
        speed_margin: float = 0.87,
        tau_v: float = 1.8,
        airbrake_drag_mult: float = 2.5,
        airbrake_threshold: float = 5.0,
    ):
        self.aircraft = aircraft
        self.terrain = terrain
        self.k_psi = k_psi
        self.k_h = k_h
        self.lookahead_dist = lookahead_dist
        self.agl_margin = agl_margin
        self.max_bank_deg = max_bank_deg
        self.min_l1_dist = 700.0
        self.max_l1_dist = 3500.0

        # Speed control params
        self.V_cruise = float(V_cruise) if V_cruise is not None else float(getattr(aircraft, "V", 220.0))
        self.speed_margin = float(speed_margin)   # safety margin on V_safe
        self.tau_v = float(tau_v)                 # time constant for speed tracking (s)

        # Optional "speedbrake" approximation via increased drag (F-16 has speed brakes).
        self._cx_drag_base = float(getattr(aircraft, "cx_drag", 0.0025))
        self.airbrake_drag_mult = float(airbrake_drag_mult)
        self.airbrake_threshold = float(airbrake_threshold)  # m/s above target to activate

        self.threats = getattr(self.terrain, "threats", [])

    # ------------------------- curvature -> target speed -------------------------

    def _idx_at_distance(self, cumdist: np.ndarray, start_idx: int, dist_ahead: float) -> int:
        """Return the smallest index such that cumdist[idx] >= cumdist[start_idx] + dist_ahead."""
        target = cumdist[start_idx] + max(0.0, float(dist_ahead))
        idx = int(np.searchsorted(cumdist, target, side="left"))
        return min(max(idx, start_idx), len(cumdist) - 1)

    def _curvature_speed_target(self, path_xy: np.ndarray, cumdist: np.ndarray, base_idx: int) -> float:
        """
        Estimate a safe speed for upcoming curvature using three points at fixed *distance* ahead.
        """
        n = len(path_xy)
        if n < 3 or base_idx >= n - 2:
            return self.V_cruise

        i1 = base_idx
        i2 = self._idx_at_distance(cumdist, base_idx, 0.8 * self.lookahead_dist)
        i3 = self._idx_at_distance(cumdist, base_idx, 1.6 * self.lookahead_dist)
        if i2 == i1 or i3 == i2:
            return self.V_cruise

        p1 = path_xy[i1]
        p2 = path_xy[i2]
        p3 = path_xy[i3]

        v1 = p2 - p1
        v2 = p3 - p2
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1.0 or n2 < 1.0:
            return self.V_cruise

        # Turn angle
        dot = float(np.dot(v1, v2) / (n1 * n2))
        dot = float(np.clip(dot, -1.0, 1.0))
        angle = float(np.arccos(dot))  # rad

        if angle < 0.05:
            return self.V_cruise

        # Radius estimate (simple geometric)
        avg_len = 0.5 * (n1 + n2)
        R_est = avg_len / (angle + 1e-6)

        # V_safe = sqrt(R * g * tan(phi_max))
        g = 9.81
        phi_max_rad = math.radians(self.max_bank_deg)
        V_safe = math.sqrt(max(1.0, R_est) * g * math.tan(phi_max_rad))

        V_min = float(getattr(self.aircraft, "V_min", 120.0))
        V_max = float(getattr(self.aircraft, "V_max", 320.0))

        V_target = np.clip(V_safe * self.speed_margin, V_min, min(self.V_cruise, V_max))
        return float(V_target)

    # ------------------------- speed tracking -> throttle -------------------------

    def _speed_to_throttle(self, V_target: float, dt: float) -> float:
        """
        Convert a speed target into a throttle command using the aircraft's longitudinal model.
        Uses a simple first-order tracking law: V_dot_des = (V_target - V) / tau_v.
        """
        V = float(getattr(self.aircraft, "V", 220.0))
        mass = float(getattr(self.aircraft, "mass", 12000.0))
        g = float(getattr(self.aircraft, "g", 9.81))
        ax_max = float(getattr(self.aircraft, "ax_max", 8.0))
        cx_drag = float(getattr(self.aircraft, "cx_drag", self._cx_drag_base))

        # Desired acceleration
        V_dot_des = (float(V_target) - V) / max(0.5, self.tau_v)
        V_dot_des = float(np.clip(V_dot_des, -ax_max, ax_max))

        # Thrust model matches Aircraft.step()
        T_max = 2.0 * mass * g
        D = cx_drag * (V ** 2)
        T_req = mass * V_dot_des + D

        throttle = float(np.clip(T_req / T_max, 0.0, 1.0))
        return throttle

    # ------------------------- guidance -------------------------

    def _guidance(self, state, wp, V_target: float, dt: float):
        x, y, h, psi = map(float, state)
        xw, yw, hw = map(float, wp)
        dx, dy = xw - x, yw - y

        # L1-style lateral guidance:
        # eta is the angle between the current velocity direction and the line-of-sight
        # to the lookahead point. The corresponding lateral acceleration command is
        # converted into a bank-angle command through coordinated-turn flight.
        psi_des = math.atan2(dy, dx)
        eta = _wrap_pi(psi_des - psi)
        V = float(getattr(self.aircraft, "V", 220.0))
        g = 9.81
        l1_dist = float(np.clip(math.hypot(dx, dy), self.min_l1_dist, self.max_l1_dist))
        a_lat_cmd = 2.0 * (V ** 2) * math.sin(eta) / max(l1_dist, 1.0)
        phi_rad_des = math.atan2(a_lat_cmd, g)
        phi_cmd_deg = math.degrees(phi_rad_des)
        phi_cmd_deg = float(np.clip(phi_cmd_deg, -self.max_bank_deg, self.max_bank_deg))

        # Vertical Guidance
        h_des = hw
        e_h = h_des - h
        gamma_cmd_deg = math.degrees(float(np.clip(self.k_h * e_h, -5.0, 5.0)))

        # Speed control -> throttle (with optional "speedbrake" drag boost)
        throttle = self._speed_to_throttle(V_target, dt)

        # If we are far above target, mimic speedbrake by increasing drag temporarily
        if V > V_target + self.airbrake_threshold:
            self.aircraft.cx_drag = self._cx_drag_base * self.airbrake_drag_mult
        else:
            self.aircraft.cx_drag = self._cx_drag_base

        return throttle, phi_cmd_deg, gamma_cmd_deg

    # ------------------------- simulation loop -------------------------

    def run(self, path, dt=0.25, max_time=None, goal_radius=3000.0, goal_center=None,
            disturbance=None, rng=None):
        if not path or len(path) < 2:
            return np.array([]), []

        if goal_center is None:
            goal_center = path[-1][:2]
        goal_center_xy = np.array(goal_center, dtype=float)

        path_arr = np.asarray(path, dtype=float)
        path_xy = path_arr[:, :2]

        # Precompute cumulative distance along the path (density-robust lookahead)
        seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
        cumdist = np.concatenate([[0.0], np.cumsum(seg)])

        if max_time is None:
            total_dist = float(np.sum(seg))
            # Conservative: allow some extra time for re-tracking / detours
            max_time = max(60.0, (total_dist / max(60.0, self.V_cruise)) * 3.0)

        # Initialize state at path start
        x0, y0, h0 = map(float, path_arr[0])
        psi0 = math.atan2(path_xy[1, 1] - y0, path_xy[1, 0] - x0)
        state = np.array([x0, y0, h0, psi0], dtype=float)
        if hasattr(self.aircraft, "reset_attitude"):
            self.aircraft.reset_attitude()

        # Disturbances (optional): used for robustness evaluation.
        # - wind_xy_mps: constant inertial wind (m/s) applied to x,y propagation
        # - phi_noise_std_deg / gamma_noise_std_deg: additive command noise
        if disturbance is None:
            disturbance = {}
        wind_xy = np.array(disturbance.get("wind_xy_mps", (0.0, 0.0)), dtype=float)
        phi_noise_std = float(disturbance.get("phi_noise_std_deg", 0.0))
        gamma_noise_std = float(disturbance.get("gamma_noise_std_deg", 0.0))
        if rng is None:
            rng = np.random.default_rng()

        traj = [state.copy()]
        history = [{
            "aircraft_state": state.copy(),
            "speed": float(getattr(self.aircraft, "V", self.V_cruise)),
            "speed_target": float(getattr(self.aircraft, "V", self.V_cruise)),
            "throttle": 0.0,
            "phi_cmd_deg": 0.0,
            "gamma_cmd_deg": 0.0,
            "phi_deg": float(getattr(self.aircraft, "phi_deg", 0.0)),
            "gamma_deg": float(getattr(self.aircraft, "gamma_deg", 0.0)),
            "wind_xy_mps": wind_xy.copy(),
            "threat_states": [(th.x, th.y, th.radius) for th in self.threats]
        }]

        t = 0.0
        closest_idx = 0

        while t < max_time:
            x, y, h, psi = state
            current_pos = np.array([x, y], dtype=float)

            # 1) SUCCESS CHECK
            if float(np.linalg.norm(goal_center_xy - current_pos)) <= float(goal_radius):
                break

            # 2) Closest waypoint in a window (fast)
            # Keep path progress predominantly forward to avoid unrealistic orbiting
            # around previously visited path sections.
            search_start = max(0, closest_idx)
            search_end = min(len(path_arr), closest_idx + 120)
            dists = np.linalg.norm(path_xy[search_start:search_end] - current_pos, axis=1)
            closest_idx = max(closest_idx, int(search_start + int(np.argmin(dists))))

            # 3) Choose lookahead target index by distance (more stable)
            target_idx = self._idx_at_distance(cumdist, closest_idx, self.lookahead_dist)
            target_wp = path_arr[target_idx]

            # 4) Fail-safe: reached path end but not in goal region
            if target_idx >= len(path_arr) - 1 and float(np.linalg.norm(path_xy[-1] - current_pos)) < 200.0:
                break

            # 5) Update threats
            for th in self.threats:
                th.update(dt)

            # 6) Curvature-based target speed (density-robust)
            V_target = self._curvature_speed_target(path_xy, cumdist, closest_idx)

            # 7) Guidance + aircraft integration
            throttle, phi_cmd_deg, gamma_cmd_deg = self._guidance(state, target_wp, V_target, dt)

            # Apply command noise (robustness studies)
            if phi_noise_std > 0.0:
                phi_cmd_deg = float(phi_cmd_deg + rng.normal(0.0, phi_noise_std))
            if gamma_noise_std > 0.0:
                gamma_cmd_deg = float(gamma_cmd_deg + rng.normal(0.0, gamma_noise_std))

            # Respect hard limits (safety): keep noise from creating impossible commands
            phi_cmd_deg = float(np.clip(phi_cmd_deg, -self.max_bank_deg, self.max_bank_deg))
            gamma_cmd_deg = float(np.clip(gamma_cmd_deg, -5.0, 5.0))

            state = self.aircraft.step(state, (throttle, phi_cmd_deg, gamma_cmd_deg), dt,
                                       terrain=self.terrain, agl_margin=self.agl_margin)

            # Apply inertial wind drift
            if wind_xy[0] != 0.0 or wind_xy[1] != 0.0:
                state[0] += float(wind_xy[0]) * float(dt)
                state[1] += float(wind_xy[1]) * float(dt)

            traj.append(state.copy())
            history.append({
                "aircraft_state": state.copy(),
                "speed": float(getattr(self.aircraft, "V", self.V_cruise)),
                "speed_target": float(V_target),
                "throttle": float(throttle),
                "phi_cmd_deg": float(phi_cmd_deg),
                "gamma_cmd_deg": float(gamma_cmd_deg),
                "phi_deg": float(getattr(self.aircraft, "phi_deg", phi_cmd_deg)),
                "gamma_deg": float(getattr(self.aircraft, "gamma_deg", gamma_cmd_deg)),
                "wind_xy_mps": wind_xy.copy(),
                "threat_states": [(th.x, th.y, th.radius) for th in self.threats]
            })

            t += dt

        # Restore base drag (clean up any temporary speedbrake)
        self.aircraft.cx_drag = self._cx_drag_base

        return np.vstack(traj), history
