import numpy as np


class Threat:
    def __init__(
        self,
        x_center,
        y_center,
        radius,
        threat_type="SAM",
        threat_level=1.0,
        dynamic=False,
        speed_mps=30.0,
        motion_pattern="static",
        motion_param=5000.0,
        predictive_horizon_s=35.0,
        corridor_weight=0.45,
    ):
        """
        Threat object with static and dynamic risk behavior.

        Dynamic threats combine:
        - instantaneous radial coverage around the current position
        - a predictive forward lobe along the motion direction
        - a patrol corridor penalty around the motion path
        """
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.radius = float(radius)
        self.type = threat_type
        self.level = float(threat_level)

        self.x = float(x_center)
        self.y = float(y_center)
        self.dynamic = bool(dynamic)
        self.speed = float(speed_mps)
        self.motion_pattern = motion_pattern
        self.motion_param = float(motion_param)
        self.predictive_horizon_s = float(predictive_horizon_s)
        self.corridor_weight = float(corridor_weight)
        self.sim_time = 0.0

        if threat_type == "SAM":
            self.soft_radius = self.radius * 3.0
            self.type_multiplier = 1.0
        elif threat_type == "SAM_Big":
            self.soft_radius = self.radius * 3.2
            self.type_multiplier = 1.15
        elif threat_type == "Radar":
            self.soft_radius = self.radius * 4.0
            self.type_multiplier = 0.8
        elif threat_type == "EW":
            self.soft_radius = self.radius * 2.8
            self.type_multiplier = 0.6
        else:
            self.soft_radius = self.radius * 2.5
            self.type_multiplier = 0.5

    def _soft_cost_from_distance(self, dist, cost_scale):
        if dist <= self.radius:
            return float("inf")
        if dist > self.soft_radius:
            return 0.0
        denom = max(self.soft_radius - self.radius, 1e-6)
        norm = (self.soft_radius - dist) / denom
        return cost_scale * (norm ** 3) * self.level * self.type_multiplier

    def _closest_point_on_segment_distance(self, px, py, ax, ay, bx, by):
        seg = np.array([bx - ax, by - ay], dtype=float)
        p = np.array([px - ax, py - ay], dtype=float)
        denom = float(np.dot(seg, seg))
        if denom <= 1e-9:
            return float(np.hypot(px - ax, py - ay))
        t = float(np.clip(np.dot(p, seg) / denom, 0.0, 1.0))
        closest = np.array([ax, ay], dtype=float) + t * seg
        return float(np.hypot(px - closest[0], py - closest[1]))

    def _current_velocity_vector(self):
        if not self.dynamic or self.speed <= 0.0:
            return np.array([0.0, 0.0], dtype=float)

        if self.motion_pattern == "patrol_line":
            half_length = self.motion_param / 2.0
            if half_length <= 1e-6:
                return np.array([0.0, 0.0], dtype=float)
            cycle_time = self.motion_param / max(self.speed, 1e-6)
            phase = (self.sim_time % (2.0 * cycle_time)) / (2.0 * cycle_time)
            direction = 1.0 if phase < 0.5 else -1.0
            return np.array([direction, 0.0], dtype=float)

        if self.motion_pattern == "patrol_circle":
            orbit_r = max(self.motion_param, 1e-6)
            omega_t = self.sim_time * self.speed / orbit_r
            tangent = np.array([-np.sin(omega_t), np.cos(omega_t)], dtype=float)
            nrm = np.linalg.norm(tangent)
            return tangent / max(nrm, 1e-6)

        if self.motion_pattern == "random":
            delta = np.array([self.x - self.x_center, self.y - self.y_center], dtype=float)
            nrm = np.linalg.norm(delta)
            if nrm > 1e-6:
                return delta / nrm

        return np.array([0.0, 0.0], dtype=float)

    def _predictive_directional_cost(self, px, py, cost_scale):
        vel_dir = self._current_velocity_vector()
        speed = max(self.speed, 0.0)
        if np.linalg.norm(vel_dir) <= 1e-6 or speed <= 0.0:
            return 0.0

        lead_dist = np.clip(speed * self.predictive_horizon_s, self.radius, self.soft_radius * 2.0)
        focus_x = self.x + vel_dir[0] * lead_dist * 0.45
        focus_y = self.y + vel_dir[1] * lead_dist * 0.45
        semi_major = self.soft_radius + lead_dist * 0.35
        semi_minor = max(self.soft_radius * 0.75, self.radius * 1.25)

        delta = np.array([px - focus_x, py - focus_y], dtype=float)
        x_local = float(np.dot(delta, vel_dir))
        y_local = float(np.dot(delta, np.array([-vel_dir[1], vel_dir[0]], dtype=float)))
        ellipse_r = (x_local / max(semi_major, 1e-6)) ** 2 + (y_local / max(semi_minor, 1e-6)) ** 2

        if ellipse_r >= 1.0:
            return 0.0

        gain = (1.0 - ellipse_r) ** 2
        type_gain = 1.15 if "SAM" in self.type else 0.85
        return cost_scale * 0.55 * gain * self.level * self.type_multiplier * type_gain

    def _motion_corridor_cost(self, px, py, cost_scale):
        if not self.dynamic or self.motion_pattern == "static":
            return 0.0

        corridor_width = max(self.soft_radius * 0.85, self.radius * 1.4)

        if self.motion_pattern == "patrol_line":
            half = self.motion_param / 2.0
            dist = self._closest_point_on_segment_distance(
                px,
                py,
                self.x_center - half,
                self.y_center,
                self.x_center + half,
                self.y_center,
            )
        elif self.motion_pattern == "patrol_circle":
            dist_to_center = float(np.hypot(px - self.x_center, py - self.y_center))
            dist = abs(dist_to_center - self.motion_param)
        elif self.motion_pattern == "random":
            dist = float(np.hypot(px - self.x_center, py - self.y_center))
            corridor_width = max(self.motion_param, corridor_width)
        else:
            return 0.0

        if dist > corridor_width:
            return 0.0

        norm = 1.0 - dist / max(corridor_width, 1e-6)
        return cost_scale * self.corridor_weight * (norm ** 2) * self.level * self.type_multiplier

    def update(self, dt):
        if not self.dynamic or self.speed <= 0.0:
            return

        self.sim_time += dt

        if self.motion_pattern == "patrol_line":
            half_length = self.motion_param / 2.0
            cycle_time = self.motion_param / max(self.speed, 1e-6)
            phase = (self.sim_time % (2.0 * cycle_time)) / (2.0 * cycle_time)

            if phase < 0.5:
                offset = -half_length + (phase * 2.0) * self.motion_param
            else:
                offset = half_length - ((phase - 0.5) * 2.0) * self.motion_param

            self.x = self.x_center + offset
            self.y = self.y_center

        elif self.motion_pattern == "patrol_circle":
            if self.motion_param <= 0.0:
                return
            omega = self.speed / self.motion_param
            self.x = self.x_center + self.motion_param * np.cos(self.sim_time * omega)
            self.y = self.y_center + self.motion_param * np.sin(self.sim_time * omega)

        elif self.motion_pattern == "random":
            self.x += np.random.uniform(-1.0, 1.0) * self.speed * dt
            self.y += np.random.uniform(-1.0, 1.0) * self.speed * dt
            if np.hypot(self.x - self.x_center, self.y - self.y_center) > self.motion_param:
                self.x = self.x_center
                self.y = self.y_center

    def get_risk_cost(self, px, py, cost_scale=5000.0):
        """Return current threat risk at point (px, py)."""
        dist = float(np.hypot(px - self.x, py - self.y))
        base_cost = self._soft_cost_from_distance(dist, cost_scale)
        if base_cost == float("inf"):
            return base_cost

        dynamic_cost = 0.0
        if self.dynamic:
            dynamic_cost += self._predictive_directional_cost(px, py, cost_scale)
            dynamic_cost += self._motion_corridor_cost(px, py, cost_scale)

        return float(base_cost + dynamic_cost)

    def reset(self):
        self.x = self.x_center
        self.y = self.y_center
        self.sim_time = 0.0
