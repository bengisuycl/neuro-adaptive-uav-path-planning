import math

import numpy as np


def process_path_for_flight(
    path,
    r_min=1500.0,
    ds_ref=200.0,
    ds_metric=50.0,
    validation_threats=None,
    apply_fillet=True,
):
    """
    Post-process a planner path for flight execution and visualization.

    1. Optionally apply bounded-curvature circular fillet smoothing.
    2. If the fillet intersects a threat hard zone, reduce the radius.
    3. Resample the result at separate resolutions for simulation and metrics.

    This stage is Dubins-inspired rather than a full Dubins shortest-path solver.
    The goal is to replace sharp planner corners with flyable circular transition
    arcs that remain consistent with minimum-turn-radius logic for fixed-wing
    aircraft, while keeping the implementation lightweight for the thesis codebase.
    """
    if not path or len(path) < 2:
        return path, path

    raw_points = [tuple(p) for p in path]
    smoothed_points = raw_points
    if apply_fillet and len(path) >= 3:
        candidate_points = _apply_circular_fillet_smoothing(raw_points, r_min, validation_threats)
        if _is_reasonable_smoothed_path(raw_points, candidate_points):
            smoothed_points = candidate_points

    ref_path = _resample_path(smoothed_points, ds_ref)
    metric_path = _resample_path(smoothed_points, ds_metric)
    return ref_path, metric_path


def _apply_circular_fillet_smoothing(path, r_min, threats=None, arc_step=50.0):
    if len(path) < 3:
        return path

    min_feasible_radius = 400.0
    new_path = [tuple(path[0])]

    for i in range(1, len(path) - 1):
        p_prev = np.array(path[i - 1], dtype=float)
        p_curr = np.array(path[i], dtype=float)
        p_next = np.array(path[i + 1], dtype=float)

        v_in = (p_curr - p_prev)[:2]
        v_out = (p_next - p_curr)[:2]
        len_in = float(np.linalg.norm(v_in))
        len_out = float(np.linalg.norm(v_out))
        if len_in < 1.0 or len_out < 1.0:
            new_path.append(tuple(p_curr))
            continue

        u_in = v_in / len_in
        u_out = v_out / len_out
        dot = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        angle = float(np.arccos(dot))

        # Skip nearly straight or sharp reversal vertices; these are poor candidates
        # for local circular fillet insertion.
        if angle < 0.05 or angle > 2.35:
            new_path.append(tuple(p_curr))
            continue

        best_arc = None
        current_r = float(r_min)
        while current_r >= min_feasible_radius:
            d_tan = current_r * math.tan(angle / 2.0)
            d_max = min(len_in, len_out) * 0.25
            effective_r = current_r

            if d_tan > d_max:
                d_tan = d_max
                effective_r = d_tan / max(math.tan(angle / 2.0), 1e-6)

            p_start = p_curr.copy()
            p_end = p_curr.copy()
            p_start[:2] = p_curr[:2] - u_in * d_tan
            p_end[:2] = p_curr[:2] + u_out * d_tan

            bisector = u_in + u_out
            bis_norm = float(np.linalg.norm(bisector))
            if bis_norm < 1e-6:
                current_r *= 0.8
                continue
            bisector /= bis_norm

            center_offset = effective_r / max(math.sin(angle / 2.0), 1e-6)
            center_xy = p_curr[:2] + bisector * center_offset

            start_vec = p_start[:2] - center_xy
            end_vec = p_end[:2] - center_xy
            start_ang = math.atan2(start_vec[1], start_vec[0])
            end_ang = math.atan2(end_vec[1], end_vec[0])

            cross_z = u_in[0] * u_out[1] - u_in[1] * u_out[0]
            if cross_z > 0.0 and end_ang < start_ang:
                end_ang += 2.0 * math.pi
            elif cross_z < 0.0 and end_ang > start_ang:
                end_ang -= 2.0 * math.pi

            span = abs(end_ang - start_ang)
            arc_len = effective_r * span
            num_steps = max(4, int(arc_len / float(arc_step)))

            arc_points = []
            z_interp = np.linspace(float(p_start[2]), float(p_end[2]), num_steps)
            ang_interp = np.linspace(start_ang, end_ang, num_steps)
            for ang, z_val in zip(ang_interp, z_interp):
                arc_points.append((
                    float(center_xy[0] + effective_r * math.cos(ang)),
                    float(center_xy[1] + effective_r * math.sin(ang)),
                    float(z_val),
                ))

            is_safe = True
            if threats:
                for pt in arc_points:
                    for th in threats:
                        if np.hypot(pt[0] - th.x, pt[1] - th.y) <= th.radius:
                            is_safe = False
                            break
                    if not is_safe:
                        break

            if is_safe:
                best_arc = arc_points
                break

            current_r *= 0.8

        if best_arc is not None:
            new_path.extend(best_arc)
        else:
            new_path.append(tuple(p_curr))

    new_path.append(tuple(path[-1]))
    return new_path


def _path_length_xy(path_points):
    if not path_points or len(path_points) < 2:
        return 0.0
    arr = np.asarray(path_points, dtype=float)
    diffs = arr[1:, :2] - arr[:-1, :2]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _segments_intersect_xy(a1, a2, b1, b2):
    eps = 1e-9

    def _orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def _on_segment(p, q, r):
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps):
        return True

    if abs(o1) <= eps and _on_segment(a1, b1, a2):
        return True
    if abs(o2) <= eps and _on_segment(a1, b2, a2):
        return True
    if abs(o3) <= eps and _on_segment(b1, a1, b2):
        return True
    if abs(o4) <= eps and _on_segment(b1, a2, b2):
        return True
    return False


def _has_self_intersection_xy(path_points):
    if not path_points or len(path_points) < 4:
        return False

    arr = np.asarray(path_points, dtype=float)
    seg_count = len(arr) - 1
    for i in range(seg_count):
        a1 = arr[i, :2]
        a2 = arr[i + 1, :2]
        for j in range(i + 2, seg_count):
            if j == i + 1:
                continue
            if i == 0 and j == seg_count - 1:
                continue
            b1 = arr[j, :2]
            b2 = arr[j + 1, :2]
            if _segments_intersect_xy(a1, a2, b1, b2):
                return True
    return False


def _is_reasonable_smoothed_path(raw_points, smoothed_points):
    if not smoothed_points or len(smoothed_points) < 2:
        return False

    raw_len = _path_length_xy(raw_points)
    smoothed_len = _path_length_xy(smoothed_points)
    if raw_len <= 1.0:
        return True

    # Reject pathological smoothing that inflates path length or introduces loops.
    if smoothed_len > raw_len * 1.12:
        return False
    if _has_self_intersection_xy(smoothed_points):
        return False
    return True


def _resample_path(path_points, spacing):
    if not path_points:
        return []

    resampled = [tuple(path_points[0])]
    accum = 0.0

    for i in range(len(path_points) - 1):
        p1 = np.array(path_points[i], dtype=float)
        p2 = np.array(path_points[i + 1], dtype=float)
        dist = float(np.linalg.norm(p2 - p1))
        if dist < 1e-6:
            continue

        vec = (p2 - p1) / dist
        needed = spacing - accum

        while needed <= dist:
            new_pt = p1 + vec * needed
            resampled.append(tuple(new_pt))
            needed += spacing
        accum = dist - (needed - spacing)

    if np.linalg.norm(np.array(resampled[-1]) - np.array(path_points[-1])) > 1.0:
        resampled.append(tuple(path_points[-1]))
    return resampled
