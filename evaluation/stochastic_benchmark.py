# evaluation/stochastic_benchmark.py
# IEEE/CODIT-ready stochastic benchmarking utilities.
# FIXED: Added 'grace_s' support with fallback for planners that don't accept it.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class RunRecord:
    run_id: int
    status: str  # "SUCCESS" | "PARTIAL" | "FAIL"
    t_total_s: float  # total planning time (wall clock)
    t_first_s: Optional[float]  # first-feasible solution time (if planner provides)
    checks: int  # feasibility check count (if available)
    path_cost_m: Optional[float]  # path length (meters) for ranking
    path: Optional[List[Tuple[float, float, float]]]  # raw planned path


@dataclass
class StochasticSummary:
    runs: int
    success: int
    partial: int
    fail: int
    best_record: Optional[RunRecord]
    records: List[RunRecord]


def _path_length_m(path: Optional[List[Tuple[float, float, float]]]) -> Optional[float]:
    if not path or len(path) < 2:
        return None
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = float(path[i][0]), float(path[i][1])
        x2, y2 = float(path[i + 1][0]), float(path[i + 1][1])
        total += math.hypot(x2 - x1, y2 - y1)
    return float(total)


def run_stochastic_planner(
        planner_factory: Callable[[], Any],
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        time_budget_s: float,
        runs: int = 10,
        grace_s: float = 0.0,  # <--- EKLENDI: Bu parametre eksikti
        success_radius_m: float = 5000.0,
) -> StochasticSummary:
    """
    Execute N stochastic runs with a fixed time budget.
    """
    records: List[RunRecord] = []
    best: Optional[RunRecord] = None
    success = partial = fail = 0

    for i in range(int(runs)):
        planner = planner_factory()

        # reset feasibility counter if available
        fe = getattr(planner, "feasibility", None)
        if fe is not None and hasattr(fe, "reset_counter"):
            fe.reset_counter()

        t0 = time.time()
        path = None

        # --- PLANLAMA ÇAĞRISI (GÜVENLİ) ---
        try:
            # 1. Öncelik: grace_s parametresini destekleyen planner (örn. RRT*)
            path = planner.plan(start, goal, time_budget_s=float(time_budget_s), grace_s=float(grace_s))
        except TypeError:
            # 2. Öncelik: grace_s desteklemeyen ama time_budget destekleyen (örn. PSO)
            try:
                path = planner.plan(start, goal, time_budget_s=float(time_budget_s))
            except TypeError:
                # 3. Öncelik: Parametre ismi farklı olabilir (eski kod uyumu)
                try:
                    path = planner.plan(start, goal, time_budget=float(time_budget_s))
                except Exception:
                    # 4. Fallback: Hiçbirini kabul etmezse düz çağır
                    path = planner.plan(start, goal)
        except Exception:
            path = None

        t_total = float(time.time() - t0)

        # planner.stats may be dict or dataclass-like
        p_stats = getattr(planner, "stats", None)
        if isinstance(p_stats, dict):
            t_first = p_stats.get("t_first", None)
        else:
            t_first = getattr(p_stats, "t_first", None) if p_stats is not None else None

        # feasibility count (if planner has feasibility engine)
        checks = 0
        feas = getattr(planner, "feasibility", None)
        if feas is not None:
            checks = int(getattr(feas, "check_count", getattr(feas, "checks", 0)) or 0)

        # coarse plan status (planning-only)
        status = "FAIL"
        if path is not None and len(path) > 0:
            dx = float(path[-1][0]) - float(goal[0])
            dy = float(path[-1][1]) - float(goal[1])
            dist = math.hypot(dx, dy)
            status = "SUCCESS" if dist <= float(success_radius_m) else "PARTIAL"

        if status == "SUCCESS":
            success += 1
        elif status == "PARTIAL":
            partial += 1
        else:
            fail += 1

        rec = RunRecord(
            run_id=i + 1,
            status=status,
            t_total_s=t_total,
            t_first_s=(float(t_first) if t_first is not None else None),
            checks=checks,
            path_cost_m=_path_length_m(path),
            path=path,
        )
        records.append(rec)

        # pick best: prefer SUCCESS > PARTIAL > FAIL, then lower path_cost_m, then lower t_total_s
        if best is None:
            best = rec
        else:
            rank = {"SUCCESS": 2, "PARTIAL": 1, "FAIL": 0}
            if rank.get(rec.status, 0) > rank.get(best.status, 0):
                best = rec
            elif rank.get(rec.status, 0) == rank.get(best.status, 0):
                # If both have a cost, prefer smaller cost; else keep existing
                if rec.path_cost_m is not None and best.path_cost_m is not None:
                    if rec.path_cost_m < best.path_cost_m:
                        best = rec
                elif rec.path_cost_m is not None and best.path_cost_m is None:
                    best = rec
                else:
                    # tie-break on faster compute
                    if rec.t_total_s < best.t_total_s:
                        best = rec

    return StochasticSummary(
        runs=int(runs),
        success=int(success),
        partial=int(partial),
        fail=int(fail),
        best_record=best,
        records=records,
    )