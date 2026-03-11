from __future__ import annotations

import random
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from backend_adapter.services.power118_service import run_power118_once


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(scenario_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"run-{scenario_id}-{ts}"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _make_trend(solve_time_ms: float) -> List[Dict[str, float]]:
    if solve_time_ms <= 0:
        return []
    multipliers = [1.15, 1.08, 1.03, 1.00, 0.95, 0.9]
    points: List[Dict[str, float]] = []
    for idx, m in enumerate(multipliers):
        points.append(
            {
                "label": f"R-{len(multipliers) - idx}",
                "value": max(0.1, solve_time_ms * m),
            }
        )
    return points


def _run_payload(
    scenario_id: str,
    solve_time_ms: float,
    infeasibility: float,
    suboptimality: float,
    strategies: List[Dict[str, Any]],
    mode: str,
    note: str | None = None,
) -> Dict[str, Any]:
    comparison = [{"label": s["name"], "value": _to_float(s["cost"])} for s in strategies[:4]]
    return {
        "runId": _run_id(scenario_id),
        "scenarioId": scenario_id,
        "generatedAt": _utc_now_iso(),
        "metrics": {
            "solveTimeMs": float(max(solve_time_ms, 0.0)),
            "infeasibilityRate": float(min(max(infeasibility, 0.0), 1.0)),
            "suboptimality": float(max(suboptimality, 0.0)),
        },
        "strategies": strategies,
        "trend": _make_trend(solve_time_ms),
        "comparison": comparison,
        "adapterMode": mode,
        "adapterNote": note,
    }


def _compat_run(scenario_id: str, note: str | None = None) -> Dict[str, Any]:
    seed = hash((scenario_id, datetime.now(timezone.utc).strftime("%Y%m%d%H"))) & 0xFFFF
    rnd = random.Random(seed)
    solve_ms = {"portfolio": 32.0, "control": 54.0, "obstacle": 68.0, "power-118": 92.0}.get(scenario_id, 40.0)
    solve_ms += rnd.uniform(-8, 8)

    strategies = [
        {
            "id": "strategy-1",
            "name": "AdapterBaseline",
            "feasible": True,
            "cost": round(rnd.uniform(8.0, 11.5), 4),
            "rank": 1,
        },
        {
            "id": "strategy-2",
            "name": "AdapterVariant",
            "feasible": True,
            "cost": round(rnd.uniform(9.0, 12.5), 4),
            "rank": 2,
        },
    ]
    return _run_payload(
        scenario_id=scenario_id,
        solve_time_ms=solve_ms,
        infeasibility=0.0,
        suboptimality=0.03,
        strategies=strategies,
        mode="compat",
        note=note or "compat mode",
    )


def _pick_solver(problem: Any) -> Tuple[str | None, str | None]:
    """Pick a feasible CVXPY solver from installed list."""
    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        return None, f"cvxpy import failed: {exc}"

    installed = set(cp.installed_solvers())
    # MIP-capable first, then continuous.
    candidates = ["GUROBI", "CPLEX", "MOSEK", "SCIP", "ECOS_BB", "CLARABEL", "OSQP", "ECOS", "SCS"]
    for name in candidates:
        if name in installed:
            return name, None
    return None, "no supported cvxpy solver found"


def _solve_problem(problem: Any, solver_name: str) -> Tuple[bool, float, float, str]:
    import cvxpy as cp

    start = perf_counter()
    solve_error = ""
    try:
        if solver_name == "SCS":
            problem.solve(solver=cp.SCS, verbose=False, max_iters=2000)
        elif solver_name == "OSQP":
            problem.solve(solver=cp.OSQP, verbose=False)
        elif solver_name == "CLARABEL":
            problem.solve(solver=cp.CLARABEL, verbose=False)
        elif solver_name == "ECOS":
            problem.solve(solver=cp.ECOS, verbose=False)
        elif solver_name == "ECOS_BB":
            problem.solve(solver=cp.ECOS_BB, verbose=False)
        elif solver_name == "SCIP":
            problem.solve(solver=cp.SCIP, verbose=False)
        elif solver_name == "MOSEK":
            problem.solve(solver=cp.MOSEK, verbose=False)
        elif solver_name == "CPLEX":
            problem.solve(solver=cp.CPLEX, verbose=False)
        elif solver_name == "GUROBI":
            problem.solve(solver=cp.GUROBI, verbose=False)
        else:
            problem.solve(verbose=False)
    except Exception as exc:
        solve_error = str(exc)

    elapsed_ms = (perf_counter() - start) * 1000.0
    status = (problem.status or "").lower()
    feasible = status in {"optimal", "optimal_inaccurate"}
    cost = _to_float(problem.value, default=float("inf"))
    if solve_error:
        return False, elapsed_ms, cost, solve_error
    if not feasible:
        return False, elapsed_ms, cost, f"status={problem.status}"
    return True, elapsed_ms, cost, ""


def _portfolio_real_run() -> Dict[str, Any]:
    import online_optimization.portfolio.utils as p_utils

    n, m = 6, 3
    problem = p_utils.create_problem(n=n, m=m, n_periods=1, k=None)

    params = {p.name(): p for p in problem.parameters()}
    rng = np.random.default_rng(7)
    w_init = rng.random(n)
    w_init = w_init / np.sum(w_init)

    if "hat_r_1" in params:
        params["hat_r_1"].value = rng.normal(0.001, 0.01, size=n)
    if "w_init" in params:
        params["w_init"].value = w_init
    if "F" in params:
        params["F"].value = rng.normal(0.0, 0.2, size=(n, m))
    if "sqrt_Sigma_F" in params:
        params["sqrt_Sigma_F"].value = np.abs(rng.normal(0.1, 0.05, size=m)) + 1e-3
    if "sqrt_D" in params:
        params["sqrt_D"].value = np.abs(rng.normal(0.08, 0.03, size=n)) + 1e-3

    solver_name, err = _pick_solver(problem)
    if not solver_name:
        return _compat_run("portfolio", note=f"portfolio real mode skipped: {err}")

    ok, elapsed_ms, cost, solve_error = _solve_problem(problem, solver_name)
    if not ok:
        return _compat_run("portfolio", note=f"portfolio solve failed: {solve_error}")

    strategies = [
        {
            "id": "strategy-1",
            "name": "SolverSolution",
            "feasible": True,
            "cost": round(cost, 6),
            "rank": 1,
        }
    ]
    return _run_payload(
        scenario_id="portfolio",
        solve_time_ms=elapsed_ms,
        infeasibility=0.0,
        suboptimality=0.0,
        strategies=strategies,
        mode="real",
        note=f"portfolio solved via {solver_name}",
    )


def _control_real_run() -> Dict[str, Any]:
    # control utils imports optional heavy dependencies (gurobi), so this can fail.
    import online_optimization.control.utils as c_utils

    T = 6
    problem, _ = c_utils.control_problem(T=T)
    params = {p.name(): p for p in problem.parameters()}
    if "E_init" in params:
        params["E_init"].value = 7.7
    if "z_init" in params:
        params["z_init"].value = 0.0
    if "s_init" in params:
        params["s_init"].value = 0.0
    if "past_d" in params:
        params["past_d"].value = np.zeros(T)
    if "P_load" in params:
        params["P_load"].value = np.maximum(c_utils.P_load_profile(T=T, seed=0), 0)

    solver_name, err = _pick_solver(problem)
    if not solver_name:
        return _compat_run("control", note=f"control real mode skipped: {err}")

    ok, elapsed_ms, cost, solve_error = _solve_problem(problem, solver_name)
    if not ok:
        return _compat_run("control", note=f"control solve failed: {solve_error}")

    strategies = [
        {
            "id": "strategy-1",
            "name": "SolverSolution",
            "feasible": True,
            "cost": round(cost, 6),
            "rank": 1,
        }
    ]
    return _run_payload(
        scenario_id="control",
        solve_time_ms=elapsed_ms,
        infeasibility=0.0,
        suboptimality=0.0,
        strategies=strategies,
        mode="real",
        note=f"control solved via {solver_name}",
    )


def _obstacle_real_run() -> Dict[str, Any]:
    import online_optimization.obstacle.utils as o_utils

    problem = o_utils.create_problem(o_utils.OBSTACLES[:2], T=20)
    params = {p.name(): p for p in problem.parameters()}
    if "p_init" in params:
        params["p_init"].value = np.array([-19.0, -19.0])

    solver_name, err = _pick_solver(problem)
    if not solver_name:
        return _compat_run("obstacle", note=f"obstacle real mode skipped: {err}")

    ok, elapsed_ms, cost, solve_error = _solve_problem(problem, solver_name)
    if not ok:
        return _compat_run("obstacle", note=f"obstacle solve failed: {solve_error}")

    strategies = [
        {
            "id": "strategy-1",
            "name": "SolverSolution",
            "feasible": True,
            "cost": round(cost, 6),
            "rank": 1,
        }
    ]
    return _run_payload(
        scenario_id="obstacle",
        solve_time_ms=elapsed_ms,
        infeasibility=0.0,
        suboptimality=0.0,
        strategies=strategies,
        mode="real",
        note=f"obstacle solved via {solver_name}",
    )


def _power118_run() -> Dict[str, Any]:
    return run_power118_once()


def _safe_runner(real_runner: Callable[[], Dict[str, Any]], scenario_id: str) -> Dict[str, Any]:
    try:
        return real_runner()
    except Exception as exc:
        return _compat_run(scenario_id, note=f"{scenario_id} adapter fallback: {exc}")


def run_scenario_once(scenario_id: str) -> Dict[str, Any]:
    if scenario_id == "portfolio":
        return _safe_runner(_portfolio_real_run, "portfolio")
    if scenario_id == "control":
        return _safe_runner(_control_real_run, "control")
    if scenario_id == "obstacle":
        return _safe_runner(_obstacle_real_run, "obstacle")
    if scenario_id == "power-118":
        return _safe_runner(_power118_run, "power-118")
    # Unknown scenario should be rejected at API layer.
    return _compat_run(scenario_id, note="unknown scenario fallback")
