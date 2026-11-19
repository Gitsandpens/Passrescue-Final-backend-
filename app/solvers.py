"""Solver selection: Google OR-Tools CP-SAT when installed, otherwise SciPy/HiGHS MILP.

CP-SAT uses the same candidate grid, the same resource-clique constraints and the same
lexicographic objectives as the MILP, so results are comparable and validated identically.
"""
from __future__ import annotations
import os
import time
from .planner import SECONDARY_VALUE, constraint_matrix, optimise as milp_optimise

try:
    import ortools
    from ortools.sat.python import cp_model
    NAME = f"Google OR-Tools CP-SAT {ortools.__version__}"
except ImportError:
    cp_model, NAME = None, "SciPy / HiGHS MILP"
if os.getenv("PASSRESCUE_SOLVER", "").lower() == "highs":
    cp_model, NAME = None, "SciPy / HiGHS MILP"


def optimise_cpsat(s, candidates, required, limit, fallback):
    if not candidates:
        return [], {"status": "OPTIMAL", "grid_optimal": True, "stages": [],
                    "message": "No eligible candidate contacts.", "runtime_s": 0}
    started = time.perf_counter()
    matrix, lower, upper = constraint_matrix(candidates, required)
    csr = matrix.tocsr()
    reqs = {r.id: r for r in s.requests}
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"c{i}") for i in range(len(candidates))]
    for r in range(csr.shape[0]):
        cols = csr.indices[csr.indptr[r]:csr.indptr[r + 1]]
        model.AddLinearConstraint(sum(x[int(i)] for i in cols), int(lower[r]), int(upper[r]))
    stages = [("critical requests", [int(reqs[c.request_id].priority == "critical") for c in candidates], True),
              ("other-request utility", [SECONDARY_VALUE[reqs[c.request_id].priority] for c in candidates], True),
              ("unchanged contacts", [int(c.preserved) for c in candidates], True),
              ("earlier payload starts", [int(c.start) for c in candidates], False)]
    logs, best, optimal, status = [], None, True, "OPTIMAL"
    for name, vals, maximise in stages:
        if not any(vals):
            continue
        remaining = limit - (time.perf_counter() - started)
        if remaining < 0.05:
            optimal, status = False, "FEASIBLE_TIME_LIMIT"
            break
        expr = sum(v * xi for v, xi in zip(vals, x))
        (model.Maximize if maximise else model.Minimize)(expr)
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = remaining
        sv.parameters.num_workers = min(4, os.cpu_count() or 1)
        code = sv.Solve(model)
        logs.append({"objective": name, "status_code": int(code), "message": sv.StatusName(code)})
        if code == cp_model.INFEASIBLE:
            raise ValueError("No feasible schedule can honour the protected existing contacts.")
        if code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            best = [int(sv.Value(xi)) for xi in x]
        if code != cp_model.OPTIMAL:
            optimal = False
            status = "FEASIBLE_TIME_LIMIT" if best is not None else "HEURISTIC_FALLBACK"
            break
        value = int(round(sv.ObjectiveValue()))
        logs[-1]["attained_value"] = -value if maximise else value
        model.Add(expr == value)
    if best is None:
        selected, optimal, status = fallback, False, "HEURISTIC_FALLBACK"
    else:
        selected = [c for i, c in enumerate(candidates) if best[i]]

    def quality(items):
        return (sum(reqs[c.request_id].priority == "critical" for c in items),
                sum(SECONDARY_VALUE[reqs[c.request_id].priority] for c in items),
                sum(c.preserved for c in items), -sum(c.start for c in items))
    if not optimal and quality(fallback) > quality(selected):
        selected, status = fallback, "HEURISTIC_FALLBACK"
    return selected, {"status": status, "grid_optimal": optimal, "stages": logs,
                      "runtime_s": round(time.perf_counter() - started, 4),
                      "message": (f"Optimal on the {s.slot_seconds}-second candidate grid; not a continuous-time optimum."
                                  if optimal else "Feasible plan; the full ordered objective was not proved optimal.")}


def run_solver(s, candidates, required, limit, fallback):
    fn = optimise_cpsat if cp_model is not None else milp_optimise
    selected, info = fn(s, candidates, required, limit, fallback)
    info["solver_name"] = NAME
    info["message"] = f"{NAME}. {info['message']}"
    return selected, info
