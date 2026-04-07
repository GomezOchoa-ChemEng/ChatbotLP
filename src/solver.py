"""Solver engine for Pyomo-based optimization models.

This module provides a high-level interface to solve Pyomo models and
extract structured results. It handles solver execution, result validation,
and formatting without modifying the input model or ProblemState.
"""

from typing import Dict, Any, Optional, List, Tuple
import json
import os

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    SolverFactory,
    value,
)


class SolveResult:
    """Structured result from solving a Pyomo model."""

    def __init__(
        self,
        model: ConcreteModel,
        status: str,
        message: str,
        objective_value: Optional[float],
        solver_time: float,
        solution: Dict[str, Any],
        success: bool,
    ):
        self.model = model
        self.status = status
        self.message = message
        self.objective_value = objective_value
        self.solver_time = solver_time
        self.solution = solution
        self.success = success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "objective_value": self.objective_value,
            "solver_time": self.solver_time,
            "solution": self.solution,
            "success": self.success,
            "dual_values": _extract_dual_values(self.model),
            "constraint_slacks": _extract_constraint_slacks(self.model),
        }

    def __repr__(self) -> str:
        return (
            f"SolveResult(status={self.status}, success={self.success}, "
            f"objective={self.objective_value:.6f})"
            if self.objective_value is not None
            else f"SolveResult(status={self.status}, success={self.success})"
        )


def _candidate_solvers(
    solver_name: str,
    fallback_solver: str,
) -> List[Tuple[str, Optional[str]]]:
    """Return solver candidates with optional explicit executable paths."""
    user_home = os.path.expanduser("~")
    windows_glpk_paths = [
        os.path.join(user_home, "miniconda3", "envs", "pyomo-solvers", "Library", "bin", "glpsol.exe"),
        os.path.join(user_home, "Miniconda3", "envs", "pyomo-solvers", "Library", "bin", "glpsol.exe"),
    ]
    windows_ipopt_paths = [
        os.path.join(user_home, "miniconda3", "envs", "pyomo-solvers", "Library", "bin", "ipopt.exe"),
        os.path.join(user_home, "Miniconda3", "envs", "pyomo-solvers", "Library", "bin", "ipopt.exe"),
    ]

    candidates: List[Tuple[str, Optional[str]]] = [
        (solver_name, None),
        (fallback_solver, None),
    ]

    # Add common explicit executable paths, useful in Colab/Linux.
    if solver_name == "glpk" or fallback_solver == "glpk":
        candidates.extend(
            [
                ("glpk", "/usr/bin/glpsol"),
                ("glpk", "/bin/glpsol"),
                *[("glpk", path) for path in windows_glpk_paths],
            ]
        )

    if solver_name == "ipopt" or fallback_solver == "ipopt":
        candidates.extend(
            [
                ("ipopt", "/usr/bin/ipopt"),
                ("ipopt", "/bin/ipopt"),
                *[("ipopt", path) for path in windows_ipopt_paths],
            ]
        )

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique_candidates.append(item)

    return unique_candidates


def _get_solver(
    solver_name: str,
    fallback_solver: str,
    verbose: bool = False,
):
    """Find the first available solver."""
    tried = []

    for name, executable in _candidate_solvers(solver_name, fallback_solver):
        try:
            if executable is not None:
                if not os.path.exists(executable):
                    tried.append(f"{name} ({executable}: not found)")
                    continue
                solver = SolverFactory(name, executable=executable)
            else:
                solver = SolverFactory(name)

            if solver is not None and solver.available(exception_flag=False):
                if verbose:
                    if executable:
                        print(f"Using solver {name} with executable {executable}")
                    else:
                        print(f"Using solver {name}")
                return solver, name, executable, tried

            if executable:
                tried.append(f"{name} ({executable})")
            else:
                tried.append(name)

        except Exception as e:
            if executable:
                tried.append(f"{name} ({executable}): {e}")
            else:
                tried.append(f"{name}: {e}")

    return None, None, None, tried


def solve_model(
    model: ConcreteModel,
    solver_name: str = "ipopt",
    solver_options: Optional[Dict[str, Any]] = None,
    fallback_solver: str = "glpk",
    verbose: bool = False,
) -> SolveResult:
    """Solve a Pyomo ConcreteModel and return structured results."""
    if solver_options is None:
        solver_options = {}

    solver, used_solver, used_executable, tried = _get_solver(
        solver_name=solver_name,
        fallback_solver=fallback_solver,
        verbose=verbose,
    )

    if solver is None:
        msg = "No available solver found. Tried: " + ", ".join(tried)
        return SolveResult(
            model=model,
            status="solver_unavailable",
            message=msg,
            objective_value=None,
            solver_time=0.0,
            solution={},
            success=False,
        )

    for key, val in solver_options.items():
        solver.options[key] = val

    if verbose:
        if used_executable:
            print(f"Solving model with {used_solver} at {used_executable}...")
        else:
            print(f"Solving model with {used_solver}...")

    import time
    start_time = time.time()
    results = solver.solve(model, tee=verbose)
    solver_time = time.time() - start_time

    term_cond = str(results.solver.termination_condition)

    status_map = {
        "optimal": "optimal",
        "feasible": "feasible",
        "infeasible": "infeasible",
        "unbounded": "unbounded",
        "infeasibilityProven": "infeasible",
        "unboundedProven": "unbounded",
    }
    status = status_map.get(term_cond, term_cond.lower())
    success = status in ("optimal", "feasible")

    objective_value = None
    if hasattr(model, "obj") and success:
        try:
            objective_value = float(value(model.obj))
        except Exception:
            objective_value = None

    solution = _extract_solution(model)

    if used_executable:
        message = (
            f"Solver {used_solver} at {used_executable} terminated with status {status} "
            f"(termination condition: {term_cond})"
        )
    else:
        message = (
            f"Solver {used_solver} terminated with status {status} "
            f"(termination condition: {term_cond})"
        )

    return SolveResult(
        model=model,
        status=status,
        message=message,
        objective_value=objective_value,
        solver_time=solver_time,
        solution=solution,
        success=success,
    )


def _extract_solution(model: ConcreteModel) -> Dict[str, Any]:
    """Extract all Pyomo Var values from a model."""
    from pyomo.core.base.var import Var

    solution: Dict[str, Any] = {}

    for var in model.component_objects(Var, descend_into=True):
        name = var.name
        if var.is_indexed():
            sub: Dict[str, Any] = {}
            for idx in var:
                try:
                    val = value(var[idx])
                except Exception:
                    val = None
                sub[str(idx)] = val
            solution[name] = sub
        else:
            try:
                solution[name] = value(var)
            except Exception:
                solution[name] = None
    return solution


def _extract_dual_values(model: ConcreteModel) -> Dict[str, Optional[float]]:
    """Extract imported dual values when available."""
    dual_suffix = getattr(model, "dual", None)
    if dual_suffix is None:
        return {}

    duals: Dict[str, Optional[float]] = {}
    try:
        for constraint_data, dual_value in dual_suffix.items():
            duals[constraint_data.name] = None if dual_value is None else float(dual_value)
    except Exception:
        return {}
    return duals


def _extract_constraint_slacks(model: ConcreteModel) -> Dict[str, Optional[float]]:
    """Extract simple primal slack values for active constraints."""
    slacks: Dict[str, Optional[float]] = {}
    try:
        for constraint in model.component_data_objects(Constraint, active=True, descend_into=True):
            lower = constraint.lower
            upper = constraint.upper
            body_value = value(constraint.body)
            slack_candidates = []
            if lower is not None:
                slack_candidates.append(abs(body_value - value(lower)))
            if upper is not None:
                slack_candidates.append(abs(value(upper) - body_value))
            slacks[constraint.name] = min(slack_candidates) if slack_candidates else 0.0
    except Exception:
        return {}
    return slacks


def extract_variable_values(
    model: ConcreteModel,
    var_name: str,
    threshold: Optional[float] = None,
) -> Dict[Any, float]:
    """Extract values for a specific variable from a solved model."""
    var = getattr(model, var_name, None)
    if var is None:
        raise KeyError(f"Variable {var_name} not found in model")

    result = {}
    if var.is_indexed():
        for idx in var:
            val = value(var[idx])
            if threshold is None or abs(val) >= threshold:
                result[idx] = val
    else:
        val = value(var)
        if threshold is None or abs(val) >= threshold:
            result["scalar"] = val

    return result


def save_result(result: SolveResult, filepath: str) -> None:
    """Save a SolveResult to a JSON file."""
    data = result.to_dict()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_result(filepath: str) -> Dict[str, Any]:
    """Load a previously saved result dictionary from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


__all__ = [
    "SolveResult",
    "solve_model",
    "extract_variable_values",
    "save_result",
    "load_result",
]
