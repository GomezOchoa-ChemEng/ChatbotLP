"""Solver engine for Pyomo-based optimization models.

This module provides a high-level interface to solve Pyomo models and
extract structured results. It handles solver execution, result validation,
and formatting without modifying the input model or ProblemState.

Design philosophy:
- Accept a Pyomo ConcreteModel as input
- Execute a solver (default: ipopt or glpk fallback)
- Return a structured, read-only result dictionary
- Report termination conditions explicitly
- Extract decision variables and objective value
- Keep the solver engine modular and testable
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    TerminationCondition,
    SolverStatus,
    value,
)


# ------------------------------------------------
# Result container classes
# ------------------------------------------------


class SolveResult:
    """Structured result from solving a Pyomo model.

    Attributes:
        model (ConcreteModel): The solved model (may be modified by solver).
        status (str): Termination status (e.g., 'optimal', 'infeasible', 'unbounded').
        message (str): Human-readable status message.
        objective_value (Optional[float]): Objective function value if solved.
        solver_time (float): Solver execution time in seconds.
        solution (Dict[str, Any]): Decision variable values extracted from model.
        success (bool): True if solution is feasible/optimal, False otherwise.

    Methods can be used to serialize results, check feasibility, and extract
    decision variables for post-solve analysis.
    """

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
        """Convert result to a JSON-serializable dictionary."""
        return {
            "status": self.status,
            "message": self.message,
            "objective_value": self.objective_value,
            "solver_time": self.solver_time,
            "solution": self.solution,
            "success": self.success,
        }

    def __repr__(self) -> str:
        return (
            f"SolveResult(status={self.status}, success={self.success}, "
            f"objective={self.objective_value:.6f})" if self.objective_value is not None
            else f"SolveResult(status={self.status}, success={self.success})"
        )


# ------------------------------------------------
# Core solver interface
# ------------------------------------------------


def solve_model(
    model: ConcreteModel,
    solver_name: str = "ipopt",
    solver_options: Optional[Dict[str, Any]] = None,
    fallback_solver: str = "glpk",
    verbose: bool = False,
) -> SolveResult:
    """Solve a Pyomo ConcreteModel and return structured results.

    This function:
    1. Creates a solver instance (with fallback if primary fails)
    2. Executes the solve action
    3. Extracts results and decision variables
    4. Returns a SolveResult object with explicit status

    Args:
        model: A Pyomo ConcreteModel with sets, variables, objective, and constraints.
        solver_name: Name of the primary solver (default: 'ipopt').
        solver_options: Dict of solver-specific options (e.g., {'tol': 1e-6}).
        fallback_solver: If primary solver unavailable, try this solver (default: 'glpk').
        verbose: If True, print solver output to console.

    Returns:
        SolveResult: Structured result object containing status, objective, solution, etc.

    Notes:
        If neither the primary nor the fallback solver is available the function
        will **not** raise an exception. Instead it returns a `SolveResult`
        instance with `status="solver_unavailable"` and `success=False`.
    """

    if solver_options is None:
        solver_options = {}

    # ------------------------------------------------
    # select a solver, but handle absence gracefully
    # ------------------------------------------------
    solver = None
    used_solver = None

    def _is_available(s):
        try:
            return s is not None and s.available()
        except Exception:
            return False

    # try primary
    try:
        solver_candidate = SolverFactory(solver_name)
    except Exception:
        solver_candidate = None
    if _is_available(solver_candidate):
        solver = solver_candidate
        used_solver = solver_name
    else:
        if verbose:
            print(f"Primary solver {solver_name} not available, checking fallback {fallback_solver}.")
        # try fallback
        try:
            solver_candidate = SolverFactory(fallback_solver)
        except Exception:
            solver_candidate = None
        if _is_available(solver_candidate):
            solver = solver_candidate
            used_solver = fallback_solver

    # if still no solver found, return structured unavailable result
    if solver is None:
        msg = (
            f"No available solver found (tried '{solver_name}' and '{fallback_solver}')."
        )
        return SolveResult(
            model=model,
            status="solver_unavailable",
            message=msg,
            objective_value=None,
            solver_time=0.0,
            solution={},
            success=False,
        )

    # Apply solver options
    for key, val in solver_options.items():
        solver.options[key] = val

    if verbose:
        print(f"Solving model with {used_solver}...")

    # Execute solve
    import time
    start_time = time.time()
    results = solver.solve(model, tee=verbose)
    solver_time = time.time() - start_time

    if verbose:
        print(f"Solver finished in {solver_time:.2f} seconds.")

    # Extract status and message
    status_str = str(results.solver.status)
    term_cond = str(results.solver.termination_condition)

    # Map termination condition to human-readable status
    status_map = {
        "optimal": "optimal",
        "feasible": "feasible",
        "infeasible": "infeasible",
        "unbounded": "unbounded",
        "infeasibilityProven": "infeasible",
        "unboundedProven": "unbounded",
    }
    status = status_map.get(term_cond, term_cond.lower())

    # Determine success flag: optimal or feasible is a success
    success = status in ("optimal", "feasible")

    # Extract objective value if solution exists
    objective_value = None
    if hasattr(model, "obj") and success:
        try:
            objective_value = float(value(model.obj))
        except Exception:
            objective_value = None

    # Extract decision variables
    solution = _extract_solution(model)

    # Build message
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


# ------------------------------------------------
# Solution extraction helpers
# ------------------------------------------------


def _extract_solution(model: ConcreteModel) -> Dict[str, Any]:
    """Extract all variable values from a solved Pyomo model.

    Returns a nested dictionary mapping variable names to their values,
    indexed by dimension keys if applicable.

    Args:
        model: A Pyomo ConcreteModel after solve.

    Returns:
        Dict[str, Any]: Flattened representation of all variable values.
                        For indexed variables, returns {var_name: {index: value, ...}}.
                        For scalar variables, returns {var_name: value}.

    Example:
        >>> solution = _extract_solution(model)
        >>> solution["q"]  # if q is indexed by bid id
        {"bid_1": 10.5, "bid_2": 20.3, ...}
    """
    solution = {}

    # Iterate over all variables in the model
    for var in model.component_objects(ctype=None):
        # Only extract Var objects
        if not hasattr(var, "is_indexed"):
            continue

        var_name = var.name
        if var.is_indexed():
            # Indexed variable: extract all indices and values
            var_dict = {}
            for idx in var:
                val = value(var[idx])
                # Convert tuple indices to list for JSON serialization
                idx_key = idx if isinstance(idx, str) else (idx,) if not isinstance(idx, tuple) else idx
                idx_str = idx_key if isinstance(idx_key, str) else str(idx_key)
                var_dict[idx_str] = val
            solution[var_name] = var_dict
        else:
            # Scalar variable
            solution[var_name] = value(var)

    return solution


def extract_variable_values(
    model: ConcreteModel,
    var_name: str,
    threshold: Optional[float] = None,
) -> Dict[Any, float]:
    """Extract values for a specific variable from a solved model.

    Useful for post-solve analysis, e.g., extracting bid quantities q above
    a certain threshold.

    Args:
        model: Solved Pyomo ConcreteModel.
        var_name: Name of the variable to extract (e.g., "q", "flow").
        threshold: If provided, filter to indices where |value| >= threshold.

    Returns:
        Dict mapping index to value for the specified variable.

    Raises:
        KeyError: If variable not found in model.
    """
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


# ------------------------------------------------
# Result serialization
# ------------------------------------------------


def save_result(result: SolveResult, filepath: str) -> None:
    """Save a SolveResult to a JSON file.

    Args:
        result: SolveResult object to save.
        filepath: Path to output JSON file.
    """
    data = result.to_dict()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_result(filepath: str) -> Dict[str, Any]:
    """Load a previously saved result dictionary from JSON.

    Args:
        filepath: Path to input JSON file.

    Returns:
        Dict containing the result (does not reconstruct SolveResult object,
        only the data dictionary).
    """
    with open(filepath, "r") as f:
        return json.load(f)


__all__ = [
    "SolveResult",
    "solve_model",
    "extract_variable_values",
    "save_result",
    "load_result",
]
