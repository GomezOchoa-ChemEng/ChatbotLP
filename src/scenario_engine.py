"""Scenario engine for what-if analysis in coordinated supply chain.

This module provides utilities to clone an existing ProblemState, apply
parameter modifications, rebuild/solve the corresponding optimization model,
and track scenarios in the state history.  The design keeps the engine
simple and extensible: more types of parameter changes and comparison metrics
can be added later.

Functions
---------
- clone_state(state): return a deep copy of a ProblemState
- apply_parameter_change(state, change_spec): mutate a state according to the
  specification dictionary
- run_scenario(state, change_spec, solve=True): perform a scenario run based
  on ``state`` and ``change_spec``; optionally solve both base and scenario
  models and record the scenario history.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from .schema import (
    ProblemState,
    ScenarioRecord,
)
from .model_builder import build_model_from_state
from .solver import solve_model, SolveResult


def clone_state(state: ProblemState) -> ProblemState:
    """Return a deep clone of the given ProblemState.

    We use ``deepcopy`` since Pydantic models are copyable, but the simplest
    reliable method is to round-trip via dict parsing to ensure no shared
    references remain.
    """
    return ProblemState.from_dict(state.to_dict())


def apply_parameter_change(state: ProblemState, change_spec: Dict[str, Any]) -> None:
    """Apply modifications described by ``change_spec`` to ``state``.

    The specification is a dictionary whose keys correspond to top-level lists
    of the ProblemState (e.g. ``suppliers``, ``bids``, ``technologies``).
    Each value should be a list of update dictionaries containing an ``id``
    field and any attributes to change.  Only existing entities are modified;
    new entities are not created by this helper.

    Example:

        change_spec = {
            "suppliers": [{"id": "s1", "capacity": 20}],
            "bids": [{"id": "b2", "price": 5.0}],
        }
    """
    # helper to update list-of-models by id
    def _apply(list_ref: List[Any], updates: List[Dict[str, Any]]) -> None:
        idx = {item.id: item for item in list_ref}
        for upd in updates:
            ident = upd.get("id")
            if ident is None:
                continue
            target = idx.get(ident)
            if target is None:
                continue
            for k, v in upd.items():
                if k == "id":
                    continue
                if hasattr(target, k):
                    setattr(target, k, v)

    for key, upd_list in change_spec.items():
        if not isinstance(upd_list, list):
            continue
        if key == "suppliers":
            _apply(state.suppliers, upd_list)
        elif key == "consumers":
            _apply(state.consumers, upd_list)
        elif key == "transport_links":
            _apply(state.transport_links, upd_list)
        elif key == "technologies":
            _apply(state.technologies, upd_list)
        elif key == "bids":
            _apply(state.bids, upd_list)
        elif key == "nodes":
            _apply(state.nodes, upd_list)
        elif key == "products":
            _apply(state.products, upd_list)
        # other keys can be supported later


def run_scenario(
    base_state: ProblemState,
    change_spec: Dict[str, Any],
    solve: bool = True,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a scenario based on ``base_state`` and ``change_spec``.

    The function performs the following steps:
    1. Clone the base state to avoid mutating it
    2. Optionally solve the base model (if ``solve`` is True)
    3. Apply ``change_spec`` to a second clone and rebuild/solve that scenario
    4. Record a ScenarioRecord in the original ``base_state`` with a name and
       description derived from ``change_spec``
    5. Return a dictionary containing the base and scenario SolveResults (or
       ``None`` if ``solve`` is False) along with a simple comparison
       dictionary.

    The returned structure looks like::

        {
            "base": SolveResult | None,
            "scenario": SolveResult | None,
            "difference": {"objective_delta": float}  # may be empty
        }

    """
    results: Dict[str, Any] = {"base": None, "scenario": None, "difference": {}}

    # prepare base clone and optional solve
    base_clone = clone_state(base_state)
    if solve:
        base_model = build_model_from_state(base_clone)
        base_res = solve_model(base_model, **(solver_options or {}))
        results["base"] = base_res

    # scenario clone and modification
    scenario_clone = clone_state(base_state)
    apply_parameter_change(scenario_clone, change_spec)
    if solve:
        scen_model = build_model_from_state(scenario_clone)
        scen_res = solve_model(scen_model, **(solver_options or {}))
        results["scenario"] = scen_res

    # comparison
    br = results.get("base")
    sr = results.get("scenario")
    if br and sr and br.objective_value is not None and sr.objective_value is not None:
        results["difference"]["objective_delta"] = sr.objective_value - br.objective_value

    # record history entry on the original base state
    name = change_spec.get("name", "scenario")
    desc = change_spec.get("description", str(change_spec))
    base_state.add_scenario(ScenarioRecord(name=name, description=desc))

    # attach scenario clone to results for inspection if desired
    results["scenario_state"] = scenario_clone
    results["base_state"] = base_clone

    return results


__all__ = ["clone_state", "apply_parameter_change", "run_scenario"]
