"""Scenario engine for what-if analysis in coordinated supply chain."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .model_builder import build_model_from_state
from .schema import ProblemState, ScenarioRecord
from .solver import SolveResult, solve_model


COMPARISON_DIMENSION_PATTERNS = {
    "objective": [r"\bobjective\b", r"\bsurplus\b", r"\bprofit\b", r"\bvalue\b"],
    "flows": [r"\bflows?\b", r"\btransport\b", r"\bship(?:ment|ments)?\b"],
    "prices": [r"\bprices?\b", r"\bdual\b", r"\bshadow\b"],
    "binding_constraints": [r"\bbinding\b", r"\bconstraint", r"\bscarcity\b"],
    "accepted_bids": [r"\baccepted bids?\b", r"\baccept(?:ed|ance)?\b", r"\bprimal\b"],
    "technology_activity": [r"\btechnolog", r"\btransformation\b", r"\byield\b"],
}

DEFAULT_REQUESTED_DIMENSIONS = [
    "objective",
    "flows",
    "prices",
    "binding_constraints",
]


def clone_state(state: ProblemState) -> ProblemState:
    """Return a deep clone of the given ProblemState."""
    return ProblemState.from_dict(state.to_dict())


def _find_entity_id(state: ProblemState, collection_name: str, message: str) -> Optional[str]:
    collection = getattr(state, collection_name, [])
    lowered = message.lower()
    for entity in collection:
        if entity.id.lower() in lowered:
            return entity.id
    if len(collection) == 1:
        return collection[0].id
    return None


def _find_bid_id(
    state: ProblemState,
    owner_type: Optional[str],
    message: str,
) -> Optional[str]:
    lowered = message.lower()
    owner_candidates = {
        "supplier": [bid for bid in state.bids if bid.owner_type == "supplier"],
        "consumer": [bid for bid in state.bids if bid.owner_type == "consumer"],
        "technology": [bid for bid in state.bids if bid.owner_type == "technology"],
    }
    if owner_type in owner_candidates:
        candidates = owner_candidates[owner_type]
    else:
        candidates = state.bids

    for bid in candidates:
        if bid.id.lower() in lowered or bid.owner_id.lower() in lowered:
            return bid.id
    if len(candidates) == 1:
        return candidates[0].id
    if owner_type is None and candidates:
        return candidates[0].id
    return None


def _extract_numeric_transition(message: str) -> Tuple[Optional[float], Optional[float]]:
    transition = re.search(
        r"from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)",
        message,
        re.IGNORECASE,
    )
    if transition:
        return float(transition.group(1)), float(transition.group(2))

    to_only = re.search(r"\bto\s+(-?\d+(?:\.\d+)?)", message, re.IGNORECASE)
    if to_only:
        return None, float(to_only.group(1))

    by_amount = re.search(
        r"\b(increase|decrease)(?:s|d)?\b.*?\bby\s+(-?\d+(?:\.\d+)?)",
        message,
        re.IGNORECASE,
    )
    if by_amount:
        direction = by_amount.group(1).lower()
        amount = float(by_amount.group(2))
        return None, amount if direction.startswith("increase") else -amount

    return None, None


def _infer_requested_dimensions(message: str) -> List[str]:
    lowered = message.lower()
    dimensions = [
        name
        for name, patterns in COMPARISON_DIMENSION_PATTERNS.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    ]
    return dimensions or list(DEFAULT_REQUESTED_DIMENSIONS)


def extract_scenario_request(state: ProblemState, user_message: str) -> Dict[str, Any]:
    """Lightweight structured scenario extraction from a natural-language request."""

    lowered = user_message.lower()
    requested_dimensions = _infer_requested_dimensions(user_message)
    result: Dict[str, Any] = {
        "is_scenario": False,
        "change_spec": {
            "name": "user_scenario",
            "description": user_message,
        },
        "parameter_type": None,
        "target_object_id": None,
        "old_value": None,
        "new_value": None,
        "requested_dimensions": requested_dimensions,
        "missing": [],
    }

    scenario_trigger = any(
        token in lowered
        for token in [
            "what happens if",
            "how would",
            "if ",
            "increase",
            "decrease",
            "drop from",
            "changes from",
            "change from",
            "set ",
            "unavailable",
            "disable",
            "remove",
            "reduce",
        ]
    )
    comparison_trigger = any(
        token in lowered
        for token in [
            "solution",
            "objective",
            "flows",
            "prices",
            "binding",
            "optimal",
            "compare",
        ]
    )
    result["is_scenario"] = scenario_trigger or comparison_trigger

    old_value, parsed_new_value = _extract_numeric_transition(user_message)

    if any(token in lowered for token in ["technology", "transformation", "yield"]):
        technology_id = _find_entity_id(state, "technologies", user_message)
        result["parameter_type"] = "technology_capacity"
        result["target_object_id"] = technology_id
        if any(token in lowered for token in ["unavailable", "disable", "disabled", "remove", "without"]):
            result["parameter_type"] = "technology_availability"
            result["old_value"] = "available"
            result["new_value"] = "disabled"
            if technology_id is None and len(state.technologies) != 1:
                result["missing"].append("technology id")
            else:
                result["change_spec"]["technologies"] = [{"id": technology_id, "capacity": 0.0}]
            return result
        if technology_id is None:
            result["missing"].append("technology id")
        if parsed_new_value is None:
            result["missing"].append("new technology capacity")
        else:
            result["old_value"] = old_value
            result["new_value"] = parsed_new_value
            result["change_spec"]["technologies"] = [{"id": technology_id, "capacity": parsed_new_value}]
        return result

    if "transport" in lowered:
        link_id = _find_entity_id(state, "transport_links", user_message)
        result["parameter_type"] = "transport_capacity"
        result["target_object_id"] = link_id
        if link_id is None:
            result["missing"].append("transport link id")
        if parsed_new_value is None:
            result["missing"].append("new transport capacity")
        else:
            result["old_value"] = old_value
            result["new_value"] = parsed_new_value
            result["change_spec"]["transport_links"] = [{"id": link_id, "capacity": parsed_new_value}]
        return result

    if "capacity" in lowered and not any(
        token in lowered for token in ["supplier", "consumer", "transport", "technology", "transformation", "yield"]
    ):
        supplier_id = _find_entity_id(state, "suppliers", user_message)
        result["parameter_type"] = "supplier_capacity"
        result["target_object_id"] = supplier_id
        if supplier_id is None:
            result["missing"].append("supplier id")
        if parsed_new_value is None:
            result["missing"].append("new supplier capacity")
        else:
            result["old_value"] = old_value
            result["new_value"] = parsed_new_value
            result["change_spec"]["suppliers"] = [{"id": supplier_id, "capacity": parsed_new_value}]
        return result

    if "supplier" in lowered and "bid" not in lowered:
        supplier_id = _find_entity_id(state, "suppliers", user_message)
        result["parameter_type"] = "supplier_capacity"
        result["target_object_id"] = supplier_id
        if supplier_id is None:
            result["missing"].append("supplier id")
        if parsed_new_value is None:
            result["missing"].append("new supplier capacity")
        else:
            result["old_value"] = old_value
            result["new_value"] = parsed_new_value
            result["change_spec"]["suppliers"] = [{"id": supplier_id, "capacity": parsed_new_value}]
        return result

    if "consumer" in lowered and "bid" not in lowered:
        consumer_id = _find_entity_id(state, "consumers", user_message)
        result["parameter_type"] = "consumer_capacity"
        result["target_object_id"] = consumer_id
        if consumer_id is None:
            result["missing"].append("consumer id")
        if parsed_new_value is None:
            result["missing"].append("new consumer capacity")
        else:
            result["old_value"] = old_value
            result["new_value"] = parsed_new_value
            result["change_spec"]["consumers"] = [{"id": consumer_id, "capacity": parsed_new_value}]
        return result

    if "bid" in lowered or "price" in lowered or "cost" in lowered:
        owner_type = None
        if "supplier" in lowered:
            owner_type = "supplier"
        elif "consumer" in lowered:
            owner_type = "consumer"
        elif "technology" in lowered:
            owner_type = "technology"
        bid_id = _find_bid_id(state, owner_type, user_message)
        result["parameter_type"] = "bid_price"
        result["target_object_id"] = bid_id
        if bid_id is None:
            result["missing"].append("bid id")
        if parsed_new_value is None:
            result["missing"].append("new bid price")
        else:
            result["old_value"] = old_value
            result["new_value"] = parsed_new_value
            result["change_spec"]["bids"] = [{"id": bid_id, "price": parsed_new_value}]
        return result

    result["missing"].append("recognized scenario parameter")
    return result


def apply_parameter_change(state: ProblemState, change_spec: Dict[str, Any]) -> None:
    """Apply modifications described by ``change_spec`` to ``state``."""

    def _apply(list_ref: List[Any], updates: List[Dict[str, Any]]) -> None:
        idx = {item.id: item for item in list_ref}
        for upd in updates:
            ident = upd.get("id")
            if ident is None:
                continue
            target = idx.get(ident)
            if target is None:
                continue
            for key, value in upd.items():
                if key == "id":
                    continue
                if hasattr(target, key):
                    setattr(target, key, value)

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

    # Keep capacity changes solver-relevant by syncing entity capacities to bid quantities.
    supplier_capacity = {supplier.id: supplier.capacity for supplier in state.suppliers}
    consumer_capacity = {consumer.id: consumer.capacity for consumer in state.consumers}
    technology_capacity = {technology.id: technology.capacity for technology in state.technologies}
    for bid in state.bids:
        if bid.owner_type == "supplier" and bid.owner_id in supplier_capacity:
            bid.quantity = supplier_capacity[bid.owner_id]
        elif bid.owner_type == "consumer" and bid.owner_id in consumer_capacity:
            bid.quantity = consumer_capacity[bid.owner_id]
        elif bid.owner_type == "technology" and bid.owner_id in technology_capacity:
            bid.quantity = technology_capacity[bid.owner_id]


def _normalize_result(result: Optional[SolveResult]) -> Optional[Dict[str, Any]]:
    return result.to_dict() if isinstance(result, SolveResult) else result


def _solution_slice(solution: Dict[str, Any], variable_name: str) -> Dict[str, float]:
    raw_values = solution.get(variable_name, {}) if isinstance(solution, dict) else {}
    if not isinstance(raw_values, dict):
        return {}
    return {str(key): float(value or 0.0) for key, value in raw_values.items()}


def _changed_entries(
    baseline: Dict[str, float],
    modified: Dict[str, float],
    tolerance: float = 1e-9,
) -> Dict[str, Dict[str, float]]:
    keys = sorted(set(baseline) | set(modified))
    changes: Dict[str, Dict[str, float]] = {}
    for key in keys:
        before = float(baseline.get(key, 0.0) or 0.0)
        after = float(modified.get(key, 0.0) or 0.0)
        delta = after - before
        if abs(delta) > tolerance:
            changes[key] = {"before": before, "after": after, "delta": delta}
    return changes


def _binding_constraint_changes(
    base_dict: Dict[str, Any],
    scenario_dict: Dict[str, Any],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    base_slacks = (base_dict or {}).get("constraint_slacks", {}) or {}
    scenario_slacks = (scenario_dict or {}).get("constraint_slacks", {}) or {}
    base_duals = (base_dict or {}).get("dual_values", {}) or {}
    scenario_duals = (scenario_dict or {}).get("dual_values", {}) or {}

    changed: Dict[str, Any] = {}
    for name in sorted(set(base_slacks) | set(scenario_slacks)):
        base_binding = base_slacks.get(name) is not None and float(base_slacks.get(name, 0.0)) <= tolerance
        scenario_binding = scenario_slacks.get(name) is not None and float(scenario_slacks.get(name, 0.0)) <= tolerance
        if base_binding != scenario_binding or name in base_duals or name in scenario_duals:
            changed[name] = {
                "base_binding": base_binding,
                "scenario_binding": scenario_binding,
                "base_dual": base_duals.get(name),
                "scenario_dual": scenario_duals.get(name),
            }
    return changed


def compare_solve_results(
    base_result: Optional[SolveResult],
    scenario_result: Optional[SolveResult],
) -> Dict[str, Any]:
    """Compare baseline and modified solves using solver-backed artifacts."""

    comparison: Dict[str, Any] = {
        "objective_delta": None,
        "flow_changes": {},
        "accepted_bid_changes": {},
        "technology_activity_changes": {},
        "price_changes": {},
        "binding_constraint_changes": {},
        "unchanged_dimensions": [],
    }
    if not isinstance(base_result, SolveResult) or not isinstance(scenario_result, SolveResult):
        return comparison

    base_dict = base_result.to_dict()
    scenario_dict = scenario_result.to_dict()

    if base_result.objective_value is not None and scenario_result.objective_value is not None:
        comparison["objective_delta"] = scenario_result.objective_value - base_result.objective_value

    comparison["accepted_bid_changes"] = _changed_entries(
        _solution_slice(base_result.solution, "q"),
        _solution_slice(scenario_result.solution, "q"),
    )
    comparison["flow_changes"] = _changed_entries(
        _solution_slice(base_result.solution, "f"),
        _solution_slice(scenario_result.solution, "f"),
    )
    comparison["technology_activity_changes"] = _changed_entries(
        _solution_slice(base_result.solution, "x"),
        _solution_slice(scenario_result.solution, "x"),
    )
    comparison["price_changes"] = _changed_entries(
        {k: float(v or 0.0) for k, v in (base_dict.get("dual_values", {}) or {}).items()},
        {k: float(v or 0.0) for k, v in (scenario_dict.get("dual_values", {}) or {}).items()},
    )
    comparison["binding_constraint_changes"] = _binding_constraint_changes(base_dict, scenario_dict)

    for dimension in ["accepted_bid_changes", "flow_changes", "technology_activity_changes", "price_changes"]:
        if not comparison[dimension]:
            comparison["unchanged_dimensions"].append(dimension)

    return comparison


def summarize_scenario_results(
    extraction: Dict[str, Any],
    results: Dict[str, Any],
) -> str:
    """Build a structured solver-grounded scenario explanation."""

    base_result = results.get("base")
    scenario_result = results.get("scenario")
    difference = results.get("difference", {})

    if extraction.get("missing"):
        missing_text = ", ".join(extraction["missing"])
        return (
            "Solver-grounded scenario analysis is not ready because the request could not be fully grounded.\n\n"
            f"Missing: {missing_text}."
        )

    if not isinstance(base_result, SolveResult) or not isinstance(scenario_result, SolveResult):
        return "Solver-grounded scenario analysis could not be completed because one of the solves did not return a structured result."

    if not base_result.success:
        return (
            "Solver-grounded scenario analysis could not compare against the baseline because the baseline model did not solve successfully.\n\n"
            f"Baseline status: {base_result.status}. {base_result.message}"
        )
    if not scenario_result.success:
        return (
            "Solver-grounded scenario analysis could not complete the modified case.\n\n"
            f"Modified status: {scenario_result.status}. {scenario_result.message}"
        )

    base_dict = base_result.to_dict()
    scenario_dict = scenario_result.to_dict()
    requested = extraction.get("requested_dimensions", DEFAULT_REQUESTED_DIMENSIONS)

    lines = [
        "Solver-grounded verification",
        "",
        "Baseline",
        f"- Status: {base_result.status}",
        f"- Objective: {base_result.objective_value}",
    ]
    if base_dict.get("dual_values"):
        lines.append(f"- Dual values extracted: {len(base_dict['dual_values'])}")

    lines.extend(
        [
            "",
            "Modified scenario",
            f"- Parameter: {extraction.get('parameter_type')}",
            f"- Target: {extraction.get('target_object_id') or 'not uniquely identified'}",
            f"- Requested change: {extraction.get('old_value')} -> {extraction.get('new_value')}",
            f"- Status: {scenario_result.status}",
            f"- Objective: {scenario_result.objective_value}",
        ]
    )

    objective_delta = difference.get("objective_delta")
    lines.extend(["", "Changes in objective"])
    if objective_delta is None:
        lines.append("- Objective comparison is unavailable.")
    else:
        lines.append(f"- Objective delta: {objective_delta}")

    if "flows" in requested:
        flow_changes = difference.get("flow_changes", {})
        lines.extend(["", "Changes in flows"])
        if flow_changes:
            for key, values in flow_changes.items():
                lines.append(f"- {key}: {values['before']} -> {values['after']} (delta {values['delta']})")
        else:
            lines.append("- No transport-flow changes were detected.")

    if "accepted_bids" in requested or "flows" in requested:
        bid_changes = difference.get("accepted_bid_changes", {})
        lines.extend(["", "Changes in accepted bids"])
        if bid_changes:
            for key, values in bid_changes.items():
                lines.append(f"- {key}: {values['before']} -> {values['after']} (delta {values['delta']})")
        else:
            lines.append("- No accepted-bid changes were detected.")

    if "technology_activity" in requested:
        tech_changes = difference.get("technology_activity_changes", {})
        lines.extend(["", "Changes in technology activity"])
        if tech_changes:
            for key, values in tech_changes.items():
                lines.append(f"- {key}: {values['before']} -> {values['after']} (delta {values['delta']})")
        else:
            lines.append("- No technology-activity changes were detected.")

    if "prices" in requested:
        price_changes = difference.get("price_changes", {})
        lines.extend(["", "Changes in prices or dual interpretation"])
        if price_changes:
            for key, values in price_changes.items():
                lines.append(f"- {key}: {values['before']} -> {values['after']} (delta {values['delta']})")
        else:
            lines.append("- No solver-imported dual-price changes were detected, or the active solver did not expose usable dual values.")

    if "binding_constraints" in requested:
        binding_changes = difference.get("binding_constraint_changes", {})
        lines.extend(["", "Binding-constraint interpretation"])
        if binding_changes:
            for key, values in binding_changes.items():
                lines.append(
                    f"- {key}: baseline binding={values['base_binding']}, modified binding={values['scenario_binding']}, "
                    f"dual {values['base_dual']} -> {values['scenario_dual']}"
                )
        else:
            lines.append("- No binding-constraint changes were detected.")

    unchanged = difference.get("unchanged_dimensions", [])
    lines.extend(["", "What did not change"])
    if unchanged:
        rendered = ", ".join(item.replace("_changes", "").replace("_", " ") for item in unchanged)
        lines.append(f"- No material change detected in: {rendered}.")
    else:
        lines.append("- Each tracked comparison bucket changed or was not requested.")

    lines.extend(
        [
            "",
            "Interpretation",
            "- This answer is grounded in a baseline solve plus a modified solve of the actual optimization model, not only generic LP theory.",
        ]
    )
    return "\n".join(lines)


def run_scenario(
    base_state: ProblemState,
    change_spec: Dict[str, Any],
    solve: bool = True,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a scenario based on ``base_state`` and ``change_spec``."""

    results: Dict[str, Any] = {"base": None, "scenario": None, "difference": {}}

    base_clone = clone_state(base_state)
    if solve:
        base_model = build_model_from_state(base_clone)
        results["base"] = solve_model(base_model, **(solver_options or {}))

    scenario_clone = clone_state(base_state)
    apply_parameter_change(scenario_clone, change_spec)
    if solve:
        scenario_model = build_model_from_state(scenario_clone)
        results["scenario"] = solve_model(scenario_model, **(solver_options or {}))

    if solve:
        results["difference"] = compare_solve_results(results.get("base"), results.get("scenario"))

    name = change_spec.get("name", "scenario")
    desc = change_spec.get("description", str(change_spec))
    base_state.add_scenario(ScenarioRecord(name=name, description=desc))

    results["scenario_state"] = scenario_clone
    results["base_state"] = base_clone
    results["base_dict"] = _normalize_result(results.get("base"))
    results["scenario_dict"] = _normalize_result(results.get("scenario"))
    return results


__all__ = [
    "apply_parameter_change",
    "clone_state",
    "compare_solve_results",
    "extract_scenario_request",
    "run_scenario",
    "summarize_scenario_results",
]
