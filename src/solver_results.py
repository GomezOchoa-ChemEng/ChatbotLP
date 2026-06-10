"""Compact solver-grounded results for explanation and figure reproduction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .figure_data import build_figure_data
from .model_builder import build_market_instance
from .schema import MarketInstance, ProblemState


class SolverResults(BaseModel):
    """Compact machine-facing results object for solved market instances."""

    solve_timestamp: datetime = Field(default_factory=datetime.utcnow)
    solver_name: str = "unknown"
    solver_status: str
    termination_condition: str
    solver_time_seconds: float
    success: bool
    objective_value: Optional[float] = None
    bid_allocations: Dict[str, float] = Field(default_factory=dict)
    transport_flows: Dict[str, float] = Field(default_factory=dict)
    technology_extents: Dict[str, float] = Field(default_factory=dict)
    technology_installations: Dict[str, float] = Field(default_factory=dict)
    node_product_duals: Dict[str, float] = Field(default_factory=dict)
    constraint_slacks: Dict[str, float] = Field(default_factory=dict)
    totals_by_product: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    plotting_data: Dict[str, Any] = Field(default_factory=dict)
    market_instance: Optional[MarketInstance] = None
    scenario_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    @classmethod
    def from_solve_result(cls, solve_result: Any, problem_state: ProblemState) -> "SolverResults":
        solve_dict = solve_result.to_dict() if hasattr(solve_result, "to_dict") else dict(solve_result)
        market_instance = build_market_instance(problem_state)

        bid_allocations = _extract_var_block(solve_dict, "q")
        transport_flows = _extract_var_block(solve_dict, "f")
        technology_extents = _extract_var_block(solve_dict, "x")
        technology_installations = _extract_var_block(solve_dict, "y")
        node_product_duals = _extract_node_product_duals(solve_dict.get("dual_values", {}))
        constraint_slacks = {
            key: float(value)
            for key, value in (solve_dict.get("constraint_slacks", {}) or {}).items()
            if value is not None
        }
        totals_by_product = _compute_totals_by_product(problem_state, bid_allocations)

        results = cls(
            solver_name=str(solve_dict.get("solver_name") or _infer_solver_name(solve_dict.get("message", ""))),
            solver_status=str(solve_dict.get("status", "unknown")),
            termination_condition=str(solve_dict.get("termination_condition") or solve_dict.get("status", "unknown")),
            solver_time_seconds=float(solve_dict.get("solver_time", 0.0) or 0.0),
            success=bool(solve_dict.get("success", False)),
            objective_value=solve_dict.get("objective_value"),
            bid_allocations=bid_allocations,
            transport_flows=transport_flows,
            technology_extents=technology_extents,
            technology_installations=technology_installations,
            node_product_duals=node_product_duals,
            constraint_slacks=constraint_slacks,
            totals_by_product=totals_by_product,
            market_instance=market_instance,
            scenario_metadata=dict(getattr(solve_result, "scenario_metadata", {}) or solve_dict.get("scenario_metadata", {}) or {}),
        )
        results.plotting_data = _build_plotting_data(problem_state, market_instance, results)
        return results


def _extract_var_block(solve_dict: Dict[str, Any], name: str) -> Dict[str, float]:
    values = solve_dict.get("solution", {}).get(name, {})
    if not isinstance(values, dict):
        return {}
    return {str(key): float(value) for key, value in values.items() if value is not None}


def _extract_node_product_duals(dual_values: Dict[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for constraint_name, value in (dual_values or {}).items():
        if value is None or not constraint_name.startswith("node_balance["):
            continue
        coordinates = constraint_name[len("node_balance[") : -1]
        parts = [part.strip().strip("'\"") for part in coordinates.split(",")]
        if len(parts) == 2:
            result[f"{parts[0]}:{parts[1]}"] = float(value)
    return result


def _compute_totals_by_product(state: ProblemState, bid_allocations: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    bid_map = {bid.id: bid for bid in state.bids}
    for bid_id, quantity in bid_allocations.items():
        bid = bid_map.get(bid_id)
        if bid is None:
            continue
        product_totals = totals.setdefault(bid.product_id, {"supply": 0.0, "demand": 0.0})
        if bid.owner_type == "supplier":
            product_totals["supply"] += quantity
        elif bid.owner_type == "consumer":
            product_totals["demand"] += quantity
    return totals


def _build_plotting_data(
    state: ProblemState,
    market_instance: MarketInstance,
    results: SolverResults,
) -> Dict[str, Any]:
    flow_rows = []
    transport_id_map = {
        (link.origin, link.destination): link.id
        for link in state.transport_links
    }
    product_map = {
        (link.origin, link.destination): link.product
        for link in state.transport_links
    }
    for arc_key, flow in results.transport_flows.items():
        origin, destination = _parse_tuple_key(arc_key)
        transport_id = transport_id_map.get((origin, destination), arc_key)
        flow_rows.append(
            {
                "transport_id": transport_id,
                "origin": origin,
                "destination": destination,
                "product": product_map.get((origin, destination)),
                "flow": flow,
            }
        )

    dual_rows = []
    for key, price in results.node_product_duals.items():
        node, product = key.split(":", 1)
        dual_rows.append({"node": node, "product": product, "shadow_price": price})

    return {
        "market_summary": {
            "title": market_instance.problem_title,
            "nodes": len(market_instance.nodes),
            "products": len(market_instance.products),
            "bids": len(market_instance.bids),
            "transport_links": len(market_instance.transport_links),
            "technologies": len(market_instance.technologies),
        },
        "bid_allocations": [
            {
                "bid_id": bid.id,
                "owner_type": bid.owner_type,
                "product_id": bid.product_id,
                "node": bid.node,
                "accepted_quantity": results.bid_allocations.get(bid.id, 0.0),
                "price": bid.price,
            }
            for bid in market_instance.bids
        ],
        "transport_flows": flow_rows,
        "nodal_prices": dual_rows,
        "product_totals": [
            {"product": product, **totals}
            for product, totals in results.totals_by_product.items()
        ],
        **build_figure_data(results),
    }


def _parse_tuple_key(key: str) -> tuple[str, str]:
    stripped = key.strip().strip("()")
    parts = [part.strip().strip("'\"") for part in stripped.split(",")]
    if len(parts) != 2:
        return key, key
    return parts[0], parts[1]


def _infer_solver_name(message: str) -> str:
    lowered = (message or "").lower()
    if "glpk" in lowered:
        return "glpk"
    if "ipopt" in lowered:
        return "ipopt"
    return "unknown"


__all__ = ["SolverResults"]
