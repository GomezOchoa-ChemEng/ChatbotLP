"""Helpers for deterministic primal and dual scaffolds."""

from __future__ import annotations

from typing import Any, Dict, List

from .schema import ProblemState


def build_primal_representation(state: ProblemState) -> Dict[str, Any]:
    """Build a compact structured representation of the current primal model."""

    supplier_map = {supplier.id: supplier for supplier in state.suppliers}
    consumer_map = {consumer.id: consumer for consumer in state.consumers}
    variables: List[Dict[str, Any]] = []
    objective_terms: List[Dict[str, Any]] = []
    constraints: List[Dict[str, Any]] = []

    supplier_bids = []
    consumer_bids = []

    for bid in state.bids:
        variable_name = f"q_{{{bid.id}}}"
        variables.append(
            {
                "name": variable_name,
                "symbol": variable_name,
                "domain": ">= 0",
                "bid_id": bid.id,
                "owner_type": bid.owner_type,
                "product_id": bid.product_id,
                "objective_coefficient": 0.0,
            }
        )

        if bid.owner_type == "supplier":
            supplier_bids.append(bid)
            owner_node = supplier_map.get(bid.owner_id).node if bid.owner_id in supplier_map else None
            variables[-1]["objective_coefficient"] = -bid.price
            variables[-1]["variable_class"] = "supplier_bid"
            variables[-1]["owner_node"] = owner_node
            objective_terms.append(
                {
                    "coefficient": -bid.price,
                    "symbol": variable_name,
                    "interpretation": "supplier acceptance term",
                    "node": owner_node,
                    "product": bid.product_id,
                }
            )
        elif bid.owner_type == "consumer":
            consumer_bids.append(bid)
            owner_node = consumer_map.get(bid.owner_id).node if bid.owner_id in consumer_map else None
            variables[-1]["objective_coefficient"] = bid.price
            variables[-1]["variable_class"] = "consumer_bid"
            variables[-1]["owner_node"] = owner_node
            objective_terms.append(
                {
                    "coefficient": bid.price,
                    "symbol": variable_name,
                    "interpretation": "consumer acceptance term",
                    "node": owner_node,
                    "product": bid.product_id,
                }
            )

        if bid.quantity is not None and bid.owner_type in {"supplier", "consumer"}:
            constraint_name = f"{bid.owner_type}_cap_{bid.id}"
            constraints.append(
                {
                    "name": constraint_name,
                    "type": "upper_bound",
                    "sense": "<=",
                    "rhs": bid.quantity,
                    "lhs_terms": [{"coefficient": 1.0, "symbol": variable_name}],
                    "dual_symbol": (
                        f"\\mu_{{{bid.id}}}"
                        if bid.owner_type == "supplier"
                        else f"\\nu_{{{bid.id}}}"
                    ),
                    "latex_label": (
                        f"\\mu_{{{bid.id}}}" if bid.owner_type == "supplier" else f"\\nu_{{{bid.id}}}"
                    ),
                }
            )

    for link in state.transport_links:
        variable_name = f"f_{{{link.origin},{link.destination}}}"
        variables.append(
            {
                "name": variable_name,
                "symbol": variable_name,
                "domain": ">= 0",
                "arc": (link.origin, link.destination),
                "product_id": link.product,
                "objective_coefficient": 0.0,
                "variable_class": "transport_flow",
            }
        )
        objective_terms.append(
            {
                "coefficient": 0.0,
                "symbol": variable_name,
                "interpretation": "transport term (explicit cost defaults to 0 in current deterministic model)",
                "node": None,
                "product": link.product,
            }
        )
        if link.capacity is not None:
            constraints.append(
                {
                    "name": f"transport_cap_{link.id}",
                    "type": "upper_bound",
                    "sense": "<=",
                    "rhs": link.capacity,
                    "lhs_terms": [{"coefficient": 1.0, "symbol": variable_name}],
                    "dual_symbol": f"\\tau_{{{link.origin},{link.destination}}}",
                    "latex_label": f"\\tau_{{{link.origin},{link.destination}}}",
                }
            )

    for tech in state.technologies:
        variable_name = f"x_{{{tech.id}}}"
        variables.append(
            {
                "name": variable_name,
                "symbol": variable_name,
                "domain": ">= 0",
                "technology_id": tech.id,
                "objective_coefficient": 0.0,
                "variable_class": "technology_activity",
                "owner_node": tech.node,
            }
        )
        objective_terms.append(
            {
                "coefficient": 0.0,
                "symbol": variable_name,
                "interpretation": "technology activity term (explicit cost defaults to 0 in current deterministic model)",
                "node": tech.node,
                "product": None,
            }
        )

    for node in state.nodes:
        for product in state.products:
            lhs_terms: List[Dict[str, Any]] = []
            for bid in supplier_bids:
                owner = supplier_map.get(bid.owner_id)
                if owner and owner.node == node.id and bid.product_id == product.id:
                    lhs_terms.append({"coefficient": 1.0, "symbol": f"q_{{{bid.id}}}"})
            for bid in consumer_bids:
                owner = consumer_map.get(bid.owner_id)
                if owner and owner.node == node.id and bid.product_id == product.id:
                    lhs_terms.append({"coefficient": -1.0, "symbol": f"q_{{{bid.id}}}"})
            for link in state.transport_links:
                flow_symbol = f"f_{{{link.origin},{link.destination}}}"
                if link.product != product.id:
                    continue
                if link.destination == node.id:
                    lhs_terms.append({"coefficient": 1.0, "symbol": flow_symbol})
                if link.origin == node.id:
                    lhs_terms.append({"coefficient": -1.0, "symbol": flow_symbol})
            for tech in state.technologies:
                coefficient = tech.yield_coefficients.get(product.id)
                if coefficient is not None and tech.node == node.id:
                    lhs_terms.append({"coefficient": coefficient, "symbol": f"x_{{{tech.id}}}"})
            constraints.append(
                {
                    "name": f"balance_{node.id}_{product.id}",
                    "type": "balance",
                    "sense": "=",
                    "rhs": 0.0,
                    "lhs_terms": lhs_terms,
                    "dual_symbol": f"\\pi_{{{node.id},{product.id}}}",
                    "latex_label": f"\\pi_{{{node.id},{product.id}}}",
                    "node_id": node.id,
                    "product_id": product.id,
                }
            )

    objective = {
        "sense": "max",
        "terms": objective_terms,
        "description": "maximize coordinated surplus using accepted bids, flows, and technology activity",
    }
    return {"objective": objective, "constraints": constraints, "variables": variables}


def build_dual_scaffold(primal_representation: Dict[str, Any]) -> Dict[str, Any]:
    """Build a canonical dual scaffold from the structured primal representation."""

    constraints = primal_representation.get("constraints", [])
    variables = primal_representation.get("variables", [])
    objective_terms = []
    dual_variables = []
    stationarity_conditions = []

    coefficient_map = {
        variable["symbol"]: [] for variable in variables
    }

    for constraint in constraints:
        dual_symbol = constraint.get("dual_symbol")
        if not dual_symbol:
            continue
        dual_variables.append(
            {
                "symbol": dual_symbol,
                "constraint_name": constraint["name"],
                "sense": "free" if constraint.get("sense") == "=" else ">= 0",
                "constraint_type": constraint.get("type"),
            }
        )
        if constraint.get("sense") == "<=" and constraint.get("rhs") is not None:
            objective_terms.append(
                {
                    "coefficient": constraint["rhs"],
                    "symbol": dual_symbol,
                    "source_constraint": constraint["name"],
                }
            )
        for term in constraint.get("lhs_terms", []):
            coefficient_map.setdefault(term["symbol"], []).append(
                {
                    "constraint_name": constraint["name"],
                    "dual_symbol": dual_symbol,
                    "coefficient": term["coefficient"],
                    "constraint_type": constraint.get("type"),
                }
            )

    for variable in variables:
        stationarity_conditions.append(
            {
                "primal_variable": variable["symbol"],
                "nonnegativity": True,
                "dual_expression_terms": coefficient_map.get(variable["symbol"], []),
                "objective_coefficient": variable.get("objective_coefficient", 0.0),
                "variable_class": variable.get("variable_class"),
                "bid_id": variable.get("bid_id"),
                "owner_type": variable.get("owner_type"),
                "product_id": variable.get("product_id"),
                "owner_node": variable.get("owner_node"),
                "arc": variable.get("arc"),
                "technology_id": variable.get("technology_id"),
            }
        )

    return {
        "sense": "min",
        "objective_terms": objective_terms,
        "dual_variables": dual_variables,
        "stationarity_conditions": stationarity_conditions,
    }


def infer_negative_bid_notes(state: ProblemState) -> List[str]:
    """Collect notes relevant to negative bids."""

    notes = []
    negative_bids = [bid for bid in state.bids if bid.price < 0]
    if negative_bids:
        notes.append(f"Detected {len(negative_bids)} negative bid(s) in the current state.")
        notes.extend(
            [
                f"Bid {bid.id} for product {bid.product_id} is negative at {bid.price}."
                for bid in negative_bids
            ]
        )
    return notes
