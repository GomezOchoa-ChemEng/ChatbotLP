"""Pyomo model builder for coordinated supply chain instances."""

from __future__ import annotations

from typing import Any, Dict

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Set,
    Suffix,
    Var,
    maximize,
)

from .schema import (
    MarketBidRecord,
    MarketInstance,
    MarketTechnologyRecord,
    MarketTransportRecord,
    ProblemState,
)
from .validator import validate_state


def build_market_instance(state: ProblemState) -> MarketInstance:
    """Normalize a teaching-facing ProblemState into a machine-facing MarketInstance."""

    supplier_nodes = {supplier.id: supplier.node for supplier in state.suppliers}
    consumer_nodes = {consumer.id: consumer.node for consumer in state.consumers}
    technology_nodes = {technology.id: technology.node for technology in state.technologies}

    bids = []
    for bid in state.bids:
        if bid.owner_type == "supplier":
            node = supplier_nodes.get(bid.owner_id)
        elif bid.owner_type == "consumer":
            node = consumer_nodes.get(bid.owner_id)
        elif bid.owner_type == "technology":
            node = technology_nodes.get(bid.owner_id)
        else:
            node = None

        bids.append(
            MarketBidRecord(
                id=bid.id,
                owner_id=bid.owner_id,
                owner_type=bid.owner_type,
                node=node,
                product_id=bid.product_id,
                price=bid.price,
                quantity=bid.quantity,
            )
        )

    transport_links = [
        MarketTransportRecord(
            id=link.id,
            origin=link.origin,
            destination=link.destination,
            product_id=link.product,
            capacity=link.capacity,
            cost=link.cost,
        )
        for link in state.transport_links
    ]

    technologies = [
        MarketTechnologyRecord(
            id=technology.id,
            node=technology.node,
            capacity=technology.capacity,
            cost=technology.cost,
            yield_coefficients=dict(technology.yield_coefficients),
        )
        for technology in state.technologies
    ]

    benchmark_case = state.benchmark.case_family if state.benchmark else None
    return MarketInstance(
        problem_title=state.problem_title,
        nodes=state.node_ids(),
        products=state.product_ids(),
        bids=bids,
        transport_links=transport_links,
        technologies=technologies,
        benchmark_case=benchmark_case,
        source="problem_state",
        metadata={
            "supplier_ids": [supplier.id for supplier in state.suppliers],
            "consumer_ids": [consumer.id for consumer in state.consumers],
            "transport_ids": [link.id for link in state.transport_links],
            "technology_ids": [technology.id for technology in state.technologies],
        },
    )


def _build_data_from_market_instance(instance: MarketInstance) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "nodes": list(instance.nodes),
        "products": list(instance.products),
        "suppliers": [
            bid.owner_id
            for bid in instance.bids
            if bid.owner_type == "supplier"
        ],
        "consumers": [
            bid.owner_id
            for bid in instance.bids
            if bid.owner_type == "consumer"
        ],
        "bids": {},
        "transport_arcs": [],
        "transport_arc_ids": {},
        "transport_products": {},
        "transport_costs": {},
        "transport_capacities": {},
        "technologies": [],
        "technology_nodes": {},
        "technology_capacities": {},
        "technology_costs": {},
        "technology_yields": {},
    }

    for bid in instance.bids:
        data["bids"][bid.id] = {
            "type": bid.owner_type,
            "owner_id": bid.owner_id,
            "node": bid.node,
            "product": bid.product_id,
            "value": bid.price,
            "quantity": bid.quantity,
        }

    for link in instance.transport_links:
        arc = (link.origin, link.destination)
        data["transport_arcs"].append(arc)
        data["transport_arc_ids"][arc] = link.id
        data["transport_products"][arc] = link.product_id
        data["transport_capacities"][arc] = link.capacity
        data["transport_costs"][arc] = link.cost

    for technology in instance.technologies:
        data["technologies"].append(technology.id)
        data["technology_nodes"][technology.id] = technology.node
        data["technology_costs"][technology.id] = technology.cost
        if technology.capacity is not None:
            data["technology_capacities"][technology.id] = technology.capacity
        for product_id, coefficient in technology.yield_coefficients.items():
            data["technology_yields"][(technology.id, product_id)] = coefficient

    return data


def _build_data_from_state(state: ProblemState) -> Dict[str, Any]:
    """Backwards-compatible data extraction helper used by tests."""

    return _build_data_from_market_instance(build_market_instance(state))


def build_model(data: Dict[str, Any]) -> ConcreteModel:
    _raise_for_missing_model_data(data)
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT)

    model.N = Set(initialize=data["nodes"])
    model.P = Set(initialize=data["products"])
    model.S = Set(initialize=list(dict.fromkeys(data["suppliers"])))
    model.C = Set(initialize=list(dict.fromkeys(data["consumers"])))
    model.B = Set(initialize=list(data["bids"].keys()))
    model.T = Set(initialize=data["transport_arcs"], dimen=2)
    model.K = Set(initialize=data["technologies"])

    model.q = Var(model.B, domain=NonNegativeReals)
    model.f = Var(model.T, domain=NonNegativeReals)
    model.x = Var(model.K, domain=NonNegativeReals)

    def obj_expr(m: ConcreteModel):
        supplier_term = sum(
            data["bids"][bid_id]["value"] * m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["type"] == "supplier"
        )
        consumer_term = sum(
            data["bids"][bid_id]["value"] * m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["type"] == "consumer"
        )
        transport_cost = sum(
            data["transport_costs"][(origin, destination)] * m.f[origin, destination]
            for origin, destination in m.T
        )
        technology_cost = sum(
            data["technology_costs"][technology_id] * m.x[technology_id]
            for technology_id in m.K
        )
        return consumer_term - supplier_term - transport_cost - technology_cost

    model.obj = Objective(rule=obj_expr, sense=maximize)

    def node_product_balance_rule(m: ConcreteModel, node: str, product: str):
        supply_sum = sum(
            m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["node"] == node
            and data["bids"][bid_id]["product"] == product
            and data["bids"][bid_id]["type"] == "supplier"
        )
        consume_sum = sum(
            m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["node"] == node
            and data["bids"][bid_id]["product"] == product
            and data["bids"][bid_id]["type"] == "consumer"
        )
        transport_in_sum = sum(
            m.f[origin, destination]
            for origin, destination in m.T
            if destination == node
            and data["transport_products"].get((origin, destination)) == product
        )
        transport_out_sum = sum(
            m.f[origin, destination]
            for origin, destination in m.T
            if origin == node
            and data["transport_products"].get((origin, destination)) == product
        )
        technology_net = sum(
            data["technology_yields"].get((technology_id, product), 0.0) * m.x[technology_id]
            for technology_id in m.K
            if data["technology_nodes"].get(technology_id) == node
        )
        balance_expr = supply_sum + transport_in_sum + technology_net - consume_sum - transport_out_sum
        if isinstance(balance_expr, (int, float)):
            return Constraint.Feasible if abs(balance_expr) <= 1e-12 else Constraint.Infeasible
        return balance_expr == 0

    model.node_balance = Constraint(model.N, model.P, rule=node_product_balance_rule)

    model.supplier_capacity = ConstraintList()
    for bid_id in model.B:
        if data["bids"][bid_id]["type"] == "supplier":
            cap = data["bids"][bid_id].get("quantity")
            if cap is not None:
                model.supplier_capacity.add(model.q[bid_id] <= cap)

    model.consumer_capacity = ConstraintList()
    for bid_id in model.B:
        if data["bids"][bid_id]["type"] == "consumer":
            cap = data["bids"][bid_id].get("quantity")
            if cap is not None:
                model.consumer_capacity.add(model.q[bid_id] <= cap)

    model.transport_capacity = ConstraintList()
    for origin, destination in model.T:
        cap = data["transport_capacities"].get((origin, destination))
        if cap is not None:
            model.transport_capacity.add(model.f[origin, destination] <= cap)

    model.technology_capacity = ConstraintList()
    for technology_id in model.K:
        cap = data["technology_capacities"].get(technology_id)
        if cap is not None:
            model.technology_capacity.add(model.x[technology_id] <= cap)

    return model


def build_model_from_market_instance(instance: MarketInstance) -> ConcreteModel:
    _raise_for_missing_market_parameters(instance)
    return build_model(_build_data_from_market_instance(instance))


def build_model_from_state(state: ProblemState) -> ConcreteModel:
    validation = validate_state(state)
    if not validation["solver_ready"]:
        details = validation["missing_parameters"] + validation["invalid_references"]
        raise ValueError(_missing_parameter_message(details))
    return build_model_from_market_instance(build_market_instance(state))


def _raise_for_missing_market_parameters(instance: MarketInstance) -> None:
    missing = []
    for bid in instance.bids:
        if bid.price is None:
            missing.append(f"bid:{bid.id} missing price")
        if bid.quantity is None:
            missing.append(f"bid:{bid.id} missing quantity")
    for link in instance.transport_links:
        if link.cost is None:
            missing.append(f"transport:{link.id} missing cost")
        if link.capacity is None:
            missing.append(f"transport:{link.id} missing capacity")
    for technology in instance.technologies:
        if technology.cost is None:
            missing.append(f"technology:{technology.id} missing cost")
        if technology.capacity is None:
            missing.append(f"technology:{technology.id} missing capacity")
        coefficients = list(technology.yield_coefficients.values())
        if (
            not coefficients
            or any(coefficient is None for coefficient in coefficients)
            or not any(coefficient is not None and coefficient < 0 for coefficient in coefficients)
            or not any(coefficient is not None and coefficient > 0 for coefficient in coefficients)
        ):
            missing.append(f"technology:{technology.id} missing input/output yield")
    if missing:
        raise ValueError(_missing_parameter_message(missing))


def _raise_for_missing_model_data(data: Dict[str, Any]) -> None:
    missing = []
    for bid_id, bid in data.get("bids", {}).items():
        if bid.get("value") is None:
            missing.append(f"bid:{bid_id} missing price")
        if bid.get("quantity") is None:
            missing.append(f"bid:{bid_id} missing quantity")
    for arc in data.get("transport_arcs", []):
        link_id = data.get("transport_arc_ids", {}).get(arc, str(arc))
        if data.get("transport_costs", {}).get(arc) is None:
            missing.append(f"transport:{link_id} missing cost")
        if data.get("transport_capacities", {}).get(arc) is None:
            missing.append(f"transport:{link_id} missing capacity")
    for technology_id in data.get("technologies", []):
        if data.get("technology_costs", {}).get(technology_id) is None:
            missing.append(f"technology:{technology_id} missing cost")
        if data.get("technology_capacities", {}).get(technology_id) is None:
            missing.append(f"technology:{technology_id} missing capacity")
        coefficients = [
            coefficient
            for (raw_id, _product_id), coefficient in data.get("technology_yields", {}).items()
            if raw_id == technology_id
        ]
        if (
            not coefficients
            or any(coefficient is None for coefficient in coefficients)
            or not any(coefficient is not None and coefficient < 0 for coefficient in coefficients)
            or not any(coefficient is not None and coefficient > 0 for coefficient in coefficients)
        ):
            missing.append(f"technology:{technology_id} missing input/output yield")
    if missing:
        raise ValueError(_missing_parameter_message(missing))


def _missing_parameter_message(missing: Any) -> str:
    details = "; ".join(dict.fromkeys(str(item) for item in missing))
    return (
        "Cannot build optimization model with missing required parameters: "
        f"{details}. Provide the values or confirm explicit assumptions before solving."
    )


__all__ = [
    "build_market_instance",
    "build_model",
    "build_model_from_market_instance",
    "build_model_from_state",
    "_build_data_from_state",
]
