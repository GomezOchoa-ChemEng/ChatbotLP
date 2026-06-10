"""Fixed-charge technology design models for coordinated supply chains.

This module extends the continuous market-clearing formulation with optional
binary technology installation decisions.  It is intentionally separate from
``model_builder.py`` so the original Case A/B/C benchmark path keeps its
historical behavior.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from pyomo.environ import (
    Binary,
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

from .schema import ProblemState
from .solver import SolveResult, solve_model
from .validator import validate_state


TOLERANCE = 1e-7


def validate_design_state(state: ProblemState) -> Dict[str, Any]:
    """Validate data needed by the fixed-charge design formulation."""

    base = validate_state(state)
    missing = list(base.get("missing_parameters", []))

    for technology in state.technologies:
        if technology.fixed_cost is None:
            missing.append(f"technology:{technology.id} missing fixed_cost")
        else:
            try:
                fixed_cost = float(technology.fixed_cost)
                if fixed_cost != fixed_cost or fixed_cost in (float("inf"), float("-inf")):
                    missing.append(f"technology:{technology.id} fixed_cost is not finite: {fixed_cost}")
            except (TypeError, ValueError):
                missing.append(f"technology:{technology.id} fixed_cost not numeric: {technology.fixed_cost}")

    missing = list(dict.fromkeys(missing))
    return {
        **base,
        "missing_parameters": missing,
        "solver_ready": not missing
        and not base.get("invalid_references")
        and not base.get("incomplete_technologies"),
    }


def build_design_data_from_state(state: ProblemState) -> Dict[str, Any]:
    """Extract model data from ``ProblemState`` without mutating it."""

    supplier_nodes = {supplier.id: supplier.node for supplier in state.suppliers}
    consumer_nodes = {consumer.id: consumer.node for consumer in state.consumers}

    data: Dict[str, Any] = {
        "nodes": state.node_ids(),
        "products": state.product_ids(),
        "bids": {},
        "transport_links": {},
        "technologies": {},
    }

    for bid in state.bids:
        node = None
        if bid.owner_type == "supplier":
            node = supplier_nodes.get(bid.owner_id)
        elif bid.owner_type == "consumer":
            node = consumer_nodes.get(bid.owner_id)
        data["bids"][bid.id] = {
            "type": bid.owner_type,
            "owner_id": bid.owner_id,
            "node": node,
            "product": bid.product_id,
            "value": bid.price,
            "quantity": bid.quantity,
        }

    for link in state.transport_links:
        data["transport_links"][link.id] = {
            "origin": link.origin,
            "destination": link.destination,
            "product": link.product,
            "capacity": link.capacity,
            "cost": link.cost,
        }

    for technology in state.technologies:
        data["technologies"][technology.id] = {
            "node": technology.node,
            "capacity": technology.capacity,
            "variable_cost": technology.cost,
            "fixed_cost": technology.fixed_cost,
            "yields": dict(technology.yield_coefficients),
        }

    return data


def build_fixed_charge_design_model_from_state(state: ProblemState) -> ConcreteModel:
    """Build a MILP with binary install variables for technology candidates."""

    diagnostics = validate_design_state(state)
    if not diagnostics["solver_ready"]:
        details = diagnostics["missing_parameters"] + diagnostics.get("invalid_references", [])
        raise ValueError(_missing_design_parameter_message(details))
    return _build_design_model(build_design_data_from_state(state), installed_technologies=None)


def build_fixed_design_management_model_from_state(
    state: ProblemState,
    installed_technologies: Mapping[str, float] | Iterable[str],
) -> ConcreteModel:
    """Build an operational LP with technology installation decisions fixed."""

    diagnostics = validate_design_state(state)
    if not diagnostics["solver_ready"]:
        details = diagnostics["missing_parameters"] + diagnostics.get("invalid_references", [])
        raise ValueError(_missing_design_parameter_message(details))
    normalized = _normalize_installed_technologies(state, installed_technologies)
    return _build_design_model(build_design_data_from_state(state), installed_technologies=normalized)


def solve_design_problem(
    state: ProblemState,
    solver_name: str = "glpk",
    solver_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> SolveResult:
    """Build and solve the fixed-charge design MILP."""

    model = build_fixed_charge_design_model_from_state(state)
    return solve_model(
        model,
        solver_name=solver_name,
        fallback_solver="glpk",
        solver_options=solver_options,
        verbose=verbose,
    )


def solve_fixed_design_management_problem(
    state: ProblemState,
    installed_technologies: Mapping[str, float] | Iterable[str],
    solver_name: str = "glpk",
    solver_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> SolveResult:
    """Build and solve the fixed-design operational LP."""

    model = build_fixed_design_management_model_from_state(state, installed_technologies)
    return solve_model(
        model,
        solver_name=solver_name,
        fallback_solver="glpk",
        solver_options=solver_options,
        verbose=verbose,
    )


def extract_design_solution(
    state: ProblemState,
    solve_result: SolveResult | Mapping[str, Any],
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    """Return a compact, ID-based solution summary."""

    solve_dict = _as_solve_dict(solve_result)
    solution = solve_dict.get("solution", {}) or {}
    q = _float_block(solution.get("q", {}))
    f = _float_block(solution.get("f", {}))
    x = _float_block(solution.get("x", {}))
    y = _float_block(solution.get("y", {}))

    supplier_ids = {supplier.id for supplier in state.suppliers}
    consumer_ids = {consumer.id for consumer in state.consumers}
    accepted_supply: Dict[str, float] = {}
    accepted_demand: Dict[str, float] = {}
    for bid in state.bids:
        quantity = q.get(bid.id, 0.0)
        if bid.owner_type == "supplier" and bid.owner_id in supplier_ids:
            accepted_supply[bid.owner_id] = accepted_supply.get(bid.owner_id, 0.0) + quantity
        elif bid.owner_type == "consumer" and bid.owner_id in consumer_ids:
            accepted_demand[bid.owner_id] = accepted_demand.get(bid.owner_id, 0.0) + quantity

    if not y:
        y = {technology.id: 1.0 if abs(x.get(technology.id, 0.0)) > tolerance else 0.0 for technology in state.technologies}

    return {
        "success": bool(solve_dict.get("success", False)),
        "status": solve_dict.get("status"),
        "termination_condition": solve_dict.get("termination_condition"),
        "solver_name": solve_dict.get("solver_name"),
        "objective_value": solve_dict.get("objective_value"),
        "accepted_supply": accepted_supply,
        "accepted_demand": accepted_demand,
        "transport_flows": f,
        "technology_activities": x,
        "technology_installations": y,
        "selected_technologies": {
            technology_id: value
            for technology_id, value in y.items()
            if value >= 1.0 - tolerance
        },
        "raw_solution": deepcopy(solution),
    }


def compute_balance_residuals(
    state: ProblemState,
    solve_result: SolveResult | Mapping[str, Any],
    tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    """Compute node-product balance residuals from a solved model."""

    solve_dict = _as_solve_dict(solve_result)
    solution = solve_dict.get("solution", {}) or {}
    q = _float_block(solution.get("q", {}))
    f = _float_block(solution.get("f", {}))
    x = _float_block(solution.get("x", {}))

    supplier_node = {supplier.id: supplier.node for supplier in state.suppliers}
    consumer_node = {consumer.id: consumer.node for consumer in state.consumers}
    supplier_product = {supplier.id: supplier.product for supplier in state.suppliers}
    consumer_product = {consumer.id: consumer.product for consumer in state.consumers}

    rows: List[Dict[str, Any]] = []
    for node in state.nodes:
        for product in state.products:
            supply = sum(
                q.get(bid.id, 0.0)
                for bid in state.bids
                if bid.owner_type == "supplier"
                and supplier_node.get(bid.owner_id) == node.id
                and supplier_product.get(bid.owner_id) == product.id
            )
            demand = sum(
                q.get(bid.id, 0.0)
                for bid in state.bids
                if bid.owner_type == "consumer"
                and consumer_node.get(bid.owner_id) == node.id
                and consumer_product.get(bid.owner_id) == product.id
            )
            incoming = sum(
                _transport_flow_value(link, f)
                for link in state.transport_links
                if link.destination == node.id and link.product == product.id
            )
            outgoing = sum(
                _transport_flow_value(link, f)
                for link in state.transport_links
                if link.origin == node.id and link.product == product.id
            )
            technology_net = sum(
                _yield_for_product(technology, product.id) * x.get(technology.id, 0.0)
                for technology in state.technologies
                if technology.node == node.id
            )
            residual = supply + incoming + technology_net - demand - outgoing
            rows.append(
                {
                    "node": node.id,
                    "product": product.id,
                    "supply": supply,
                    "accepted_demand": demand,
                    "incoming_flow": incoming,
                    "outgoing_flow": outgoing,
                    "technology_net": technology_net,
                    "residual": residual,
                    "holds": abs(residual) <= tolerance,
                }
            )
    return rows


def compute_objective_decomposition(
    state: ProblemState,
    solve_result: SolveResult | Mapping[str, Any],
    installed_technologies: Optional[Mapping[str, float] | Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute objective components from state data and solution values."""

    solve_dict = _as_solve_dict(solve_result)
    solution = solve_dict.get("solution", {}) or {}
    q = _float_block(solution.get("q", {}))
    f = _float_block(solution.get("f", {}))
    x = _float_block(solution.get("x", {}))
    y = _float_block(solution.get("y", {}))
    if installed_technologies is not None:
        y = _normalize_installed_technologies(state, installed_technologies)
    elif not y:
        y = {technology.id: 1.0 if abs(x.get(technology.id, 0.0)) > TOLERANCE else 0.0 for technology in state.technologies}

    bid_by_id = {bid.id: bid for bid in state.bids}
    demand_revenue = sum(
        _required_number(bid_by_id[bid_id].price, f"bid:{bid_id} price") * quantity
        for bid_id, quantity in q.items()
        if bid_id in bid_by_id and bid_by_id[bid_id].owner_type == "consumer"
    )
    supplier_cost = sum(
        _required_number(bid_by_id[bid_id].price, f"bid:{bid_id} price") * quantity
        for bid_id, quantity in q.items()
        if bid_id in bid_by_id and bid_by_id[bid_id].owner_type == "supplier"
    )
    transport_cost = sum(
        _required_number(link.cost, f"transport:{link.id} cost") * _transport_flow_value(link, f)
        for link in state.transport_links
    )
    operating_cost = sum(
        _required_number(technology.cost, f"technology:{technology.id} cost") * x.get(technology.id, 0.0)
        for technology in state.technologies
    )
    investment_cost = sum(
        _required_number(technology.fixed_cost, f"technology:{technology.id} fixed_cost") * y.get(technology.id, 0.0)
        for technology in state.technologies
    )
    total_profit = demand_revenue - supplier_cost - transport_cost - operating_cost - investment_cost

    return {
        "demand_revenue": demand_revenue,
        "supplier_cost": supplier_cost,
        "transport_cost": transport_cost,
        "operating_cost": operating_cost,
        "investment_cost": investment_cost,
        "total_profit": total_profit,
    }


def extract_node_product_prices(
    solve_result: SolveResult | Mapping[str, Any],
    dual_sign: float = -1.0,
) -> Dict[str, float]:
    """Extract economic node-product prices from ``node_balance`` duals.

    The model writes balances as ``supply + inflow + technology_net - demand
    - outflow == 0``.  With this convention, GLPK reports the negative of the
    usual market-clearing price, so ``dual_sign=-1`` returns economic prices.
    """

    solve_dict = _as_solve_dict(solve_result)
    prices: Dict[str, float] = {}
    for constraint_name, dual_value in (solve_dict.get("dual_values", {}) or {}).items():
        if dual_value is None or not str(constraint_name).startswith("node_balance["):
            continue
        coordinates = str(constraint_name)[len("node_balance[") : -1]
        parts = [part.strip().strip("'\"") for part in coordinates.split(",")]
        if len(parts) == 2:
            prices[f"{parts[0]}:{parts[1]}"] = float(dual_sign) * float(dual_value)
    return prices


def compute_participant_economics(
    state: ProblemState,
    solve_result: SolveResult | Mapping[str, Any],
    node_product_prices: Mapping[str, float],
    installed_technologies: Optional[Mapping[str, float] | Iterable[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Summarize participant revenues, costs, and profits from solved quantities."""

    solve_dict = _as_solve_dict(solve_result)
    solution = solve_dict.get("solution", {}) or {}
    q = _float_block(solution.get("q", {}))
    f = _float_block(solution.get("f", {}))
    x = _float_block(solution.get("x", {}))
    y = _float_block(solution.get("y", {}))
    if installed_technologies is not None:
        y = _normalize_installed_technologies(state, installed_technologies)

    supplier_by_id = {supplier.id: supplier for supplier in state.suppliers}
    consumer_by_id = {consumer.id: consumer for consumer in state.consumers}

    suppliers = []
    consumers = []
    for bid in state.bids:
        quantity = q.get(bid.id, 0.0)
        if bid.owner_type == "supplier" and bid.owner_id in supplier_by_id:
            supplier = supplier_by_id[bid.owner_id]
            price = node_product_prices.get(f"{supplier.node}:{supplier.product}")
            revenue = None if price is None else price * quantity
            cost = _required_number(bid.price, f"bid:{bid.id} price") * quantity
            suppliers.append(
                {
                    "participant": supplier.id,
                    "node": supplier.node,
                    "product": supplier.product,
                    "quantity": quantity,
                    "market_price": price,
                    "revenue": revenue,
                    "cost": cost,
                    "profit": None if revenue is None else revenue - cost,
                }
            )
        elif bid.owner_type == "consumer" and bid.owner_id in consumer_by_id:
            consumer = consumer_by_id[bid.owner_id]
            price = node_product_prices.get(f"{consumer.node}:{consumer.product}")
            payment = None if price is None else price * quantity
            value_received = _required_number(bid.price, f"bid:{bid.id} price") * quantity
            consumers.append(
                {
                    "participant": consumer.id,
                    "node": consumer.node,
                    "product": consumer.product,
                    "quantity": quantity,
                    "market_price": price,
                    "willingness_value": value_received,
                    "payment": payment,
                    "surplus": None if payment is None else value_received - payment,
                }
            )

    transporters = []
    for link in state.transport_links:
        flow = _transport_flow_value(link, f)
        origin_price = node_product_prices.get(f"{link.origin}:{link.product}")
        destination_price = node_product_prices.get(f"{link.destination}:{link.product}")
        revenue = None
        if origin_price is not None and destination_price is not None:
            revenue = (destination_price - origin_price) * flow
        cost = _required_number(link.cost, f"transport:{link.id} cost") * flow
        transporters.append(
            {
                "participant": link.id,
                "origin": link.origin,
                "destination": link.destination,
                "product": link.product,
                "flow": flow,
                "revenue": revenue,
                "cost": cost,
                "profit": None if revenue is None else revenue - cost,
            }
        )

    technologies = []
    for technology in state.technologies:
        activity = x.get(technology.id, 0.0)
        input_cost = 0.0
        output_revenue = 0.0
        prices_available = True
        for product_id, coefficient in technology.yield_coefficients.items():
            price = node_product_prices.get(f"{technology.node}:{product_id}")
            if price is None:
                prices_available = False
                continue
            quantity = _required_number(
                coefficient,
                f"technology:{technology.id} yield for {product_id}",
            ) * activity
            if quantity >= 0:
                output_revenue += price * quantity
            else:
                input_cost += price * (-quantity)
        operating_cost = _required_number(technology.cost, f"technology:{technology.id} cost") * activity
        investment_cost = _required_number(
            technology.fixed_cost,
            f"technology:{technology.id} fixed_cost",
        ) * y.get(technology.id, 0.0)
        profit = None
        if prices_available:
            profit = output_revenue - input_cost - operating_cost - investment_cost
        technologies.append(
            {
                "participant": technology.id,
                "node": technology.node,
                "activity": activity,
                "installed": y.get(technology.id, 0.0),
                "output_revenue": output_revenue if prices_available else None,
                "input_cost": input_cost if prices_available else None,
                "operating_cost": operating_cost,
                "investment_cost": investment_cost,
                "profit": profit,
            }
        )

    return {
        "suppliers": suppliers,
        "consumers": consumers,
        "transporters": transporters,
        "technologies": technologies,
    }


def clone_with_consumer_bid_price(
    state: ProblemState,
    product_id: str,
    price: float,
) -> ProblemState:
    """Clone ``state`` and set all consumer bids for ``product_id`` to ``price``."""

    clone = ProblemState.from_dict(state.to_dict())
    for bid in clone.bids:
        if bid.owner_type == "consumer" and bid.product_id == product_id:
            bid.price = float(price)
    return clone


def _build_design_model(
    data: Dict[str, Any],
    installed_technologies: Optional[Mapping[str, float]],
) -> ConcreteModel:
    _raise_for_missing_design_data(data)
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT)

    model.N = Set(initialize=data["nodes"])
    model.P = Set(initialize=data["products"])
    model.B = Set(initialize=list(data["bids"].keys()))
    model.L = Set(initialize=list(data["transport_links"].keys()))
    model.K = Set(initialize=list(data["technologies"].keys()))

    model.q = Var(model.B, domain=NonNegativeReals)
    model.f = Var(model.L, domain=NonNegativeReals)
    model.x = Var(model.K, domain=NonNegativeReals)
    if installed_technologies is None:
        model.y = Var(model.K, domain=Binary)

    def obj_expr(m: ConcreteModel):
        consumer_term = sum(
            data["bids"][bid_id]["value"] * m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["type"] == "consumer"
        )
        supplier_term = sum(
            data["bids"][bid_id]["value"] * m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["type"] == "supplier"
        )
        transport_cost = sum(
            data["transport_links"][link_id]["cost"] * m.f[link_id]
            for link_id in m.L
        )
        operating_cost = sum(
            data["technologies"][technology_id]["variable_cost"] * m.x[technology_id]
            for technology_id in m.K
        )
        if installed_technologies is None:
            fixed_cost = sum(
                data["technologies"][technology_id]["fixed_cost"] * m.y[technology_id]
                for technology_id in m.K
            )
        else:
            fixed_cost = sum(
                data["technologies"][technology_id]["fixed_cost"]
                * installed_technologies.get(technology_id, 0.0)
                for technology_id in m.K
            )
        return consumer_term - supplier_term - transport_cost - operating_cost - fixed_cost

    model.obj = Objective(rule=obj_expr, sense=maximize)

    def node_product_balance_rule(m: ConcreteModel, node: str, product: str):
        supply_sum = sum(
            m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["node"] == node
            and data["bids"][bid_id]["product"] == product
            and data["bids"][bid_id]["type"] == "supplier"
        )
        demand_sum = sum(
            m.q[bid_id]
            for bid_id in m.B
            if data["bids"][bid_id]["node"] == node
            and data["bids"][bid_id]["product"] == product
            and data["bids"][bid_id]["type"] == "consumer"
        )
        transport_in_sum = sum(
            m.f[link_id]
            for link_id in m.L
            if data["transport_links"][link_id]["destination"] == node
            and data["transport_links"][link_id]["product"] == product
        )
        transport_out_sum = sum(
            m.f[link_id]
            for link_id in m.L
            if data["transport_links"][link_id]["origin"] == node
            and data["transport_links"][link_id]["product"] == product
        )
        technology_net = sum(
            data["technologies"][technology_id]["yields"].get(product, 0.0) * m.x[technology_id]
            for technology_id in m.K
            if data["technologies"][technology_id]["node"] == node
        )
        balance_expr = supply_sum + transport_in_sum + technology_net - demand_sum - transport_out_sum
        if isinstance(balance_expr, (int, float)):
            return Constraint.Feasible if abs(balance_expr) <= 1e-12 else Constraint.Infeasible
        return balance_expr == 0

    model.node_balance = Constraint(model.N, model.P, rule=node_product_balance_rule)

    model.bid_capacity = ConstraintList()
    for bid_id in model.B:
        cap = data["bids"][bid_id].get("quantity")
        if cap is not None:
            model.bid_capacity.add(model.q[bid_id] <= cap)

    model.transport_capacity = ConstraintList()
    for link_id in model.L:
        cap = data["transport_links"][link_id].get("capacity")
        if cap is not None:
            model.transport_capacity.add(model.f[link_id] <= cap)

    model.technology_capacity = ConstraintList()
    for technology_id in model.K:
        cap = data["technologies"][technology_id].get("capacity")
        if cap is None:
            continue
        if installed_technologies is None:
            model.technology_capacity.add(model.x[technology_id] <= cap * model.y[technology_id])
        else:
            model.technology_capacity.add(
                model.x[technology_id] <= cap * installed_technologies.get(technology_id, 0.0)
            )

    return model


def _raise_for_missing_design_data(data: Dict[str, Any]) -> None:
    missing = []
    for bid_id, bid in data.get("bids", {}).items():
        if bid.get("value") is None:
            missing.append(f"bid:{bid_id} missing price")
        if bid.get("quantity") is None:
            missing.append(f"bid:{bid_id} missing quantity")
    for link_id, link in data.get("transport_links", {}).items():
        if link.get("cost") is None:
            missing.append(f"transport:{link_id} missing cost")
        if link.get("capacity") is None:
            missing.append(f"transport:{link_id} missing capacity")
    for technology_id, technology in data.get("technologies", {}).items():
        if technology.get("variable_cost") is None:
            missing.append(f"technology:{technology_id} missing cost")
        if technology.get("fixed_cost") is None:
            missing.append(f"technology:{technology_id} missing fixed_cost")
        if technology.get("capacity") is None:
            missing.append(f"technology:{technology_id} missing capacity")
        coefficients = list((technology.get("yields") or {}).values())
        if (
            not coefficients
            or any(coefficient is None for coefficient in coefficients)
            or not any(coefficient is not None and coefficient < 0 for coefficient in coefficients)
            or not any(coefficient is not None and coefficient > 0 for coefficient in coefficients)
        ):
            missing.append(f"technology:{technology_id} missing input/output yield")
    if missing:
        raise ValueError(_missing_design_parameter_message(missing))


def _missing_design_parameter_message(missing: Sequence[Any]) -> str:
    details = "; ".join(dict.fromkeys(str(item) for item in missing))
    return (
        "Cannot build fixed-charge design model with missing required parameters: "
        f"{details}. Provide values or confirm explicit assumptions before solving."
    )


def _normalize_installed_technologies(
    state: ProblemState,
    installed_technologies: Mapping[str, float] | Iterable[str],
) -> Dict[str, float]:
    if isinstance(installed_technologies, Mapping):
        normalized = {}
        for technology in state.technologies:
            raw_value = installed_technologies.get(technology.id, 0.0)
            if raw_value is None:
                raise ValueError(f"installed technology value for {technology.id} is missing")
            normalized[technology.id] = float(raw_value)
        return normalized
    installed = set(installed_technologies)
    return {technology.id: 1.0 if technology.id in installed else 0.0 for technology in state.technologies}


def _as_solve_dict(solve_result: SolveResult | Mapping[str, Any]) -> Dict[str, Any]:
    if hasattr(solve_result, "to_dict"):
        return solve_result.to_dict()
    return dict(solve_result)


def _float_block(values: Any) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if value is not None
    }


def _transport_flow_value(link: Any, flows: Mapping[str, float]) -> float:
    if link.id in flows:
        return float(flows[link.id])
    for key in (
        str((link.origin, link.destination)),
        f"({link.origin}, {link.destination})",
        f"{link.origin}->{link.destination}",
        f"{link.origin}_to_{link.destination}",
        f"{link.origin}_to_{link.destination}:{link.product}",
    ):
        if key in flows:
            return float(flows[key])
    return 0.0


def _yield_for_product(technology: Any, product_id: str) -> float:
    if product_id not in technology.yield_coefficients:
        return 0.0
    coefficient = technology.yield_coefficients[product_id]
    return _required_number(coefficient, f"technology:{technology.id} yield for {product_id}")


def _required_number(value: Optional[float], label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is missing")
    return float(value)


def solve_consumer_price_sweep(
    state: ProblemState,
    product_id: str,
    prices: Sequence[float],
    quantity_getter: Callable[[ProblemState, SolveResult], float],
    solver_name: str = "glpk",
) -> List[Dict[str, Any]]:
    """Solve a design problem over a sequence of consumer bid prices."""

    rows = []
    for price in prices:
        scenario_state = clone_with_consumer_bid_price(state, product_id, float(price))
        result = solve_design_problem(scenario_state, solver_name=solver_name)
        summary = extract_design_solution(scenario_state, result)
        quantity = quantity_getter(scenario_state, result) if result.success else None
        rows.append(
            {
                "price": float(price),
                "success": result.success,
                "objective_value": result.objective_value,
                "quantity": quantity,
                "selected_technologies": sorted(summary["selected_technologies"]),
                "result": result,
                "state": scenario_state,
            }
        )
    return rows


def find_minimum_price_for_target(
    state: ProblemState,
    product_id: str,
    target_quantity: float,
    quantity_getter: Callable[[ProblemState, SolveResult], float],
    low: float,
    high: float,
    coarse_steps: int = 20,
    bisection_iterations: int = 24,
    solver_name: str = "glpk",
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Find the minimum consumer bid price that reaches a target quantity."""

    if coarse_steps < 2:
        raise ValueError("coarse_steps must be at least 2")

    step = (high - low) / (coarse_steps - 1)
    coarse_prices = [low + index * step for index in range(coarse_steps)]
    coarse_rows = solve_consumer_price_sweep(
        state,
        product_id=product_id,
        prices=coarse_prices,
        quantity_getter=quantity_getter,
        solver_name=solver_name,
    )

    bracket_low = None
    bracket_high = None
    for row in coarse_rows:
        quantity = row["quantity"]
        if quantity is not None and quantity >= target_quantity - tolerance:
            bracket_high = row["price"]
            break
        bracket_low = row["price"]

    if bracket_high is None:
        return {
            "threshold": None,
            "coarse_rows": coarse_rows,
            "refined_rows": [],
            "message": "Target quantity was not reached within the supplied price range.",
        }
    if bracket_low is None:
        bracket_low = low

    refined_rows = []
    lo = float(bracket_low)
    hi = float(bracket_high)
    for _ in range(bisection_iterations):
        mid = (lo + hi) / 2.0
        row = solve_consumer_price_sweep(
            state,
            product_id=product_id,
            prices=[mid],
            quantity_getter=quantity_getter,
            solver_name=solver_name,
        )[0]
        refined_rows.append(row)
        quantity = row["quantity"]
        if quantity is not None and quantity >= target_quantity - tolerance:
            hi = mid
        else:
            lo = mid

    threshold_state = clone_with_consumer_bid_price(state, product_id, hi)
    threshold_result = solve_design_problem(threshold_state, solver_name=solver_name)
    return {
        "threshold": hi,
        "coarse_rows": coarse_rows,
        "refined_rows": refined_rows,
        "threshold_state": threshold_state,
        "threshold_result": threshold_result,
        "message": "Target quantity reached.",
    }


__all__ = [
    "build_design_data_from_state",
    "build_fixed_charge_design_model_from_state",
    "build_fixed_design_management_model_from_state",
    "clone_with_consumer_bid_price",
    "compute_balance_residuals",
    "compute_objective_decomposition",
    "compute_participant_economics",
    "extract_design_solution",
    "extract_node_product_prices",
    "find_minimum_price_for_target",
    "solve_consumer_price_sweep",
    "solve_design_problem",
    "solve_fixed_design_management_problem",
    "validate_design_state",
]
