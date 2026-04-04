from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    NonNegativeReals,
    Objective,
    Constraint,
    ConstraintList,
    maximize,
)


def _build_data_from_state(state):
    data = {
        "nodes": [n.id for n in state.nodes],
        "products": [p.id for p in state.products],
        "suppliers": [s.id for s in state.suppliers],
        "consumers": [c.id for c in state.consumers],
        "bids": {},
        "transport_arcs": [],
        "transport_costs": {},
        "transport_capacities": {},
        "technologies": [],
        "technology_nodes": {},
        "technology_capacities": {},
        "technology_costs": {},
        "technology_yields": {},
    }

    supplier_map = {s.id: s for s in state.suppliers}
    consumer_map = {c.id: c for c in state.consumers}
    technology_map = {t.id: t for t in state.technologies}

    for b in state.bids:
        if b.owner_type == "supplier":
            node = supplier_map[b.owner_id].node
        elif b.owner_type == "consumer":
            node = consumer_map[b.owner_id].node
        elif b.owner_type == "technology":
            node = technology_map[b.owner_id].node
        else:
            node = None

        data["bids"][b.id] = {
            "type": b.owner_type,
            "owner_id": b.owner_id,
            "node": node,
            "product": b.product_id,
            "value": b.price,
            "quantity": b.quantity,
        }

    for t in state.transport_links:
        arc = (t.origin, t.destination)
        data["transport_arcs"].append(arc)
        data["transport_capacities"][arc] = t.capacity
        data["transport_costs"][arc] = getattr(t, "cost", 0.0)

    for tech in state.technologies:
        data["technologies"].append(tech.id)
        data["technology_nodes"][tech.id] = tech.node
        data["technology_costs"][tech.id] = 0.0
        if tech.capacity is not None:
            data["technology_capacities"][tech.id] = tech.capacity
        for product_id, coeff in tech.yield_coefficients.items():
            data["technology_yields"][(tech.id, product_id)] = coeff

    return data


def build_model(data):
    model = ConcreteModel()

    # =========================
    # SETS
    # =========================
    model.N = Set(initialize=data["nodes"])
    model.P = Set(initialize=data["products"])
    model.S = Set(initialize=data["suppliers"])
    model.C = Set(initialize=data["consumers"])
    model.B = Set(initialize=list(data["bids"].keys()))
    model.T = Set(initialize=data["transport_arcs"], dimen=2)
    model.K = Set(initialize=data["technologies"])

    # =========================
    # VARIABLES
    # =========================
    model.q = Var(model.B, domain=NonNegativeReals)
    model.f = Var(model.T, domain=NonNegativeReals)
    model.x = Var(model.K, domain=NonNegativeReals)

    # =========================
    # OBJECTIVE
    # =========================
    def obj_expr(m):
        supplier_term = sum(
            data["bids"][b]["value"] * m.q[b]
            for b in m.B
            if data["bids"][b]["type"] == "supplier"
        )

        consumer_term = sum(
            data["bids"][b]["value"] * m.q[b]
            for b in m.B
            if data["bids"][b]["type"] == "consumer"
        )

        transport_cost = sum(
            data["transport_costs"].get((i, j), 0.0) * m.f[i, j]
            for (i, j) in m.T
        )

        tech_cost = sum(
            data["technology_costs"].get(k, 0.0) * m.x[k]
            for k in m.K
        )

        return consumer_term - supplier_term - transport_cost - tech_cost

    model.obj = Objective(rule=obj_expr, sense=maximize)

    # =========================
    # CONSTRAINTS
    # =========================
    def node_product_balance_rule(m, n, p):
        supply_sum = sum(
            m.q[b]
            for b in m.B
            if data["bids"][b]["node"] == n
            and data["bids"][b]["product"] == p
            and data["bids"][b]["type"] == "supplier"
        )

        consume_sum = sum(
            m.q[b]
            for b in m.B
            if data["bids"][b]["node"] == n
            and data["bids"][b]["product"] == p
            and data["bids"][b]["type"] == "consumer"
        )

        transport_in_sum = sum(
            m.f[i, j]
            for (i, j) in m.T
            if j == n
        )

        transport_out_sum = sum(
            m.f[i, j]
            for (i, j) in m.T
            if i == n
        )

        tech_net = sum(
            data["technology_yields"].get((k, p), 0.0) * m.x[k]
            for k in m.K
            if data["technology_nodes"].get(k) == n
        )

        return (
            supply_sum
            + transport_in_sum
            + tech_net
            - consume_sum
            - transport_out_sum
            == 0
        )

    model.node_balance = Constraint(model.N, model.P, rule=node_product_balance_rule)

    # =========================
    # CAPACITY CONSTRAINTS
    # =========================

    # Supplier bounds
    model.supplier_capacity = ConstraintList()
    for b in model.B:
        if data["bids"][b]["type"] == "supplier":
            cap = data["bids"][b].get("quantity", None)
            if cap is not None:
                model.supplier_capacity.add(model.q[b] <= cap)

    # Consumer bounds
    model.consumer_capacity = ConstraintList()
    for b in model.B:
        if data["bids"][b]["type"] == "consumer":
            cap = data["bids"][b].get("quantity", None)
            if cap is not None:
                model.consumer_capacity.add(model.q[b] <= cap)

    # Transport bounds
    model.transport_capacity = ConstraintList()
    for (i, j) in model.T:
        cap = data["transport_capacities"].get((i, j), None)
        if cap is not None:
            model.transport_capacity.add(model.f[i, j] <= cap)

    # Technology bounds
    model.technology_capacity = ConstraintList()
    for k in model.K:
        cap = data["technology_capacities"].get(k, None)
        if cap is not None:
            model.technology_capacity.add(model.x[k] <= cap)

    return model


def build_model_from_state(state):
    return build_model(_build_data_from_state(state))
