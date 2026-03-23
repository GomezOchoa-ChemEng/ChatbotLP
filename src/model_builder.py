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


def build_model(data):
    model = ConcreteModel()

    # =========================
    # SETS
    # =========================
    model.N = Set(initialize=data["nodes"])
    model.P = Set(initialize=data["products"])
    model.S = Set(initialize=data["suppliers"])
    model.C = Set(initialize=data["consumers"])
    model.B = Set(initialize=data["bids"])
    model.T = Set(initialize=data["transport_arcs"], dimen=2)
    model.K = Set(initialize=data["technologies"])

    # =========================
    # VARIABLES
    # =========================
    model.q = Var(model.B, domain=NonNegativeReals)
    model.f = Var(model.T, domain=NonNegativeReals)
    model.x = Var(model.K, domain=NonNegativeReals)

    # =========================
    # OBJECTIVE (maximize surplus)
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
            data["transport_costs"][(i, j)] * m.f[i, j]
            for (i, j) in m.T
        )

        tech_cost = sum(
            data["technology_costs"][k] * m.x[k]
            for k in m.K
        )

        return consumer_term - supplier_term - transport_cost - tech_cost

    model.obj = Objective(rule=obj_expr, sense=maximize)

    # =========================
    # CONSTRAINTS
    # =========================

    # --- Node-product balance ---
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
            if data["technology_nodes"][k] == n
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

    # --- Supplier capacity constraints ---
    model.supplier_capacity = ConstraintList()

    for b in model.B:
        if data["bids"][b]["type"] == "supplier":
            cap = data["bids"][b].get("capacity", None)
            if cap is not None:
                model.supplier_capacity.add(model.q[b] <= cap)

    return model