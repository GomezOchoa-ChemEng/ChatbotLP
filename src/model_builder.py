"""Build a Pyomo model from a ProblemState.

This module constructs a Pyomo `ConcreteModel` with sets, parameters,
variables, objective, and constraints suitable for the benchmark cases
described in the supporting information (Case A/B/C). It does NOT run a
solver; it only returns a built model ready to be solved by a Pyomo
solver.

Design notes:
- Bids are modeled at the bid-level with decision variable `q[b]`.
- Owner capacities (suppliers, consumers, transport links, technologies)
  are enforced by summing associated bid quantities.
- Technologies use a single activity level equal to the sum of their
  associated bid quantities; yield coefficients convert activity into
  product flows.
- Node-product material balances ensure conservation across suppliers,
  consumers, transports and technologies.
"""

from typing import Dict, List, Tuple

from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Var,
    NonNegativeReals,
    Reals,
    Objective,
    Constraint,
    summation,
    value,
)

from .schema import ProblemState


def build_model_from_state(state: ProblemState) -> ConcreteModel:
    """Construct and return a Pyomo ConcreteModel for the given ProblemState.

    The returned model contains the following components (naming shown):
    - model.NODES, model.PRODUCTS, model.BIDS, model.SUPPLIERS, model.CONSUMERS,
      model.TRANSPORTS, model.TECHNOLOGIES
    - model.bid_price[b], model.bid_quantity_upper[b]
    - model.q[b]: decision variable for bid quantity
    - capacity constraints per owner where capacities are provided
    - model.node_product_balance[n,p]: conservation constraints
    - model.obj: objective to maximize social surplus

    The function assumes `state` has been validated by `validate_state`.
    """

    model = ConcreteModel()

    # Sets
    model.NODES = Set(initialize=[n.id for n in state.nodes])
    model.PRODUCTS = Set(initialize=[p.id for p in state.products])
    model.SUPPLIERS = Set(initialize=[s.id for s in state.suppliers])
    model.CONSUMERS = Set(initialize=[c.id for c in state.consumers])
    model.TRANSPORTS = Set(initialize=[t.id for t in state.transport_links])
    model.TECHNOLOGIES = Set(initialize=[t.id for t in state.technologies])

    # Bids
    bid_ids = [b.id for b in state.bids]
    model.BIDS = Set(initialize=bid_ids)

    # prepare Python dictionaries for bid data (no Pyomo Params required)
    owner_map = {b.id: b.owner_id for b in state.bids}
    owner_type_map = {b.id: b.owner_type for b in state.bids}
    product_map = {b.id: b.product_id for b in state.bids}
    price_map = {b.id: b.price for b in state.bids}
    bid_qcap_dict = {b.id: (b.quantity if b.quantity is not None else None) for b in state.bids}

    # Variables: bid quantities
    def q_bounds(model, b):
        cap = bid_qcap_dict.get(b)
        if cap is None:
            return (0.0, None)
        return (0.0, float(cap))

    model.q = Var(model.BIDS, domain=NonNegativeReals, bounds=q_bounds)

    # Owner capacity constraints: sum of bids for owner <= capacity (if provided)
    # Build mappings of bids per owner
    bids_by_supplier: Dict[str, List[str]] = {}
    bids_by_consumer: Dict[str, List[str]] = {}
    bids_by_transport: Dict[str, List[str]] = {}
    bids_by_technology: Dict[str, List[str]] = {}

    for b in state.bids:
        if b.owner_type == "supplier":
            bids_by_supplier.setdefault(b.owner_id, []).append(b.id)
        elif b.owner_type == "consumer":
            bids_by_consumer.setdefault(b.owner_id, []).append(b.id)
        elif b.owner_type == "transport":
            bids_by_transport.setdefault(b.owner_id, []).append(b.id)
        elif b.owner_type == "technology":
            bids_by_technology.setdefault(b.owner_id, []).append(b.id)

    # Add capacity constraints for suppliers
    model.supplier_capacity = ConstraintList = None
    from pyomo.environ import ConstraintList
    model.supplier_capacity = ConstraintList()
    supplier_cap_map = {s.id: s.capacity for s in state.suppliers}
    for sid, bids in bids_by_supplier.items():
        cap = supplier_cap_map.get(sid)
        if cap is None:
            continue
        model.supplier_capacity.add(sum(model.q[b] for b in bids) <= float(cap))

    # Consumers
    model.consumer_capacity = ConstraintList()
    consumer_cap_map = {c.id: c.capacity for c in state.consumers}
    for cid, bids in bids_by_consumer.items():
        cap = consumer_cap_map.get(cid)
        if cap is None:
            continue
        model.consumer_capacity.add(sum(model.q[b] for b in bids) <= float(cap))

    # Transports
    model.transport_capacity = ConstraintList()
    transport_cap_map = {t.id: t.capacity for t in state.transport_links}
    for tid, bids in bids_by_transport.items():
        cap = transport_cap_map.get(tid)
        if cap is None:
            continue
        model.transport_capacity.add(sum(model.q[b] for b in bids) <= float(cap))

    # Technologies
    model.technology_capacity = ConstraintList()
    tech_cap_map = {t.id: t.capacity for t in state.technologies}
    for tid, bids in bids_by_technology.items():
        cap = tech_cap_map.get(tid)
        if cap is None:
            continue
        model.technology_capacity.add(sum(model.q[b] for b in bids) <= float(cap))

    # Technology net contribution per product: we will compute inside node-product balance
    # Build quick lookup maps for transports and technologies
    transport_map = {t.id: t for t in state.transport_links}
    tech_map = {t.id: t for t in state.technologies}

    # Objective: maximize social surplus
    # sign: consumer -> +1 (willingness to pay), others -> -1 (costs)
    sign_map = {"consumer": 1.0, "supplier": -1.0, "transport": -1.0, "technology": -1.0}

    def obj_expr(m):
        total = 0
        for b in m.BIDS:
            ot = owner_type_map.get(b, "supplier")
            price = price_map.get(b, 0.0)
            total += sign_map.get(ot, -1.0) * float(price) * m.q[b]
        return total

    model.obj = Objective(rule=obj_expr, sense=1)  # maximize

    # Node-product balance constraints
    def node_product_balance_rule(m, n, p):
        # supplies at node n for product p
        supply_sum = 0
        for sid, bids in bids_by_supplier.items():
            # find supplier node
            # locate supplier object
            # we can find supplier by id in state; create a small map
            pass

    # To implement node-product balances efficiently, precompute mappings
    supplier_map = {s.id: s for s in state.suppliers}
    consumer_map = {c.id: c for c in state.consumers}

    # Build index lists for quick aggregation
    bids_for_supplier_product: Dict[Tuple[str, str], List[str]] = {}
    for b in state.bids:
        if b.owner_type == "supplier":
            sup = supplier_map.get(b.owner_id)
            if sup is None:
                continue
            key = (sup.node, b.product_id)
            bids_for_supplier_product.setdefault(key, []).append(b.id)

    bids_for_consumer_product: Dict[Tuple[str, str], List[str]] = {}
    for b in state.bids:
        if b.owner_type == "consumer":
            cons = consumer_map.get(b.owner_id)
            if cons is None:
                continue
            key = (cons.node, b.product_id)
            bids_for_consumer_product.setdefault(key, []).append(b.id)

    bids_for_transport_product_origin: Dict[Tuple[str, str], List[str]] = {}
    bids_for_transport_product_destination: Dict[Tuple[str, str], List[str]] = {}
    for b in state.bids:
        if b.owner_type == "transport":
            t = transport_map.get(b.owner_id)
            if t is None:
                continue
            key_o = (t.origin, b.product_id)
            key_d = (t.destination, b.product_id)
            bids_for_transport_product_origin.setdefault(key_o, []).append(b.id)
            bids_for_transport_product_destination.setdefault(key_d, []).append(b.id)

    bids_for_tech_product_by_node: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    # Map technology bids to node and product with yield coefficient
    for b in state.bids:
        if b.owner_type == "technology":
            t = tech_map.get(b.owner_id)
            if t is None:
                continue
            for pid, coef in t.yield_coefficients.items():
                key = (t.node, pid)
                # store tuple (bid_id, coef)
                bids_for_tech_product_by_node.setdefault(key, []).append((b.id, coef))

    from pyomo.environ import Constraint

    def node_product_balance_rule(m, n, p):
        # supply
        sup_list = bids_for_supplier_product.get((n, p), [])
        supply_sum = sum(m.q[b] for b in sup_list)

        # transport in
        trans_in = bids_for_transport_product_destination.get((n, p), [])
        transport_in_sum = sum(m.q[b] for b in trans_in)

        # transport out
        trans_out = bids_for_transport_product_origin.get((n, p), [])
        transport_out_sum = sum(m.q[b] for b in trans_out)

        # consumer demand
        cons_list = bids_for_consumer_product.get((n, p), [])
        consume_sum = sum(m.q[b] for b in cons_list)

        # technology net contribution
        tech_entries = bids_for_tech_product_by_node.get((n, p), [])
        tech_net = sum(coef * m.q[bid_id] for (bid_id, coef) in tech_entries)

        return supply_sum + transport_in_sum + tech_net - consume_sum - transport_out_sum == 0

    model.node_product_balance = Constraint(model.NODES, model.PRODUCTS, rule=node_product_balance_rule)

    return model


__all__ = ["build_model_from_state"]
