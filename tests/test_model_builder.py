import sys
from pathlib import Path

import pytest
from pyomo.environ import value

# ensure src importable
sys.path.insert(0, str(Path.cwd()))

from src.model_builder import (
    _build_data_from_state,
    build_market_instance,
    build_model_from_market_instance,
    build_model_from_state,
)
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology, TransportLink


def make_simple_state():
    state = ProblemState()
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p", capacity=10))
    state.add_consumer(Consumer(id="c1", node="n1", product="p", capacity=5))
    state.add_bid(
        Bid(
            id="b1",
            owner_id="s1",
            owner_type="supplier",
            product_id="p",
            price=1.0,
            quantity=4,
        )
    )
    state.add_bid(
        Bid(
            id="b2",
            owner_id="c1",
            owner_type="consumer",
            product_id="p",
            price=2.0,
            quantity=4,
        )
    )
    return state


def make_transformation_state():
    state = ProblemState(problem_title="Case C smoke test")
    state.add_node(Node(id="plant"))
    state.add_product(Product(id="P1"))
    state.add_product(Product(id="P2"))
    state.add_supplier(Supplier(id="raw_supplier", node="plant", product="P1", capacity=10))
    state.add_consumer(Consumer(id="processed_consumer", node="plant", product="P2", capacity=10))
    state.add_technology(
        Technology(
            id="tech1",
            node="plant",
            capacity=3.0,
            cost=0.5,
            yield_coefficients={"P1": -1.0, "P2": 0.8},
        )
    )
    state.add_bid(
        Bid(
            id="raw_bid",
            owner_id="raw_supplier",
            owner_type="supplier",
            product_id="P1",
            price=1.0,
            quantity=3.0,
        )
    )
    state.add_bid(
        Bid(
            id="processed_bid",
            owner_id="processed_consumer",
            owner_type="consumer",
            product_id="P2",
            price=5.0,
            quantity=2.4,
        )
    )
    return state


def test_build_simple_model():
    state = make_simple_state()
    model = build_model_from_state(state)
    assert "q" in model.component_map()
    assert len(list(model.N)) == 1
    assert len(list(model.P)) == 1
    assert len(list(model.B)) == 2


def test_capacity_constraint_exists():
    state = make_simple_state()
    model = build_model_from_state(state)
    cons = [c for c in model.component_data_objects() if c.name.startswith("supplier_capacity")]
    assert cons


def test_node_balance_valid():
    state = make_simple_state()
    model = build_model_from_state(state)
    for n in model.N:
        for p in model.P:
            expr = model.node_balance[n, p].expr
            val = value(expr, exception=False)
            assert val in (None, 0)


def test_build_data_from_state_includes_technology_fields():
    state = make_transformation_state()
    data = _build_data_from_state(state)

    assert data["technologies"] == ["tech1"]
    assert data["technology_nodes"]["tech1"] == "plant"
    assert data["technology_capacities"]["tech1"] == 3.0
    assert data["technology_costs"]["tech1"] == 0.5
    assert data["technology_yields"][("tech1", "P1")] == -1.0
    assert data["technology_yields"][("tech1", "P2")] == 0.8


def test_generic_technology_cost_is_not_tied_to_q4_names():
    state = ProblemState(problem_title="Generic smelting transformation")
    state.add_node(Node(id="mine"))
    state.add_node(Node(id="mill"))
    state.add_node(Node(id="factory"))
    state.add_product(Product(id="Ore"))
    state.add_product(Product(id="Ingot"))
    state.add_supplier(Supplier(id="ore_supplier", node="mine", product="Ore", capacity=40.0))
    state.add_consumer(Consumer(id="ingot_buyer", node="factory", product="Ingot", capacity=12.0))
    state.add_technology(
        Technology(
            id="smelter",
            node="mill",
            capacity=20.0,
            cost=7.25,
            yield_coefficients={"Ore": -2.0, "Ingot": 0.6},
        )
    )
    state.add_bid(
        Bid(
            id="ore_bid",
            owner_id="ore_supplier",
            owner_type="supplier",
            product_id="Ore",
            price=3.0,
            quantity=40.0,
        )
    )
    state.add_bid(
        Bid(
            id="ingot_bid",
            owner_id="ingot_buyer",
            owner_type="consumer",
            product_id="Ingot",
            price=50.0,
            quantity=12.0,
        )
    )

    data = _build_data_from_state(state)
    instance = build_market_instance(state)
    model = build_model_from_state(state)

    assert data["technologies"] == ["smelter"]
    assert data["technology_capacities"]["smelter"] == 20.0
    assert data["technology_costs"]["smelter"] == 7.25
    assert data["technology_yields"][("smelter", "Ore")] == -2.0
    assert data["technology_yields"][("smelter", "Ingot")] == 0.6
    assert instance.technologies[0].cost == 7.25
    assert instance.technologies[0].yield_coefficients == {"Ore": -2.0, "Ingot": 0.6}
    assert "smelter" in model.K


def test_build_market_instance_creates_lightweight_machine_object():
    state = make_transformation_state()
    instance = build_market_instance(state)

    assert instance.problem_title == state.problem_title
    assert instance.nodes == ["plant"]
    assert instance.products == ["P1", "P2"]
    assert len(instance.bids) == 2
    assert len(instance.technologies) == 1
    assert instance.technologies[0].cost == 0.5
    assert instance.technologies[0].yield_coefficients["P2"] == 0.8


def test_build_model_from_market_instance_matches_state_path():
    state = make_simple_state()
    instance = build_market_instance(state)

    model_from_state = build_model_from_state(state)
    model_from_instance = build_model_from_market_instance(instance)

    assert list(model_from_state.B) == list(model_from_instance.B)
    assert list(model_from_state.N) == list(model_from_instance.N)
    assert list(model_from_state.P) == list(model_from_instance.P)


def test_transformation_model_includes_technology_components_and_capacity():
    state = make_transformation_state()
    model = build_model_from_state(state)

    assert "tech1" in model.K
    assert ("tech1" in model.x) is True
    assert len(model.technology_capacity) == 1

    model.x["tech1"].set_value(3.0)
    cap_expr = model.technology_capacity[1].expr
    assert value(cap_expr, exception=False) is True


def test_transformation_yields_appear_in_node_balance_structure():
    state = make_transformation_state()
    model = build_model_from_state(state)

    model.q["raw_bid"].set_value(3.0)
    model.q["processed_bid"].set_value(2.4)
    model.x["tech1"].set_value(3.0)

    p1_balance = value(model.node_balance["plant", "P1"].body)
    p2_balance = value(model.node_balance["plant", "P2"].body)

    assert abs(p1_balance) < 1e-9
    assert abs(p2_balance) < 1e-9


def test_model_builder_rejects_missing_route_cost_instead_of_defaulting_to_zero():
    state = make_simple_state()
    state.add_transport(
        TransportLink(id="t1", origin="n1", destination="n1", product="p", capacity=10.0)
    )

    instance = build_market_instance(state)

    assert instance.transport_links[0].cost is None
    with pytest.raises(ValueError, match="transport:t1 missing cost"):
        build_model_from_state(state)
    with pytest.raises(ValueError, match="transport:t1 missing cost"):
        build_model_from_market_instance(instance)


def test_model_builder_accepts_explicit_zero_route_and_technology_costs():
    state = make_simple_state()
    state.add_transport(
        TransportLink(id="t1", origin="n1", destination="n1", product="p", capacity=10.0, cost=0.0)
    )

    model = build_model_from_state(state)

    assert "t1" in build_market_instance(state).metadata["transport_ids"]
    assert len(model.transport_capacity) == 1

    technology_state = make_transformation_state()
    technology_state.technologies[0].cost = 0.0

    assert build_market_instance(technology_state).technologies[0].cost == 0.0
    assert "tech1" in build_model_from_state(technology_state).K
