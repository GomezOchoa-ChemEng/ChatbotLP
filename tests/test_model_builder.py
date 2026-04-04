import sys
from pathlib import Path

from pyomo.environ import value

# ensure src importable
sys.path.insert(0, str(Path.cwd()))

from src.model_builder import _build_data_from_state, build_model_from_state
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology


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
    assert data["technology_costs"]["tech1"] == 0.0
    assert data["technology_yields"][("tech1", "P1")] == -1.0
    assert data["technology_yields"][("tech1", "P2")] == 0.8


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
