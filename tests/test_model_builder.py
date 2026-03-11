import sys
from pathlib import Path

# ensure src importable
sys.path.insert(0, str(Path.cwd()))

from src.schema import (
    ProblemState,
    Node,
    Product,
    Supplier,
    Consumer,
    TransportLink,
    Technology,
    Bid,
)
from src.model_builder import build_model_from_state
from pyomo.environ import value


def make_simple_state():
    state = ProblemState()
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p", capacity=10))
    state.add_consumer(Consumer(id="c1", node="n1", product="p", capacity=5))
    state.add_bid(Bid(id="b1", owner_id="s1", owner_type="supplier", product_id="p", price=1.0, quantity=4))
    state.add_bid(Bid(id="b2", owner_id="c1", owner_type="consumer", product_id="p", price=2.0, quantity=4))
    return state


def test_build_simple_model():
    state = make_simple_state()
    model = build_model_from_state(state)
    assert "q" in model.component_map()
    assert len(list(model.NODES)) == 1
    assert len(list(model.PRODUCTS)) == 1
    assert len(list(model.BIDS)) == 2


def test_capacity_constraint_exists():
    state = make_simple_state()
    model = build_model_from_state(state)
    # there should be at least one constraint for supplier capacity
    cons = [c for c in model.component_data_objects() if c.name.startswith("supplier_capacity")]
    assert cons


def test_node_balance_valid():
    state = make_simple_state()
    m = build_model_from_state(state)
    # evaluate balance expression at zero
    for n in m.NODES:
        for p in m.PRODUCTS:
            # evaluate expression without forcing var values (allow None)
            expr = m.node_product_balance[n, p].expr
            val = value(expr, exception=False)
            # when all variables are uninitialized we expect None or 0
            assert val in (None, 0)
