import sys
from pathlib import Path

# ensure src is importable
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


def test_basic_entities_and_ids():
    state = ProblemState()
    state.add_node(Node(id="n1", name="Node 1"))
    state.add_node(Node(id="n2"))
    state.add_product(Product(id="p1", name="Prod 1"))
    state.add_product(Product(id="p2"))

    assert state.node_ids() == ["n1", "n2"]
    assert state.product_ids() == ["p1", "p2"]

    # suppliers and consumers
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=10))
    state.add_consumer(Consumer(id="c1", node="n2", product="p2", capacity=5))
    assert [s.id for s in state.suppliers] == ["s1"]
    assert [c.id for c in state.consumers] == ["c1"]

    # duplicates are ignored
    state.add_node(Node(id="n1"))
    assert state.node_ids() == ["n1", "n2"]


def test_bid_model():
    state = ProblemState()
    state.add_node(Node(id="n"))
    state.add_product(Product(id="p"))
    state.add_supplier(Supplier(id="s", node="n", product="p"))
    b = Bid(id="b1", owner_id="s", owner_type="supplier", product_id="p", price=-3.5)
    state.add_bid(b)
    assert state.bids[0].price == -3.5
    assert state.bids[0].owner_type == "supplier"
