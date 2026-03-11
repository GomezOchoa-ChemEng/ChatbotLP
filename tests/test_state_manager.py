import sys
from pathlib import Path
import json

def _add_minimal_entities(state):
    from src.schema import Node, Product, Supplier, Consumer, Bid
    state.add_node(Node(id="n"))
    state.add_product(Product(id="p"))
    state.add_supplier(Supplier(id="s", node="n", product="p", capacity=10))
    state.add_consumer(Consumer(id="c", node="n", product="p", capacity=5))
    state.add_bid(Bid(id="b", owner_id="s", owner_type="supplier", product_id="p", price=1.0))

# ensure src importable
sys.path.insert(0, str(Path.cwd()))

from src.state_manager import StateManager
from src.schema import ProblemState


def test_load_save_roundtrip(tmp_path):
    mgr = StateManager()
    _add_minimal_entities(mgr.state)
    path = tmp_path / "state.json"
    mgr.save_to_file(str(path))
    assert path.exists()

    mgr2 = StateManager()
    mgr2.load_from_file(str(path))
    assert mgr2.state.problem_title == mgr.state.problem_title
    assert mgr2.state.node_ids() == mgr.state.node_ids()


def test_validate_fills_missing():
    mgr = StateManager()
    # start with empty state, should indicate missing nodes/products
    missing = mgr.validate()
    assert "no nodes defined" in missing
    assert "no products defined" in missing

    # add minimal structure and revalidate
    _add_minimal_entities(mgr.state)
    missing2 = mgr.validate()
    assert missing2 == []
