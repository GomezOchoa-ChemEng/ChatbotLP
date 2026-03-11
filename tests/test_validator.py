import sys
from pathlib import Path

# ensure imports work
sys.path.insert(0, str(Path.cwd()))

from src.schema import ProblemState, Node, Product, Supplier, Consumer, Technology, Bid
from src.validator import validate_state


def make_case_a():
    state = ProblemState()
    state.add_node(Node(id="n"))
    state.add_product(Product(id="p"))
    state.add_supplier(Supplier(id="s", node="n", product="p", capacity=10))
    state.add_consumer(Consumer(id="c", node="n", product="p", capacity=5))
    state.add_bid(Bid(id="b1", owner_id="s", owner_type="supplier", product_id="p", price=1.0, quantity=5))
    state.add_bid(Bid(id="b2", owner_id="c", owner_type="consumer", product_id="p", price=2.0, quantity=5))
    return state


def make_case_b():
    state = make_case_a()
    # add negative bid
    state.add_bid(Bid(id="b_neg", owner_id="s", owner_type="supplier", product_id="p", price=-1.0, quantity=2))
    return state


def make_case_c():
    state = make_case_a()
    # add second product for output
    state.add_product(Product(id="p2"))
    state.add_technology(Technology(
        id="t1",
        node="n",
        capacity=10,
        yield_coefficients={"p": -1.0, "p2": 0.8},
    ))
    state.add_bid(Bid(id="b_tech_in", owner_id="t1", owner_type="technology", product_id="p", price=-0.5, quantity=3))
    state.add_bid(Bid(id="b_tech_out", owner_id="t1", owner_type="technology", product_id="p2", price=0.5, quantity=3))
    state.add_bid(Bid(id="b_cons2", owner_id="c", owner_type="consumer", product_id="p2", price=3.0, quantity=3))
    return state


def test_case_a_compat():
    s = make_case_a()
    diag = validate_state(s)
    assert diag["benchmark_compatibility"]["Case A"]["compatible"]
    assert not diag["benchmark_compatibility"]["Case B"]["compatible"]
    assert not diag["benchmark_compatibility"]["Case C"]["compatible"]
    assert diag["solver_ready"]


def test_case_b_compat():
    s = make_case_b()
    diag = validate_state(s)
    assert diag["benchmark_compatibility"]["Case B"]["compatible"]


def test_case_c_compat():
    s = make_case_c()
    diag = validate_state(s)
    assert diag["benchmark_compatibility"]["Case C"]["compatible"]
    # also solver ready should be true for complete case
    assert diag["solver_ready"]


def test_invalid_references():
    s = ProblemState()
    s.add_node(Node(id="n"))
    s.add_product(Product(id="p"))
    s.add_supplier(Supplier(id="s", node="n", product="p"))
    # add bid with wrong product
    s.add_bid(Bid(id="bad", owner_id="s", owner_type="supplier", product_id="x", price=1.0))
    diag = validate_state(s)
    assert diag["invalid_references"]
