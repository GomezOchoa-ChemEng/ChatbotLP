import sys
from pathlib import Path

# ensure imports work
sys.path.insert(0, str(Path.cwd()))

from src.schema import ProblemState, Node, Product, Supplier, Consumer, TransportLink, Technology, Bid
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
        cost=0.0,
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


def test_missing_route_cost_blocks_solver_readiness_but_explicit_zero_is_allowed():
    s = make_case_a()
    s.add_transport(TransportLink(id="t1", origin="n", destination="n", product="p", capacity=10))

    missing = validate_state(s)

    assert missing["solver_ready"] is False
    assert "transport:t1 missing cost" in missing["missing_parameters"]

    s.transport_links[0].cost = 0.0
    explicit_zero = validate_state(s)

    assert explicit_zero["solver_ready"] is True
    assert "transport:t1 missing cost" not in explicit_zero["missing_parameters"]


def test_missing_route_capacity_and_incomplete_yield_are_reported_precisely():
    s = make_case_a()
    s.add_product(Product(id="p2"))
    s.add_transport(TransportLink(id="t1", origin="n", destination="n", product="p", cost=0.0))
    s.add_technology(
        Technology(
            id="tech1",
            node="n",
            capacity=10,
            cost=0.0,
            yield_coefficients={"p": -1.0},
        )
    )

    missing = validate_state(s)

    assert missing["solver_ready"] is False
    assert "transport:t1 missing capacity" in missing["missing_parameters"]
    assert "technology:tech1 missing input/output yield" in missing["missing_parameters"]


def test_missing_technology_cost_and_bid_values_block_solver_readiness():
    s = make_case_a()
    s.add_product(Product(id="p2"))
    s.add_technology(
        Technology(
            id="t1",
            node="n",
            capacity=10,
            yield_coefficients={"p": -1.0, "p2": 0.8},
        )
    )
    s.add_bid(Bid(id="missing_bid", owner_id="s", owner_type="supplier", product_id="p"))

    missing = validate_state(s)

    assert missing["solver_ready"] is False
    assert "technology:t1 missing cost" in missing["missing_parameters"]
    assert "bid:missing_bid missing price" in missing["missing_parameters"]
    assert "bid:missing_bid missing quantity" in missing["missing_parameters"]

    s.technologies[0].cost = 0.0
    s.bids[-1].price = 0.0
    s.bids[-1].quantity = 0.0
    explicit_zero = validate_state(s)

    assert explicit_zero["solver_ready"] is True
