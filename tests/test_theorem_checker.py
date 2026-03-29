import sys
from pathlib import Path

# ensure imports from workspace work correctly
sys.path.insert(0, str(Path.cwd()))

from src.schema import (
    ProblemState,
    Node,
    Product,
    Supplier,
    Consumer,
    Technology,
    Bid,
)
from src.theorem_checker import check_theorems


def make_case_a():
    state = ProblemState()
    state.add_node(Node(id="n"))
    state.add_product(Product(id="p"))
    state.add_supplier(Supplier(id="s", node="n", product="p", capacity=10))
    state.add_consumer(Consumer(id="c", node="n", product="p", capacity=5))
    state.add_bid(
        Bid(
            id="b1",
            owner_id="s",
            owner_type="supplier",
            product_id="p",
            price=1.0,
            quantity=5,
        )
    )
    state.add_bid(
        Bid(
            id="b2",
            owner_id="c",
            owner_type="consumer",
            product_id="p",
            price=2.0,
            quantity=5,
        )
    )
    return state


def make_case_b():
    state = make_case_a()
    state.add_bid(
        Bid(
            id="b_neg",
            owner_id="s",
            owner_type="supplier",
            product_id="p",
            price=-1.0,
            quantity=2,
        )
    )
    return state


def make_case_c():
    state = make_case_a()
    state.add_product(Product(id="p2"))
    state.add_technology(
        Technology(
            id="t1",
            node="n",
            capacity=10,
            yield_coefficients={"p": -1.0, "p2": 0.8},
        )
    )
    # add bids for the technology
    state.add_bid(
        Bid(
            id="b_tech_in",
            owner_id="t1",
            owner_type="technology",
            product_id="p",
            price=-0.5,
            quantity=3,
        )
    )
    state.add_bid(
        Bid(
            id="b_tech_out",
            owner_id="t1",
            owner_type="technology",
            product_id="p2",
            price=0.5,
            quantity=3,
        )
    )
    state.add_bid(
        Bid(
            id="b_cons2",
            owner_id="c",
            owner_type="consumer",
            product_id="p2",
            price=3.0,
            quantity=3,
        )
    )
    return state


def test_empty_state():
    s = ProblemState()
    checks = check_theorems(s)
    # three case checks plus structure
    assert len(checks) >= 4
    struct = next(c for c in checks if "Basic supply-demand" in c.theorem_name)
    assert not struct.applies
    # with an empty state, Case A vacuously holds (no technologies),
    # while B and C cannot hold
    assert any("Case A" in c.theorem_name and c.applies for c in checks)
    assert any("Case B" in c.theorem_name and not c.applies for c in checks)
    assert any("Case C" in c.theorem_name and not c.applies for c in checks)


def test_case_a_theorems():
    s = make_case_a()
    checks = check_theorems(s)
    # ensure state updated
    assert s.theorem_checks == checks
    assert any(c.theorem_name.startswith("Case A") and c.applies for c in checks)
    assert any(c.theorem_name.startswith("Case B") and not c.applies for c in checks)
    assert any(c.theorem_name.startswith("Case C") and not c.applies for c in checks)
    theorem_1 = next(c for c in checks if c.theorem_id == "theorem_1")
    assert theorem_1.applies is True
    assert "validated_linear_problem_state" in theorem_1.assumptions_verified


def test_case_b_theorems():
    s = make_case_b()
    checks = check_theorems(s)
    assert any(c.theorem_name.startswith("Case B") and c.applies for c in checks)
    assert any(c.theorem_name.startswith("Case A") and c.applies for c in checks)
    assert any(c.theorem_name.startswith("Case C") and not c.applies for c in checks)


def test_case_c_theorems():
    s = make_case_c()
    checks = check_theorems(s)
    assert any(c.theorem_name.startswith("Case C") and c.applies for c in checks)
    assert any(c.theorem_name.startswith("Case A") and not c.applies for c in checks)


def test_structure_warning():
    s = ProblemState()
    s.add_supplier(Supplier(id="s", node="n", product="p"))
    checks = check_theorems(s)
    struct = next(c for c in checks if "Basic supply-demand" in c.theorem_name)
    assert not struct.applies
    assert "no consumers" in struct.explanation.lower()
    theorem_1 = next(c for c in checks if c.theorem_id == "theorem_1")
    assert theorem_1.applies is False
    assert "basic_supply_demand_structure" in theorem_1.assumptions_missing
