import sys
from pathlib import Path

# ensure workspace import path
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
from src.scenario_engine import (
    clone_state,
    apply_parameter_change,
    extract_scenario_request,
    run_scenario,
    summarize_scenario_results,
)
from src.solver import SolveResult


def make_simple_state():
    s = ProblemState()
    s.add_node(Node(id="n"))
    s.add_product(Product(id="p"))
    s.add_supplier(Supplier(id="sup", node="n", product="p", capacity=10))
    s.add_consumer(Consumer(id="con", node="n", product="p", capacity=5))
    s.add_bid(
        Bid(
            id="b1",
            owner_id="sup",
            owner_type="supplier",
            product_id="p",
            price=1.0,
            quantity=5,
        )
    )
    s.add_bid(
        Bid(
            id="b2",
            owner_id="con",
            owner_type="consumer",
            product_id="p",
            price=2.0,
            quantity=5,
        )
    )
    return s


def make_case_c_state():
    s = make_simple_state()
    s.add_product(Product(id="p2"))
    s.add_transport(TransportLink(id="t12", origin="n", destination="n", product="p", capacity=100))
    s.add_technology(
        Technology(
            id="tech1",
            node="n",
            capacity=100,
            yield_coefficients={"p": -1.0, "p2": 0.8},
        )
    )
    return s


def test_clone_independence():
    base = make_simple_state()
    clone = clone_state(base)
    assert clone is not base
    # modify clone should not affect base
    clone.suppliers[0].capacity = 999
    assert base.suppliers[0].capacity == 10


def test_apply_parameter_change_basic():
    s = make_simple_state()
    change = {"suppliers": [{"id": "sup", "capacity": 20}], "bids": [{"id": "b1", "price": 5.0}]}
    apply_parameter_change(s, change)
    assert s.suppliers[0].capacity == 20
    assert any(b.price == 5.0 for b in s.bids if b.id == "b1")
    assert any(b.quantity == 20 for b in s.bids if b.id == "b1")


def test_extract_scenario_request_for_supplier_capacity_increase():
    extraction = extract_scenario_request(
        make_simple_state(),
        "What happens if supplier capacity increases from 100 to 150?",
    )
    assert extraction["parameter_type"] == "supplier_capacity"
    assert extraction["target_object_id"] == "sup"
    assert extraction["old_value"] == 100
    assert extraction["new_value"] == 150
    assert "flows" in extraction["requested_dimensions"] or "objective" in extraction["requested_dimensions"]


def test_extract_scenario_request_for_transport_price_question():
    extraction = extract_scenario_request(
        make_case_c_state(),
        "What happens to the dual prices if transport capacity decreases from 100 to 30?",
    )
    assert extraction["parameter_type"] == "transport_capacity"
    assert extraction["target_object_id"] == "t12"
    assert extraction["new_value"] == 30
    assert "prices" in extraction["requested_dimensions"]


def test_extract_scenario_request_for_disabling_technology():
    extraction = extract_scenario_request(
        make_case_c_state(),
        "How would the optimal flows and prices change if the transformation technology were unavailable?",
    )
    assert extraction["parameter_type"] == "technology_availability"
    assert extraction["target_object_id"] == "tech1"
    assert extraction["change_spec"]["technologies"][0]["capacity"] == 0.0


def test_run_scenario_without_solver_records_history():
    base = make_simple_state()
    assert len(base.scenario_history) == 0
    change = {"name": "increase cap", "suppliers": [{"id": "sup", "capacity": 20}]}  # simple change
    res = run_scenario(base, change, solve=False)
    # history should have new record
    assert len(base.scenario_history) == 1
    rec = base.scenario_history[0]
    assert rec.name == "increase cap"
    assert "suppliers" in rec.description
    # results should have None for base and scenario when solve=False
    assert res["base"] is None
    assert res["scenario"] is None
    assert res["difference"] == {}
    # scenario state should reflect change
    scen = res["scenario_state"]
    assert scen.suppliers[0].capacity == 20
    # original base unchanged
    assert base.suppliers[0].capacity == 10


def test_run_scenario_with_mock_solver(monkeypatch):
    base = make_simple_state()
    # prepare two fake results with different objective
    fake1 = SolveResult(model=None, status="opt", message="", objective_value=10, solver_time=0, solution={}, success=True)
    fake2 = SolveResult(model=None, status="opt", message="", objective_value=25, solver_time=0, solution={}, success=True)
    calls = []

    def fake_solve(model, **kwargs):
        calls.append(model)
        return fake1 if len(calls) == 1 else fake2

    monkeypatch.setattr("src.scenario_engine.solve_model", fake_solve)
    change = {"bids": [{"id": "b1", "price": 2.0}], "name": "price up"}
    res = run_scenario(base, change, solve=True)
    assert isinstance(res["base"], SolveResult)
    assert isinstance(res["scenario"], SolveResult)
    assert res["difference"]["objective_delta"] == 15
    # ensure two solver calls were made
    assert len(calls) == 2


def test_summarize_scenario_results_mentions_baseline_and_modified():
    extraction = extract_scenario_request(
        make_simple_state(),
        "If the consumer bid drops from 20 to 15, how do the primal and dual solutions change?",
    )
    base = SolveResult(
        model=None,
        status="optimal",
        message="",
        objective_value=20.0,
        solver_time=0.0,
        solution={"q": {"b1": 5.0, "b2": 5.0}, "f": {}, "x": {}},
        success=True,
    )
    modified = SolveResult(
        model=None,
        status="optimal",
        message="",
        objective_value=15.0,
        solver_time=0.0,
        solution={"q": {"b1": 4.0, "b2": 4.0}, "f": {}, "x": {}},
        success=True,
    )
    results = {
        "base": base,
        "scenario": modified,
        "difference": {
            "objective_delta": -5.0,
            "accepted_bid_changes": {"b2": {"before": 5.0, "after": 4.0, "delta": -1.0}},
            "flow_changes": {},
            "technology_activity_changes": {},
            "price_changes": {},
            "binding_constraint_changes": {},
            "unchanged_dimensions": ["flow_changes", "technology_activity_changes", "price_changes"],
        },
    }

    summary = summarize_scenario_results(extraction, results)
    assert "Baseline" in summary
    assert "Modified scenario" in summary
    assert "Changes in objective" in summary
    assert "What did not change" in summary
