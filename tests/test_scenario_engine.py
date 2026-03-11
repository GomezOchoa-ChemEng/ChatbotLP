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
    Technology,
    Bid,
)
from src.scenario_engine import (
    clone_state,
    apply_parameter_change,
    run_scenario,
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
