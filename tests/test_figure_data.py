import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology, TransportLink
from src.solver import SolveResult
from src.solver_results import SolverResults


def make_case_state() -> ProblemState:
    state = ProblemState(problem_title="Figure Data Test")
    state.add_node(Node(id="N1"))
    state.add_node(Node(id="N2"))
    state.add_product(Product(id="P1"))
    state.add_product(Product(id="P2"))
    state.add_supplier(Supplier(id="S1", node="N1", product="P1", capacity=10.0))
    state.add_consumer(Consumer(id="C1", node="N2", product="P1", capacity=8.0))
    state.add_transport(TransportLink(id="T1", origin="N1", destination="N2", product="P1", capacity=10.0, cost=0.0))
    state.add_technology(Technology(id="K1", node="N2", capacity=4.0, cost=0.0, yield_coefficients={"P1": -1.0, "P2": 0.5}))
    state.add_bid(Bid(id="B_SUP", owner_id="S1", owner_type="supplier", product_id="P1", price=2.0, quantity=10.0))
    state.add_bid(Bid(id="B_CON", owner_id="C1", owner_type="consumer", product_id="P1", price=9.0, quantity=8.0))
    return state


def make_solver_results() -> SolverResults:
    state = make_case_state()
    solve_result = SolveResult(
        model=None,
        status="optimal",
        message="Solver glpk terminated with status optimal",
        objective_value=56.0,
        solver_time=0.01,
        solution={
            "q": {"B_SUP": 8.0, "B_CON": 8.0},
            "f": {"('N1', 'N2')": 8.0},
            "x": {"K1": 2.0},
        },
        success=True,
        termination_condition="optimal",
        solver_name="glpk",
    )
    solve_result.to_dict = lambda: {
        "status": "optimal",
        "message": "Solver glpk terminated with status optimal",
        "objective_value": 56.0,
        "solver_time": 0.01,
        "solution": {
            "q": {"B_SUP": 8.0, "B_CON": 8.0},
            "f": {"('N1', 'N2')": 8.0},
            "x": {"K1": 2.0},
        },
        "success": True,
        "termination_condition": "optimal",
        "solver_name": "glpk",
        "dual_values": {
            "node_balance[N1,P1]": -2.0,
            "node_balance[N2,P1]": -2.0,
        },
        "constraint_slacks": {
            "node_balance[N1,P1]": 0.0,
            "node_balance[N2,P1]": 0.0,
            "transport_capacity[1]": 2.0,
        },
        "scenario_metadata": {
            "run_id": "run_base",
            "scenario_id": "base",
            "parameter_name": "transport_capacity",
            "parameter_value": 10.0,
            "baseline_id": None,
            "comparison_metrics": {"objective_delta": 0.0},
        },
    }
    return SolverResults.from_solve_result(solve_result, state)


def test_plotting_data_contains_figure_tables():
    results = make_solver_results()

    assert "scenario_sweep_table" in results.plotting_data
    assert "utilization_table" in results.plotting_data
    assert "nodal_price_table" in results.plotting_data
    assert "bid_acceptance_table" in results.plotting_data
    assert "derived_metrics_table" in results.plotting_data


def test_utilization_table_covers_supplier_transport_and_technology():
    results = make_solver_results()
    rows = results.plotting_data["utilization_table"]

    asset_types = {row["asset_type"] for row in rows}
    assert asset_types == {"supplier", "transport", "technology"}

    supplier_row = next(row for row in rows if row["asset_type"] == "supplier")
    assert supplier_row["capacity"] == 10.0
    assert supplier_row["cleared_quantity"] == 8.0
    assert abs(supplier_row["utilization_fraction"] - 0.8) < 1e-9


def test_nodal_price_table_has_explicit_normalization():
    results = make_solver_results()
    rows = results.plotting_data["nodal_price_table"]

    assert rows
    assert rows[0]["raw_shadow_price"] == -2.0
    assert rows[0]["normalized_price"] == 2.0
    assert "normalized_price = - raw_shadow_price" in rows[0]["sign_convention_note"]


def test_bid_acceptance_table_has_acceptance_fraction():
    results = make_solver_results()
    rows = results.plotting_data["bid_acceptance_table"]

    supplier_bid = next(row for row in rows if row["bid_id"] == "B_SUP")
    consumer_bid = next(row for row in rows if row["bid_id"] == "B_CON")
    assert abs(supplier_bid["acceptance_fraction"] - 0.8) < 1e-9
    assert abs(consumer_bid["acceptance_fraction"] - 1.0) < 1e-9


def test_scenario_sweep_table_is_long_form():
    results = make_solver_results()
    rows = results.plotting_data["scenario_sweep_table"]

    assert rows
    first = rows[0]
    assert set(
        ["run_id", "scenario_id", "parameter_name", "parameter_value", "baseline_id", "product", "node", "metric_name", "metric_value"]
    ).issubset(first.keys())


def test_derived_metrics_table_contains_core_metrics():
    results = make_solver_results()
    rows = results.plotting_data["derived_metrics_table"]
    metric_names = {row["metric_name"] for row in rows}

    assert "total_welfare" in metric_names
    assert "total_cleared_quantity" in metric_names
    assert "total_transport_flow" in metric_names
    assert "max_constraint_slack" in metric_names
    assert "objective_delta" in metric_names
