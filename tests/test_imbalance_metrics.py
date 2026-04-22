import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.llm_problem_interpreter import build_state_from_semantic_plan
from src.solver import SolveResult
from src.solver_results import SolverResults


def build_case_state():
    return build_state_from_semantic_plan(
        {
            "problem_title": "Imbalance Test",
            "nodes": [{"id": "N1"}, {"id": "N2"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 10.0}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 8.0}],
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 10.0}],
            "bids": [
                {"id": "B_SUP", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 2.0, "quantity": 10.0},
                {"id": "B_CON", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 9.0, "quantity": 8.0},
            ],
            "technologies": [],
        }
    )


def make_results(flow: float, scenario_id: str, baseline_results=None):
    state = build_case_state()
    quantity = flow
    solve_result = SolveResult(
        model=None,
        status="optimal",
        message="Solver glpk terminated with status optimal",
        objective_value=7.0 * quantity,
        solver_time=0.01,
        solution={"q": {"B_SUP": quantity, "B_CON": quantity}, "f": {"('N1', 'N2')": flow}, "x": {}},
        success=True,
        termination_condition="optimal",
        solver_name="glpk",
    )
    solve_result.to_dict = lambda: {
        "status": "optimal",
        "message": "Solver glpk terminated with status optimal",
        "objective_value": 7.0 * quantity,
        "solver_time": 0.01,
        "solution": {"q": {"B_SUP": quantity, "B_CON": quantity}, "f": {"('N1', 'N2')": flow}, "x": {}},
        "success": True,
        "termination_condition": "optimal",
        "solver_name": "glpk",
        "dual_values": {"node_balance[N1,P1]": -2.0, "node_balance[N2,P1]": -2.0},
        "constraint_slacks": {"node_balance[N1,P1]": 0.0, "node_balance[N2,P1]": 0.0},
        "scenario_metadata": {
            "run_id": f"run_{scenario_id}",
            "scenario_id": scenario_id,
            "parameter_name": "transport_capacity",
            "parameter_value": flow,
            "baseline_id": "baseline",
            "region_map": {"N1": "upstream", "N2": "downstream"},
            "baseline_results": baseline_results,
        },
    }
    return SolverResults.from_solve_result(solve_result, state)


def test_imbalance_metrics_table_contains_node_region_and_comparison_rows():
    baseline = make_results(8.0, "baseline")
    scenario = make_results(4.0, "scenario", baseline_results=baseline)

    rows = scenario.plotting_data["imbalance_metrics_table"]
    metric_names = {row["metric_name"] for row in rows}

    assert "node_product_net_surplus" in metric_names
    assert "node_product_local_surplus" in metric_names
    assert "regional_local_surplus" in metric_names
    assert "node_product_local_surplus_delta" in metric_names


def test_imbalance_metrics_capture_local_surplus_and_zero_residual():
    scenario = make_results(4.0, "scenario")
    rows = scenario.plotting_data["imbalance_metrics_table"]

    n1_local = next(row for row in rows if row["metric_name"] == "node_product_local_surplus" and row["node"] == "N1")
    n2_local = next(row for row in rows if row["metric_name"] == "node_product_local_surplus" and row["node"] == "N2")
    n1_residual = next(row for row in rows if row["metric_name"] == "node_product_net_surplus" and row["node"] == "N1")

    assert n1_local["metric_value"] == 4.0
    assert n2_local["metric_value"] == -4.0
    assert n1_residual["metric_value"] == 0.0
    assert "Positive imbalance means net surplus" in n1_local["sign_convention_note"]


def test_imbalance_metrics_include_normalized_ratios_and_baseline_deltas():
    baseline = make_results(8.0, "baseline")
    scenario = make_results(4.0, "scenario", baseline_results=baseline)
    rows = scenario.plotting_data["imbalance_metrics_table"]

    n1_ratio = next(
        row
        for row in rows
        if row["metric_name"] == "node_product_local_normalized_imbalance_ratio" and row["node"] == "N1"
    )
    n1_delta = next(
        row
        for row in rows
        if row["metric_name"] == "node_product_local_surplus_delta" and row["node"] == "N1"
    )

    assert n1_ratio["metric_value"] == 1.0
    assert n1_delta["metric_value"] == -4.0


def test_derived_metrics_table_includes_imbalance_rows():
    scenario = make_results(4.0, "scenario")
    rows = scenario.plotting_data["derived_metrics_table"]
    metric_names = {row["metric_name"] for row in rows}

    assert "node_product_local_surplus" in metric_names
    assert "regional_local_surplus" in metric_names
