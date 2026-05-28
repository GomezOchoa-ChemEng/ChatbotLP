import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path.cwd()))

from src.llm_problem_interpreter import build_state_from_semantic_plan
from src.midterm_benchmark import (
    DEFAULT_Q2_BENCHMARK_DIR,
    DEFAULT_Q3_BENCHMARK_DIR,
    DEFAULT_Q4_BENCHMARK_DIR,
    MidtermBenchmarkConfig,
    build_supplier_removal_incentive_diagnostics,
    build_midterm_manure_expected_plan,
    build_midterm_manure_q2_expected_plan,
    build_midterm_manure_q3_expected_plan,
    build_midterm_manure_q4_expected_plan,
    compare_problem_states_for_midterm,
    compare_solution_to_reference,
    evaluate_primary_semantic_metrics,
    extract_midterm_solution_components,
    load_benchmark_files,
    run_midterm_manure_q1_benchmark,
    run_midterm_manure_q3_benchmark,
    run_midterm_manure_q4_benchmark,
)


def test_midterm_benchmark_files_load_correctly():
    files = load_benchmark_files()

    assert "Manure Management" in files["problem_statement"]
    assert files["reference_solution"]["benchmark_id"] == "midterm1_manure_q1"
    assert {prompt["id"] for prompt in files["prompts"]} >= {
        "canonical",
        "paraphrased",
        "incomplete",
        "ambiguous",
    }


def test_reference_solution_has_expected_objective():
    files = load_benchmark_files()

    assert files["reference_solution"]["objective_value"] == 850.0


def test_q2_benchmark_files_load_with_negative_bid_reference():
    files = load_benchmark_files(DEFAULT_Q2_BENCHMARK_DIR)

    assert "DNR" in files["problem_statement"]
    assert files["reference_solution"]["benchmark_id"] == "midterm1_manure_q2"
    assert files["reference_solution"]["objective_value"] == 650.0
    assert files["reference_solution"]["expected_semantic_metrics"]["consumer_bid_prices"] == [-0.5, 1.5]


def test_q3_benchmark_files_load_with_supplier_payment_reference():
    files = load_benchmark_files(DEFAULT_Q3_BENCHMARK_DIR)

    assert "0.7 dollars per ton" in files["problem_statement"]
    assert files["reference_solution"]["benchmark_id"] == "midterm1_manure_q3"
    assert files["reference_solution"]["objective_value"] == 1050.0
    metrics = files["reference_solution"]["expected_semantic_metrics"]
    assert metrics["consumer_bid_prices"] == [-0.5, 1.5]
    assert metrics["supplier_bid_prices"] == [-0.7]
    assert metrics["supplier_removal_payment"] == 0.7


def test_q4_benchmark_files_load_with_compost_reference():
    files = load_benchmark_files(DEFAULT_Q4_BENCHMARK_DIR)

    assert "Composter" in files["problem_statement"]
    assert files["reference_solution"]["benchmark_id"] == "midterm1_manure_q4"
    assert files["reference_solution"]["objective_value"] == 5800.0
    metrics = files["reference_solution"]["expected_semantic_metrics"]
    assert metrics["entity_counts"]["products"] == 2
    assert metrics["technology"]["input_coefficients"] == [-1.0]
    assert metrics["technology"]["output_coefficients"] == [0.1]
    assert metrics["solver_aggregates"]["technology_cost"] == 500.0
    assert len(metrics["transport_links"]) == 4
    assert metrics["route_association"]["compost_pathway"]["expected_pathway_net_value"] == 9.6


def test_q4_paraphrased_prompt_explicitly_states_all_four_transport_capacities():
    prompts = {
        prompt["id"]: prompt["text"]
        for prompt in load_benchmark_files(DEFAULT_Q4_BENCHMARK_DIR)["prompts"]
    }
    paraphrased = prompts["paraphrased"]
    expected_state = build_state_from_semantic_plan(build_midterm_manure_q4_expected_plan("paraphrased"))

    assert "all four transport links have capacity 1000" in paraphrased
    assert "composter to Madison for compost" in paraphrased
    assert [link.capacity for link in expected_state.transport_links] == [1000.0] * 4


def test_solution_comparison_detects_objective_and_flow_match():
    state = build_state_from_semantic_plan(build_midterm_manure_expected_plan("canonical"))
    reference = load_benchmark_files()["reference_solution"]
    solve_result = {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "objective_value": 850.0,
        "message": "mock solve",
        "solution": {
            "q": {
                "B_Dairy_EauClaire": 1000.0,
                "B_Menomonie": 500.0,
                "B_BlackRiverFalls": 500.0,
            },
            "f": {
                "('EauClaire', 'Menomonie')": 500.0,
                "('EauClaire', 'BlackRiverFalls')": 500.0,
            },
            "x": {},
        },
    }

    checks = compare_solution_to_reference(state, solve_result, reference)

    assert checks["objective_match"] is True
    assert checks["accepted_supply_match"] is True
    assert checks["accepted_demand_match"] is True
    assert checks["transport_flow_match"] is True
    assert checks["balance_match"] is True


def test_q2_solution_comparison_detects_policy_allocation_match():
    state = build_state_from_semantic_plan(build_midterm_manure_q2_expected_plan("canonical"))
    reference = load_benchmark_files(DEFAULT_Q2_BENCHMARK_DIR)["reference_solution"]
    solve_result = {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "objective_value": 650.0,
        "message": "mock solve",
        "solution": {
            "q": {
                "B_Dairy_EauClaire": 500.0,
                "B_Menomonie": 0.0,
                "B_BlackRiverFalls": 500.0,
            },
            "f": {
                "('EauClaire', 'Menomonie')": 0.0,
                "('EauClaire', 'BlackRiverFalls')": 500.0,
            },
            "x": {},
        },
    }

    checks = compare_solution_to_reference(state, solve_result, reference)

    assert checks["objective_match"] is True
    assert checks["demand_revenue_match"] is True
    assert checks["transport_cost_match"] is True
    assert checks["accepted_supply_match"] is True
    assert checks["accepted_demand_match"] is True
    assert checks["transport_flow_match"] is True
    assert checks["balance_match"] is True


def test_q3_solution_comparison_detects_removal_incentive_allocation_match():
    state = build_state_from_semantic_plan(build_midterm_manure_q3_expected_plan("canonical"))
    reference = load_benchmark_files(DEFAULT_Q3_BENCHMARK_DIR)["reference_solution"]
    solve_result = {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "objective_value": 1050.0,
        "message": "mock solve",
        "solution": {
            "q": {
                "B_Dairy_EauClaire": 1000.0,
                "B_Menomonie": 500.0,
                "B_BlackRiverFalls": 500.0,
            },
            "f": {
                "('EauClaire', 'Menomonie')": 500.0,
                "('EauClaire', 'BlackRiverFalls')": 500.0,
            },
            "x": {},
        },
    }

    checks = compare_solution_to_reference(state, solve_result, reference)

    assert checks["objective_match"] is True
    assert checks["demand_revenue_match"] is True
    assert checks["transport_cost_match"] is True
    assert checks["supply_cost_match"] is True
    assert checks["accepted_supply_match"] is True
    assert checks["accepted_demand_match"] is True
    assert checks["transport_flow_match"] is True
    assert checks["balance_match"] is True


def test_incomplete_prompt_is_not_solver_ready_or_flags_missing_information():
    report = run_midterm_manure_q1_benchmark(
        config=MidtermBenchmarkConfig(
            prompt_ids=("incomplete",),
            use_llm=False,
            fallback_to_reference_fixture=True,
            attempt_solve=False,
            run_reasoning=False,
        )
    )
    case = report["cases"][0]

    assert case["solver_ready"] is False
    assert case["validation_result"]["missing_parameters"]
    assert "missing capacity" in "; ".join(case["validation_result"]["missing_parameters"])


@pytest.mark.parametrize(
    "notebook_path",
    [
        Path("notebooks/MidtermManureQ1Benchmark.ipynb"),
        Path("notebooks/MidtermManureQ2Benchmark.ipynb"),
        Path("notebooks/MidtermManureQ3Benchmark.ipynb"),
        Path("notebooks/MidtermManureQ4Benchmark.ipynb"),
    ],
)
def test_no_real_api_key_is_stored_in_midterm_notebooks(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assignments = re.findall(r'os\.environ\["GEMINI_API_KEY"\]\s*=\s*"([^"]*)"', source)
    assert assignments
    assert all(value == "" for value in assignments)


def _build_alias_state():
    return build_state_from_semantic_plan(
        {
            "problem_title": "Alias manure state",
            "problem_type": "case_a",
            "nodes": [
                {"id": "EC", "name": "Eau Claire"},
                {"id": "ME", "name": "Menomonie"},
                {"id": "BlackRiverFalls", "name": "Black River Falls"},
            ],
            "products": [{"id": "P1", "name": "manure"}],
            "suppliers": [
                {"id": "S_DAIRY", "node": "EC", "product": "P1", "capacity": 1000.0}
            ],
            "consumers": [
                {"id": "C_CORN", "node": "ME", "product": "P1", "capacity": 500.0},
                {
                    "id": "BlackRiverFalls",
                    "node": "BlackRiverFalls",
                    "product": "P1",
                    "capacity": 500.0,
                },
            ],
            "transport_links": [
                {
                    "id": "EC_to_ME",
                    "origin": "EC",
                    "destination": "ME",
                    "product": "P1",
                    "capacity": None,
                    "cost": 0.1,
                },
                {
                    "id": "EC_to_BlackRiverFalls",
                    "origin": "EC",
                    "destination": "BlackRiverFalls",
                    "product": "P1",
                    "capacity": None,
                    "cost": 0.2,
                },
            ],
            "technologies": [],
            "bids": [
                {
                    "id": "B_SUPPLY",
                    "owner_id": "S_DAIRY",
                    "owner_type": "supplier",
                    "product_id": "P1",
                    "price": 0.0,
                    "quantity": 1000.0,
                },
                {
                    "id": "B_CORN",
                    "owner_id": "C_CORN",
                    "owner_type": "consumer",
                    "product_id": "P1",
                    "price": 0.5,
                    "quantity": 500.0,
                },
                {
                    "id": "B_SOYBEAN",
                    "owner_id": "BlackRiverFalls",
                    "owner_type": "consumer",
                    "product_id": "P1",
                    "price": 1.5,
                    "quantity": 500.0,
                },
            ],
        }
    )


def _alias_solve_result():
    return {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "objective_value": 850.0,
        "message": "mock solve",
        "solution": {
            "q": {
                "B_SUPPLY": 1000.0,
                "B_CORN": 500.0,
                "B_SOYBEAN": 500.0,
            },
            "f": {
                "EC_to_ME": 500.0,
                "EC_to_BlackRiverFalls": 500.0,
            },
            "x": {},
        },
    }


def _build_id_independent_state(
    menomonie_node="M",
    corn_consumer_id="C1",
    first_link_id="T1",
    supplier_price=0.0,
    menomonie_price=0.5,
    black_river_price=1.5,
    problem_type="case_a",
):
    return build_state_from_semantic_plan(
        {
            "problem_title": "ID-independent manure state",
            "problem_type": problem_type,
            "nodes": [
                {"id": "SRC", "name": "dairy source"},
                {"id": menomonie_node, "name": "corn demand"},
                {"id": "SINK2", "name": "soybean demand"},
            ],
            "products": [{"id": "P1", "name": "manure"}],
            "suppliers": [
                {"id": "S1", "node": "SRC", "product": "P1", "capacity": 1000.0}
            ],
            "consumers": [
                {"id": corn_consumer_id, "node": menomonie_node, "product": "P1", "capacity": 500.0},
                {"id": "C2", "node": "SINK2", "product": "P1", "capacity": 500.0},
            ],
            "transport_links": [
                {
                    "id": first_link_id,
                    "origin": "SRC",
                    "destination": menomonie_node,
                    "product": "P1",
                    "capacity": None,
                    "cost": 0.1,
                },
                {
                    "id": "T2",
                    "origin": "SRC",
                    "destination": "SINK2",
                    "product": "P1",
                    "capacity": None,
                    "cost": 0.2,
                },
            ],
            "technologies": [],
            "bids": [
                {
                    "id": "B_SUPPLY",
                    "owner_id": "S1",
                    "owner_type": "supplier",
                    "product_id": "P1",
                    "price": supplier_price,
                    "quantity": 1000.0,
                },
                {
                    "id": "B_CORN",
                    "owner_id": corn_consumer_id,
                    "owner_type": "consumer",
                    "product_id": "P1",
                    "price": menomonie_price,
                    "quantity": 500.0,
                },
                {
                    "id": "B_SOYBEAN",
                    "owner_id": "C2",
                    "owner_type": "consumer",
                    "product_id": "P1",
                    "price": black_river_price,
                    "quantity": 500.0,
                },
            ],
        }
    )


def _id_independent_solve_result(
    first_link_id="T1",
    first_flow=500.0,
    second_flow=500.0,
    objective=850.0,
    accepted_supply=1000.0,
    first_demand=500.0,
    second_demand=500.0,
):
    return {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "objective_value": objective,
        "message": "mock solve",
        "solution": {
            "q": {
                "B_SUPPLY": accepted_supply,
                "B_CORN": first_demand,
                "B_SOYBEAN": second_demand,
            },
            "f": {
                first_link_id: first_flow,
                "T2": second_flow,
            },
            "x": {},
        },
    }


def _build_q4_id_independent_state(
    compost_output_coefficient=0.1,
    technology_capacity=500.0,
    madison_price=100.0,
    compost_transport_cost=1.0,
    technology_cost=1.0,
):
    return build_state_from_semantic_plan(
        {
            "problem_title": "Q4 ID-independent manure compost state",
            "problem_type": "case_c",
            "nodes": [
                {"id": "SRC", "name": "Eau Claire dairy source"},
                {"id": "M", "name": "Menomonie receiver"},
                {"id": "BRF", "name": "Black River Falls receiver"},
                {"id": "K_NODE", "name": "Composter"},
                {"id": "MAD", "name": "Madison compost demand"},
            ],
            "products": [
                {"id": "P1", "name": "dairy manure"},
                {"id": "P2", "name": "compost"},
            ],
            "suppliers": [
                {"id": "S_DAIRY", "node": "SRC", "product": "P1", "capacity": 1000.0}
            ],
            "consumers": [
                {"id": "C_MEN", "node": "M", "product": "P1", "capacity": 500.0},
                {"id": "C_BRF", "node": "BRF", "product": "P1", "capacity": 500.0},
                {"id": "C_MAD", "node": "MAD", "product": "P2", "capacity": 100.0},
            ],
            "transport_links": [
                {
                    "id": "A",
                    "origin": "SRC",
                    "destination": "M",
                    "product": "P1",
                    "capacity": 1000.0,
                    "cost": 0.1,
                },
                {
                    "id": "B",
                    "origin": "SRC",
                    "destination": "BRF",
                    "product": "P1",
                    "capacity": 1000.0,
                    "cost": 0.2,
                },
                {
                    "id": "C",
                    "origin": "SRC",
                    "destination": "K_NODE",
                    "product": "P1",
                    "capacity": 1000.0,
                    "cost": 0.0,
                },
                {
                    "id": "D",
                    "origin": "K_NODE",
                    "destination": "MAD",
                    "product": "P2",
                    "capacity": 1000.0,
                    "cost": compost_transport_cost,
                },
            ],
            "technologies": [
                {
                    "id": "K",
                    "node": "K_NODE",
                    "capacity": technology_capacity,
                    "cost": technology_cost,
                    "yield_coefficients": {"P1": -1.0, "P2": compost_output_coefficient},
                }
            ],
            "bids": [
                {
                    "id": "B_SUPPLY",
                    "owner_id": "S_DAIRY",
                    "owner_type": "supplier",
                    "product_id": "P1",
                    "price": -0.7,
                    "quantity": 1000.0,
                },
                {
                    "id": "B_MEN",
                    "owner_id": "C_MEN",
                    "owner_type": "consumer",
                    "product_id": "P1",
                    "price": -0.5,
                    "quantity": 500.0,
                },
                {
                    "id": "B_BRF",
                    "owner_id": "C_BRF",
                    "owner_type": "consumer",
                    "product_id": "P1",
                    "price": 1.5,
                    "quantity": 500.0,
                },
                {
                    "id": "B_MAD",
                    "owner_id": "C_MAD",
                    "owner_type": "consumer",
                    "product_id": "P2",
                    "price": madison_price,
                    "quantity": 100.0,
                },
            ],
        }
    )


def _q4_id_independent_solve_result(
    objective=5800.0,
    menomonie_flow=0.0,
    black_river_flow=500.0,
    composter_inflow=500.0,
    compost_flow=50.0,
    technology_activity=500.0,
    madison_demand=50.0,
):
    return {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "objective_value": objective,
        "message": "mock solve",
        "solution": {
            "q": {
                "B_SUPPLY": 1000.0,
                "B_MEN": menomonie_flow,
                "B_BRF": black_river_flow,
                "B_MAD": madison_demand,
            },
            "f": {
                "A": menomonie_flow,
                "B": black_river_flow,
                "C": composter_inflow,
                "D": compost_flow,
            },
            "x": {
                "K": technology_activity,
            },
        },
    }


def _primary_metrics_for(state, solve_result=None):
    return evaluate_primary_semantic_metrics(
        state,
        solve_result or _id_independent_solve_result(),
        load_benchmark_files()["reference_solution"],
    )


def _primary_q2_metrics_for(state, solve_result):
    return evaluate_primary_semantic_metrics(
        state,
        solve_result,
        load_benchmark_files(DEFAULT_Q2_BENCHMARK_DIR)["reference_solution"],
    )


def _primary_q3_metrics_for(state, solve_result):
    return evaluate_primary_semantic_metrics(
        state,
        solve_result,
        load_benchmark_files(DEFAULT_Q3_BENCHMARK_DIR)["reference_solution"],
    )


def _primary_q4_metrics_for(state, solve_result):
    return evaluate_primary_semantic_metrics(
        state,
        solve_result,
        load_benchmark_files(DEFAULT_Q4_BENCHMARK_DIR)["reference_solution"],
    )


def _metric(metrics, group, name):
    return next(row for row in metrics[group] if row["metric"] == name)


def test_midterm_aliases_match_reference_solution_components():
    state = _build_alias_state()
    reference = load_benchmark_files()["reference_solution"]

    checks = compare_solution_to_reference(state, _alias_solve_result(), reference)

    assert checks["accepted_supply_match"] is True
    assert checks["accepted_demand_match"] is True
    assert checks["transport_flow_match"] is True
    assert checks["balance_match"] is True
    assert checks["accepted_supply_total_match"] is True

    components = checks["actual_components"]
    assert components["accepted_supply"] == {"Dairy/EauClaire": 1000.0}
    assert components["accepted_demands"]["Menomonie"] == 500.0
    assert components["transport_flows"]["EauClaire_to_Menomonie"] == 500.0


def test_midterm_aliases_are_benign_identifier_mismatches_structurally():
    expected = build_state_from_semantic_plan(build_midterm_manure_expected_plan("canonical"))
    actual = _build_alias_state()

    comparison = compare_problem_states_for_midterm(expected, actual)

    assert comparison["structural_match"] is True
    assert comparison["blocking_errors"] == []
    assert comparison["benign_identifier_mismatches"]
    assert "benign_identifier_mismatch" in comparison["error_categories"]


def test_balance_checks_pass_under_alias_resolved_ids():
    components = extract_midterm_solution_components(_build_alias_state(), _alias_solve_result())

    assert components["balance_checks"]["EauClaire"]["supply"] == 1000.0
    assert components["balance_checks"]["EauClaire"]["total_outgoing_flow"] == 1000.0
    assert components["balance_checks"]["EauClaire"]["holds"] is True
    assert components["balance_checks"]["Menomonie"]["incoming_flow"] == 500.0
    assert components["balance_checks"]["Menomonie"]["accepted_demand"] == 500.0
    assert components["balance_checks"]["Menomonie"]["holds"] is True
    assert components["balance_checks"]["BlackRiverFalls"]["incoming_flow"] == 500.0
    assert components["balance_checks"]["BlackRiverFalls"]["accepted_demand"] == 500.0
    assert components["balance_checks"]["BlackRiverFalls"]["holds"] is True


@pytest.mark.parametrize("menomonie_node", ["M", "MN"])
def test_primary_metrics_pass_with_id_artifacts(menomonie_node):
    state = _build_id_independent_state(
        menomonie_node=menomonie_node,
        corn_consumer_id="C1",
        first_link_id="T1",
    )
    solve_result = _id_independent_solve_result(first_link_id="T1")

    metrics = _primary_metrics_for(state, solve_result)

    assert metrics["semantic_structure_pass"] is True
    assert metrics["solver_aggregate_pass"] is True
    assert metrics["balance_residual_pass"] is True
    assert metrics["primary_success"] is True
    assert _metric(metrics, "route_economics_metrics", "sorted_route_net_values")["pass"] is True
    assert _metric(metrics, "solver_aggregate_metrics", "sorted_active_flow_values")["pass"] is True


@pytest.mark.parametrize(
    ("menomonie_node", "corn_consumer_id", "first_link_id"),
    [("M", "C1", "T1"), ("MN", "CornFarmer", "T1")],
)
def test_q2_primary_metrics_pass_with_id_artifacts_and_negative_bid(
    menomonie_node,
    corn_consumer_id,
    first_link_id,
):
    state = _build_id_independent_state(
        menomonie_node=menomonie_node,
        corn_consumer_id=corn_consumer_id,
        first_link_id=first_link_id,
        menomonie_price=-0.5,
        problem_type="case_b",
    )
    solve_result = _id_independent_solve_result(
        first_link_id=first_link_id,
        first_flow=0.0,
        second_flow=500.0,
        objective=650.0,
        accepted_supply=500.0,
        first_demand=0.0,
        second_demand=500.0,
    )

    metrics = _primary_q2_metrics_for(state, solve_result)

    assert metrics["semantic_structure_pass"] is True
    assert metrics["solver_aggregate_pass"] is True
    assert metrics["balance_residual_pass"] is True
    assert metrics["primary_success"] is True
    assert _metric(metrics, "parameter_multiset_metrics", "consumer_bid_prices")["pass"] is True
    assert _metric(metrics, "topology_metrics", "negative_bid_detection")["pass"] is True
    assert _metric(metrics, "route_economics_metrics", "sorted_route_net_values")["pass"] is True
    assert _metric(metrics, "solver_aggregate_metrics", "sorted_active_flow_values")["pass"] is True


def test_q2_primary_metrics_fail_when_negative_bid_is_missing():
    state = _build_id_independent_state(problem_type="case_b", menomonie_price=0.5)
    solve_result = _id_independent_solve_result(
        first_flow=0.0,
        second_flow=500.0,
        objective=650.0,
        accepted_supply=500.0,
        first_demand=0.0,
        second_demand=500.0,
    )

    metrics = _primary_q2_metrics_for(state, solve_result)

    assert metrics["semantic_structure_pass"] is False
    assert _metric(metrics, "parameter_multiset_metrics", "consumer_bid_prices")["pass"] is False
    assert _metric(metrics, "topology_metrics", "negative_bid_detection")["pass"] is False


@pytest.mark.parametrize(
    ("menomonie_node", "corn_consumer_id", "first_link_id"),
    [("M", "C1", "T1"), ("ME", "CornFarmer", "EC_to_ME")],
)
def test_q3_primary_metrics_pass_with_id_artifacts_and_supplier_payment(
    menomonie_node,
    corn_consumer_id,
    first_link_id,
):
    state = _build_id_independent_state(
        menomonie_node=menomonie_node,
        corn_consumer_id=corn_consumer_id,
        first_link_id=first_link_id,
        supplier_price=-0.7,
        menomonie_price=-0.5,
        problem_type="case_b",
    )
    solve_result = _id_independent_solve_result(
        first_link_id=first_link_id,
        first_flow=500.0,
        second_flow=500.0,
        objective=1050.0,
        accepted_supply=1000.0,
        first_demand=500.0,
        second_demand=500.0,
    )

    metrics = _primary_q3_metrics_for(state, solve_result)

    assert metrics["semantic_structure_pass"] is True
    assert metrics["solver_aggregate_pass"] is True
    assert metrics["balance_residual_pass"] is True
    assert metrics["primary_success"] is True
    assert _metric(metrics, "parameter_multiset_metrics", "supplier_bid_prices")["pass"] is True
    assert _metric(metrics, "topology_metrics", "negative_supplier_bid_detection")["pass"] is True
    assert _metric(metrics, "route_economics_metrics", "sorted_route_net_values")["pass"] is True
    assert _metric(metrics, "solver_aggregate_metrics", "supply_cost")["pass"] is True


def test_q3_primary_metrics_fail_when_supplier_payment_is_missing():
    state = _build_id_independent_state(
        supplier_price=0.0,
        menomonie_price=-0.5,
        problem_type="case_b",
    )
    solve_result = _id_independent_solve_result(
        objective=1050.0,
        accepted_supply=1000.0,
    )

    metrics = _primary_q3_metrics_for(state, solve_result)

    assert metrics["semantic_structure_pass"] is False
    assert _metric(metrics, "parameter_multiset_metrics", "supplier_bid_prices")["pass"] is False
    assert _metric(metrics, "topology_metrics", "negative_supplier_bid_detection")["pass"] is False
    assert _metric(metrics, "route_economics_metrics", "sorted_route_net_values")["pass"] is False


def test_q3_supplier_removal_incentive_diagnostics_are_id_independent():
    state = _build_id_independent_state(
        menomonie_node="MN",
        corn_consumer_id="CornFarmer",
        first_link_id="T1",
        supplier_price=-0.7,
        menomonie_price=-0.5,
        problem_type="case_b",
    )
    solve_result = _id_independent_solve_result(
        objective=1050.0,
        accepted_supply=1000.0,
    )
    reference = load_benchmark_files(DEFAULT_Q3_BENCHMARK_DIR)["reference_solution"]

    rows = build_supplier_removal_incentive_diagnostics(state, solve_result, reference)

    assert _metric({"diagnostics": rows}, "diagnostics", "supplier_removal_payment_per_ton")["pass"] is True
    assert _metric({"diagnostics": rows}, "diagnostics", "total_removal_incentive_value")["pass"] is True
    assert _metric({"diagnostics": rows}, "diagnostics", "all_routes_profitable_after_payment")["pass"] is True


def test_q4_solution_comparison_detects_compost_allocation_match():
    state = build_state_from_semantic_plan(build_midterm_manure_q4_expected_plan("canonical"))
    reference = load_benchmark_files(DEFAULT_Q4_BENCHMARK_DIR)["reference_solution"]
    solve_result = {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "objective_value": 5800.0,
        "message": "mock solve",
        "solution": {
            "q": {
                "B_DF_DM": 1000.0,
                "B_CF_DM": 0.0,
                "B_SF_DM": 500.0,
                "B_DC_Compost": 50.0,
            },
            "f": {
                "DF_to_CF": 0.0,
                "DF_to_SF": 500.0,
                "DF_to_Composter": 500.0,
                "Composter_to_DC": 50.0,
            },
            "x": {"Composter": 500.0},
        },
    }

    checks = compare_solution_to_reference(state, solve_result, reference)

    assert checks["objective_match"] is True
    assert checks["demand_revenue_match"] is True
    assert checks["transport_cost_match"] is True
    assert checks["supply_cost_match"] is True
    assert checks["supply_contribution_match"] is True
    assert checks["technology_cost_match"] is True
    assert checks["technology_activity_match"] is True
    assert checks["technology_outputs_match"] is True
    assert checks["balance_match"] is True


def test_q4_primary_metrics_pass_with_id_artifacts_and_compost_technology():
    state = _build_q4_id_independent_state()
    solve_result = _q4_id_independent_solve_result()

    metrics = _primary_q4_metrics_for(state, solve_result)

    assert metrics["formulation_completeness_pass"] is True
    assert metrics["solve_correctness_pass"] is True
    assert metrics["reasoning_ready_pass"] is True
    assert metrics["route_association_pass"] is True
    assert metrics["semantic_structure_pass"] is True
    assert metrics["technology_structure_pass"] is True
    assert metrics["solver_aggregate_pass"] is True
    assert metrics["balance_residual_pass"] is True
    assert metrics["primary_success"] is True
    assert metrics["failure_type"] == "none"
    assert _metric(metrics, "semantic_count_metrics", "products_include_compost")["pass"] is True
    assert _metric(metrics, "technology_yield_metrics", "technology_output_coefficients")["pass"] is True
    assert _metric(metrics, "route_economics_metrics", "technology_pathway_net_values_per_input")["pass"] is True
    assert _metric(metrics, "route_association_metrics", "route_association:Menomonie")["pass"] is True
    assert _metric(metrics, "route_association_metrics", "route_association:Black River Falls")["pass"] is True
    assert _metric(metrics, "route_association_metrics", "route_association:compost_pathway")["pass"] is True
    assert _metric(metrics, "solver_aggregate_metrics", "sorted_active_manure_flow_values")["pass"] is True
    assert _metric(metrics, "solver_aggregate_metrics", "sorted_active_compost_flow_values")["pass"] is True
    assert _metric(metrics, "solver_aggregate_metrics", "technology_activity")["pass"] is True


def test_q4_route_association_fails_when_menomonie_and_black_river_costs_are_swapped():
    state = _build_q4_id_independent_state()
    state.transport_links[0].cost = 0.2
    state.transport_links[1].cost = 0.1
    solve_result = _q4_id_independent_solve_result(objective=5850.0)

    metrics = _primary_q4_metrics_for(state, solve_result)

    assert _metric(metrics, "parameter_multiset_metrics", "transport_costs")["pass"] is True
    assert metrics["route_association_pass"] is False
    assert metrics["formulation_completeness_pass"] is False
    assert metrics["solve_correctness_pass"] is False
    assert metrics["primary_success"] is False
    assert metrics["failure_type"] == "route_cost_association_error"
    assert _metric(metrics, "route_association_metrics", "route_association:Menomonie")["pass"] is False
    assert _metric(metrics, "route_association_metrics", "route_association:Black River Falls")["pass"] is False


def test_q4_sorted_transport_cost_multiset_alone_is_insufficient_for_route_association():
    state = _build_q4_id_independent_state()
    state.transport_links[0].cost = 0.2
    state.transport_links[1].cost = 0.1

    metrics = _primary_q4_metrics_for(state, _q4_id_independent_solve_result(objective=5850.0))

    assert _metric(metrics, "parameter_multiset_metrics", "transport_costs")["pass"] is True
    assert _metric(metrics, "route_economics_metrics", "sorted_route_or_pathway_net_values")["pass"] is False
    assert metrics["route_association_pass"] is False
    assert metrics["primary_success"] is False


def test_q4_missing_nonbinding_transport_capacity_is_not_primary_success():
    state = _build_q4_id_independent_state()
    state.transport_links[0].capacity = None
    solve_result = _q4_id_independent_solve_result()

    metrics = _primary_q4_metrics_for(state, solve_result)

    assert metrics["solve_correctness_pass"] is True
    assert metrics["formulation_completeness_pass"] is False
    assert metrics["reasoning_ready_pass"] is False
    assert metrics["primary_success"] is False
    assert metrics["failure_type"] == "incomplete_formulation_but_solution_equivalent"
    capacity_row = _metric(
        metrics,
        "formulation_completeness_metrics",
        "transport_capacity:EauClaire_to_Menomonie:Manure",
    )
    assert capacity_row["pass"] is False
    assert "Menomonie" in capacity_row["expected"]


def test_q4_missing_nonbinding_transport_capacity_blocks_reasoning_readiness():
    state = _build_q4_id_independent_state()
    state.transport_links[0].capacity = None

    metrics = _primary_q4_metrics_for(state, _q4_id_independent_solve_result())

    assert metrics["reasoning_ready_pass"] is False
    for row in metrics["reasoning_readiness_metrics"]:
        assert row["pass"] is False


def test_q4_missing_route_cost_fails_solution_and_formulation_completeness():
    state = _build_q4_id_independent_state()
    state.transport_links[1].cost = 0.0
    solve_result = _q4_id_independent_solve_result(objective=5900.0)

    metrics = _primary_q4_metrics_for(state, solve_result)

    assert metrics["solve_correctness_pass"] is False
    assert metrics["formulation_completeness_pass"] is False
    assert metrics["primary_success"] is False
    assert _metric(metrics, "solve_correctness_metrics", "transport_cost_match")["pass"] is False
    assert _metric(
        metrics,
        "formulation_completeness_metrics",
        "transport_cost:EauClaire_to_BlackRiverFalls:Manure",
    )["pass"] is False


def test_q4_fully_recovered_formulation_solves_and_is_reasoning_ready():
    state = _build_q4_id_independent_state()
    metrics = _primary_q4_metrics_for(state, _q4_id_independent_solve_result())

    assert metrics["formulation_completeness_pass"] is True
    assert metrics["solve_correctness_pass"] is True
    assert metrics["reasoning_ready_pass"] is True
    assert metrics["primary_success"] is True


@pytest.mark.parametrize(
    ("state_kwargs", "metric_group", "metric_name"),
    [
        ({"compost_output_coefficient": 0.2}, "technology_yield_metrics", "technology_output_coefficients"),
        ({"technology_capacity": 400.0}, "technology_yield_metrics", "technology_capacities"),
        ({"madison_price": 90.0}, "parameter_multiset_metrics", "consumer_bid_prices"),
        ({"compost_transport_cost": 2.0}, "parameter_multiset_metrics", "transport_costs"),
    ],
)
def test_q4_primary_metrics_fail_for_wrong_strict_parameters(state_kwargs, metric_group, metric_name):
    state = _build_q4_id_independent_state(**state_kwargs)
    solve_result = _q4_id_independent_solve_result()

    metrics = _primary_q4_metrics_for(state, solve_result)

    assert metrics["primary_success"] is False
    assert _metric(metrics, metric_group, metric_name)["pass"] is False


def test_q4_primary_metrics_fail_for_wrong_objective_active_flows_and_balance():
    state = _build_q4_id_independent_state()

    wrong_objective = _q4_id_independent_solve_result(objective=5799.0)
    objective_metrics = _primary_q4_metrics_for(state, wrong_objective)
    assert objective_metrics["solver_aggregate_pass"] is False
    assert _metric(objective_metrics, "solver_aggregate_metrics", "objective_value")["pass"] is False

    wrong_flows = _q4_id_independent_solve_result(
        black_river_flow=400.0,
        composter_inflow=600.0,
        compost_flow=50.0,
        technology_activity=500.0,
    )
    flow_metrics = _primary_q4_metrics_for(state, wrong_flows)
    assert flow_metrics["solver_aggregate_pass"] is False
    assert flow_metrics["balance_residual_pass"] is False
    assert _metric(flow_metrics, "solver_aggregate_metrics", "sorted_active_manure_flow_values")["pass"] is False
    assert _metric(flow_metrics, "balance_residual_metrics", "max_abs_balance_residual")["pass"] is False


def test_q4_benchmark_runner_includes_technology_table_without_llm(monkeypatch):
    class FakeSolveResult:
        def to_dict(self):
            return {
                "success": True,
                "status": "optimal",
                "termination_condition": "optimal",
                "solver_name": "mock",
                "objective_value": 5800.0,
                "message": "mock solve",
                "solution": {
                    "q": {
                        "B_DF_DM": 1000.0,
                        "B_CF_DM": 0.0,
                        "B_SF_DM": 500.0,
                        "B_DC_Compost": 50.0,
                    },
                    "f": {
                        "DF_to_CF": 0.0,
                        "DF_to_SF": 500.0,
                        "DF_to_Composter": 500.0,
                        "Composter_to_DC": 50.0,
                    },
                    "x": {"Composter": 500.0},
                },
            }

    monkeypatch.setattr("src.midterm_benchmark.solve_model", lambda model: FakeSolveResult())
    report = run_midterm_manure_q4_benchmark(
        config=MidtermBenchmarkConfig(
            prompt_ids=("canonical", "paraphrased"),
            use_llm=False,
            fallback_to_reference_fixture=True,
            run_reasoning=False,
        )
    )

    summary = report["tables"]["case_summary"]
    technology_rows = report["tables"]["technology_yield_metrics"]
    route_association_rows = report["tables"]["route_association_metrics"]
    formulation_rows = report["tables"]["formulation_completeness_metrics"]
    solve_correctness_rows = report["tables"]["solve_correctness_metrics"]
    readiness_rows = report["tables"]["reasoning_readiness_metrics"]

    assert set(summary["prompt_id"]) == {"canonical", "paraphrased"}
    assert summary["primary_success"].all()
    assert summary["formulation_completeness_pass"].all()
    assert summary["solve_correctness_pass"].all()
    assert summary["reasoning_ready_pass"].all()
    assert not technology_rows.empty
    assert not route_association_rows.empty
    assert not formulation_rows.empty
    assert not solve_correctness_rows.empty
    assert not readiness_rows.empty
    assert set(technology_rows["prompt_id"]) == {"canonical", "paraphrased"}
    assert technology_rows["pass"].all()
    assert route_association_rows["pass"].all()


def test_q3_benchmark_runner_includes_removal_incentive_table_without_llm(monkeypatch):
    class FakeSolveResult:
        def to_dict(self):
            return {
                "success": True,
                "status": "optimal",
                "termination_condition": "optimal",
                "solver_name": "mock",
                "objective_value": 1050.0,
                "message": "mock solve",
                "solution": {
                    "q": {
                        "B_Dairy_EauClaire": 1000.0,
                        "B_Menomonie": 500.0,
                        "B_BlackRiverFalls": 500.0,
                    },
                    "f": {
                        "('EauClaire', 'Menomonie')": 500.0,
                        "('EauClaire', 'BlackRiverFalls')": 500.0,
                    },
                    "x": {},
                },
            }

    monkeypatch.setattr("src.midterm_benchmark.solve_model", lambda model: FakeSolveResult())
    report = run_midterm_manure_q3_benchmark(
        config=MidtermBenchmarkConfig(
            prompt_ids=("canonical", "paraphrased"),
            use_llm=False,
            fallback_to_reference_fixture=True,
            run_reasoning=False,
        )
    )

    summary = report["tables"]["case_summary"]
    diagnostics = report["tables"]["removal_incentive_diagnostics"]

    assert set(summary["prompt_id"]) == {"canonical", "paraphrased"}
    assert summary["primary_success"].all()
    assert not diagnostics.empty
    assert set(diagnostics["prompt_id"]) == {"canonical", "paraphrased"}
    assert diagnostics["pass"].all()


def test_primary_metrics_fail_for_wrong_number_of_consumers():
    state = _build_id_independent_state()
    state.consumers = state.consumers[:1]

    metrics = _primary_metrics_for(state)

    assert metrics["semantic_structure_pass"] is False
    assert _metric(metrics, "semantic_count_metrics", "demand_consumer_entities")["pass"] is False


def test_primary_metrics_fail_for_wrong_consumer_price_and_transport_cost():
    state = _build_id_independent_state()
    state.bids[1].price = 0.6
    price_metrics = _primary_metrics_for(state)

    assert price_metrics["semantic_structure_pass"] is False
    assert _metric(price_metrics, "parameter_multiset_metrics", "consumer_bid_prices")["pass"] is False

    state = _build_id_independent_state()
    state.transport_links[0].cost = 0.15
    cost_metrics = _primary_metrics_for(state)

    assert cost_metrics["semantic_structure_pass"] is False
    assert _metric(cost_metrics, "parameter_multiset_metrics", "transport_costs")["pass"] is False


def test_primary_metrics_fail_for_wrong_objective_and_active_flow_values():
    state = _build_id_independent_state()
    wrong_objective = _id_independent_solve_result(objective=840.0)

    objective_metrics = _primary_metrics_for(state, wrong_objective)

    assert objective_metrics["solver_aggregate_pass"] is False
    assert _metric(objective_metrics, "solver_aggregate_metrics", "objective_value")["pass"] is False

    wrong_flows = _id_independent_solve_result(first_flow=600.0, second_flow=400.0)
    flow_metrics = _primary_metrics_for(state, wrong_flows)

    assert flow_metrics["solver_aggregate_pass"] is False
    assert _metric(flow_metrics, "solver_aggregate_metrics", "sorted_active_flow_values")["pass"] is False


def test_primary_balance_residual_metrics_fail_when_balances_do_not_hold():
    state = _build_id_independent_state()
    wrong_flows = _id_independent_solve_result(first_flow=600.0, second_flow=400.0)

    metrics = _primary_metrics_for(state, wrong_flows)

    assert metrics["balance_residual_pass"] is False
    assert _metric(metrics, "balance_residual_metrics", "max_abs_balance_residual")["pass"] is False
    assert _metric(metrics, "balance_residual_metrics", "balance_violation_count")["actual"] > 0
