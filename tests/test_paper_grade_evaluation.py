import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.llm_problem_interpreter import build_state_from_semantic_plan
from src.model_builder import build_model_from_state
from src.paper_grade_evaluation import (
    EvaluationConfig,
    build_evaluation_config,
    build_output_tables,
    build_paper_grade_cases,
    compare_problem_states_paper_grade,
    evaluate_case,
    evaluate_solve_accuracy,
    run_paper_grade_evaluation,
    run_reasoning_prompt_battery,
)
from pyomo.environ import value


def test_paper_grade_case_catalog_covers_requested_cases():
    cases = build_paper_grade_cases()
    names = {case["name"] for case in cases}

    assert "canonical_case_a" in names
    assert "paraphrased_case_a" in names
    assert "negative_bid_case_b" in names
    assert "transformation_case_c" in names
    assert "incomplete_case_a" in names
    assert "ambiguous_case_a" in names

    case_b = next(case for case in cases if case["name"] == "negative_bid_case_b")
    case_c = next(case for case in cases if case["name"] == "transformation_case_c")

    assert case_b["expected_solution"]["objective_value"] == 240.0
    assert case_c["expected_solution"]["technology_extents"] == {"K1": 50.0}


def test_compare_separates_benign_extra_names_from_blocking_errors():
    expected = build_state_from_semantic_plan(
        {
            "problem_title": "Expected",
            "nodes": [{"id": "N1"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }
    )
    actual = build_state_from_semantic_plan(
        {
            "problem_title": "Actual",
            "nodes": [{"id": "N1", "name": "Supply node"}],
            "products": [{"id": "P1", "name": "Product"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 90.0}],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }
    )

    comparison = compare_problem_states_paper_grade(expected, actual)

    assert len(comparison["benign_extra_name_fields"]) == 2
    assert "benign_extra_name_field" in comparison["error_categories"]
    assert "wrong_numeric_value" in comparison["blocking_error_categories"]
    assert comparison["structural_match"] is False


def test_evaluate_case_fixture_fallback_builds_tables_without_live_llm():
    case = next(case for case in build_paper_grade_cases() if case["name"] == "canonical_case_a")
    result = evaluate_case(
        case,
        config=EvaluationConfig(
            use_llm=False,
            fallback_to_expected_fixture=True,
            attempt_solve=False,
            run_reasoning=False,
        ),
    )
    tables = build_output_tables([result])

    assert result["semantic_plan_created"] is True
    assert result["problem_state_created"] is True
    assert result["comparison"]["structural_match"] is True
    assert result["interpretation_metadata"]["fallback_used"] is True
    assert result["interpretation_metadata"]["interpretation_source"] == "deterministic_fallback"
    assert tables["case_level_summary"].iloc[0]["case"] == "canonical_case_a"
    assert tables["interpretation_accuracy"].iloc[0]["blocking_error_categories"] == ""


def test_evaluate_solve_accuracy_checks_objective_and_solution_blocks():
    solve_result = {
        "success": True,
        "status": "optimal",
        "termination_condition": "optimal",
        "solver_name": "mock",
        "message": "mock solve",
        "objective_value": 500.0,
        "solution": {
            "q": {"B1": 50.0, "B2": 50.0},
            "f": {"('N1', 'N2')": 50.0},
            "x": {},
        },
    }
    expected = {
        "objective_value": 500.0,
        "bid_allocations": {"B1": 50.0, "B2": 50.0},
        "transport_flows": {"('N1', 'N2')": 50.0},
        "technology_extents": {},
    }

    checks = evaluate_solve_accuracy(solve_result, expected, solve_expected=True)

    assert checks["objective_match"] is True
    assert checks["accepted_bid_match"] is True
    assert checks["transport_flow_match"] is True
    assert checks["technology_activity_match"] is True


def test_case_c_expected_solution_satisfies_product_specific_balances():
    case = next(case for case in build_paper_grade_cases() if case["name"] == "transformation_case_c")
    model = build_model_from_state(case["expected_state"])

    model.q["B1"].set_value(50.0)
    model.q["B2"].set_value(40.0)
    model.f["N1", "N2"].set_value(40.0)
    model.x["K1"].set_value(50.0)

    for node in model.N:
        for product in model.P:
            assert abs(value(model.node_balance[node, product].body)) <= 1e-9


def test_selected_case_filter_limits_evaluation_scope():
    report = run_paper_grade_evaluation(
        config=EvaluationConfig(
            selected_cases=("canonical_case_a", "negative_bid_case_b"),
            use_llm=False,
            attempt_solve=False,
            run_reasoning=False,
        )
    )

    assert [case["name"] for case in report["cases"]] == [
        "canonical_case_a",
        "negative_bid_case_b",
    ]
    assert list(report["tables"]["case_level_summary"]["case"]) == [
        "canonical_case_a",
        "negative_bid_case_b",
    ]


def test_run_interpretation_false_marks_stage_skipped_but_supplies_fixture_state():
    case = next(case for case in build_paper_grade_cases() if case["name"] == "canonical_case_a")
    result = evaluate_case(
        case,
        config=EvaluationConfig(
            run_interpretation=False,
            use_llm=False,
            attempt_solve=False,
            run_reasoning=False,
        ),
    )

    assert result["semantic_plan_created"] is False
    assert result["problem_state_created"] is True
    assert result["comparison"]["structural_match"] is True
    assert result["interpretation_metadata"]["interpretation_source"] == "skipped_by_user_config"


def test_reasoning_prompt_subset_runs_only_requested_prompts():
    case = next(case for case in build_paper_grade_cases() if case["name"] == "canonical_case_a")
    rows = run_reasoning_prompt_battery(
        state=case["expected_state"],
        case_name=case["name"],
        use_llm=False,
        prompt_subset=("primal_lp", "dual_lp"),
    )

    assert [row["prompt_id"] for row in rows] == ["primal_lp", "dual_lp"]


def test_quota_safe_demo_preset_uses_three_interpretation_cases_without_reasoning():
    config = build_evaluation_config("quota_safe_demo", use_llm=False)
    report = run_paper_grade_evaluation(config=config)

    assert [case["name"] for case in report["cases"]] == [
        "canonical_case_a",
        "negative_bid_case_b",
        "transformation_case_c",
    ]
    assert report["tables"]["reasoning_prompt_success"].empty
    assert set(report["tables"]["solve_accuracy"]["status"]) == {"skipped"}


def test_strict_live_llm_without_configuration_does_not_use_fixture_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    case = next(case for case in build_paper_grade_cases() if case["name"] == "canonical_case_a")

    result = evaluate_case(
        case,
        config=EvaluationConfig(
            use_llm=True,
            fallback_to_expected_fixture=False,
            attempt_solve=False,
            run_reasoning=False,
        ),
    )

    assert result["problem_state_created"] is False
    assert result["interpretation_metadata"]["interpretation_source"] == "live_llm_pipeline"
    assert result["interpretation_metadata"]["fallback_used"] is False
    assert "Gemini is not configured" in result["interpretation_metadata"]["llm_failure"]
