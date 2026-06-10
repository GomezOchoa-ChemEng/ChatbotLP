import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

from src.llm_problem_interpreter import (  # noqa: E402
    LLMInvalidJSONError,
    LLMSchemaError,
    build_problem_artifacts_from_semantic_plan,
    build_state_from_semantic_plan,
)
from src.paper_grade_evaluation import EvaluationConfig, build_paper_grade_cases, evaluate_case  # noqa: E402


def _case(name: str = "canonical_case_a"):
    return next(case for case in build_paper_grade_cases() if case["name"] == name)


def _live_config(**overrides):
    data = {
        "use_llm": True,
        "fallback_to_expected_fixture": True,
        "attempt_solve": False,
        "run_reasoning": False,
    }
    data.update(overrides)
    return EvaluationConfig(**data)


def test_live_missing_api_key_reports_not_configured_without_fixture(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    result = evaluate_case(_case(), config=_live_config())
    metadata = result["interpretation_metadata"]

    assert result["problem_state_created"] is False
    assert metadata["evaluation_mode"] == "live_llm"
    assert metadata["live_llm_attempted"] is True
    assert metadata["deterministic_fixture_used"] is False
    assert metadata["failure_type"] == "llm_not_configured"
    assert result["primary_success"] is False
    assert "Live LLM evaluation failed" in capsys.readouterr().out


def test_live_llm_exception_reports_call_failed_without_fixture():
    with patch(
        "src.paper_grade_evaluation.interpret_problem_from_text",
        side_effect=RuntimeError("provider outage"),
    ):
        result = evaluate_case(_case(), config=_live_config())

    metadata = result["interpretation_metadata"]
    assert result["problem_state_created"] is False
    assert metadata["failure_type"] == "llm_call_failed"
    assert metadata["llm_error_type"] == "RuntimeError"
    assert metadata["deterministic_fixture_used"] is False
    assert result["primary_success"] is False


def test_live_invalid_json_reports_invalid_json_without_fixture():
    with patch(
        "src.paper_grade_evaluation.interpret_problem_from_text",
        side_effect=LLMInvalidJSONError("LLM output is not valid JSON"),
    ):
        result = evaluate_case(_case(), config=_live_config())

    metadata = result["interpretation_metadata"]
    assert result["problem_state_created"] is False
    assert metadata["failure_type"] == "llm_invalid_json"
    assert metadata["deterministic_fixture_used"] is False


def test_live_schema_error_reports_schema_error_without_fixture():
    with patch(
        "src.paper_grade_evaluation.interpret_problem_from_text",
        side_effect=LLMSchemaError("LLM output failed schema validation"),
    ):
        result = evaluate_case(_case(), config=_live_config())

    metadata = result["interpretation_metadata"]
    assert result["problem_state_created"] is False
    assert metadata["failure_type"] == "llm_schema_error"
    assert metadata["deterministic_fixture_used"] is False


def test_live_incomplete_state_reports_incomplete_and_does_not_solve_or_repair():
    incomplete_plan = {
        "problem_title": "Incomplete live interpretation",
        "nodes": [{"id": "N1"}, {"id": "N2"}],
        "products": [{"id": "P1"}],
        "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": None}],
        "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
        "transport_links": [
            {"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 50.0, "cost": None}
        ],
        "bids": [
            {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 1.0, "quantity": None},
            {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 2.0, "quantity": 50.0},
        ],
        "technologies": [],
    }
    artifacts = build_problem_artifacts_from_semantic_plan(incomplete_plan)

    with patch("src.paper_grade_evaluation.interpret_problem_from_text", return_value=artifacts):
        result = evaluate_case(_case(), config=_live_config(attempt_solve=True))

    state = result["problem_state"]
    metadata = result["interpretation_metadata"]
    assert state.suppliers[0].capacity is None
    assert state.transport_links[0].cost is None
    assert state.bids[0].quantity is None
    assert result["solve_result"]["status"] == "skipped"
    assert metadata["failure_type"] == "incomplete_problem_state"
    assert metadata["solver_ready"] is False
    assert metadata["missing_fields"]


def test_deterministic_fixture_mode_is_not_live_llm_performance():
    result = evaluate_case(
        _case(),
        config=EvaluationConfig(
            use_llm=False,
            use_deterministic_fixture=True,
            attempt_solve=False,
            run_reasoning=False,
        ),
    )

    metadata = result["interpretation_metadata"]
    assert result["problem_state_created"] is True
    assert metadata["evaluation_mode"] == "deterministic_fixture"
    assert metadata["interpretation_source"] == "deterministic_fixture"
    assert metadata["live_llm_attempted"] is False
    assert metadata["deterministic_fixture_used"] is True


def test_reference_data_do_not_mutate_or_complete_interpreted_problem_state():
    case = _case()
    expected_snapshot = case["expected_state"].to_dict()
    incomplete_plan = {
        "problem_title": "Incomplete live interpretation",
        "nodes": [{"id": "N1"}, {"id": "N2"}],
        "products": [{"id": "P1"}],
        "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": None}],
        "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
        "transport_links": [
            {"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 50.0, "cost": None}
        ],
        "bids": [
            {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 1.0, "quantity": None},
            {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 2.0, "quantity": 50.0},
        ],
        "technologies": [],
    }

    with patch(
        "src.paper_grade_evaluation.interpret_problem_from_text",
        return_value=build_problem_artifacts_from_semantic_plan(incomplete_plan),
    ):
        result = evaluate_case(case, config=_live_config())

    assert case["expected_state"].to_dict() == expected_snapshot
    assert result["problem_state"].suppliers[0].capacity is None
    assert result["problem_state"].transport_links[0].cost is None
    assert result["comparison"]["structural_match"] is False


def test_missing_numerical_values_remain_none_and_explicit_zero_remains_zero():
    state = build_state_from_semantic_plan(
        {
            "problem_title": "None versus zero",
            "nodes": [{"id": "N1"}, {"id": "N2"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": None}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 0.0}],
            "transport_links": [
                {"id": "missing", "origin": "N1", "destination": "N2", "product": "P1", "capacity": None, "cost": None},
                {"id": "zero", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 0.0, "cost": 0.0},
            ],
            "bids": [],
            "technologies": [],
        }
    )

    assert state.suppliers[0].capacity is None
    assert state.consumers[0].capacity == 0.0
    assert state.transport_links[0].capacity is None
    assert state.transport_links[0].cost is None
    assert state.transport_links[1].capacity == 0.0
    assert state.transport_links[1].cost == 0.0
