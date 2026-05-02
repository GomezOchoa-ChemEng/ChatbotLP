import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.llm_problem_interpreter import build_state_from_semantic_plan
from src.prose_interpreter_evaluation import (
    build_benchmark_evaluation_cases,
    compare_problem_states,
    summarize_failure_modes,
)


def _build_state(plan):
    return build_state_from_semantic_plan(plan)


def test_benchmark_cases_cover_a_b_c_shapes():
    cases = build_benchmark_evaluation_cases()

    names = {case["name"] for case in cases}

    assert "canonical_case_a" in names
    assert "negative_bid_case_b" in names
    assert "transformation_case_c" in names
    assert len(cases) >= 6


def test_compare_problem_states_reports_numeric_and_mapping_errors():
    expected = _build_state(
        {
            "problem_title": "Expected",
            "nodes": [{"id": "N1"}, {"id": "N2"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
            "transport_links": [],
            "technologies": [],
            "bids": [
                {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0, "quantity": 100.0},
                {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0},
            ],
        }
    )
    actual = _build_state(
        {
            "problem_title": "Actual",
            "nodes": [{"id": "N1"}, {"id": "N3"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N3", "product": "P1", "capacity": 90.0}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
            "transport_links": [],
            "technologies": [],
            "bids": [
                {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 9.0, "quantity": None},
                {"id": "B2", "owner_id": "S1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0},
            ],
        }
    )

    comparison = compare_problem_states(expected, actual)

    categories = comparison["error_categories"]
    assert "missing_entity" in categories
    assert "extra_entity" in categories
    assert "wrong_node_mapping" in categories
    assert "wrong_numeric_value" in categories
    assert "wrong_owner_relation" in categories
    assert "missing_quantity" in categories


def test_compare_problem_states_reports_missing_and_extra_fields():
    expected = _build_state(
        {
            "problem_title": "Expected",
            "nodes": [{"id": "N1"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": None}],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }
    )
    actual = _build_state(
        {
            "problem_title": "Actual",
            "nodes": [{"id": "N1"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }
    )

    comparison = compare_problem_states(expected, actual)

    assert comparison["extra_fields"]
    assert "extra_field" in comparison["error_categories"]


def test_summarize_failure_modes_aggregates_case_counts():
    summary = summarize_failure_modes(
        [
            {"comparison": {"error_categories": ["missing_entity", "wrong_numeric_value", "wrong_numeric_value"]}},
            {"comparison": {"error_categories": ["missing_entity", "wrong_owner_relation"]}},
        ]
    )

    assert summary[0]["category"] == "missing_entity"
    assert summary[0]["count"] == 2
    assert summary[0]["cases"] == 2
