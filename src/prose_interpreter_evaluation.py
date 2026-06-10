"""Lightweight evaluation helpers for the prose interpreter milestone."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List

from .chatbot_engine import run_chatbot_session
from .llm_problem_interpreter import summarize_problem_state
from .schema import ProblemState
from .validator import validate_state


ENTITY_COLLECTIONS = (
    "nodes",
    "products",
    "suppliers",
    "consumers",
    "transport_links",
    "technologies",
    "bids",
)

FIELD_MAP = {
    "nodes": ("name",),
    "products": ("name",),
    "suppliers": ("node", "product", "capacity"),
    "consumers": ("node", "product", "capacity"),
    "transport_links": ("origin", "destination", "product", "capacity", "cost"),
    "technologies": ("node", "capacity", "cost", "yield_coefficients"),
    "bids": ("owner_id", "owner_type", "product_id", "price", "quantity"),
}


def build_benchmark_evaluation_cases() -> List[Dict[str, Any]]:
    """Return a small benchmark-oriented prose evaluation set."""

    return [
        {
            "name": "canonical_case_a",
            "label": "Canonical Case A",
            "prose": (
                "There are two nodes, N1 and N2. Supplier S1 at N1 can supply up to 100 units "
                "of product P1. Consumer C1 at N2 is willing to buy up to 50 units of P1 at price 20. "
                "Supplier S1 offers P1 at price 10. A free transport link T1 carries P1 from N1 to N2 with capacity 100."
            ),
            "expected_state": _build_expected_state(
                problem_title="Canonical Case A",
                nodes=[{"id": "N1"}, {"id": "N2"}],
                products=[{"id": "P1"}],
                suppliers=[{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
                consumers=[{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
                transport_links=[{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 100.0, "cost": 0.0}],
                bids=[
                    {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0, "quantity": 100.0},
                    {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0},
                ],
            ),
        },
        {
            "name": "paraphrased_case_a",
            "label": "Paraphrased Case A",
            "prose": (
                "Think of a simple two-location market. At N1, supplier S1 can provide as many as 100 units of P1 "
                "and asks 10 per unit. At N2, consumer C1 would take at most 50 units of that same product and would pay 20. "
                "The commodity can move at zero cost on link T1 from N1 to N2, with a shipping capacity of 100."
            ),
            "expected_state": _build_expected_state(
                problem_title="Paraphrased Case A",
                nodes=[{"id": "N1"}, {"id": "N2"}],
                products=[{"id": "P1"}],
                suppliers=[{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
                consumers=[{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
                transport_links=[{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 100.0, "cost": 0.0}],
                bids=[
                    {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0, "quantity": 100.0},
                    {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0},
                ],
            ),
        },
        {
            "name": "incomplete_case_a",
            "label": "Incomplete Case A",
            "prose": (
                "There are nodes N1 and N2. Supplier S1 at N1 can supply product P1. "
                "Consumer C1 at N2 wants P1. A transport link goes from N1 to N2."
            ),
            "expected_state": _build_expected_state(
                problem_title="Incomplete Case A",
                nodes=[{"id": "N1"}, {"id": "N2"}],
                products=[{"id": "P1"}],
                suppliers=[{"id": "S1", "node": "N1", "product": "P1", "capacity": None}],
                consumers=[{"id": "C1", "node": "N2", "product": "P1", "capacity": None}],
                transport_links=[{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": None, "cost": None}],
                bids=[],
            ),
        },
        {
            "name": "ambiguous_case_a",
            "label": "Ambiguous Case A",
            "prose": (
                "At N1, supplier S1 can release up to 100 units of P1. At N2, buyer C1 would take 50 units "
                "of that product for 20. The material can move across the network from N1 to N2, and S1's offer is 10."
            ),
            "expected_state": _build_expected_state(
                problem_title="Ambiguous Case A",
                nodes=[{"id": "N1"}, {"id": "N2"}],
                products=[{"id": "P1"}],
                suppliers=[{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
                consumers=[{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
                transport_links=[{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": None, "cost": None}],
                bids=[
                    {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0, "quantity": None},
                    {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0},
                ],
            ),
        },
        {
            "name": "negative_bid_case_b",
            "label": "Negative-Bid Case B",
            "prose": (
                "At node N1, supplier S1 can provide up to 40 units of waste product P1 and is willing to pay 5 "
                "per unit to have it accepted, so its bid price is -5. At node N2, consumer C1 can accept up to 40 units "
                "of P1 at bid price 1. Free transport link T1 moves P1 from N1 to N2 with capacity 40."
            ),
            "expected_state": _build_expected_state(
                problem_title="Negative-Bid Case B",
                nodes=[{"id": "N1"}, {"id": "N2"}],
                products=[{"id": "P1"}],
                suppliers=[{"id": "S1", "node": "N1", "product": "P1", "capacity": 40.0}],
                consumers=[{"id": "C1", "node": "N2", "product": "P1", "capacity": 40.0}],
                transport_links=[{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 40.0, "cost": 0.0}],
                bids=[
                    {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": -5.0, "quantity": 40.0},
                    {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 1.0, "quantity": 40.0},
                ],
            ),
        },
        {
            "name": "transformation_case_c",
            "label": "Transformation Case C",
            "prose": (
                "Node N1 has supplier S1 that can provide up to 60 units of feedstock P1 at price 4. "
                "At the same node, zero-cost technology K1 can process up to 60 units and converts 1 unit of P1 into 0.8 units of P2. "
                "Consumer C1 at node N2 will buy up to 40 units of P2 at price 15. Free transport link T2 ships P2 from N1 to N2 with capacity 40."
            ),
            "expected_state": _build_expected_state(
                problem_title="Transformation Case C",
                nodes=[{"id": "N1"}, {"id": "N2"}],
                products=[{"id": "P1"}, {"id": "P2"}],
                suppliers=[{"id": "S1", "node": "N1", "product": "P1", "capacity": 60.0}],
                consumers=[{"id": "C1", "node": "N2", "product": "P2", "capacity": 40.0}],
                transport_links=[{"id": "T2", "origin": "N1", "destination": "N2", "product": "P2", "capacity": 40.0, "cost": 0.0}],
                technologies=[{"id": "K1", "node": "N1", "capacity": 60.0, "cost": 0.0, "yield_coefficients": {"P1": -1.0, "P2": 0.8}}],
                bids=[
                    {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 4.0, "quantity": 60.0},
                    {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P2", "price": 15.0, "quantity": 40.0},
                ],
            ),
        },
    ]


def compare_problem_states(expected: ProblemState, actual: ProblemState) -> Dict[str, Any]:
    """Compare two states and report lightweight structural mismatches."""

    passed_fields: List[Dict[str, Any]] = []
    failed_fields: List[Dict[str, Any]] = []
    missing_fields: List[Dict[str, Any]] = []
    extra_fields: List[Dict[str, Any]] = []
    error_categories: List[str] = []

    for collection in ENTITY_COLLECTIONS:
        expected_entities = _entity_map(getattr(expected, collection))
        actual_entities = _entity_map(getattr(actual, collection))

        for entity_id, expected_entity in expected_entities.items():
            path_prefix = f"{collection}.{entity_id}"
            actual_entity = actual_entities.get(entity_id)
            if actual_entity is None:
                missing_fields.append({"path": path_prefix, "expected": expected_entity})
                error_categories.append("missing_entity")
                continue

            for field_name in FIELD_MAP[collection]:
                expected_value = expected_entity.get(field_name)
                actual_value = actual_entity.get(field_name)
                field_path = f"{path_prefix}.{field_name}"

                if expected_value is None and actual_value is None:
                    passed_fields.append({"path": field_path, "expected": None, "actual": None})
                    continue
                if expected_value is None and actual_value is not None:
                    extra_fields.append({"path": field_path, "expected": None, "actual": actual_value})
                    error_categories.append(_extra_field_category(field_name))
                    continue
                if expected_value is not None and actual_value is None:
                    missing_fields.append({"path": field_path, "expected": expected_value, "actual": None})
                    error_categories.append(_missing_field_category(field_name))
                    continue
                if _values_match(expected_value, actual_value):
                    passed_fields.append({"path": field_path, "expected": expected_value, "actual": actual_value})
                else:
                    failed_fields.append({"path": field_path, "expected": expected_value, "actual": actual_value})
                    error_categories.append(_failed_field_category(field_name, expected_value, actual_value))

        for entity_id, actual_entity in actual_entities.items():
            if entity_id not in expected_entities:
                extra_fields.append({"path": f"{collection}.{entity_id}", "actual": actual_entity})
                error_categories.append("extra_entity")

    return {
        "passed_fields": passed_fields,
        "failed_fields": failed_fields,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "error_categories": sorted(error_categories),
        "error_category_counts": dict(sorted(Counter(error_categories).items())),
    }


def evaluate_benchmark_cases(mode: str = "guided") -> Dict[str, Any]:
    """Run the prose interpreter evaluation set if Gemini is configured."""

    cases = build_benchmark_evaluation_cases()
    if not _gemini_is_configured():
        return {
            "ran": False,
            "cases": [],
            "summary": (
                "No live evaluation was run because GEMINI_API_KEY and LLM_PROVIDER=gemini "
                "were not both available."
            ),
            "dominant_failure_modes": [],
        }

    results = [evaluate_single_case(case, mode=mode) for case in cases]
    dominant_failure_modes = summarize_failure_modes(results)
    return {
        "ran": True,
        "cases": results,
        "summary": _format_failure_mode_summary(dominant_failure_modes),
        "dominant_failure_modes": dominant_failure_modes,
    }


def evaluate_single_case(case: Dict[str, Any], mode: str = "guided") -> Dict[str, Any]:
    """Run one evaluation case through the live prose-to-solve flow."""

    session_result = run_chatbot_session(
        state=ProblemState(),
        user_message=case["prose"],
        mode=mode,
        use_llm=True,
    )

    interpreted_state = session_result.get("state", ProblemState())
    comparison = compare_problem_states(case["expected_state"], interpreted_state)
    validation = session_result.get("validation_result") or validate_state(interpreted_state)
    solve_result = session_result.get("solve_result", {}) or {}

    return {
        "name": case["name"],
        "label": case["label"],
        "prose_input": case["prose"],
        "semantic_plan": session_result.get("semantic_plan"),
        "problem_state_summary": summarize_problem_state(interpreted_state),
        "validation_result": validation,
        "solve_succeeded": bool(solve_result.get("success", False)),
        "solve_result": solve_result,
        "explanation_output": session_result.get("response", ""),
        "comparison": comparison,
    }


def summarize_failure_modes(case_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate failure categories across cases."""

    counter: Counter[str] = Counter()
    case_hits: Counter[str] = Counter()
    for case in case_results:
        categories = list(case.get("comparison", {}).get("error_categories", []))
        if not categories:
            continue
        category_counter = Counter(categories)
        counter.update(category_counter)
        case_hits.update(set(categories))

    return [
        {"category": category, "count": count, "cases": case_hits.get(category, 0)}
        for category, count in counter.most_common()
    ]


def format_evaluation_report(report: Dict[str, Any]) -> str:
    """Return a readable text summary for notebooks or scripts."""

    if not report.get("ran"):
        return report["summary"]

    lines = [report["summary"], ""]
    for case in report["cases"]:
        lines.append(f"=== {case['label']} ===")
        lines.append(f"Solve succeeded: {case['solve_succeeded']}")
        lines.append("Error categories: " + json.dumps(case["comparison"]["error_category_counts"], indent=2))
        lines.append("")
    return "\n".join(lines).strip()


def _build_expected_state(**plan: Any) -> ProblemState:
    from .llm_problem_interpreter import build_state_from_semantic_plan

    normalized_plan = {
        "problem_title": plan.get("problem_title", "Benchmark Evaluation Case"),
        "nodes": plan.get("nodes", []),
        "products": plan.get("products", []),
        "suppliers": plan.get("suppliers", []),
        "consumers": plan.get("consumers", []),
        "transport_links": plan.get("transport_links", []),
        "bids": plan.get("bids", []),
        "technologies": plan.get("technologies", []),
    }
    return build_state_from_semantic_plan(normalized_plan)


def _entity_map(entities: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        raw = entity.model_dump() if hasattr(entity, "model_dump") else entity.dict()
        result[str(raw["id"])] = raw
    return result


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= 1e-9
    return expected == actual


def _failed_field_category(field_name: str, expected: Any, actual: Any) -> str:
    if field_name in {"node", "origin", "destination"}:
        return "wrong_node_mapping"
    if field_name in {"product", "product_id"}:
        return "wrong_product_mapping"
    if field_name == "owner_id":
        return "wrong_owner_relation"
    if field_name in {"capacity", "quantity", "price"} and isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return "wrong_numeric_value"
    return "wrong_numeric_value" if isinstance(expected, (int, float)) or isinstance(actual, (int, float)) else "wrong_value"


def _missing_field_category(field_name: str) -> str:
    return {
        "capacity": "missing_capacity",
        "quantity": "missing_quantity",
        "price": "missing_price",
    }.get(field_name, "missing_field")


def _extra_field_category(field_name: str) -> str:
    return "extra_field"


def _gemini_is_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY")) and os.getenv("LLM_PROVIDER", "").lower() == "gemini"


def _format_failure_mode_summary(dominant_failure_modes: List[Dict[str, Any]]) -> str:
    if not dominant_failure_modes:
        return "No structural mismatches were observed across the evaluated cases."

    preview = ", ".join(
        f"{item['category']} ({item['count']})"
        for item in dominant_failure_modes[:4]
    )
    return f"Dominant observed failure modes: {preview}."


__all__ = [
    "build_benchmark_evaluation_cases",
    "compare_problem_states",
    "evaluate_benchmark_cases",
    "evaluate_single_case",
    "format_evaluation_report",
    "summarize_failure_modes",
]
