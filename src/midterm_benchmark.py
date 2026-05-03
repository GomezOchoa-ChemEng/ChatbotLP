"""Focused benchmark layer for Midterm 1, Problem 1: manure management.

This module evaluates the existing ChatbotLP flow on a realistic midterm-style
problem without changing the Case A/B/C paper benchmark behavior:

natural prose -> semantic plan -> ProblemState -> validation -> solve -> reasoning
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .chatbot_engine import run_chatbot_session
from .llm_problem_interpreter import (
    build_problem_artifacts_from_semantic_plan,
    build_state_from_semantic_plan,
    interpret_problem_from_text,
    summarize_problem_state,
)
from .model_builder import build_model_from_state
from .schema import ProblemState
from .solver import solve_model
from .solver_results import SolverResults
from .validator import validate_state


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "Benchmarks" / "midterm1" / "manure_q1"
TOLERANCE = 1e-6

MIDTERM_REASONING_PROMPTS: List[Dict[str, str]] = [
    {
        "id": "primal_lp",
        "label": "Primal LP formulation",
        "prompt": "Formulate this manure management problem as a primal linear program in LaTeX.",
    },
    {
        "id": "economic_intuition",
        "label": "Economic intuition",
        "prompt": "Explain the optimal allocation using simple economic intuition.",
    },
    {
        "id": "all_manure",
        "label": "All manure feasibility",
        "prompt": "Can the dairy farmer get rid of all its manure? Why or why not?",
    },
    {
        "id": "node_balances",
        "label": "Node balance verification",
        "prompt": "Verify that all node balances hold.",
    },
]

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
    "nodes": (),
    "products": (),
    "suppliers": ("node", "product", "capacity"),
    "consumers": ("node", "product", "capacity"),
    "transport_links": ("origin", "destination", "product", "cost"),
    "technologies": ("node", "capacity", "yield_coefficients"),
    "bids": ("owner_id", "owner_type", "product_id", "price", "quantity"),
}

CANONICAL_SUPPLY_ID = "Dairy/EauClaire"
CANONICAL_MENOMONIE_DEMAND_ID = "Menomonie"
CANONICAL_BLACK_RIVER_DEMAND_ID = "BlackRiverFalls"
CANONICAL_MANURE_PRODUCT_ID = "Manure"

REFERENCE_COMPONENT_IDS = {
    "accepted_supply": {
        CANONICAL_SUPPLY_ID: CANONICAL_SUPPLY_ID,
    },
    "accepted_demands": {
        CANONICAL_MENOMONIE_DEMAND_ID: CANONICAL_MENOMONIE_DEMAND_ID,
        CANONICAL_BLACK_RIVER_DEMAND_ID: CANONICAL_BLACK_RIVER_DEMAND_ID,
    },
    "transport_flows": {
        "EauClaire_to_Menomonie": "EauClaire_to_Menomonie",
        "EauClaire_to_BlackRiverFalls": "EauClaire_to_BlackRiverFalls",
    },
}


@dataclass(frozen=True)
class MidtermBenchmarkConfig:
    """Runtime switches for the manure Q1 benchmark."""

    mode: str = "guided"
    prompt_ids: Optional[Tuple[str, ...]] = None
    use_llm: bool = True
    use_llm_for_reasoning: bool = True
    fallback_to_reference_fixture: bool = True
    attempt_solve: bool = True
    run_reasoning: bool = True


def load_benchmark_files(benchmark_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the benchmark text, prompt set, and reference solution."""

    directory = Path(benchmark_dir or DEFAULT_BENCHMARK_DIR)
    problem_statement_path = directory / "problem_statement.md"
    reference_solution_path = directory / "reference_solution.json"
    prompts_path = directory / "prompts.json"

    return {
        "benchmark_dir": directory,
        "problem_statement": problem_statement_path.read_text(encoding="utf-8"),
        "reference_solution": json.loads(reference_solution_path.read_text(encoding="utf-8")),
        "prompts": json.loads(prompts_path.read_text(encoding="utf-8"))["prompts"],
    }


def build_midterm_manure_expected_plan(prompt_id: str = "canonical") -> Dict[str, Any]:
    """Return the deterministic fixture plan used for local no-LLM runs."""

    incomplete = prompt_id == "incomplete"
    menomonie_capacity = None if incomplete else 500.0
    black_river_capacity = None if incomplete else 500.0
    missing_information = []
    ambiguities = []
    if incomplete:
        missing_information = [
            "maximum manure capacity for the Menomonie receiving farm",
            "maximum manure capacity for the Black River Falls receiving farm",
        ]

    return {
        "problem_title": "Midterm 1 Problem 1 Question 1: Manure Management",
        "problem_type": "case_a",
        "nodes": [
            {"id": "EauClaire", "name": "Eau Claire"},
            {"id": "Menomonie", "name": "Menomonie"},
            {"id": "BlackRiverFalls", "name": "Black River Falls"},
        ],
        "products": [{"id": "Manure", "name": "waste manure"}],
        "suppliers": [
            {
                "id": "Dairy/EauClaire",
                "node": "EauClaire",
                "product": "Manure",
                "capacity": 1000.0,
            }
        ],
        "consumers": [
            {
                "id": "Menomonie",
                "node": "Menomonie",
                "product": "Manure",
                "capacity": menomonie_capacity,
            },
            {
                "id": "BlackRiverFalls",
                "node": "BlackRiverFalls",
                "product": "Manure",
                "capacity": black_river_capacity,
            },
        ],
        "transport_links": [
            {
                "id": "EauClaire_to_Menomonie",
                "origin": "EauClaire",
                "destination": "Menomonie",
                "product": "Manure",
                "capacity": None,
                "cost": 0.1,
            },
            {
                "id": "EauClaire_to_BlackRiverFalls",
                "origin": "EauClaire",
                "destination": "BlackRiverFalls",
                "product": "Manure",
                "capacity": None,
                "cost": 0.2,
            },
        ],
        "technologies": [],
        "bids": [
            {
                "id": "B_Dairy_EauClaire",
                "owner_id": "Dairy/EauClaire",
                "owner_type": "supplier",
                "product_id": "Manure",
                "price": 0.0,
                "quantity": 1000.0,
            },
            {
                "id": "B_Menomonie",
                "owner_id": "Menomonie",
                "owner_type": "consumer",
                "product_id": "Manure",
                "price": 0.5,
                "quantity": menomonie_capacity,
            },
            {
                "id": "B_BlackRiverFalls",
                "owner_id": "BlackRiverFalls",
                "owner_type": "consumer",
                "product_id": "Manure",
                "price": 1.5,
                "quantity": black_river_capacity,
            },
        ],
        "missing_information": missing_information,
        "ambiguities": ambiguities,
    }


def build_midterm_manure_cases(
    prompt_ids: Optional[Sequence[str]] = None,
    benchmark_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build prompt cases with deterministic expected ProblemState fixtures."""

    files = load_benchmark_files(benchmark_dir)
    prompts = list(files["prompts"])
    by_id = {prompt["id"]: prompt for prompt in prompts}
    requested = tuple(prompt_ids) if prompt_ids is not None else tuple(by_id)
    unknown = [prompt_id for prompt_id in requested if prompt_id not in by_id]
    if unknown:
        valid = ", ".join(sorted(by_id))
        raise ValueError(f"Unknown prompt id(s): {', '.join(unknown)}. Valid ids: {valid}")

    cases = []
    for prompt_id in requested:
        prompt = by_id[prompt_id]
        expected_plan = build_midterm_manure_expected_plan(prompt_id)
        expected_state = build_state_from_semantic_plan(expected_plan)
        cases.append(
            {
                "name": prompt["id"],
                "label": prompt["label"],
                "prompt_id": prompt["id"],
                "prose": prompt["text"],
                "expected_solver_ready": bool(prompt.get("expected_solver_ready", True)),
                "expected_plan": expected_plan,
                "expected_state": expected_state,
            }
        )
    return cases


def run_midterm_manure_q1_benchmark(
    config: Optional[MidtermBenchmarkConfig] = None,
    benchmark_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the manure Q1 benchmark and return raw results plus tables."""

    runtime_config = config or MidtermBenchmarkConfig()
    files = load_benchmark_files(benchmark_dir)
    reference_solution = files["reference_solution"]
    cases = build_midterm_manure_cases(
        prompt_ids=runtime_config.prompt_ids,
        benchmark_dir=benchmark_dir,
    )
    case_results = [
        evaluate_midterm_prompt_case(case, reference_solution, runtime_config)
        for case in cases
    ]
    tables = build_midterm_output_tables(case_results)
    return {
        "metadata": {
            "benchmark_id": reference_solution.get("benchmark_id", "midterm1_manure_q1"),
            "benchmark_dir": str(files["benchmark_dir"]),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "gemini_model": os.getenv("GEMINI_MODEL"),
            "gemini_configured": gemini_is_configured(),
            "config": runtime_config.__dict__,
        },
        "problem_statement": files["problem_statement"],
        "reference_solution": reference_solution,
        "cases": case_results,
        "tables": tables,
    }


def evaluate_midterm_prompt_case(
    case: Dict[str, Any],
    reference_solution: Dict[str, Any],
    config: Optional[MidtermBenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Evaluate one manure prompt through interpretation, validation, solve, and reasoning."""

    runtime_config = config or MidtermBenchmarkConfig()
    interpretation = _run_interpretation(case, runtime_config)
    actual_state = interpretation.get("state")
    problem_state_created = isinstance(actual_state, ProblemState)

    validation = (
        validate_state(actual_state)
        if problem_state_created
        else {
            "solver_ready": False,
            "missing_parameters": ["ProblemState was not created"],
            "invalid_references": [],
            "incomplete_technologies": [],
            "issues": [],
        }
    )

    state_comparison = (
        compare_problem_states_for_midterm(case["expected_state"], actual_state)
        if problem_state_created
        else _empty_failed_state_comparison("ProblemState was not created")
    )

    solve_result, solver_results = _select_or_run_solve_result(
        state=actual_state,
        validation=validation,
        attempt_solve=runtime_config.attempt_solve,
    )
    solution_checks = compare_solution_to_reference(
        state=actual_state,
        solve_result=solve_result,
        reference_solution=reference_solution,
    )
    primary_metrics = evaluate_primary_semantic_metrics(
        state=actual_state,
        solve_result=solve_result,
        reference_solution=reference_solution,
    )

    reasoning_results: List[Dict[str, Any]] = []
    if (
        runtime_config.run_reasoning
        and problem_state_created
        and bool(validation.get("solver_ready", False))
    ):
        reasoning_results = run_midterm_reasoning_prompt_battery(
            state=actual_state,
            case_name=case["name"],
            mode=runtime_config.mode,
            use_llm=runtime_config.use_llm_for_reasoning and gemini_is_configured(),
        )

    explanation_generated = any(row.get("success", False) for row in reasoning_results)
    solver_ready_actual = bool(validation.get("solver_ready", False))

    return {
        "name": case["name"],
        "label": case["label"],
        "prompt_id": case["prompt_id"],
        "prose_input": case["prose"],
        "expected_solver_ready": bool(case["expected_solver_ready"]),
        "semantic_plan": interpretation.get("semantic_plan"),
        "semantic_plan_created": bool(interpretation.get("semantic_plan")),
        "problem_state": actual_state,
        "problem_state_created": problem_state_created,
        "problem_state_summary": summarize_problem_state(actual_state) if problem_state_created else None,
        "validation_result": validation,
        "solver_ready": solver_ready_actual,
        "solver_ready_correct": solver_ready_actual == bool(case["expected_solver_ready"]),
        "solve_result": solve_result,
        "solver_results": solver_results,
        "solve_success": bool(solve_result.get("success", False)),
        "solution_checks": solution_checks,
        "primary_metrics": primary_metrics,
        "state_comparison": state_comparison,
        "reasoning_results": reasoning_results,
        "explanation_generated": explanation_generated,
        "blocking_errors": state_comparison["blocking_errors"],
        "benign_extra_name_fields": state_comparison["benign_extra_name_fields"],
        "interpretation_metadata": interpretation.get("metadata", {}),
    }


def compare_problem_states_for_midterm(
    expected: ProblemState,
    actual: ProblemState,
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    """Compare expected and actual states for the manure benchmark.

    Entity IDs are compared semantically for this benchmark. For example,
    ``S_DAIRY`` and ``Dairy/EauClaire`` are the same supplier role if they
    resolve to the Eau Claire manure source. Numeric and relationship mistakes
    remain blocking.
    """

    passed_fields: List[Dict[str, Any]] = []
    benign_identifier_mismatches: List[Dict[str, Any]] = []
    blocking_errors: List[Dict[str, Any]] = []
    error_categories: List[str] = []

    for collection in ENTITY_COLLECTIONS:
        expected_entities = _semantic_entity_map(collection, expected)
        actual_entities = _semantic_entity_map(collection, actual)

        for semantic_key, expected_entity in expected_entities.items():
            path_prefix = f"{collection}.{_format_semantic_key(semantic_key)}"
            actual_entity = actual_entities.get(semantic_key)
            if actual_entity is None:
                _append_error(
                    blocking_errors,
                    error_categories,
                    "missing_entity",
                    path_prefix,
                    expected=expected_entity,
                    actual=None,
                )
                continue

            _record_benign_identifier_mismatch(
                collection=collection,
                semantic_key=semantic_key,
                expected_entity=expected_entity,
                actual_entity=actual_entity,
                benign_identifier_mismatches=benign_identifier_mismatches,
                error_categories=error_categories,
            )

            for field_name in _semantic_fields_for_collection(collection):
                expected_value = _semantic_field_value(collection, field_name, expected_entity)
                actual_value = _semantic_field_value(collection, field_name, actual_entity)
                field_path = f"{path_prefix}.{field_name}"

                if expected_value is None and actual_value is None:
                    passed_fields.append({"path": field_path, "expected": None, "actual": None})
                    continue
                if expected_value is None and actual_value is not None:
                    _append_error(
                        blocking_errors,
                        error_categories,
                        "invented_value",
                        field_path,
                        expected=expected_value,
                        actual=actual_value,
                    )
                    continue
                if expected_value is not None and actual_value is None:
                    _append_error(
                        blocking_errors,
                        error_categories,
                        _missing_field_category(field_name),
                        field_path,
                        expected=expected_value,
                        actual=actual_value,
                    )
                    continue
                if _values_match(expected_value, actual_value, tolerance=tolerance):
                    passed_fields.append(
                        {"path": field_path, "expected": expected_value, "actual": actual_value}
                    )
                else:
                    _append_error(
                        blocking_errors,
                        error_categories,
                        _failed_field_category(field_name, expected_value, actual_value),
                        field_path,
                        expected=expected_value,
                        actual=actual_value,
                    )

        for semantic_key, actual_entity in actual_entities.items():
            if semantic_key not in expected_entities:
                _append_error(
                    blocking_errors,
                    error_categories,
                    "extra_invented_entity",
                    f"{collection}.{_format_semantic_key(semantic_key)}",
                    expected=None,
                    actual=actual_entity,
                )

    blocking_categories = [item["category"] for item in blocking_errors]
    return {
        "passed_fields": passed_fields,
        "benign_extra_name_fields": benign_identifier_mismatches,
        "benign_identifier_mismatches": benign_identifier_mismatches,
        "blocking_errors": blocking_errors,
        "structural_match": len(blocking_errors) == 0,
        "error_categories": sorted(error_categories),
        "blocking_error_categories": sorted(blocking_categories),
        "error_category_counts": dict(sorted(Counter(error_categories).items())),
        "blocking_error_category_counts": dict(sorted(Counter(blocking_categories).items())),
    }


def compare_solution_to_reference(
    state: Optional[ProblemState],
    solve_result: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    """Compare objective, accepted quantities, flows, and balances to reference."""

    solve_success = bool(solve_result.get("success", False))
    objective_value = solve_result.get("objective_value")
    expected_objective = reference_solution.get("objective_value")
    objective_match = None
    if solve_success and objective_value is not None and expected_objective is not None:
        objective_match = abs(float(objective_value) - float(expected_objective)) <= tolerance

    components = extract_midterm_solution_components(state, solve_result)
    accepted_supply_match = _dict_matches_expected(
        components["accepted_supply"],
        reference_solution.get("accepted_supply", {}),
        tolerance,
    )
    accepted_demand_match = _dict_matches_expected(
        components["accepted_demands"],
        reference_solution.get("accepted_demands", {}),
        tolerance,
    )
    transport_flow_match = _dict_matches_expected(
        components["transport_flows"],
        reference_solution.get("transport_flows", {}),
        tolerance,
    )
    balance_match = _balance_checks_match(
        components["balance_checks"],
        reference_solution.get("balance_checks", {}),
        tolerance,
    )
    demand_revenue_match = _float_or_none_matches(
        components.get("demand_revenue"),
        reference_solution.get("demand_revenue"),
        tolerance,
    )
    transport_cost_match = _float_or_none_matches(
        components.get("transport_cost"),
        reference_solution.get("transport_cost"),
        tolerance,
    )
    supply_cost_match = _float_or_none_matches(
        components.get("supply_cost"),
        reference_solution.get("supply_cost"),
        tolerance,
    )
    accepted_supply_total_match = _float_or_none_matches(
        sum(components["accepted_supply"].values()),
        sum(reference_solution.get("accepted_supply", {}).values()),
        tolerance,
    )
    diagnostics = _build_solution_diagnostics(components, reference_solution, tolerance)

    return {
        "solve_success": solve_success,
        "status": solve_result.get("status"),
        "termination_condition": solve_result.get("termination_condition"),
        "solver_name": solve_result.get("solver_name"),
        "expected_objective": expected_objective,
        "actual_objective": objective_value,
        "objective_match": objective_match,
        "accepted_supply_match": accepted_supply_match,
        "accepted_demand_match": accepted_demand_match,
        "transport_flow_match": transport_flow_match,
        "balance_match": balance_match,
        "demand_revenue_match": demand_revenue_match,
        "transport_cost_match": transport_cost_match,
        "supply_cost_match": supply_cost_match,
        "accepted_supply_total_match": accepted_supply_total_match,
        "actual_components": components,
        "diagnostics": diagnostics,
        "solver_message": solve_result.get("message"),
    }


def evaluate_primary_semantic_metrics(
    state: Optional[ProblemState],
    solve_result: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    """Evaluate Midterm Q1 without depending on exact entity IDs."""

    if not isinstance(state, ProblemState):
        empty = _empty_primary_metrics("ProblemState was not created")
        return empty

    count_rows = _semantic_count_metric_rows(state)
    parameter_rows = _parameter_multiset_metric_rows(state, solve_result, tolerance)
    topology_rows = _topology_metric_rows(state)
    route_rows = _route_economics_metric_rows(state, tolerance)
    solver_rows = _solver_aggregate_metric_rows(state, solve_result, reference_solution, tolerance)
    balance_rows = _balance_residual_metric_rows(state, solve_result, tolerance)

    semantic_structure_pass = all(
        bool(row["pass"])
        for rows in (count_rows, parameter_rows, topology_rows, route_rows)
        for row in rows
    )
    solver_aggregate_pass = all(bool(row["pass"]) for row in solver_rows)
    balance_residual_pass = all(bool(row["pass"]) for row in balance_rows)
    primary_success = semantic_structure_pass and solver_aggregate_pass and balance_residual_pass

    return {
        "semantic_count_metrics": count_rows,
        "parameter_multiset_metrics": parameter_rows,
        "topology_metrics": topology_rows,
        "route_economics_metrics": route_rows,
        "solver_aggregate_metrics": solver_rows,
        "balance_residual_metrics": balance_rows,
        "semantic_structure_pass": semantic_structure_pass,
        "solver_aggregate_pass": solver_aggregate_pass,
        "balance_residual_pass": balance_residual_pass,
        "primary_success": primary_success,
    }


def extract_midterm_solution_components(
    state: Optional[ProblemState],
    solve_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract manure-specific economic components from a solved ProblemState."""

    if not isinstance(state, ProblemState) or not solve_result.get("success", False):
        return {
            "accepted_supply": {},
            "accepted_demands": {},
            "transport_flows": {},
            "demand_revenue": None,
            "transport_cost": None,
            "supply_cost": None,
            "balance_checks": {},
            "raw_component_rows": [],
        }

    solution = solve_result.get("solution", {}) if isinstance(solve_result, dict) else {}
    q_values = _solution_block(solution, "q")
    f_values = _solution_block(solution, "f")

    bids_by_id = {bid.id: bid for bid in state.bids}
    suppliers_by_id = {supplier.id: supplier for supplier in state.suppliers}
    consumers_by_id = {consumer.id: consumer for consumer in state.consumers}

    accepted_supply: Dict[str, float] = {}
    accepted_demands: Dict[str, float] = {}
    demand_revenue = 0.0
    supply_cost = 0.0
    raw_component_rows: List[Dict[str, Any]] = []

    for bid_id, quantity in q_values.items():
        bid = bids_by_id.get(str(bid_id))
        if bid is None:
            continue
        if bid.owner_type == "supplier":
            supplier = suppliers_by_id.get(bid.owner_id)
            label = _supplier_reference_key(supplier, bid.owner_id, bid.product_id)
            accepted_supply[label] = accepted_supply.get(label, 0.0) + quantity
            supply_cost += float(bid.price) * quantity
            raw_component_rows.append(
                {
                    "component": "accepted_supply",
                    "raw_id": bid.owner_id,
                    "raw_bid_id": bid.id,
                    "raw_node": getattr(supplier, "node", None),
                    "raw_product": bid.product_id,
                    "canonical_resolved_id": label,
                    "actual_value": quantity,
                }
            )
        elif bid.owner_type == "consumer":
            consumer = consumers_by_id.get(bid.owner_id)
            label = _consumer_reference_key(consumer, bid.owner_id, bid.product_id)
            accepted_demands[label] = accepted_demands.get(label, 0.0) + quantity
            demand_revenue += float(bid.price) * quantity
            raw_component_rows.append(
                {
                    "component": "accepted_demands",
                    "raw_id": bid.owner_id,
                    "raw_bid_id": bid.id,
                    "raw_node": getattr(consumer, "node", None),
                    "raw_product": bid.product_id,
                    "canonical_resolved_id": label,
                    "actual_value": quantity,
                }
            )

    transport_flows: Dict[str, float] = {}
    transport_cost = 0.0
    for activity in _transport_activity_rows(state, f_values):
        origin = activity["origin"]
        destination = activity["destination"]
        product_id = activity["product"]
        quantity = activity["flow"]
        route_key = _route_reference_key(origin, destination, product_id)
        transport_flows[route_key] = transport_flows.get(route_key, 0.0) + quantity
        transport_cost += activity["cost"] * quantity
        raw_component_rows.append(
            {
                "component": "transport_flows",
                "raw_id": activity["link_id"],
                "raw_arc": activity["raw_key"],
                "raw_origin": origin,
                "raw_destination": destination,
                "raw_product": product_id,
                "canonical_resolved_id": route_key,
                "actual_value": quantity,
            }
        )

    balance_checks = _compute_midterm_balance_checks(
        accepted_supply=accepted_supply,
        accepted_demands=accepted_demands,
        transport_flows=transport_flows,
    )

    return {
        "accepted_supply": accepted_supply,
        "accepted_demands": accepted_demands,
        "transport_flows": transport_flows,
        "demand_revenue": demand_revenue,
        "transport_cost": transport_cost,
        "supply_cost": supply_cost,
        "balance_checks": balance_checks,
        "raw_component_rows": raw_component_rows,
    }


def run_midterm_reasoning_prompt_battery(
    state: ProblemState,
    case_name: str,
    mode: str = "guided",
    use_llm: bool = False,
) -> List[Dict[str, Any]]:
    """Run the requested manure-specific reasoning prompts."""

    rows = []
    for prompt_spec in MIDTERM_REASONING_PROMPTS:
        result = run_chatbot_session(
            state=_copy_problem_state(state),
            user_message=prompt_spec["prompt"],
            mode=mode,
            use_llm=use_llm,
        )
        metadata = result.get("response_metadata", {}) or {}
        response_text = result.get("response", "") or ""
        success = (
            bool(result.get("success", False))
            and bool(str(response_text).strip())
            and not metadata.get("validation_fatal")
            and result.get("intent") != "problem_formulation"
        )
        rows.append(
            {
                "case": case_name,
                "prompt_id": prompt_spec["id"],
                "prompt_label": prompt_spec["label"],
                "success": success,
                "intent": result.get("intent"),
                "render_mode": result.get("render_mode"),
                "response_source": metadata.get("response_source"),
                "fallback_triggered": metadata.get("fallback_triggered"),
                "fallback_reason": metadata.get("fallback_reason"),
                "llm_exception_type": metadata.get("llm_exception_type"),
                "response_preview": str(response_text)[:240],
            }
        )
    return rows


def build_midterm_output_tables(case_results: Sequence[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Create notebook/script friendly output tables."""

    case_summary_rows = []
    solve_rows = []
    interpretation_rows = []
    reasoning_rows = []
    metadata_rows = []
    diagnostic_rows = []
    count_rows = []
    parameter_rows = []
    topology_rows = []
    route_rows = []
    aggregate_rows = []
    residual_rows = []

    for result in case_results:
        checks = result["solution_checks"]
        primary_metrics = result.get("primary_metrics", {})
        comparison = result["state_comparison"]
        metadata = result["interpretation_metadata"]
        blocking_count = len(comparison["blocking_errors"])
        benign_count = len(comparison["benign_extra_name_fields"])
        benign_identifier_count = len(comparison.get("benign_identifier_mismatches", []))
        semantic_structure_pass = bool(primary_metrics.get("semantic_structure_pass", False))
        solver_aggregate_pass = bool(primary_metrics.get("solver_aggregate_pass", False))
        balance_residual_pass = bool(primary_metrics.get("balance_residual_pass", False))
        primary_success = bool(primary_metrics.get("primary_success", False))

        pass_flags = [
            result["semantic_plan_created"],
            result["problem_state_created"],
            semantic_structure_pass,
            result["solver_ready_correct"],
        ]
        if result["expected_solver_ready"]:
            pass_flags.extend(
                [
                    result["solve_success"],
                    solver_aggregate_pass,
                    balance_residual_pass,
                ]
            )

        case_summary_rows.append(
            {
                "prompt_id": result["prompt_id"],
                "label": result["label"],
                "semantic_plan_created": result["semantic_plan_created"],
                "problem_state_created": result["problem_state_created"],
                "structural_match": semantic_structure_pass,
                "semantic_structure_pass": semantic_structure_pass,
                "solver_aggregate_pass": solver_aggregate_pass,
                "balance_residual_pass": balance_residual_pass,
                "primary_success": primary_success,
                "alias_structural_match": comparison["structural_match"],
                "blocking_error_count": _primary_failure_count(primary_metrics),
                "alias_blocking_error_count": blocking_count,
                "benign_extra_name_fields": benign_count,
                "benign_identifier_mismatches": benign_identifier_count,
                "solver_ready_expected": result["expected_solver_ready"],
                "solver_ready_actual": result["solver_ready"],
                "solver_ready_correct": result["solver_ready_correct"],
                "solve_success": result["solve_success"],
                "objective_match": checks["objective_match"],
                "transport_flow_match": checks["transport_flow_match"],
                "balance_match": checks["balance_match"],
                "explanation_generated": result["explanation_generated"],
                "passed_checks": sum(1 for flag in pass_flags if flag is True),
                "total_applicable_checks": len(pass_flags),
            }
        )

        solve_rows.append(
            {
                "prompt_id": result["prompt_id"],
                "solve_success": checks["solve_success"],
                "status": checks["status"],
                "solver_name": checks["solver_name"],
                "expected_objective": checks["expected_objective"],
                "actual_objective": checks["actual_objective"],
                "objective_match": checks["objective_match"],
                "accepted_supply_match": checks["accepted_supply_match"],
                "accepted_demand_match": checks["accepted_demand_match"],
                "transport_flow_match": checks["transport_flow_match"],
                "balance_match": checks["balance_match"],
                "demand_revenue_match": checks["demand_revenue_match"],
                "transport_cost_match": checks["transport_cost_match"],
                "supply_cost_match": checks["supply_cost_match"],
                "accepted_supply_total_match": checks["accepted_supply_total_match"],
                "solver_message": checks["solver_message"],
            }
        )

        interpretation_rows.append(
            {
                "prompt_id": result["prompt_id"],
                "structural_match": comparison["structural_match"],
                "blocking_error_count": blocking_count,
                "benign_extra_name_fields": benign_count,
                "benign_identifier_mismatches": benign_identifier_count,
                "blocking_error_categories": ", ".join(comparison["blocking_error_categories"]),
                "blocking_errors": _format_blocking_errors(comparison["blocking_errors"]),
                "missing_parameters": "; ".join(result["validation_result"].get("missing_parameters", [])),
                "invalid_references": "; ".join(result["validation_result"].get("invalid_references", [])),
            }
        )

        metadata_rows.append(
            {
                "prompt_id": result["prompt_id"],
                "interpretation_source": metadata.get("interpretation_source"),
                "fallback_used": metadata.get("fallback_used"),
                "fallback_reason": metadata.get("fallback_reason"),
                "llm_failure": metadata.get("llm_failure"),
                "llm_provider": metadata.get("llm_provider"),
                "gemini_model": metadata.get("gemini_model"),
            }
        )

        reasoning_rows.extend(result.get("reasoning_results", []))
        for row in checks.get("diagnostics", []):
            diagnostic_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("semantic_count_metrics", []):
            count_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("parameter_multiset_metrics", []):
            parameter_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("topology_metrics", []):
            topology_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("route_economics_metrics", []):
            route_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("solver_aggregate_metrics", []):
            aggregate_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("balance_residual_metrics", []):
            residual_rows.append({"prompt_id": result["prompt_id"], **row})

    return {
        "case_summary": pd.DataFrame(case_summary_rows),
        "solve_accuracy": pd.DataFrame(solve_rows),
        "interpretation_errors": pd.DataFrame(interpretation_rows),
        "semantic_count_metrics": pd.DataFrame(count_rows),
        "parameter_multiset_metrics": pd.DataFrame(parameter_rows),
        "topology_metrics": pd.DataFrame(topology_rows),
        "route_economics_metrics": pd.DataFrame(route_rows),
        "solver_aggregate_metrics": pd.DataFrame(aggregate_rows),
        "balance_residual_metrics": pd.DataFrame(residual_rows),
        "alias_resolution_diagnostics": pd.DataFrame(diagnostic_rows),
        "reasoning_prompt_success": pd.DataFrame(reasoning_rows),
        "interpretation_metadata": pd.DataFrame(metadata_rows),
    }


def write_midterm_outputs(report: Dict[str, Any], output_dir: Path) -> None:
    """Write benchmark tables and summary metadata to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in report["tables"].items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    summary = {
        "metadata": report["metadata"],
        "reference_solution": report["reference_solution"],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )


def gemini_is_configured() -> bool:
    """Return whether live Gemini interpretation can be attempted."""

    return bool(os.getenv("GEMINI_API_KEY")) and os.getenv("LLM_PROVIDER", "").lower() == "gemini"


def _run_interpretation(
    case: Dict[str, Any],
    config: MidtermBenchmarkConfig,
) -> Dict[str, Any]:
    metadata = {
        "interpretation_source": "not_run",
        "llm_provider": os.getenv("LLM_PROVIDER"),
        "gemini_model": os.getenv("GEMINI_MODEL"),
        "gemini_configured": gemini_is_configured(),
        "fallback_used": False,
        "fallback_reason": None,
        "llm_failure": None,
    }

    if config.use_llm:
        metadata["interpretation_source"] = "live_llm_pipeline"
        if gemini_is_configured():
            try:
                artifacts = interpret_problem_from_text(case["prose"])
                return {
                    "semantic_plan": artifacts.get("semantic_plan"),
                    "state": artifacts.get("problem_state"),
                    "market_instance": artifacts.get("market_instance"),
                    "metadata": metadata,
                }
            except Exception as exc:
                metadata["llm_failure"] = f"{exc.__class__.__name__}: {exc}"
                if not config.fallback_to_reference_fixture:
                    return {"semantic_plan": None, "state": None, "metadata": metadata}
        else:
            metadata["llm_failure"] = (
                "Gemini is not configured; GEMINI_API_KEY is empty or LLM_PROVIDER is not gemini."
            )
            if not config.fallback_to_reference_fixture:
                return {"semantic_plan": None, "state": None, "metadata": metadata}
    elif not config.fallback_to_reference_fixture:
        metadata["interpretation_source"] = "skipped_by_user_config"
        return {"semantic_plan": None, "state": None, "metadata": metadata}

    if config.fallback_to_reference_fixture:
        artifacts = build_problem_artifacts_from_semantic_plan(case["expected_plan"])
        metadata["interpretation_source"] = (
            "expected_fixture_fallback_after_llm_failure"
            if metadata.get("llm_failure")
            else "deterministic_fallback"
        )
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = metadata.get("llm_failure") or "live LLM disabled"
        return {
            "semantic_plan": artifacts["semantic_plan"],
            "state": artifacts["problem_state"],
            "market_instance": artifacts["market_instance"],
            "metadata": metadata,
        }

    return {"semantic_plan": None, "state": None, "metadata": metadata}


def _select_or_run_solve_result(
    state: Optional[ProblemState],
    validation: Dict[str, Any],
    attempt_solve: bool,
) -> Tuple[Dict[str, Any], Optional[SolverResults]]:
    if not attempt_solve:
        return _skipped_solve_result("solve attempt disabled"), None
    if not isinstance(state, ProblemState):
        return _skipped_solve_result("ProblemState was not created"), None
    if not validation.get("solver_ready", False):
        return _skipped_solve_result("state is not solver-ready"), None

    model = build_model_from_state(state)
    raw_result = solve_model(model)
    solve_dict = raw_result.to_dict() if hasattr(raw_result, "to_dict") else dict(raw_result)
    solver_results = None
    try:
        solver_results = SolverResults.from_solve_result(raw_result, state)
    except Exception:
        solver_results = None
    return solve_dict, solver_results


def _skipped_solve_result(reason: str) -> Dict[str, Any]:
    return {
        "success": False,
        "status": "skipped",
        "message": reason,
        "objective_value": None,
        "solver_time": 0.0,
        "solution": {},
        "termination_condition": "skipped",
        "solver_name": None,
    }


def _copy_problem_state(state: ProblemState) -> ProblemState:
    if hasattr(state, "model_copy"):
        return state.model_copy(deep=True)
    return state.copy(deep=True)


def _empty_primary_metrics(reason: str) -> Dict[str, Any]:
    row = {
        "metric": "problem_state_created",
        "expected": True,
        "actual": False,
        "pass": False,
        "details": reason,
    }
    return {
        "semantic_count_metrics": [row],
        "parameter_multiset_metrics": [],
        "topology_metrics": [],
        "route_economics_metrics": [],
        "solver_aggregate_metrics": [],
        "balance_residual_metrics": [],
        "semantic_structure_pass": False,
        "solver_aggregate_pass": False,
        "balance_residual_pass": False,
        "primary_success": False,
    }


def _semantic_count_metric_rows(state: ProblemState) -> List[Dict[str, Any]]:
    expected_counts = {
        "products": 1,
        "source_supplier_entities": 1,
        "demand_consumer_entities": 2,
        "transport_paths": 2,
        "technologies": 0,
    }
    actual_counts = {
        "products": len(state.products),
        "source_supplier_entities": len(state.suppliers),
        "demand_consumer_entities": len(state.consumers),
        "transport_paths": len(state.transport_links),
        "technologies": len(state.technologies),
    }
    return [
        _metric_row(metric, expected_counts[metric], actual_counts[metric], expected_counts[metric] == actual_counts[metric])
        for metric in expected_counts
    ]


def _parameter_multiset_metric_rows(
    state: ProblemState,
    solve_result: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    supplier_bid_prices = [bid.price for bid in state.bids if bid.owner_type == "supplier"]
    consumer_bid_prices = [bid.price for bid in state.bids if bid.owner_type == "consumer"]
    transport_costs = [float(getattr(link, "cost", 0.0) or 0.0) for link in state.transport_links]
    finite_transport_capacities = [
        float(link.capacity)
        for link in state.transport_links
        if link.capacity is not None
    ]
    solve_success = bool(solve_result.get("success", False))
    transport_capacity_pass = (
        _multiset_matches(finite_transport_capacities, [500.0, 500.0], tolerance)
        or (not finite_transport_capacities and solve_success)
    )

    return [
        _metric_row(
            "supplier_capacities",
            [1000.0],
            [supplier.capacity for supplier in state.suppliers],
            _multiset_matches([supplier.capacity for supplier in state.suppliers], [1000.0], tolerance),
        ),
        _metric_row(
            "consumer_capacities",
            [500.0, 500.0],
            [consumer.capacity for consumer in state.consumers],
            _multiset_matches([consumer.capacity for consumer in state.consumers], [500.0, 500.0], tolerance),
        ),
        _metric_row(
            "consumer_bid_prices",
            [0.5, 1.5],
            consumer_bid_prices,
            _multiset_matches(consumer_bid_prices, [0.5, 1.5], tolerance),
        ),
        _metric_row(
            "supplier_bid_prices",
            [0.0],
            supplier_bid_prices,
            _multiset_matches(supplier_bid_prices, [0.0], tolerance),
        ),
        _metric_row(
            "transport_costs",
            [0.1, 0.2],
            transport_costs,
            _multiset_matches(transport_costs, [0.1, 0.2], tolerance),
        ),
        _metric_row(
            "transport_capacities",
            "absent/unbounded with successful solve, or [500.0, 500.0]",
            finite_transport_capacities or "absent/unbounded",
            transport_capacity_pass,
            details="Transport capacities are not part of the Q1 economics if routes remain nonbinding.",
        ),
    ]


def _topology_metric_rows(state: ProblemState) -> List[Dict[str, Any]]:
    source_nodes = {supplier.node for supplier in state.suppliers}
    sink_nodes = {consumer.node for consumer in state.consumers}
    source_node = next(iter(source_nodes), None) if len(source_nodes) == 1 else None
    outgoing_destinations = {
        link.destination
        for link in state.transport_links
        if source_node is not None and link.origin == source_node
    }
    outgoing_sink_destinations = outgoing_destinations & sink_nodes

    rows = [
        _metric_row("one_source_node", True, len(source_nodes) == 1, len(source_nodes) == 1, details=sorted(source_nodes)),
        _metric_row("two_distinct_sink_nodes", True, len(sink_nodes) == 2, len(sink_nodes) == 2, details=sorted(sink_nodes)),
        _metric_row(
            "source_outgoing_to_two_sinks",
            True,
            len(outgoing_sink_destinations) == 2,
            len(outgoing_sink_destinations) == 2,
            details={"source": source_node, "destinations": sorted(outgoing_destinations)},
        ),
        _metric_row("no_technologies", True, len(state.technologies) == 0, len(state.technologies) == 0),
        _metric_row("no_negative_bids", True, all(bid.price >= 0 for bid in state.bids), all(bid.price >= 0 for bid in state.bids)),
        _metric_row(
            "all_transport_arcs_carry_manure",
            True,
            all(_canonical_product_key(link.product) == CANONICAL_MANURE_PRODUCT_ID for link in state.transport_links),
            all(_canonical_product_key(link.product) == CANONICAL_MANURE_PRODUCT_ID for link in state.transport_links),
            details=[link.product for link in state.transport_links],
        ),
    ]
    return rows


def _route_economics_metric_rows(state: ProblemState, tolerance: float) -> List[Dict[str, Any]]:
    route_net_values = _route_net_values(state)
    return [
        _metric_row(
            "sorted_route_net_values",
            [0.4, 1.3],
            route_net_values,
            _multiset_matches(route_net_values, [0.4, 1.3], tolerance),
            details="consumer willingness-to-pay - transport cost - source cost",
        )
    ]


def _solver_aggregate_metric_rows(
    state: ProblemState,
    solve_result: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    aggregates = _solver_aggregates(state, solve_result)
    expected = {
        "objective_value": reference_solution.get("objective_value", 850.0),
        "demand_revenue": reference_solution.get("demand_revenue", 1000.0),
        "transport_cost": reference_solution.get("transport_cost", 150.0),
        "supply_cost": reference_solution.get("supply_cost", 0.0),
        "total_accepted_supply": 1000.0,
        "total_accepted_demand": 1000.0,
        "total_transport_flow": 1000.0,
        "active_transport_routes": 2,
    }
    rows = [
        _numeric_metric_row(metric, expected[metric], aggregates.get(metric), tolerance)
        for metric in expected
    ]
    rows.extend(
        [
            _metric_row(
                "sorted_active_flow_values",
                [500.0, 500.0],
                aggregates.get("sorted_active_flow_values", []),
                _multiset_matches(aggregates.get("sorted_active_flow_values", []), [500.0, 500.0], tolerance),
            ),
            _metric_row(
                "sorted_accepted_demand_values",
                [500.0, 500.0],
                aggregates.get("sorted_accepted_demand_values", []),
                _multiset_matches(aggregates.get("sorted_accepted_demand_values", []), [500.0, 500.0], tolerance),
            ),
        ]
    )
    return rows


def _balance_residual_metric_rows(
    state: ProblemState,
    solve_result: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    stats = _balance_residual_stats(state, solve_result, tolerance)
    return [
        _metric_row(
            "max_abs_balance_residual",
            f"<= {tolerance}",
            stats["max_abs_balance_residual"],
            stats["max_abs_balance_residual"] <= tolerance,
            details=stats["residuals"],
        ),
        _metric_row(
            "sum_abs_balance_residual",
            f"<= {tolerance}",
            stats["sum_abs_balance_residual"],
            stats["sum_abs_balance_residual"] <= tolerance,
        ),
        _metric_row(
            "balance_violation_count",
            0,
            stats["balance_violation_count"],
            stats["balance_violation_count"] == 0,
        ),
    ]


def _route_net_values(state: ProblemState) -> List[float]:
    values: List[float] = []
    for link in state.transport_links:
        source_cost = _supplier_price_at(state, link.origin, link.product)
        consumer_price = _consumer_price_at(state, link.destination, link.product)
        if source_cost is None or consumer_price is None:
            continue
        values.append(float(consumer_price) - float(getattr(link, "cost", 0.0) or 0.0) - float(source_cost))
    return sorted(values)


def _solver_aggregates(state: ProblemState, solve_result: Dict[str, Any]) -> Dict[str, Any]:
    if not solve_result.get("success", False):
        return {
            "objective_value": solve_result.get("objective_value"),
            "demand_revenue": None,
            "transport_cost": None,
            "supply_cost": None,
            "total_accepted_supply": None,
            "total_accepted_demand": None,
            "total_transport_flow": None,
            "active_transport_routes": None,
            "sorted_active_flow_values": [],
            "sorted_accepted_demand_values": [],
        }

    solution = solve_result.get("solution", {}) if isinstance(solve_result, dict) else {}
    q_values = _solution_block(solution, "q")
    f_values = _solution_block(solution, "f")
    bids_by_id = {bid.id: bid for bid in state.bids}

    demand_revenue = 0.0
    supply_cost = 0.0
    total_accepted_supply = 0.0
    total_accepted_demand = 0.0
    accepted_demand_values = []
    for bid_id, quantity in q_values.items():
        bid = bids_by_id.get(str(bid_id))
        if bid is None:
            continue
        if bid.owner_type == "supplier":
            total_accepted_supply += quantity
            supply_cost += float(bid.price) * quantity
        elif bid.owner_type == "consumer":
            total_accepted_demand += quantity
            demand_revenue += float(bid.price) * quantity
            if abs(quantity) > TOLERANCE:
                accepted_demand_values.append(quantity)

    transport_activities = _transport_activity_rows(state, f_values)
    active_flows = [activity["flow"] for activity in transport_activities if abs(activity["flow"]) > TOLERANCE]
    transport_cost = sum(activity["cost"] * activity["flow"] for activity in transport_activities)
    total_transport_flow = sum(activity["flow"] for activity in transport_activities)

    return {
        "objective_value": solve_result.get("objective_value"),
        "demand_revenue": demand_revenue,
        "transport_cost": transport_cost,
        "supply_cost": supply_cost,
        "total_accepted_supply": total_accepted_supply,
        "total_accepted_demand": total_accepted_demand,
        "total_transport_flow": total_transport_flow,
        "active_transport_routes": len(active_flows),
        "sorted_active_flow_values": sorted(active_flows),
        "sorted_accepted_demand_values": sorted(accepted_demand_values),
    }


def _balance_residual_stats(
    state: ProblemState,
    solve_result: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    if not solve_result.get("success", False):
        return {
            "max_abs_balance_residual": float("inf"),
            "sum_abs_balance_residual": float("inf"),
            "balance_violation_count": len(state.nodes) * max(1, len(state.products)),
            "residuals": {},
        }

    solution = solve_result.get("solution", {}) if isinstance(solve_result, dict) else {}
    q_values = _solution_block(solution, "q")
    f_values = _solution_block(solution, "f")
    bids_by_id = {bid.id: bid for bid in state.bids}
    suppliers_by_id = {supplier.id: supplier for supplier in state.suppliers}
    consumers_by_id = {consumer.id: consumer for consumer in state.consumers}

    residuals: Dict[Tuple[str, str], float] = {
        (node.id, product.id): 0.0
        for node in state.nodes
        for product in state.products
    }

    def add(node: str, product: str, value: float) -> None:
        residuals[(node, product)] = residuals.get((node, product), 0.0) + float(value)

    for bid_id, quantity in q_values.items():
        bid = bids_by_id.get(str(bid_id))
        if bid is None:
            continue
        if bid.owner_type == "supplier":
            supplier = suppliers_by_id.get(bid.owner_id)
            if supplier is not None:
                add(supplier.node, supplier.product, quantity)
        elif bid.owner_type == "consumer":
            consumer = consumers_by_id.get(bid.owner_id)
            if consumer is not None:
                add(consumer.node, consumer.product, -quantity)

    for activity in _transport_activity_rows(state, f_values):
        product = activity["product"]
        if product is None:
            continue
        add(activity["origin"], product, -activity["flow"])
        add(activity["destination"], product, activity["flow"])

    rendered_residuals = {
        f"{node}:{product}": value
        for (node, product), value in sorted(residuals.items())
    }
    abs_values = [abs(value) for value in residuals.values()]
    return {
        "max_abs_balance_residual": max(abs_values) if abs_values else 0.0,
        "sum_abs_balance_residual": sum(abs_values),
        "balance_violation_count": sum(1 for value in abs_values if value > tolerance),
        "residuals": rendered_residuals,
    }


def _supplier_price_at(state: ProblemState, node: str, product: str) -> Optional[float]:
    supplier_ids = {
        supplier.id
        for supplier in state.suppliers
        if supplier.node == node and _same_product(supplier.product, product)
    }
    prices = [
        bid.price
        for bid in state.bids
        if bid.owner_type == "supplier"
        and bid.owner_id in supplier_ids
        and _same_product(bid.product_id, product)
    ]
    return float(prices[0]) if prices else None


def _consumer_price_at(state: ProblemState, node: str, product: str) -> Optional[float]:
    consumer_ids = {
        consumer.id
        for consumer in state.consumers
        if consumer.node == node and _same_product(consumer.product, product)
    }
    prices = [
        bid.price
        for bid in state.bids
        if bid.owner_type == "consumer"
        and bid.owner_id in consumer_ids
        and _same_product(bid.product_id, product)
    ]
    return float(prices[0]) if prices else None


def _same_product(left: Any, right: Any) -> bool:
    return str(left) == str(right) or _canonical_product_key(left) == _canonical_product_key(right)


def _transport_activity_rows(
    state: ProblemState,
    f_values: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows = []
    for raw_key, flow in f_values.items():
        parsed_origin, parsed_destination = _parse_arc_key(str(raw_key))
        link = _find_transport_link_for_arc(state, parsed_origin, parsed_destination, str(raw_key))
        if link is not None:
            origin = link.origin
            destination = link.destination
            product = link.product
            cost = float(getattr(link, "cost", 0.0) or 0.0)
            link_id = link.id
        else:
            origin = parsed_origin
            destination = parsed_destination
            product = state.products[0].id if len(state.products) == 1 else None
            cost = _transport_cost_for_arc(state, parsed_origin, parsed_destination)
            link_id = str(raw_key)
        rows.append(
            {
                "raw_key": str(raw_key),
                "link_id": link_id,
                "origin": origin,
                "destination": destination,
                "product": product,
                "cost": cost,
                "flow": float(flow),
            }
        )
    return rows


def _metric_row(
    metric: str,
    expected: Any,
    actual: Any,
    passed: bool,
    details: Any = None,
) -> Dict[str, Any]:
    return {
        "metric": metric,
        "expected": _render_metric_value(expected),
        "actual": _render_metric_value(actual),
        "pass": bool(passed),
        "details": _render_metric_value(details),
    }


def _numeric_metric_row(metric: str, expected: Optional[float], actual: Any, tolerance: float) -> Dict[str, Any]:
    passed = actual is not None and expected is not None and abs(float(actual) - float(expected)) <= tolerance
    return _metric_row(metric, expected, actual, passed)


def _multiset_matches(actual: Sequence[Any], expected: Sequence[float], tolerance: float) -> bool:
    if len(actual) != len(expected):
        return False
    try:
        actual_values = sorted(float(value) for value in actual if value is not None)
    except (TypeError, ValueError):
        return False
    if len(actual_values) != len(expected):
        return False
    expected_values = sorted(float(value) for value in expected)
    return all(abs(actual_value - expected_value) <= tolerance for actual_value, expected_value in zip(actual_values, expected_values))


def _render_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), sort_keys=True)
    return value


def _primary_failure_count(primary_metrics: Dict[str, Any]) -> int:
    return sum(
        1
        for key in (
            "semantic_count_metrics",
            "parameter_multiset_metrics",
            "topology_metrics",
            "route_economics_metrics",
            "solver_aggregate_metrics",
            "balance_residual_metrics",
        )
        for row in primary_metrics.get(key, [])
        if not row.get("pass", False)
    )


def _semantic_entity_map(collection: str, state: ProblemState) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    if collection == "nodes":
        return {
            ("node", _canonical_node_key(node.id)): _dump_model(node)
            for node in state.nodes
        }
    if collection == "products":
        return {
            ("product", _canonical_product_key(product.id)): _dump_model(product)
            for product in state.products
        }
    if collection == "suppliers":
        return {
            (
                "supplier",
                _canonical_node_key(supplier.node),
                _canonical_product_key(supplier.product),
            ): _dump_model(supplier)
            for supplier in state.suppliers
        }
    if collection == "consumers":
        return {
            (
                "consumer",
                _canonical_node_key(consumer.node),
                _canonical_product_key(consumer.product),
            ): _dump_model(consumer)
            for consumer in state.consumers
        }
    if collection == "transport_links":
        return {
            (
                "transport",
                _canonical_node_key(link.origin),
                _canonical_node_key(link.destination),
                _canonical_product_key(link.product),
            ): _dump_model(link)
            for link in state.transport_links
        }
    if collection == "technologies":
        return {
            ("technology", _canonical_node_key(technology.node), technology.id): _dump_model(technology)
            for technology in state.technologies
        }
    if collection == "bids":
        return _semantic_bid_map(state)
    raise KeyError(f"Unsupported collection: {collection}")


def _semantic_bid_map(state: ProblemState) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    suppliers_by_id = {supplier.id: supplier for supplier in state.suppliers}
    consumers_by_id = {consumer.id: consumer for consumer in state.consumers}
    technologies_by_id = {technology.id: technology for technology in state.technologies}
    result: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for bid in state.bids:
        product = _canonical_product_key(bid.product_id)
        if bid.owner_type == "supplier":
            supplier = suppliers_by_id.get(bid.owner_id)
            owner_key = _supplier_reference_key(supplier, bid.owner_id, bid.product_id)
        elif bid.owner_type == "consumer":
            consumer = consumers_by_id.get(bid.owner_id)
            owner_key = _consumer_reference_key(consumer, bid.owner_id, bid.product_id)
        elif bid.owner_type == "technology":
            technology = technologies_by_id.get(bid.owner_id)
            owner_key = (
                "technology",
                _canonical_node_key(getattr(technology, "node", bid.owner_id)),
                bid.owner_id,
            )
        else:
            owner_key = bid.owner_id
        result[("bid", bid.owner_type, owner_key, product)] = _dump_model(bid)
    return result


def _semantic_fields_for_collection(collection: str) -> Tuple[str, ...]:
    return {
        "nodes": (),
        "products": (),
        "suppliers": ("capacity",),
        "consumers": ("capacity",),
        "transport_links": ("cost",),
        "technologies": ("capacity", "yield_coefficients"),
        "bids": ("price", "quantity"),
    }[collection]


def _semantic_field_value(collection: str, field_name: str, entity: Dict[str, Any]) -> Any:
    if collection == "technologies" and field_name == "yield_coefficients":
        return {
            _canonical_product_key(product_id): coefficient
            for product_id, coefficient in (entity.get("yield_coefficients") or {}).items()
        }
    return entity.get(field_name)


def _format_semantic_key(semantic_key: Tuple[Any, ...]) -> str:
    return ":".join(str(part) for part in semantic_key)


def _record_benign_identifier_mismatch(
    collection: str,
    semantic_key: Tuple[Any, ...],
    expected_entity: Dict[str, Any],
    actual_entity: Dict[str, Any],
    benign_identifier_mismatches: List[Dict[str, Any]],
    error_categories: List[str],
) -> None:
    expected_id = expected_entity.get("id")
    actual_id = actual_entity.get("id")
    if expected_id == actual_id:
        return
    benign_identifier_mismatches.append(
        {
            "path": f"{collection}.{_format_semantic_key(semantic_key)}.id",
            "category": "benign_identifier_mismatch",
            "expected": expected_id,
            "actual": actual_id,
            "canonical_resolved_id": _format_semantic_key(semantic_key),
            "blocking": False,
        }
    )
    error_categories.append("benign_identifier_mismatch")


def _entity_map(entities: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        raw = _dump_model(entity)
        result[str(raw["id"])] = raw
    return result


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        raw = model.model_dump()
    elif hasattr(model, "dict"):
        raw = model.dict()
    else:
        raw = dict(model)
    return raw


def _record_benign_name_difference(
    collection: str,
    path_prefix: str,
    expected_entity: Dict[str, Any],
    actual_entity: Dict[str, Any],
    benign_extra_name_fields: List[Dict[str, Any]],
    error_categories: List[str],
) -> None:
    if collection not in {"nodes", "products"}:
        return
    expected_name = expected_entity.get("name")
    actual_name = actual_entity.get("name")
    if actual_name is None or expected_name == actual_name:
        return
    benign_extra_name_fields.append(
        {
            "path": f"{path_prefix}.name",
            "expected": expected_name,
            "actual": actual_name,
        }
    )
    error_categories.append("benign_extra_name_field")


def _values_match(expected: Any, actual: Any, tolerance: float) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(_values_match(expected[key], actual[key], tolerance) for key in expected)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= tolerance
    return expected == actual


def _append_error(
    errors: List[Dict[str, Any]],
    categories: List[str],
    category: str,
    path: str,
    expected: Any,
    actual: Any,
) -> None:
    errors.append(
        {
            "path": path,
            "category": category,
            "expected": expected,
            "actual": actual,
            "blocking": True,
        }
    )
    categories.append(category)


def _failed_field_category(field_name: str, expected: Any, actual: Any) -> str:
    if field_name in {"node", "origin", "destination"}:
        return "wrong_node_mapping"
    if field_name in {"product", "product_id", "yield_coefficients"}:
        return "wrong_product_mapping"
    if field_name in {"owner_id", "owner_type"}:
        return "wrong_owner_relation"
    if field_name in {"capacity", "quantity", "price", "cost"} and (
        isinstance(expected, (int, float)) or isinstance(actual, (int, float))
    ):
        return "wrong_numeric_value"
    return "wrong_value"


def _missing_field_category(field_name: str) -> str:
    return {
        "capacity": "missing_capacity",
        "quantity": "missing_quantity",
        "price": "missing_price",
        "cost": "missing_transport_cost",
        "yield_coefficients": "missing_yield_coefficients",
    }.get(field_name, "missing_field")


def _empty_failed_state_comparison(reason: str) -> Dict[str, Any]:
    return {
        "passed_fields": [],
        "benign_extra_name_fields": [],
        "blocking_errors": [
            {
                "path": "problem_state",
                "category": "problem_state_missing",
                "expected": "ProblemState",
                "actual": None,
                "blocking": True,
                "reason": reason,
            }
        ],
        "structural_match": False,
        "error_categories": ["problem_state_missing"],
        "blocking_error_categories": ["problem_state_missing"],
        "error_category_counts": {"problem_state_missing": 1},
        "blocking_error_category_counts": {"problem_state_missing": 1},
    }


def _solution_block(solution: Dict[str, Any], block_name: str) -> Dict[str, float]:
    block = solution.get(block_name, {}) if isinstance(solution, dict) else {}
    if not isinstance(block, dict):
        return {}
    result = {}
    for key, value in block.items():
        if value is None:
            continue
        result[str(key)] = float(value)
    return result


def _supplier_reference_key(supplier: Any, owner_id: str, product_id: Optional[str] = None) -> str:
    candidates = [owner_id]
    if supplier is not None:
        candidates.append(getattr(supplier, "node", ""))
        product_id = product_id or getattr(supplier, "product", None)
    product_ok = product_id is None or _canonical_product_key(product_id) == CANONICAL_MANURE_PRODUCT_ID
    if product_ok and any(_canonical_node_key(candidate) == "EauClaire" for candidate in candidates):
        return CANONICAL_SUPPLY_ID
    return str(owner_id)


def _consumer_reference_key(consumer: Any, owner_id: str, product_id: Optional[str] = None) -> str:
    candidates = [owner_id]
    if consumer is not None:
        candidates.append(getattr(consumer, "node", ""))
        product_id = product_id or getattr(consumer, "product", None)
    product_ok = product_id is None or _canonical_product_key(product_id) == CANONICAL_MANURE_PRODUCT_ID
    if not product_ok:
        return str(owner_id)
    for candidate in candidates:
        key = _canonical_node_key(candidate)
        if key in {"Menomonie", "BlackRiverFalls"}:
            return key
    return str(owner_id)


def _route_reference_key(origin: str, destination: str, product_id: Optional[str] = None) -> str:
    canonical_origin = _canonical_node_key(origin)
    canonical_destination = _canonical_node_key(destination)
    route_key = f"{canonical_origin}_to_{canonical_destination}"
    if product_id is not None and _canonical_product_key(product_id) != CANONICAL_MANURE_PRODUCT_ID:
        return f"{route_key}:{_canonical_product_key(product_id)}"
    return route_key


def _canonical_node_key(value: Any) -> str:
    text = _normalize_token(value)
    aliases = {
        "eauclaire": "EauClaire",
        "eau": "EauClaire",
        "ec": "EauClaire",
        "df": "EauClaire",
        "dairy": "EauClaire",
        "dairyfarm": "EauClaire",
        "sdairy": "EauClaire",
        "supplierdairy": "EauClaire",
        "dairyeauclaire": "EauClaire",
        "menomonie": "Menomonie",
        "m": "Menomonie",
        "mn": "Menomonie",
        "me": "Menomonie",
        "cf": "Menomonie",
        "corn": "Menomonie",
        "cornfarm": "Menomonie",
        "cornfarmer": "Menomonie",
        "ccorn": "Menomonie",
        "consumercorn": "Menomonie",
        "blackriverfalls": "BlackRiverFalls",
        "blackriver": "BlackRiverFalls",
        "brf": "BlackRiverFalls",
        "sf": "BlackRiverFalls",
        "soybean": "BlackRiverFalls",
        "soybeanfarm": "BlackRiverFalls",
        "soybeanfarmer": "BlackRiverFalls",
        "csoybean": "BlackRiverFalls",
        "consumersoybean": "BlackRiverFalls",
    }
    return aliases.get(text, str(value).replace(" ", ""))


def _canonical_product_key(value: Any) -> str:
    text = _normalize_token(value)
    aliases = {
        "manure": CANONICAL_MANURE_PRODUCT_ID,
        "wastemanure": CANONICAL_MANURE_PRODUCT_ID,
        "dairymanure": CANONICAL_MANURE_PRODUCT_ID,
        "p1": CANONICAL_MANURE_PRODUCT_ID,
        "dm": CANONICAL_MANURE_PRODUCT_ID,
    }
    return aliases.get(text, str(value).replace(" ", ""))


def _normalize_token(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _parse_arc_key(key: str) -> Tuple[str, str]:
    stripped = key.strip().strip("()")
    if "_to_" in stripped:
        origin, destination = stripped.split("_to_", 1)
        return origin.strip().strip("'\""), destination.strip().strip("'\"")
    if "->" in stripped:
        origin, destination = stripped.split("->", 1)
        return origin.strip().strip("'\""), destination.strip().strip("'\"")
    if " to " in stripped.lower():
        origin, destination = stripped.lower().split(" to ", 1)
        return origin.strip().strip("'\""), destination.strip().strip("'\"")
    parts = [part.strip().strip("'\"") for part in stripped.split(",")]
    if len(parts) != 2:
        return key, key
    return parts[0], parts[1]


def _find_transport_link_for_arc(
    state: ProblemState,
    origin: str,
    destination: str,
    arc_key: str,
) -> Any:
    canonical_origin = _canonical_node_key(origin)
    canonical_destination = _canonical_node_key(destination)
    normalized_arc_key = _normalize_token(arc_key)
    for link in state.transport_links:
        link_origin = _canonical_node_key(link.origin)
        link_destination = _canonical_node_key(link.destination)
        if link_origin == canonical_origin and link_destination == canonical_destination:
            return link
        if _normalize_token(link.id) == normalized_arc_key:
            return link
    parsed_origin, parsed_destination = _parse_arc_key(arc_key)
    canonical_parsed_origin = _canonical_node_key(parsed_origin)
    canonical_parsed_destination = _canonical_node_key(parsed_destination)
    for link in state.transport_links:
        if (
            _canonical_node_key(link.origin) == canonical_parsed_origin
            and _canonical_node_key(link.destination) == canonical_parsed_destination
        ):
            return link
    return None


def _transport_cost_for_arc(state: ProblemState, origin: str, destination: str) -> float:
    canonical_origin = _canonical_node_key(origin)
    canonical_destination = _canonical_node_key(destination)
    for link in state.transport_links:
        link_origin = _canonical_node_key(link.origin)
        link_destination = _canonical_node_key(link.destination)
        if link_origin == canonical_origin and link_destination == canonical_destination:
            return float(getattr(link, "cost", 0.0) or 0.0)
    return 0.0


def _compute_midterm_balance_checks(
    accepted_supply: Dict[str, float],
    accepted_demands: Dict[str, float],
    transport_flows: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    source_supply = accepted_supply.get("Dairy/EauClaire", 0.0)
    to_menomonie = transport_flows.get("EauClaire_to_Menomonie", 0.0)
    to_black_river = transport_flows.get("EauClaire_to_BlackRiverFalls", 0.0)
    menomonie_demand = accepted_demands.get("Menomonie", 0.0)
    black_river_demand = accepted_demands.get("BlackRiverFalls", 0.0)

    return {
        "EauClaire": {
            "supply": source_supply,
            "total_outgoing_flow": to_menomonie + to_black_river,
            "holds": abs(source_supply - to_menomonie - to_black_river) <= TOLERANCE,
        },
        "Menomonie": {
            "incoming_flow": to_menomonie,
            "accepted_demand": menomonie_demand,
            "holds": abs(to_menomonie - menomonie_demand) <= TOLERANCE,
        },
        "BlackRiverFalls": {
            "incoming_flow": to_black_river,
            "accepted_demand": black_river_demand,
            "holds": abs(to_black_river - black_river_demand) <= TOLERANCE,
        },
    }


def _build_solution_diagnostics(
    components: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_rows = components.get("raw_component_rows", [])
    rows.extend(
        _component_diagnostic_rows(
            component="accepted_supply",
            actual=components.get("accepted_supply", {}),
            expected=reference_solution.get("accepted_supply", {}),
            raw_rows=raw_rows,
            tolerance=tolerance,
        )
    )
    rows.extend(
        _component_diagnostic_rows(
            component="accepted_demands",
            actual=components.get("accepted_demands", {}),
            expected=reference_solution.get("accepted_demands", {}),
            raw_rows=raw_rows,
            tolerance=tolerance,
        )
    )
    rows.extend(
        _component_diagnostic_rows(
            component="transport_flows",
            actual=components.get("transport_flows", {}),
            expected=reference_solution.get("transport_flows", {}),
            raw_rows=raw_rows,
            tolerance=tolerance,
        )
    )
    rows.extend(
        _balance_diagnostic_rows(
            actual=components.get("balance_checks", {}),
            expected=reference_solution.get("balance_checks", {}),
            tolerance=tolerance,
        )
    )
    return rows


def _component_diagnostic_rows(
    component: str,
    actual: Dict[str, float],
    expected: Dict[str, float],
    raw_rows: Sequence[Dict[str, Any]],
    tolerance: float,
) -> List[Dict[str, Any]]:
    rows = []
    all_keys = sorted(set(actual.keys()) | set(expected.keys()))
    for canonical_id in all_keys:
        related_raw_rows = [
            row for row in raw_rows
            if row.get("component") == component and row.get("canonical_resolved_id") == canonical_id
        ]
        actual_value = actual.get(canonical_id, 0.0)
        expected_value = expected.get(canonical_id)
        numeric_match = (
            expected_value is not None
            and abs(float(actual_value) - float(expected_value)) <= tolerance
        )
        rows.append(
            {
                "component": component,
                "raw_interpreted_ids": _join_unique(row.get("raw_id") for row in related_raw_rows),
                "raw_bid_ids": _join_unique(row.get("raw_bid_id") for row in related_raw_rows),
                "raw_nodes": _join_unique(row.get("raw_node") for row in related_raw_rows),
                "raw_origin": _join_unique(row.get("raw_origin") for row in related_raw_rows),
                "raw_destination": _join_unique(row.get("raw_destination") for row in related_raw_rows),
                "raw_product": _join_unique(row.get("raw_product") for row in related_raw_rows),
                "canonical_resolved_id": canonical_id,
                "reference_id": canonical_id if canonical_id in expected else None,
                "actual_value": actual_value,
                "expected_value": expected_value,
                "numeric_match": numeric_match,
            }
        )
    return rows


def _balance_diagnostic_rows(
    actual: Dict[str, Dict[str, Any]],
    expected: Dict[str, Dict[str, Any]],
    tolerance: float,
) -> List[Dict[str, Any]]:
    rows = []
    for node in sorted(set(actual.keys()) | set(expected.keys())):
        actual_values = actual.get(node, {})
        expected_values = expected.get(node, {})
        numeric_match = True
        for key, expected_value in expected_values.items():
            if key == "holds":
                continue
            actual_value = actual_values.get(key)
            if actual_value is None or abs(float(actual_value) - float(expected_value)) > tolerance:
                numeric_match = False
        numeric_match = numeric_match and bool(actual_values.get("holds", False)) == bool(expected_values.get("holds", False))
        rows.append(
            {
                "component": "balance_checks",
                "raw_interpreted_ids": None,
                "raw_bid_ids": None,
                "raw_nodes": node,
                "raw_origin": None,
                "raw_destination": None,
                "raw_product": CANONICAL_MANURE_PRODUCT_ID,
                "canonical_resolved_id": node,
                "reference_id": node if node in expected else None,
                "actual_value": json.dumps(actual_values, sort_keys=True),
                "expected_value": json.dumps(expected_values, sort_keys=True),
                "numeric_match": numeric_match,
            }
        )
    return rows


def _join_unique(values: Iterable[Any]) -> Optional[str]:
    rendered = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in rendered:
            rendered.append(text)
    return ", ".join(rendered) if rendered else None


def _dict_matches_expected(
    actual: Dict[str, float],
    expected: Dict[str, float],
    tolerance: float,
) -> Optional[bool]:
    if expected is None:
        return None
    actual_filtered = {
        key: value
        for key, value in actual.items()
        if abs(float(value)) > tolerance or key in expected
    }
    if set(actual_filtered.keys()) != set(expected.keys()):
        return False
    return all(abs(float(actual_filtered[key]) - float(expected[key])) <= tolerance for key in expected)


def _balance_checks_match(
    actual: Dict[str, Dict[str, Any]],
    expected: Dict[str, Dict[str, Any]],
    tolerance: float,
) -> Optional[bool]:
    if expected is None:
        return None
    if set(actual.keys()) != set(expected.keys()):
        return False
    for node, expected_values in expected.items():
        actual_values = actual.get(node, {})
        if bool(actual_values.get("holds", False)) != bool(expected_values.get("holds", False)):
            return False
        for key, expected_value in expected_values.items():
            if key == "holds":
                continue
            actual_value = actual_values.get(key)
            if actual_value is None or abs(float(actual_value) - float(expected_value)) > tolerance:
                return False
    return True


def _float_or_none_matches(actual: Optional[float], expected: Optional[float], tolerance: float) -> Optional[bool]:
    if expected is None:
        return None
    if actual is None:
        return False
    return abs(float(actual) - float(expected)) <= tolerance


def _format_blocking_errors(errors: Sequence[Dict[str, Any]]) -> str:
    return "; ".join(
        f"{error.get('category')} at {error.get('path')}"
        for error in errors[:8]
    )


__all__ = [
    "DEFAULT_BENCHMARK_DIR",
    "MIDTERM_REASONING_PROMPTS",
    "MidtermBenchmarkConfig",
    "build_midterm_manure_cases",
    "build_midterm_manure_expected_plan",
    "build_midterm_output_tables",
    "compare_problem_states_for_midterm",
    "compare_solution_to_reference",
    "evaluate_primary_semantic_metrics",
    "evaluate_midterm_prompt_case",
    "extract_midterm_solution_components",
    "gemini_is_configured",
    "load_benchmark_files",
    "run_midterm_manure_q1_benchmark",
    "run_midterm_reasoning_prompt_battery",
    "write_midterm_outputs",
]
