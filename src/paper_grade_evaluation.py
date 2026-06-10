"""Paper-grade evaluation harness for the prose-to-solver workflow.

This module evaluates the existing ChatbotLP pipeline without changing the
architecture:

natural prose -> semantic plan -> ProblemState -> validation -> solve -> reasoning

Live LLM evaluation reports LLM/configuration/schema failures directly. Benchmark
fixtures are available only as explicit deterministic fixture inputs for tests,
offline reproducibility checks, and downstream solver demonstrations.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .chatbot_engine import run_chatbot_session
from .llm_adapter import LLMConfigurationError
from .llm_problem_interpreter import (
    LLMInvalidJSONError,
    LLMSchemaError,
    build_problem_artifacts_from_semantic_plan,
    interpret_problem_from_text,
    summarize_problem_state,
)
from .model_builder import build_model_from_state
from .prose_interpreter_evaluation import build_benchmark_evaluation_cases
from .schema import ProblemState
from .solver import SolveResult, solve_model
from .solver_results import SolverResults
from .validator import validate_state


TOLERANCE = 1e-6

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
    "transport_links": ("origin", "destination", "product", "capacity"),
    "technologies": ("node", "capacity", "yield_coefficients"),
    "bids": ("owner_id", "owner_type", "product_id", "price", "quantity"),
}

REASONING_PROMPTS = [
    {
        "id": "primal_lp",
        "label": "Primal LP formulation",
        "prompt": "Write the primal LP formulation for this current coordinated clearing instance.",
    },
    {
        "id": "dual_lp",
        "label": "Dual LP formulation",
        "prompt": "Write the dual LP formulation for this current coordinated clearing instance.",
    },
    {
        "id": "theorem_1_proof",
        "label": "Theorem 1 proof",
        "prompt": "Prove Theorem 1 for this current coordinated clearing problem.",
    },
    {
        "id": "strong_duality",
        "label": "Strong duality explanation/proof",
        "prompt": "Explain and prove strong duality for this current model.",
    },
    {
        "id": "dual_variable_interpretation",
        "label": "Dual variable interpretation",
        "prompt": "Interpret the dual variables economically for this current coordinated model.",
    },
    {
        "id": "node_product_price_interpretation",
        "label": "Node-product price interpretation",
        "prompt": "Interpret node-product prices for this solved case.",
    },
    {
        "id": "complementary_slackness",
        "label": "Complementary slackness verification",
        "prompt": (
            "Verify the complementary slackness conditions for the current primal-dual pair "
            "and explain their economic interpretation."
        ),
    },
]


@dataclass(frozen=True)
class EvaluationConfig:
    """Runtime switches for the paper-grade evaluator."""

    mode: str = "guided"
    selected_cases: Optional[Tuple[str, ...]] = None
    run_interpretation: bool = True
    use_llm: bool = True
    use_llm_for_reasoning: bool = True
    use_deterministic_fixture: bool = False
    fallback_to_expected_fixture: bool = False
    attempt_solve: bool = True
    run_reasoning: bool = True
    reasoning_prompt_subset: Optional[Tuple[str, ...]] = None


EVALUATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "interpretation_only": {
        "run_interpretation": True,
        "attempt_solve": False,
        "run_reasoning": False,
    },
    "case_a_full": {
        "selected_cases": ("canonical_case_a",),
        "run_interpretation": True,
        "attempt_solve": True,
        "run_reasoning": True,
    },
    "case_b_interpretation": {
        "selected_cases": ("negative_bid_case_b",),
        "run_interpretation": True,
        "attempt_solve": False,
        "run_reasoning": False,
    },
    "case_c_interpretation": {
        "selected_cases": ("transformation_case_c",),
        "run_interpretation": True,
        "attempt_solve": False,
        "run_reasoning": False,
    },
    "reasoning_one_case": {
        "selected_cases": ("canonical_case_a",),
        "run_interpretation": False,
        "attempt_solve": True,
        "run_reasoning": True,
        "reasoning_prompt_subset": ("primal_lp",),
    },
    "full_evaluation": {},
    "quota_safe_demo": {
        "selected_cases": ("canonical_case_a", "negative_bid_case_b", "transformation_case_c"),
        "run_interpretation": True,
        "attempt_solve": False,
        "run_reasoning": False,
    },
}


def build_evaluation_config(
    preset: str = "full_evaluation",
    **overrides: Any,
) -> EvaluationConfig:
    """Build an EvaluationConfig from a named preset and explicit overrides."""

    if preset not in EVALUATION_PRESETS:
        valid = ", ".join(sorted(EVALUATION_PRESETS))
        raise ValueError(f"Unknown evaluation preset {preset!r}. Valid presets: {valid}")
    config_data = dict(EVALUATION_PRESETS[preset])
    config_data.update({key: value for key, value in overrides.items() if value is not None})
    for key in ("selected_cases", "reasoning_prompt_subset"):
        if config_data.get(key) is not None:
            config_data[key] = tuple(config_data[key])
    return EvaluationConfig(**config_data)


def build_paper_grade_cases() -> List[Dict[str, Any]]:
    """Return the requested benchmark-style ABC prose evaluation set."""

    base_cases = build_benchmark_evaluation_cases()
    expected_by_name = _expected_solution_catalog()

    cases: List[Dict[str, Any]] = []
    for case in base_cases:
        name = case["name"]
        expected_state = case["expected_state"]
        expected_solution = expected_by_name.get(name, {})
        expected_solver_ready = name in {
            "canonical_case_a",
            "paraphrased_case_a",
            "negative_bid_case_b",
            "transformation_case_c",
        }

        cases.append(
            {
                **case,
                "case_family": _case_family_for_name(name),
                "expected_plan": semantic_plan_from_state(
                    expected_state,
                    problem_type=_problem_type_for_name(name),
                ),
                "expected_solver_ready": expected_solver_ready,
                "solve_expected": expected_solver_ready,
                "expected_solution": expected_solution,
            }
        )
    return cases


def semantic_plan_from_state(
    state: ProblemState,
    problem_type: str = "unknown",
) -> Dict[str, Any]:
    """Create a JSON-like semantic plan from a ProblemState fixture."""

    return {
        "problem_title": state.problem_title,
        "problem_type": problem_type,
        "nodes": [_dump_model(item) for item in state.nodes],
        "products": [_dump_model(item) for item in state.products],
        "suppliers": [_dump_model(item) for item in state.suppliers],
        "consumers": [_dump_model(item) for item in state.consumers],
        "transport_links": [_dump_model(item) for item in state.transport_links],
        "technologies": [_dump_model(item) for item in state.technologies],
        "bids": [_dump_model(item) for item in state.bids],
        "missing_information": list(state.missing_parameters),
        "ambiguities": [],
    }


def compare_problem_states_paper_grade(
    expected: ProblemState,
    actual: ProblemState,
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    """Compare ProblemStates with paper-grade error categories.

    Human-readable names on nodes/products are treated as benign metadata when
    the expected benchmark omits them. Numeric, ownership, mapping, missing,
    and invented-entity errors remain blocking.
    """

    passed_fields: List[Dict[str, Any]] = []
    benign_extra_name_fields: List[Dict[str, Any]] = []
    blocking_errors: List[Dict[str, Any]] = []
    error_categories: List[str] = []

    for collection in ENTITY_COLLECTIONS:
        expected_entities = _entity_map(getattr(expected, collection))
        actual_entities = _entity_map(getattr(actual, collection))

        for entity_id, expected_entity in expected_entities.items():
            path_prefix = f"{collection}.{entity_id}"
            actual_entity = actual_entities.get(entity_id)
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

            for field_name in FIELD_MAP[collection]:
                field_path = f"{path_prefix}.{field_name}"
                expected_value = expected_entity.get(field_name)
                actual_value = actual_entity.get(field_name)

                if expected_value is None and actual_value is None:
                    passed_fields.append({"path": field_path, "expected": None, "actual": None})
                    continue
                if expected_value is None and actual_value is not None:
                    if field_name == "name" and collection in {"nodes", "products"}:
                        benign_extra_name_fields.append(
                            {"path": field_path, "expected": expected_value, "actual": actual_value}
                        )
                        error_categories.append("benign_extra_name_field")
                    else:
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

        for entity_id, actual_entity in actual_entities.items():
            if entity_id not in expected_entities:
                _append_error(
                    blocking_errors,
                    error_categories,
                    "extra_invented_entity",
                    f"{collection}.{entity_id}",
                    expected=None,
                    actual=actual_entity,
                )

    blocking_categories = [item["category"] for item in blocking_errors]
    return {
        "passed_fields": passed_fields,
        "benign_extra_name_fields": benign_extra_name_fields,
        "blocking_errors": blocking_errors,
        "structural_match": len(blocking_errors) == 0,
        "error_categories": sorted(error_categories),
        "blocking_error_categories": sorted(blocking_categories),
        "error_category_counts": dict(sorted(Counter(error_categories).items())),
        "blocking_error_category_counts": dict(sorted(Counter(blocking_categories).items())),
        "missing_fields": [
            item
            for item in blocking_errors
            if item["category"].startswith("missing_") and item["category"] != "missing_entity"
        ],
        "wrong_numeric_values": [
            item for item in blocking_errors if item["category"] == "wrong_numeric_value"
        ],
        "wrong_ownership_relations": [
            item for item in blocking_errors if item["category"] == "wrong_owner_relation"
        ],
        "wrong_node_product_mappings": [
            item
            for item in blocking_errors
            if item["category"] in {"wrong_node_mapping", "wrong_product_mapping"}
        ],
        "extra_invented_entities": [
            item for item in blocking_errors if item["category"] == "extra_invented_entity"
        ],
    }


def run_paper_grade_evaluation(
    config: Optional[EvaluationConfig] = None,
    cases: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run all paper-grade cases and return raw results plus tables."""

    runtime_config = config or EvaluationConfig()
    selected_cases = list(cases) if cases is not None else build_paper_grade_cases()
    selected_cases = _filter_cases(selected_cases, runtime_config.selected_cases)
    case_results = [
        evaluate_case(case, config=runtime_config)
        for case in selected_cases
    ]
    tables = build_output_tables(case_results)
    return {
        "cases": case_results,
        "tables": tables,
        "metadata": {
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "gemini_model": os.getenv("GEMINI_MODEL"),
            "gemini_configured": gemini_is_configured(),
            "config": runtime_config.__dict__,
        },
    }


def evaluate_case(
    case: Dict[str, Any],
    config: Optional[EvaluationConfig] = None,
) -> Dict[str, Any]:
    """Evaluate one prose benchmark case through the current pipeline."""

    runtime_config = config or EvaluationConfig()
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
    comparison = (
        compare_problem_states_paper_grade(case["expected_state"], actual_state)
        if problem_state_created
        else _empty_failed_comparison("ProblemState was not created")
    )

    solve_result = _select_or_run_solve_result(
        state=actual_state,
        validation=validation,
        interpretation=interpretation,
        attempt_solve=runtime_config.attempt_solve,
    )
    solve_checks = evaluate_solve_accuracy(
        solve_result=solve_result,
        expected_solution=case.get("expected_solution", {}),
        solve_expected=bool(case.get("solve_expected", False)),
    )
    metadata = interpretation.get("metadata", {})
    metadata["validation_passed"] = bool(validation.get("solver_ready", False))
    metadata["missing_fields"] = _validation_missing_fields(validation)
    metadata["solver_ready"] = solver_ready_actual = bool(validation.get("solver_ready", False))
    metadata["solver_status"] = solve_result.get("status") if isinstance(solve_result, dict) else None

    primary_success = bool(comparison["structural_match"]) and solver_ready_actual == bool(
        case.get("expected_solver_ready", False)
    )
    if solve_checks["solve_expected"]:
        primary_success = primary_success and bool(solve_checks["solve_success"]) and all(
            check is True
            for check in (
                solve_checks["objective_match"],
                solve_checks["accepted_bid_match"],
                solve_checks["transport_flow_match"],
                solve_checks["technology_activity_match"],
            )
        )

    failure_type = metadata.get("failure_type")
    if not failure_type:
        if not problem_state_created:
            failure_type = "incomplete_problem_state"
        elif not validation.get("solver_ready", False):
            failure_type = "incomplete_problem_state"
        elif solve_checks["solve_expected"] and not solve_checks["solve_success"]:
            failure_type = "solver_failed"
        elif not comparison["structural_match"] or (
            solve_checks["solve_expected"]
            and any(
                check is False
                for check in (
                    solve_checks["objective_match"],
                    solve_checks["accepted_bid_match"],
                    solve_checks["transport_flow_match"],
                    solve_checks["technology_activity_match"],
                )
            )
        ):
            failure_type = "benchmark_mismatch"
        else:
            failure_type = "none"
    metadata["failure_type"] = failure_type
    metadata["primary_success"] = bool(primary_success and failure_type == "none")

    reasoning_results: List[Dict[str, Any]] = []
    if (
        runtime_config.run_reasoning
        and problem_state_created
        and bool(validation.get("solver_ready", False))
    ):
        reasoning_results = run_reasoning_prompt_battery(
            state=actual_state,
            case_name=case["name"],
            mode=runtime_config.mode,
            use_llm=runtime_config.use_llm_for_reasoning,
            prompt_subset=runtime_config.reasoning_prompt_subset,
        )

    solver_ready_correct = solver_ready_actual == bool(case.get("expected_solver_ready", False))

    return {
        "name": case["name"],
        "label": case["label"],
        "case_family": case.get("case_family"),
        "prose_input": case["prose"],
        "expected_solver_ready": bool(case.get("expected_solver_ready", False)),
        "semantic_plan": interpretation.get("semantic_plan"),
        "semantic_plan_created": bool(interpretation.get("semantic_plan")),
        "problem_state": actual_state,
        "problem_state_created": problem_state_created,
        "problem_state_summary": summarize_problem_state(actual_state) if problem_state_created else None,
        "interpretation_metadata": metadata,
        "validation_result": validation,
        "comparison": comparison,
        "solver_ready_actual": solver_ready_actual,
        "solver_ready_correct": solver_ready_correct,
        "primary_success": metadata["primary_success"],
        "failure_type": metadata["failure_type"],
        "solve_result": solve_result,
        "solve_checks": solve_checks,
        "reasoning_results": reasoning_results,
    }


def evaluate_solve_accuracy(
    solve_result: Dict[str, Any],
    expected_solution: Dict[str, Any],
    solve_expected: bool,
    tolerance: float = TOLERANCE,
) -> Dict[str, Any]:
    """Compare objective and decision-variable blocks against expectations."""

    solve_success = bool(solve_result.get("success", False))
    expected_objective = expected_solution.get("objective_value")
    actual_objective = solve_result.get("objective_value")

    objective_match = None
    if solve_expected and expected_objective is not None and solve_success and actual_objective is not None:
        objective_match = abs(float(actual_objective) - float(expected_objective)) <= tolerance

    solution = solve_result.get("solution", {}) if isinstance(solve_result, dict) else {}
    accepted_bid_match = _dict_matches_expected(
        actual=_extract_solution_block(solution, "q"),
        expected=expected_solution.get("bid_allocations"),
        tolerance=tolerance,
    )
    transport_flow_match = _dict_matches_expected(
        actual=_extract_solution_block(solution, "f"),
        expected=expected_solution.get("transport_flows"),
        tolerance=tolerance,
    )
    technology_activity_match = _dict_matches_expected(
        actual=_extract_solution_block(solution, "x"),
        expected=expected_solution.get("technology_extents"),
        tolerance=tolerance,
    )

    return {
        "solve_expected": solve_expected,
        "solve_success": solve_success,
        "status": solve_result.get("status"),
        "termination_condition": solve_result.get("termination_condition"),
        "solver_name": solve_result.get("solver_name"),
        "solver_message": solve_result.get("message"),
        "expected_objective": expected_objective,
        "actual_objective": actual_objective,
        "objective_match": objective_match,
        "accepted_bid_match": accepted_bid_match,
        "transport_flow_match": transport_flow_match,
        "technology_activity_match": technology_activity_match,
    }


def run_reasoning_prompt_battery(
    state: ProblemState,
    case_name: str,
    mode: str = "guided",
    use_llm: bool = True,
    prompt_subset: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Run the requested formal reasoning prompt battery for a solver-ready state."""

    rows = []
    prompt_specs = _filter_reasoning_prompts(prompt_subset)
    for prompt_spec in prompt_specs:
        result = run_chatbot_session(
            state=state,
            user_message=prompt_spec["prompt"],
            mode=mode,
            use_llm=use_llm,
        )
        metadata = result.get("response_metadata", {}) or {}
        response_text = result.get("response", "") or ""
        success = (
            bool(result.get("success", False))
            and bool(response_text.strip())
            and not metadata.get("validation_fatal")
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
                "raw_llm_output_present": metadata.get("raw_llm_output_present"),
                "llm_output_length": metadata.get("llm_output_length"),
                "validation_warnings": metadata.get("validation_warnings", []),
                "validation_fatal": metadata.get("validation_fatal", []),
                "response_preview": response_text[:240],
            }
        )
    return rows


def build_output_tables(case_results: Sequence[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Create the requested case summary, accuracy, solve, reasoning, and error tables."""

    case_summary_rows = []
    interpretation_rows = []
    readiness_rows = []
    solve_rows = []
    reasoning_rows = []
    metadata_rows = []
    error_counter: Counter[str] = Counter()

    for result in case_results:
        comparison = result["comparison"]
        solve_checks = result["solve_checks"]
        blocking_count = len(comparison["blocking_errors"])
        benign_count = len(comparison["benign_extra_name_fields"])

        pass_flags = [
            result["semantic_plan_created"],
            result["problem_state_created"],
            comparison["structural_match"],
            result["solver_ready_correct"],
        ]
        if solve_checks["solve_expected"]:
            pass_flags.extend(
                [
                    solve_checks["solve_success"],
                    solve_checks["objective_match"],
                    solve_checks["accepted_bid_match"],
                    solve_checks["transport_flow_match"],
                    solve_checks["technology_activity_match"],
                ]
            )

        case_summary_rows.append(
            {
                "case": result["name"],
                "label": result["label"],
                "family": result["case_family"],
                "evaluation_mode": result["interpretation_metadata"].get("evaluation_mode"),
                "interpretation_source": result["interpretation_metadata"].get("interpretation_source"),
                "semantic_plan_created": result["semantic_plan_created"],
                "problem_state_created": result["problem_state_created"],
                "structural_match": comparison["structural_match"],
                "blocking_error_count": blocking_count,
                "benign_extra_name_fields": benign_count,
                "solver_ready_expected": result["expected_solver_ready"],
                "solver_ready_actual": result["solver_ready_actual"],
                "solver_ready_correct": result["solver_ready_correct"],
                "solve_expected": solve_checks["solve_expected"],
                "solve_success": solve_checks["solve_success"],
                "objective_match": solve_checks["objective_match"],
                "primary_success": result.get("primary_success", False),
                "failure_type": result.get("failure_type"),
                "passed_checks": sum(1 for flag in pass_flags if flag is True),
                "total_applicable_checks": len(pass_flags),
            }
        )

        interpretation_rows.append(
            {
                "case": result["name"],
                "structural_match": comparison["structural_match"],
                "missing_fields": len(comparison["missing_fields"]),
                "wrong_numeric_values": len(comparison["wrong_numeric_values"]),
                "wrong_ownership_relations": len(comparison["wrong_ownership_relations"]),
                "wrong_node_product_mappings": len(comparison["wrong_node_product_mappings"]),
                "extra_invented_entities": len(comparison["extra_invented_entities"]),
                "benign_extra_name_fields": benign_count,
                "blocking_error_categories": ", ".join(comparison["blocking_error_categories"]),
            }
        )

        readiness_rows.append(
            {
                "case": result["name"],
                "expected_solver_ready": result["expected_solver_ready"],
                "actual_solver_ready": result["solver_ready_actual"],
                "correct": result["solver_ready_correct"],
                "missing_parameters": "; ".join(result["validation_result"].get("missing_parameters", [])),
                "invalid_references": "; ".join(result["validation_result"].get("invalid_references", [])),
                "incomplete_technologies": "; ".join(result["validation_result"].get("incomplete_technologies", [])),
            }
        )

        metadata = result["interpretation_metadata"]
        metadata_rows.append({
            "case": result["name"],
            "evaluation_mode": metadata.get("evaluation_mode"),
            "interpretation_source": metadata.get("interpretation_source"),
            "live_llm_attempted": metadata.get("live_llm_attempted"),
            "llm_provider": metadata.get("llm_provider"),
            "llm_model": metadata.get("llm_model"),
            "llm_error_type": metadata.get("llm_error_type"),
            "llm_error_message": metadata.get("llm_error_message"),
            "validation_passed": metadata.get("validation_passed"),
            "missing_fields": "; ".join(metadata.get("missing_fields") or []),
            "solver_ready": metadata.get("solver_ready"),
            "solver_status": metadata.get("solver_status"),
            "failure_type": metadata.get("failure_type"),
            "primary_success": metadata.get("primary_success"),
            "deterministic_fixture_used": metadata.get("deterministic_fixture_used"),
            "fallback_used": metadata.get("fallback_used"),
            "fallback_reason": metadata.get("fallback_reason"),
        })

        solve_rows.append(
            {
                "case": result["name"],
                "solve_expected": solve_checks["solve_expected"],
                "solve_success": solve_checks["solve_success"],
                "status": solve_checks["status"],
                "solver_name": solve_checks["solver_name"],
                "expected_objective": solve_checks["expected_objective"],
                "actual_objective": solve_checks["actual_objective"],
                "objective_match": solve_checks["objective_match"],
                "accepted_bid_match": solve_checks["accepted_bid_match"],
                "transport_flow_match": solve_checks["transport_flow_match"],
                "technology_activity_match": solve_checks["technology_activity_match"],
                "solver_message": solve_checks["solver_message"],
            }
        )

        reasoning_rows.extend(result.get("reasoning_results", []))
        error_counter.update(comparison["error_category_counts"])

    error_rows = [
        {"category": category, "count": count}
        for category, count in error_counter.most_common()
    ]

    return {
        "case_level_summary": pd.DataFrame(case_summary_rows),
        "interpretation_accuracy": pd.DataFrame(interpretation_rows),
        "solver_readiness_accuracy": pd.DataFrame(readiness_rows),
        "solve_accuracy": pd.DataFrame(solve_rows),
        "reasoning_prompt_success": pd.DataFrame(reasoning_rows),
        "error_category_counts": pd.DataFrame(error_rows),
        "interpretation_metadata": pd.DataFrame(metadata_rows),
    }


def plot_passed_checks_by_case(case_summary: pd.DataFrame, ax: Any = None) -> Any:
    """Matplotlib bar chart of passed checks by case."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    plot_data = case_summary.copy()
    ax.bar(plot_data["case"], plot_data["passed_checks"], color="#2f7f72")
    ax.set_title("Passed Checks by Case")
    ax.set_xlabel("Case")
    ax.set_ylabel("Passed checks")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, max(1, int(plot_data["total_applicable_checks"].max() if not plot_data.empty else 1)))
    return ax


def plot_error_categories(error_counts: pd.DataFrame, ax: Any = None) -> Any:
    """Matplotlib bar chart of error categories across cases."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    if error_counts.empty:
        ax.bar(["none"], [0], color="#7a869a")
        ax.set_ylim(0, 1)
    else:
        ax.bar(error_counts["category"], error_counts["count"], color="#8a4f7d")
    ax.set_title("Error Categories Across Cases")
    ax.set_xlabel("Error category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    return ax


def gemini_is_configured() -> bool:
    """Return whether live Gemini interpretation can be attempted."""

    return bool(os.getenv("GEMINI_API_KEY")) and os.getenv("LLM_PROVIDER", "").lower() == "gemini"


def _deterministic_fixture_enabled(config: EvaluationConfig) -> bool:
    """Return whether benchmark fixtures were explicitly requested."""

    return bool(
        config.use_deterministic_fixture
        or (not config.use_llm and config.fallback_to_expected_fixture)
        or (not config.run_interpretation and config.fallback_to_expected_fixture)
    )


def _base_interpretation_metadata(config: EvaluationConfig) -> Dict[str, Any]:
    fixture_enabled = _deterministic_fixture_enabled(config)
    if config.use_llm and config.run_interpretation:
        evaluation_mode = "live_llm"
    elif fixture_enabled:
        evaluation_mode = "deterministic_fixture"
    else:
        evaluation_mode = "not_run"

    return {
        "evaluation_mode": evaluation_mode,
        "interpretation_source": "not_run",
        "live_llm_attempted": False,
        "llm_provider": os.getenv("LLM_PROVIDER"),
        "llm_model": os.getenv("GEMINI_MODEL"),
        "gemini_model": os.getenv("GEMINI_MODEL"),
        "gemini_configured": gemini_is_configured(),
        "llm_error_type": None,
        "llm_error_message": None,
        "validation_passed": False,
        "missing_fields": [],
        "solver_ready": False,
        "solver_status": None,
        "failure_type": None,
        "primary_success": False,
        "deterministic_fixture_used": False,
        # Legacy fields retained for old notebooks/tables. Live LLM mode never
        # sets these to recover from failure.
        "fallback_used": False,
        "fallback_reason": None,
        "llm_failure": None,
    }


def _record_llm_failure(metadata: Dict[str, Any], exc: Exception) -> None:
    metadata["llm_error_type"] = exc.__class__.__name__
    metadata["llm_error_message"] = str(exc)
    metadata["llm_failure"] = f"{exc.__class__.__name__}: {exc}"
    if isinstance(exc, LLMConfigurationError):
        metadata["failure_type"] = "llm_not_configured"
    elif isinstance(exc, LLMInvalidJSONError):
        metadata["failure_type"] = "llm_invalid_json"
    elif isinstance(exc, LLMSchemaError):
        metadata["failure_type"] = "llm_schema_error"
    else:
        metadata["failure_type"] = "llm_call_failed"
    metadata["primary_success"] = False


def _validation_missing_fields(validation: Dict[str, Any]) -> List[str]:
    return list(
        dict.fromkeys(
            list(validation.get("missing_parameters", []))
            + list(validation.get("invalid_references", []))
            + list(validation.get("incomplete_technologies", []))
        )
    )


def _run_interpretation(
    case: Dict[str, Any],
    config: EvaluationConfig,
) -> Dict[str, Any]:
    metadata = _base_interpretation_metadata(config)
    fixture_enabled = _deterministic_fixture_enabled(config)

    if not config.run_interpretation:
        if not fixture_enabled:
            metadata["interpretation_source"] = "skipped_by_user_config"
            metadata["failure_type"] = "solver_not_ready"
            return {"semantic_plan": None, "state": None, "metadata": metadata}
        artifacts = build_problem_artifacts_from_semantic_plan(case["expected_plan"])
        metadata["interpretation_source"] = "deterministic_fixture_skipped_interpretation"
        metadata["deterministic_fixture_used"] = True
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = "interpretation disabled by user config"
        return {
            "semantic_plan": None,
            "state": artifacts["problem_state"],
            "market_instance": artifacts["market_instance"],
            "metadata": metadata,
        }

    if config.use_llm:
        metadata["interpretation_source"] = "live_llm_pipeline"
        metadata["live_llm_attempted"] = True
        try:
            artifacts = interpret_problem_from_text(case["prose"])
        except Exception as exc:
            _record_llm_failure(metadata, exc)
            if metadata["failure_type"] == "llm_not_configured":
                print(f"Live LLM evaluation failed: {metadata['llm_error_message']}")
            return {"semantic_plan": None, "state": None, "metadata": metadata}

        state = artifacts.get("problem_state")
        validation = validate_state(state) if isinstance(state, ProblemState) else {}
        metadata["validation_passed"] = bool(validation.get("solver_ready", False))
        metadata["missing_fields"] = _validation_missing_fields(validation)
        metadata["solver_ready"] = bool(validation.get("solver_ready", False))
        if not metadata["solver_ready"]:
            metadata["failure_type"] = "incomplete_problem_state"
        return {
            "semantic_plan": artifacts.get("semantic_plan"),
            "state": state,
            "market_instance": artifacts.get("market_instance"),
            "validation_result": validation,
            "metadata": metadata,
        }

    if fixture_enabled:
        artifacts = build_problem_artifacts_from_semantic_plan(case["expected_plan"])
        metadata["evaluation_mode"] = "deterministic_fixture"
        metadata["interpretation_source"] = "deterministic_fixture"
        metadata["deterministic_fixture_used"] = True
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = "live LLM disabled; deterministic fixture explicitly requested"
        return {
            "semantic_plan": artifacts["semantic_plan"],
            "state": artifacts["problem_state"],
            "market_instance": artifacts["market_instance"],
            "metadata": metadata,
        }

    metadata["interpretation_source"] = "skipped_by_user_config"
    metadata["failure_type"] = "solver_not_ready"
    return {"semantic_plan": None, "state": None, "metadata": metadata}


def _filter_cases(
    cases: Sequence[Dict[str, Any]],
    selected_names: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if selected_names is None:
        return list(cases)
    requested = [name for name in selected_names if name]
    by_name = {case["name"]: case for case in cases}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        valid = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown selected case(s): {', '.join(unknown)}. Valid cases: {valid}")
    return [by_name[name] for name in requested]


def _filter_reasoning_prompts(
    prompt_subset: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if prompt_subset is None:
        return list(REASONING_PROMPTS)
    requested = [prompt_id for prompt_id in prompt_subset if prompt_id]
    by_id = {prompt["id"]: prompt for prompt in REASONING_PROMPTS}
    unknown = [prompt_id for prompt_id in requested if prompt_id not in by_id]
    if unknown:
        valid = ", ".join(sorted(by_id))
        raise ValueError(f"Unknown reasoning prompt(s): {', '.join(unknown)}. Valid prompts: {valid}")
    return [by_id[prompt_id] for prompt_id in requested]


def _select_or_run_solve_result(
    state: Optional[ProblemState],
    validation: Dict[str, Any],
    interpretation: Dict[str, Any],
    attempt_solve: bool,
) -> Dict[str, Any]:
    existing = interpretation.get("solve_result")
    if isinstance(existing, dict) and existing:
        return existing
    if not attempt_solve:
        return _skipped_solve_result("solve attempt disabled")
    if not isinstance(state, ProblemState):
        return _skipped_solve_result("ProblemState was not created")
    if not validation.get("solver_ready", False):
        return _skipped_solve_result("state is not solver-ready")

    try:
        model = build_model_from_state(state)
        raw_result = solve_model(model)
    except Exception as exc:
        return _failed_solve_result(exc)
    solve_dict = raw_result.to_dict() if hasattr(raw_result, "to_dict") else dict(raw_result)
    try:
        SolverResults.from_solve_result(raw_result, state)
    except Exception:
        pass
    return solve_dict


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


def _failed_solve_result(exc: Exception) -> Dict[str, Any]:
    return {
        "success": False,
        "status": "failed",
        "message": f"{exc.__class__.__name__}: {exc}",
        "objective_value": None,
        "solver_time": 0.0,
        "solution": {},
        "termination_condition": "solver_failed",
        "solver_name": None,
    }


def _expected_solution_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        "canonical_case_a": {
            "objective_value": 500.0,
            "bid_allocations": {"B1": 50.0, "B2": 50.0},
            "transport_flows": {"('N1', 'N2')": 50.0},
            "technology_extents": {},
        },
        "paraphrased_case_a": {
            "objective_value": 500.0,
            "bid_allocations": {"B1": 50.0, "B2": 50.0},
            "transport_flows": {"('N1', 'N2')": 50.0},
            "technology_extents": {},
        },
        "negative_bid_case_b": {
            "objective_value": 240.0,
            "bid_allocations": {"B1": 40.0, "B2": 40.0},
            "transport_flows": {"('N1', 'N2')": 40.0},
            "technology_extents": {},
        },
        "transformation_case_c": {
            "objective_value": 400.0,
            "bid_allocations": {"B1": 50.0, "B2": 40.0},
            "transport_flows": {"('N1', 'N2')": 40.0},
            "technology_extents": {"K1": 50.0},
        },
    }


def _case_family_for_name(name: str) -> str:
    if "negative" in name:
        return "Case B"
    if "transformation" in name:
        return "Case C"
    return "Case A"


def _problem_type_for_name(name: str) -> str:
    if "negative" in name:
        return "case_b"
    if "transformation" in name:
        return "case_c"
    return "case_a"


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
    if raw.get("name") is None:
        raw.pop("name", None)
    return raw


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
    if field_name in {"capacity", "quantity", "price"} and (
        isinstance(expected, (int, float)) or isinstance(actual, (int, float))
    ):
        return "wrong_numeric_value"
    return "wrong_value"


def _missing_field_category(field_name: str) -> str:
    return {
        "capacity": "missing_capacity",
        "quantity": "missing_quantity",
        "price": "missing_price",
        "yield_coefficients": "missing_yield_coefficients",
    }.get(field_name, "missing_field")


def _empty_failed_comparison(reason: str) -> Dict[str, Any]:
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
        "missing_fields": [],
        "wrong_numeric_values": [],
        "wrong_ownership_relations": [],
        "wrong_node_product_mappings": [],
        "extra_invented_entities": [],
    }


def _extract_solution_block(solution: Dict[str, Any], block_name: str) -> Dict[str, float]:
    block = solution.get(block_name, {}) if isinstance(solution, dict) else {}
    if not isinstance(block, dict):
        return {}
    return {str(key): float(value) for key, value in block.items() if value is not None}


def _dict_matches_expected(
    actual: Dict[str, float],
    expected: Optional[Dict[str, float]],
    tolerance: float,
) -> Optional[bool]:
    if expected is None:
        return None
    if set(actual.keys()) != set(expected.keys()):
        return False
    return all(abs(float(actual[key]) - float(expected[key])) <= tolerance for key in expected)


__all__ = [
    "EVALUATION_PRESETS",
    "EvaluationConfig",
    "REASONING_PROMPTS",
    "build_evaluation_config",
    "build_output_tables",
    "build_paper_grade_cases",
    "compare_problem_states_paper_grade",
    "evaluate_case",
    "evaluate_solve_accuracy",
    "gemini_is_configured",
    "plot_error_categories",
    "plot_passed_checks_by_case",
    "run_paper_grade_evaluation",
    "run_reasoning_prompt_battery",
    "semantic_plan_from_state",
]
