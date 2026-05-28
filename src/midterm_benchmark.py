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
DEFAULT_Q1_BENCHMARK_DIR = REPO_ROOT / "Benchmarks" / "midterm1" / "manure_q1"
DEFAULT_Q2_BENCHMARK_DIR = REPO_ROOT / "Benchmarks" / "midterm1" / "manure_q2"
DEFAULT_Q3_BENCHMARK_DIR = REPO_ROOT / "Benchmarks" / "midterm1" / "manure_q3"
DEFAULT_Q4_BENCHMARK_DIR = REPO_ROOT / "Benchmarks" / "midterm1" / "manure_q4"
DEFAULT_BENCHMARK_DIR = DEFAULT_Q1_BENCHMARK_DIR
TOLERANCE = 1e-6

MIDTERM_REASONING_PROMPTS: List[Dict[str, Any]] = [
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

MIDTERM_Q2_REASONING_PROMPTS: List[Dict[str, Any]] = [
    *MIDTERM_REASONING_PROMPTS,
    {
        "id": "menomonie_diversion",
        "label": "Menomonie diversion economics",
        "prompt": (
            "Explain economically why manure is diverted away from Menomonie under the DNR remediation charge."
        ),
        "required_terms": ["Menomonie", "negative"],
    },
]

MIDTERM_Q3_REASONING_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "all_manure_with_payment",
        "label": "All manure with removal incentive",
        "prompt": "Explain why the dairy farmer can now get rid of all manure.",
        "required_terms": ["0.7", "all"],
    },
    {
        "id": "supplier_payment_role",
        "label": "Supplier payment role",
        "prompt": "Explain the role of the 0.7 $/ton payment from the dairy farmer.",
        "required_terms": ["0.7", "payment"],
    },
    {
        "id": "route_economics",
        "label": "Route economics",
        "prompt": "Explain the route economics for Menomonie and Black River Falls.",
        "required_terms": ["Menomonie", "Black River Falls"],
    },
    {
        "id": "node_balances",
        "label": "Node balance verification",
        "prompt": "Verify the node balances.",
        "required_terms": ["balance"],
    },
]

MIDTERM_Q4_REASONING_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "technology_use",
        "label": "Compost technology use",
        "prompt": "Explain why the compost technology is used in the optimal solution.",
        "required_terms": ["compost", "technology"],
    },
    {
        "id": "pathway_economics",
        "label": "Pathway economics",
        "prompt": "Explain the pathway economics for Menomonie, Black River Falls, and Madison compost.",
        "required_terms": ["Menomonie", "Black River Falls", "Madison"],
    },
    {
        "id": "all_manure_q4",
        "label": "All manure in Q4",
        "prompt": "Can the dairy farmer get rid of all its manure in Question 4? Why or why not?",
        "required_terms": ["all", "manure"],
    },
    {
        "id": "q3_q4_comparison",
        "label": "Q3 versus Q4 profit",
        "prompt": "Compare the total profit in Question 4 with Question 3. Did the technology increase total profit? By how much?",
        "required_terms": ["4750", "5800", "1050"],
    },
    {
        "id": "primal_lp_q4",
        "label": "Q4 primal LP",
        "prompt": "Formulate the Q4 manure-compost problem as a primal linear program in LaTeX.",
        "required_terms": ["Compost", "yield"],
    },
    {
        "id": "yield_balance",
        "label": "Yield and balances",
        "prompt": "Explain how the technology yield coefficient links manure and compost balances.",
        "required_terms": ["0.1", "balance"],
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
CANONICAL_MADISON_COMPOST_DEMAND_ID = "Madison"
CANONICAL_MANURE_PRODUCT_ID = "Manure"
CANONICAL_COMPOST_PRODUCT_ID = "Compost"
CANONICAL_COMPOSTER_ID = "Composter"

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
    """Runtime switches for the midterm manure benchmarks."""

    mode: str = "guided"
    prompt_ids: Optional[Tuple[str, ...]] = None
    reasoning_prompt_ids: Optional[Tuple[str, ...]] = None
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


def build_midterm_manure_q2_expected_plan(prompt_id: str = "canonical") -> Dict[str, Any]:
    """Return the deterministic fixture plan for the DNR policy Q2 benchmark."""

    plan = build_midterm_manure_expected_plan(prompt_id="canonical")
    plan["problem_title"] = "Midterm 1 Problem 1 Question 2: Manure Management With DNR Policy"
    plan["problem_type"] = "case_b"
    for bid in plan["bids"]:
        if bid["owner_id"] == "Menomonie":
            bid["price"] = -0.5
    plan["missing_information"] = []
    plan["ambiguities"] = []
    return plan


def build_midterm_manure_q3_expected_plan(prompt_id: str = "canonical") -> Dict[str, Any]:
    """Return the deterministic fixture plan for Q3 with a manure removal incentive."""

    plan = build_midterm_manure_q2_expected_plan(prompt_id=prompt_id)
    plan["problem_title"] = "Midterm 1 Problem 1 Question 3: Manure Management With Removal Incentive"
    plan["problem_type"] = "case_b"
    for bid in plan["bids"]:
        if bid["owner_id"] == "Dairy/EauClaire":
            bid["price"] = -0.7
    plan["missing_information"] = []
    plan["ambiguities"] = []
    return plan


def build_midterm_manure_q4_expected_plan(prompt_id: str = "canonical") -> Dict[str, Any]:
    """Return the deterministic fixture plan for Q4 with compost transformation."""

    plan = {
        "problem_title": "Midterm 1 Problem 1 Question 4: Manure Management With Composting",
        "problem_type": "case_c",
        "nodes": [
            {"id": "DF", "name": "Eau Claire dairy farmer"},
            {"id": "CF", "name": "Menomonie"},
            {"id": "SF", "name": "Black River Falls"},
            {"id": "Composter", "name": "Composter"},
            {"id": "DC", "name": "Madison compost consumer"},
        ],
        "products": [
            {"id": "DM", "name": "dairy manure"},
            {"id": "Compost", "name": "compost"},
        ],
        "suppliers": [
            {
                "id": "DF",
                "node": "DF",
                "product": "DM",
                "capacity": 1000.0,
            }
        ],
        "consumers": [
            {
                "id": "CF",
                "node": "CF",
                "product": "DM",
                "capacity": 500.0,
            },
            {
                "id": "SF",
                "node": "SF",
                "product": "DM",
                "capacity": 500.0,
            },
            {
                "id": "DC",
                "node": "DC",
                "product": "Compost",
                "capacity": 100.0,
            },
        ],
        "transport_links": [
            {
                "id": "DF_to_CF",
                "origin": "DF",
                "destination": "CF",
                "product": "DM",
                "capacity": 1000.0,
                "cost": 0.1,
            },
            {
                "id": "DF_to_SF",
                "origin": "DF",
                "destination": "SF",
                "product": "DM",
                "capacity": 1000.0,
                "cost": 0.2,
            },
            {
                "id": "DF_to_Composter",
                "origin": "DF",
                "destination": "Composter",
                "product": "DM",
                "capacity": 1000.0,
                "cost": 0.0,
            },
            {
                "id": "Composter_to_DC",
                "origin": "Composter",
                "destination": "DC",
                "product": "Compost",
                "capacity": 1000.0,
                "cost": 1.0,
            },
        ],
        "technologies": [
            {
                "id": "Composter",
                "node": "Composter",
                "capacity": 500.0,
                "cost": 1.0,
                "yield_coefficients": {"DM": -1.0, "Compost": 0.1},
            }
        ],
        "bids": [
            {
                "id": "B_DF_DM",
                "owner_id": "DF",
                "owner_type": "supplier",
                "product_id": "DM",
                "price": -0.7,
                "quantity": 1000.0,
            },
            {
                "id": "B_CF_DM",
                "owner_id": "CF",
                "owner_type": "consumer",
                "product_id": "DM",
                "price": -0.5,
                "quantity": 500.0,
            },
            {
                "id": "B_SF_DM",
                "owner_id": "SF",
                "owner_type": "consumer",
                "product_id": "DM",
                "price": 1.5,
                "quantity": 500.0,
            },
            {
                "id": "B_DC_Compost",
                "owner_id": "DC",
                "owner_type": "consumer",
                "product_id": "Compost",
                "price": 100.0,
                "quantity": 100.0,
            },
        ],
        "missing_information": [],
        "ambiguities": [],
    }
    return plan


def build_midterm_manure_cases(
    prompt_ids: Optional[Sequence[str]] = None,
    benchmark_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build prompt cases with deterministic expected ProblemState fixtures."""

    return _build_midterm_manure_cases(
        prompt_ids=prompt_ids,
        benchmark_dir=benchmark_dir,
        expected_plan_builder=build_midterm_manure_expected_plan,
        reasoning_prompts=MIDTERM_REASONING_PROMPTS,
    )


def build_midterm_manure_q2_cases(
    prompt_ids: Optional[Sequence[str]] = None,
    benchmark_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build Q2 prompt cases with deterministic expected ProblemState fixtures."""

    return _build_midterm_manure_cases(
        prompt_ids=prompt_ids,
        benchmark_dir=benchmark_dir or DEFAULT_Q2_BENCHMARK_DIR,
        expected_plan_builder=build_midterm_manure_q2_expected_plan,
        reasoning_prompts=MIDTERM_Q2_REASONING_PROMPTS,
    )


def build_midterm_manure_q3_cases(
    prompt_ids: Optional[Sequence[str]] = None,
    benchmark_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build Q3 prompt cases with deterministic expected ProblemState fixtures."""

    return _build_midterm_manure_cases(
        prompt_ids=prompt_ids,
        benchmark_dir=benchmark_dir or DEFAULT_Q3_BENCHMARK_DIR,
        expected_plan_builder=build_midterm_manure_q3_expected_plan,
        reasoning_prompts=MIDTERM_Q3_REASONING_PROMPTS,
    )


def build_midterm_manure_q4_cases(
    prompt_ids: Optional[Sequence[str]] = None,
    benchmark_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build Q4 prompt cases with deterministic expected ProblemState fixtures."""

    return _build_midterm_manure_cases(
        prompt_ids=prompt_ids,
        benchmark_dir=benchmark_dir or DEFAULT_Q4_BENCHMARK_DIR,
        expected_plan_builder=build_midterm_manure_q4_expected_plan,
        reasoning_prompts=MIDTERM_Q4_REASONING_PROMPTS,
    )


def _build_midterm_manure_cases(
    prompt_ids: Optional[Sequence[str]],
    benchmark_dir: Optional[Path],
    expected_plan_builder: Any,
    reasoning_prompts: Sequence[Dict[str, Any]],
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
        expected_plan = expected_plan_builder(prompt_id)
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
                "reasoning_prompts": list(reasoning_prompts),
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


def run_midterm_manure_q2_benchmark(
    config: Optional[MidtermBenchmarkConfig] = None,
    benchmark_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the manure Q2 DNR policy benchmark and return raw results plus tables."""

    runtime_config = config or MidtermBenchmarkConfig()
    files = load_benchmark_files(benchmark_dir or DEFAULT_Q2_BENCHMARK_DIR)
    reference_solution = files["reference_solution"]
    cases = build_midterm_manure_q2_cases(
        prompt_ids=runtime_config.prompt_ids,
        benchmark_dir=benchmark_dir or DEFAULT_Q2_BENCHMARK_DIR,
    )
    case_results = [
        evaluate_midterm_prompt_case(case, reference_solution, runtime_config)
        for case in cases
    ]
    tables = build_midterm_output_tables(case_results)
    return {
        "metadata": {
            "benchmark_id": reference_solution.get("benchmark_id", "midterm1_manure_q2"),
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


def run_midterm_manure_q3_benchmark(
    config: Optional[MidtermBenchmarkConfig] = None,
    benchmark_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the manure Q3 removal-incentive benchmark and return raw results plus tables."""

    runtime_config = config or MidtermBenchmarkConfig()
    files = load_benchmark_files(benchmark_dir or DEFAULT_Q3_BENCHMARK_DIR)
    reference_solution = files["reference_solution"]
    cases = build_midterm_manure_q3_cases(
        prompt_ids=runtime_config.prompt_ids,
        benchmark_dir=benchmark_dir or DEFAULT_Q3_BENCHMARK_DIR,
    )
    case_results = [
        evaluate_midterm_prompt_case(case, reference_solution, runtime_config)
        for case in cases
    ]
    tables = build_midterm_output_tables(case_results)
    return {
        "metadata": {
            "benchmark_id": reference_solution.get("benchmark_id", "midterm1_manure_q3"),
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


def run_midterm_manure_q4_benchmark(
    config: Optional[MidtermBenchmarkConfig] = None,
    benchmark_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the manure Q4 composting benchmark and return raw results plus tables."""

    runtime_config = config or MidtermBenchmarkConfig()
    files = load_benchmark_files(benchmark_dir or DEFAULT_Q4_BENCHMARK_DIR)
    reference_solution = files["reference_solution"]
    cases = build_midterm_manure_q4_cases(
        prompt_ids=runtime_config.prompt_ids,
        benchmark_dir=benchmark_dir or DEFAULT_Q4_BENCHMARK_DIR,
    )
    case_results = [
        evaluate_midterm_prompt_case(case, reference_solution, runtime_config)
        for case in cases
    ]
    tables = build_midterm_output_tables(case_results)
    return {
        "metadata": {
            "benchmark_id": reference_solution.get("benchmark_id", "midterm1_manure_q4"),
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
        semantic_plan=interpretation.get("semantic_plan"),
        prose_input=case.get("prose"),
    )
    removal_incentive_diagnostics = build_supplier_removal_incentive_diagnostics(
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
        prompt_specs = _filter_reasoning_prompts(
            case.get("reasoning_prompts"),
            runtime_config.reasoning_prompt_ids,
        )
        reasoning_results = run_midterm_reasoning_prompt_battery(
            state=actual_state,
            case_name=case["name"],
            mode=runtime_config.mode,
            use_llm=runtime_config.use_llm_for_reasoning and gemini_is_configured(),
            prompt_specs=prompt_specs,
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
        "removal_incentive_diagnostics": removal_incentive_diagnostics,
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
    technology_activity_match = _dict_matches_expected(
        components.get("technology_activity", {}),
        reference_solution.get("technology_activity", {}),
        tolerance,
    )
    technology_outputs_match = _dict_matches_expected(
        components.get("technology_outputs", {}),
        reference_solution.get("technology_outputs", {}),
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
    expected_supply_contribution = reference_solution.get("supply_contribution")
    if expected_supply_contribution is None and reference_solution.get("supply_cost") is not None:
        expected_supply_contribution = -float(reference_solution.get("supply_cost"))
    supply_contribution_match = _float_or_none_matches(
        components.get("supply_contribution"),
        expected_supply_contribution,
        tolerance,
    )
    technology_cost_match = _float_or_none_matches(
        components.get("technology_cost"),
        reference_solution.get("technology_cost", 0.0),
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
        "technology_activity_match": technology_activity_match,
        "technology_outputs_match": technology_outputs_match,
        "balance_match": balance_match,
        "demand_revenue_match": demand_revenue_match,
        "transport_cost_match": transport_cost_match,
        "supply_cost_match": supply_cost_match,
        "supply_contribution_match": supply_contribution_match,
        "technology_cost_match": technology_cost_match,
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
    semantic_plan: Optional[Dict[str, Any]] = None,
    prose_input: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the midterm manure benchmark without depending on exact entity IDs."""

    if not isinstance(state, ProblemState):
        empty = _empty_primary_metrics("ProblemState was not created")
        return empty

    count_rows = _semantic_count_metric_rows(state, reference_solution)
    parameter_rows = _parameter_multiset_metric_rows(state, solve_result, reference_solution, tolerance)
    topology_rows = _topology_metric_rows(state, reference_solution)
    technology_rows = _technology_yield_metric_rows(state, reference_solution, tolerance)
    route_rows = _route_economics_metric_rows(state, reference_solution, tolerance)
    route_association_rows = _route_association_metric_rows(state, reference_solution, tolerance)
    solver_rows = _solver_aggregate_metric_rows(state, solve_result, reference_solution, tolerance)
    balance_rows = _balance_residual_metric_rows(state, solve_result, tolerance)
    solve_correctness_rows = _solve_correctness_metric_rows(
        state,
        solve_result,
        reference_solution,
        tolerance,
    )
    formulation_rows = _formulation_completeness_metric_rows(
        state=state,
        reference_solution=reference_solution,
        count_rows=count_rows,
        parameter_rows=parameter_rows,
        topology_rows=topology_rows,
        technology_rows=technology_rows,
        route_association_rows=route_association_rows,
        tolerance=tolerance,
        semantic_plan=semantic_plan,
        prose_input=prose_input,
    )

    semantic_structure_pass = all(
        bool(row["pass"])
        for rows in (count_rows, parameter_rows, topology_rows, route_rows)
        for row in rows
    )
    technology_structure_pass = all(bool(row["pass"]) for row in technology_rows)
    route_association_pass = all(bool(row["pass"]) for row in route_association_rows)
    solver_aggregate_pass = all(bool(row["pass"]) for row in solver_rows)
    balance_residual_pass = all(bool(row["pass"]) for row in balance_rows)
    formulation_completeness_pass = all(bool(row["pass"]) for row in formulation_rows)
    solve_correctness_pass = all(bool(row["pass"]) for row in solve_correctness_rows)
    reasoning_rows = _reasoning_readiness_metric_rows(
        formulation_completeness_pass=formulation_completeness_pass,
        route_association_pass=route_association_pass,
        technology_structure_pass=technology_structure_pass,
    )
    reasoning_ready_pass = all(bool(row["pass"]) for row in reasoning_rows)
    primary_success = (
        formulation_completeness_pass
        and solve_correctness_pass
        and reasoning_ready_pass
    )
    failure_type = _classify_primary_failure(
        formulation_completeness_pass=formulation_completeness_pass,
        solve_correctness_pass=solve_correctness_pass,
        reasoning_ready_pass=reasoning_ready_pass,
        route_association_rows=route_association_rows,
    )

    return {
        "semantic_count_metrics": count_rows,
        "parameter_multiset_metrics": parameter_rows,
        "topology_metrics": topology_rows,
        "technology_yield_metrics": technology_rows,
        "route_economics_metrics": route_rows,
        "route_association_metrics": route_association_rows,
        "solver_aggregate_metrics": solver_rows,
        "balance_residual_metrics": balance_rows,
        "formulation_completeness_metrics": formulation_rows,
        "solve_correctness_metrics": solve_correctness_rows,
        "reasoning_readiness_metrics": reasoning_rows,
        "semantic_structure_pass": semantic_structure_pass,
        "technology_structure_pass": technology_structure_pass,
        "route_association_pass": route_association_pass,
        "solver_aggregate_pass": solver_aggregate_pass,
        "balance_residual_pass": balance_residual_pass,
        "formulation_completeness_pass": formulation_completeness_pass,
        "solve_correctness_pass": solve_correctness_pass,
        "reasoning_ready_pass": reasoning_ready_pass,
        "primary_success": primary_success,
        "failure_type": failure_type,
    }


def build_supplier_removal_incentive_diagnostics(
    state: Optional[ProblemState],
    solve_result: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float = TOLERANCE,
) -> List[Dict[str, Any]]:
    """Build Q3-specific diagnostics for a supplier payment to remove manure."""

    metrics = _expected_semantic_metrics(reference_solution)
    expects_supplier_payment = (
        "supplier_removal_payment" in metrics
        or bool(metrics.get("negative_supplier_bid_expected", False))
    )
    if not expects_supplier_payment:
        return []

    if not isinstance(state, ProblemState):
        return [
            _metric_row(
                "supplier_removal_incentive_state_created",
                True,
                False,
                False,
                details="ProblemState was not created.",
            )
        ]

    expected_payment = metrics.get("supplier_removal_payment")
    expected_payments = (
        [float(expected_payment)]
        if expected_payment is not None
        else sorted(
            abs(float(value))
            for value in metrics.get("supplier_bid_prices", [])
            if float(value) < 0
        )
    )
    supplier_bid_prices = [
        float(bid.price)
        for bid in state.bids
        if bid.owner_type == "supplier"
    ]
    actual_payments = sorted(abs(price) for price in supplier_bid_prices if price < 0)
    aggregates = _solver_aggregates(state, solve_result)
    actual_supply_cost = aggregates.get("supply_cost")
    actual_total_payment = (
        -float(actual_supply_cost)
        if actual_supply_cost is not None and float(actual_supply_cost) < 0
        else 0.0
        if actual_supply_cost is not None
        else None
    )
    expected_supply_cost = reference_solution.get("supply_cost")
    expected_total_payment = (
        -float(expected_supply_cost)
        if expected_supply_cost is not None and float(expected_supply_cost) < 0
        else None
    )
    route_net_values = _route_net_values(state)
    expected_route_values = _expected_route_net_values(reference_solution)
    route_values_match = _multiset_matches(route_net_values, expected_route_values, tolerance)

    rows = [
        _metric_row(
            "supplier_bid_represents_payment",
            "at least one negative supplier bid",
            supplier_bid_prices,
            any(price < 0 for price in supplier_bid_prices),
            details="The dairy payment is encoded as a negative supplier bid.",
        ),
        _metric_row(
            "supplier_removal_payment_per_ton",
            expected_payments,
            actual_payments,
            _multiset_matches(actual_payments, expected_payments, tolerance),
        ),
        _metric_row(
            "all_routes_profitable_after_payment",
            all(value >= -tolerance for value in expected_route_values),
            all(value >= -tolerance for value in route_net_values),
            bool(route_net_values) and all(value >= -tolerance for value in route_net_values) and route_values_match,
            details={"route_net_values": route_net_values},
        ),
    ]
    if expected_total_payment is not None:
        rows.append(
            _numeric_metric_row(
                "total_removal_incentive_value",
                expected_total_payment,
                actual_total_payment,
                tolerance,
            )
        )
    return rows


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
            "technology_activity": {},
            "technology_outputs": {},
            "demand_revenue": None,
            "transport_cost": None,
            "supply_cost": None,
            "technology_cost": None,
            "supply_contribution": None,
            "balance_checks": {},
            "raw_component_rows": [],
        }

    solution = solve_result.get("solution", {}) if isinstance(solve_result, dict) else {}
    q_values = _solution_block(solution, "q")
    f_values = _solution_block(solution, "f")
    x_values = _solution_block(solution, "x")

    bids_by_id = {bid.id: bid for bid in state.bids}
    suppliers_by_id = {supplier.id: supplier for supplier in state.suppliers}
    consumers_by_id = {consumer.id: consumer for consumer in state.consumers}
    technologies_by_id = {technology.id: technology for technology in state.technologies}

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
            label = _supplier_reference_key(supplier, bid.owner_id, bid.product_id, state=state)
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
            label = _consumer_reference_key(consumer, bid.owner_id, bid.product_id, state=state)
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
        route_key = _route_reference_key(origin, destination, product_id, state=state)
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

    technology_activity: Dict[str, float] = {}
    technology_outputs: Dict[str, float] = {}
    technology_cost = 0.0
    for raw_key, activity in x_values.items():
        technology = technologies_by_id.get(str(raw_key))
        if technology is None:
            technology = _find_technology_for_key(state, str(raw_key))
        label = _technology_reference_key(technology, str(raw_key))
        technology_activity[label] = technology_activity.get(label, 0.0) + activity
        if technology is not None:
            technology_cost += float(getattr(technology, "cost", 0.0) or 0.0) * activity
            for product_id, coefficient in technology.yield_coefficients.items():
                if coefficient > 0:
                    product_key = _canonical_product_key_for_state(state, product_id)
                    technology_outputs[product_key] = technology_outputs.get(product_key, 0.0) + (
                        float(coefficient) * activity
                    )
        raw_component_rows.append(
            {
                "component": "technology_activity",
                "raw_id": str(raw_key),
                "raw_bid_id": None,
                "raw_node": getattr(technology, "node", None),
                "raw_product": None,
                "canonical_resolved_id": label,
                "actual_value": activity,
            }
        )

    balance_checks = (
        _compute_midterm_balance_checks(
            accepted_supply=accepted_supply,
            accepted_demands=accepted_demands,
            transport_flows=transport_flows,
        )
        if len(state.products) == 1 and not state.technologies
        else _compute_generic_balance_checks(state, solve_result)
    )

    return {
        "accepted_supply": accepted_supply,
        "accepted_demands": accepted_demands,
        "transport_flows": transport_flows,
        "technology_activity": technology_activity,
        "technology_outputs": technology_outputs,
        "demand_revenue": demand_revenue,
        "transport_cost": transport_cost,
        "supply_cost": supply_cost,
        "technology_cost": technology_cost,
        "supply_contribution": -supply_cost,
        "balance_checks": balance_checks,
        "raw_component_rows": raw_component_rows,
    }


def run_midterm_reasoning_prompt_battery(
    state: ProblemState,
    case_name: str,
    mode: str = "guided",
    use_llm: bool = False,
    prompt_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run the requested manure-specific reasoning prompts."""

    rows = []
    for prompt_spec in (prompt_specs or MIDTERM_REASONING_PROMPTS):
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
        required_terms = list(prompt_spec.get("required_terms", []))
        term_hits = {
            term: term.lower() in str(response_text).lower()
            for term in required_terms
        }
        required_terms_present = all(term_hits.values()) if required_terms else None
        rows.append(
            {
                "case": case_name,
                "prompt_id": prompt_spec["id"],
                "prompt_label": prompt_spec["label"],
                "success": success,
                "required_terms": ", ".join(required_terms),
                "required_terms_present": required_terms_present,
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


def _filter_reasoning_prompts(
    prompt_specs: Optional[Sequence[Dict[str, Any]]],
    prompt_ids: Optional[Sequence[str]],
) -> Sequence[Dict[str, Any]]:
    prompts = list(prompt_specs or MIDTERM_REASONING_PROMPTS)
    if prompt_ids is None:
        return prompts
    by_id = {prompt["id"]: prompt for prompt in prompts}
    unknown = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
    if unknown:
        valid = ", ".join(sorted(by_id))
        raise ValueError(f"Unknown reasoning prompt id(s): {', '.join(unknown)}. Valid ids: {valid}")
    return [by_id[prompt_id] for prompt_id in prompt_ids]


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
    technology_rows = []
    route_rows = []
    route_association_rows = []
    aggregate_rows = []
    residual_rows = []
    formulation_rows = []
    solve_correctness_rows = []
    reasoning_readiness_rows = []
    removal_rows = []

    for result in case_results:
        checks = result["solution_checks"]
        primary_metrics = result.get("primary_metrics", {})
        comparison = result["state_comparison"]
        metadata = result["interpretation_metadata"]
        blocking_count = len(comparison["blocking_errors"])
        benign_count = len(comparison["benign_extra_name_fields"])
        benign_identifier_count = len(comparison.get("benign_identifier_mismatches", []))
        semantic_structure_pass = bool(primary_metrics.get("semantic_structure_pass", False))
        technology_structure_pass = bool(primary_metrics.get("technology_structure_pass", True))
        route_association_pass = bool(primary_metrics.get("route_association_pass", True))
        solver_aggregate_pass = bool(primary_metrics.get("solver_aggregate_pass", False))
        balance_residual_pass = bool(primary_metrics.get("balance_residual_pass", False))
        formulation_completeness_pass = bool(primary_metrics.get("formulation_completeness_pass", False))
        solve_correctness_pass = bool(primary_metrics.get("solve_correctness_pass", False))
        reasoning_ready_pass = bool(primary_metrics.get("reasoning_ready_pass", False))
        primary_success = bool(primary_metrics.get("primary_success", False))
        failure_type = primary_metrics.get("failure_type", "unknown")

        pass_flags = [
            result["semantic_plan_created"],
            result["problem_state_created"],
            formulation_completeness_pass,
            solve_correctness_pass,
            reasoning_ready_pass,
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
                "technology_structure_pass": technology_structure_pass,
                "route_association_pass": route_association_pass,
                "solver_aggregate_pass": solver_aggregate_pass,
                "balance_residual_pass": balance_residual_pass,
                "formulation_completeness_pass": formulation_completeness_pass,
                "solve_correctness_pass": solve_correctness_pass,
                "reasoning_ready_pass": reasoning_ready_pass,
                "primary_success": primary_success,
                "failure_type": failure_type,
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
                "technology_activity_match": checks["technology_activity_match"],
                "technology_outputs_match": checks["technology_outputs_match"],
                "balance_match": checks["balance_match"],
                "demand_revenue_match": checks["demand_revenue_match"],
                "transport_cost_match": checks["transport_cost_match"],
                "supply_cost_match": checks["supply_cost_match"],
                "supply_contribution_match": checks["supply_contribution_match"],
                "technology_cost_match": checks["technology_cost_match"],
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
        for row in primary_metrics.get("technology_yield_metrics", []):
            technology_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("route_economics_metrics", []):
            route_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("route_association_metrics", []):
            route_association_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("solver_aggregate_metrics", []):
            aggregate_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("balance_residual_metrics", []):
            residual_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("formulation_completeness_metrics", []):
            formulation_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("solve_correctness_metrics", []):
            solve_correctness_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in primary_metrics.get("reasoning_readiness_metrics", []):
            reasoning_readiness_rows.append({"prompt_id": result["prompt_id"], **row})
        for row in result.get("removal_incentive_diagnostics", []):
            removal_rows.append({"prompt_id": result["prompt_id"], **row})

    return {
        "case_summary": pd.DataFrame(case_summary_rows),
        "solve_accuracy": pd.DataFrame(solve_rows),
        "interpretation_errors": pd.DataFrame(interpretation_rows),
        "semantic_count_metrics": pd.DataFrame(count_rows),
        "parameter_multiset_metrics": pd.DataFrame(parameter_rows),
        "topology_metrics": pd.DataFrame(topology_rows),
        "technology_yield_metrics": pd.DataFrame(technology_rows),
        "route_economics_metrics": pd.DataFrame(route_rows),
        "route_association_metrics": pd.DataFrame(route_association_rows),
        "solver_aggregate_metrics": pd.DataFrame(aggregate_rows),
        "balance_residual_metrics": pd.DataFrame(residual_rows),
        "formulation_completeness_metrics": pd.DataFrame(formulation_rows),
        "solve_correctness_metrics": pd.DataFrame(solve_correctness_rows),
        "reasoning_readiness_metrics": pd.DataFrame(reasoning_readiness_rows),
        "removal_incentive_diagnostics": pd.DataFrame(removal_rows),
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
        "technology_yield_metrics": [],
        "route_economics_metrics": [],
        "route_association_metrics": [],
        "solver_aggregate_metrics": [],
        "balance_residual_metrics": [],
        "formulation_completeness_metrics": [
            _formulation_metric_row(
                "problem_state_created",
                True,
                False,
                False,
                reason,
            )
        ],
        "solve_correctness_metrics": [],
        "reasoning_readiness_metrics": _reasoning_readiness_metric_rows(
            formulation_completeness_pass=False,
            route_association_pass=False,
            technology_structure_pass=False,
        ),
        "semantic_structure_pass": False,
        "technology_structure_pass": False,
        "route_association_pass": False,
        "solver_aggregate_pass": False,
        "balance_residual_pass": False,
        "formulation_completeness_pass": False,
        "solve_correctness_pass": False,
        "reasoning_ready_pass": False,
        "primary_success": False,
        "failure_type": "incomplete_formulation_and_wrong_solution",
    }


def _expected_semantic_metrics(reference_solution: Dict[str, Any]) -> Dict[str, Any]:
    metrics = reference_solution.get("expected_semantic_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _semantic_count_metric_rows(
    state: ProblemState,
    reference_solution: Dict[str, Any],
) -> List[Dict[str, Any]]:
    default_counts = {
        "products": 1,
        "source_supplier_entities": 1,
        "demand_consumer_entities": 2,
        "transport_paths": 2,
        "technologies": 0,
    }
    expected_counts = {
        **default_counts,
        **_expected_semantic_metrics(reference_solution).get("entity_counts", {}),
    }
    product_keys = _product_semantic_keys(state)
    manure_supplier_count = sum(
        1
        for supplier in state.suppliers
        if _canonical_product_key_for_state(state, supplier.product) == CANONICAL_MANURE_PRODUCT_ID
    )
    manure_consumer_count = sum(
        1
        for consumer in state.consumers
        if _canonical_product_key_for_state(state, consumer.product) == CANONICAL_MANURE_PRODUCT_ID
    )
    compost_consumer_count = sum(
        1
        for consumer in state.consumers
        if _canonical_product_key_for_state(state, consumer.product) == CANONICAL_COMPOST_PRODUCT_ID
    )
    source_nodes = {supplier.node for supplier in state.suppliers}
    compost_consumer_nodes = {
        consumer.node
        for consumer in state.consumers
        if _canonical_product_key_for_state(state, consumer.product) == CANONICAL_COMPOST_PRODUCT_ID
    }
    actual_counts = {
        "products": len(state.products),
        "source_supplier_entities": len(state.suppliers),
        "demand_consumer_entities": len(state.consumers),
        "transport_paths": len(state.transport_links),
        "technologies": len(state.technologies),
        "products_include_manure": CANONICAL_MANURE_PRODUCT_ID in product_keys,
        "products_include_compost": CANONICAL_COMPOST_PRODUCT_ID in product_keys,
        "manure_source_supplier_entities": manure_supplier_count,
        "manure_demand_consumer_entities": manure_consumer_count,
        "compost_demand_consumer_entities": compost_consumer_count,
        "manure_transport_paths_from_source": sum(
            1
            for link in state.transport_links
            if link.origin in source_nodes
            and _canonical_product_key_for_state(state, link.product) == CANONICAL_MANURE_PRODUCT_ID
        ),
        "compost_transport_paths_to_compost_sink": sum(
            1
            for link in state.transport_links
            if link.destination in compost_consumer_nodes
            and _canonical_product_key_for_state(state, link.product) == CANONICAL_COMPOST_PRODUCT_ID
        ),
    }
    return [
        _metric_row(metric, expected_counts[metric], actual_counts[metric], expected_counts[metric] == actual_counts[metric])
        for metric in expected_counts
    ]


def _parameter_multiset_metric_rows(
    state: ProblemState,
    solve_result: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    metrics = _expected_semantic_metrics(reference_solution)
    expected_supplier_capacities = metrics.get("supplier_capacities", [1000.0])
    expected_consumer_capacities = metrics.get("consumer_capacities", [500.0, 500.0])
    expected_consumer_bid_prices = metrics.get("consumer_bid_prices", [0.5, 1.5])
    expected_supplier_bid_prices = metrics.get("supplier_bid_prices", [0.0])
    expected_transport_costs = metrics.get("transport_costs", [0.1, 0.2])
    expected_transport_capacities = metrics.get(
        "transport_capacities",
        "allow_absent_or_unbounded",
    )

    supplier_bid_prices = [bid.price for bid in state.bids if bid.owner_type == "supplier"]
    consumer_bid_prices = [bid.price for bid in state.bids if bid.owner_type == "consumer"]
    supplier_capacities = [supplier.capacity for supplier in state.suppliers]
    consumer_capacities = [consumer.capacity for consumer in state.consumers]
    transport_costs = [float(getattr(link, "cost", 0.0) or 0.0) for link in state.transport_links]
    finite_transport_capacities = [
        float(link.capacity)
        for link in state.transport_links
        if link.capacity is not None
    ]
    solve_success = bool(solve_result.get("success", False))
    if expected_transport_capacities == "allow_absent_or_unbounded":
        bounded_expected_capacities = _bounded_transport_capacity_default(reference_solution)
        transport_capacity_expected_label = (
            "absent/unbounded with successful solve"
            f", or {bounded_expected_capacities}"
        )
        transport_capacity_pass = (
            _multiset_matches(finite_transport_capacities, bounded_expected_capacities, tolerance)
            or (not finite_transport_capacities and solve_success)
        )
    elif isinstance(expected_transport_capacities, (list, tuple)):
        bounded_expected_capacities = [float(value) for value in expected_transport_capacities]
        transport_capacity_expected_label = bounded_expected_capacities
        transport_capacity_pass = _multiset_matches(
            finite_transport_capacities,
            bounded_expected_capacities,
            tolerance,
        )
    else:
        bounded_expected_capacities = []
        transport_capacity_expected_label = expected_transport_capacities
        transport_capacity_pass = _multiset_matches(
            finite_transport_capacities,
            bounded_expected_capacities,
            tolerance,
        )

    return [
        _metric_row(
            "supplier_capacities",
            expected_supplier_capacities,
            supplier_capacities,
            _multiset_matches(supplier_capacities, expected_supplier_capacities, tolerance),
        ),
        _metric_row(
            "consumer_capacities",
            expected_consumer_capacities,
            consumer_capacities,
            _multiset_matches(consumer_capacities, expected_consumer_capacities, tolerance),
        ),
        _metric_row(
            "consumer_bid_prices",
            expected_consumer_bid_prices,
            consumer_bid_prices,
            _multiset_matches(consumer_bid_prices, expected_consumer_bid_prices, tolerance),
        ),
        _metric_row(
            "supplier_bid_prices",
            expected_supplier_bid_prices,
            supplier_bid_prices,
            _multiset_matches(supplier_bid_prices, expected_supplier_bid_prices, tolerance),
        ),
        _metric_row(
            "transport_costs",
            expected_transport_costs,
            transport_costs,
            _multiset_matches(transport_costs, expected_transport_costs, tolerance),
        ),
        _metric_row(
            "transport_capacities",
            transport_capacity_expected_label,
            finite_transport_capacities or "absent/unbounded",
            transport_capacity_pass,
            details="Transport capacities may be absent when routes remain nonbinding.",
        ),
    ]


def _topology_metric_rows(
    state: ProblemState,
    reference_solution: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metrics = _expected_semantic_metrics(reference_solution)
    expected_counts = metrics.get("entity_counts", {})
    negative_bid_expected = bool(metrics.get("negative_bid_expected", False))
    negative_supplier_bid_expected = bool(metrics.get("negative_supplier_bid_expected", False))
    has_negative_bid = any(bid.price < 0 for bid in state.bids)
    has_negative_supplier_bid = any(
        bid.price < 0
        for bid in state.bids
        if bid.owner_type == "supplier"
    )
    source_nodes = {supplier.node for supplier in state.suppliers}
    sink_nodes = {consumer.node for consumer in state.consumers}
    source_node = next(iter(source_nodes), None) if len(source_nodes) == 1 else None
    tech_nodes = {technology.node for technology in state.technologies}
    compost_sink_nodes = {
        consumer.node
        for consumer in state.consumers
        if _canonical_product_key_for_state(state, consumer.product) == CANONICAL_COMPOST_PRODUCT_ID
    }
    outgoing_destinations = {
        link.destination
        for link in state.transport_links
        if source_node is not None and link.origin == source_node
    }
    outgoing_sink_destinations = outgoing_destinations & sink_nodes
    expected_sink_count = int(expected_counts.get("demand_consumer_entities", 2))
    expected_technology_count = int(expected_counts.get("technologies", 0))

    rows = [
        _metric_row("one_source_node", True, len(source_nodes) == 1, len(source_nodes) == 1, details=sorted(source_nodes)),
        _metric_row(
            "distinct_sink_nodes",
            expected_sink_count,
            len(sink_nodes),
            len(sink_nodes) == expected_sink_count,
            details=sorted(sink_nodes),
        ),
    ]
    if expected_technology_count == 0:
        rows.extend([
        _metric_row(
            "source_outgoing_to_two_sinks",
            True,
            len(outgoing_sink_destinations) == 2,
            len(outgoing_sink_destinations) == 2,
            details={"source": source_node, "destinations": sorted(outgoing_destinations)},
        ),
        _metric_row("no_technologies", True, len(state.technologies) == 0, len(state.technologies) == 0),
        ])
    else:
        expected_manure_paths = expected_counts.get("manure_transport_paths_from_source")
        expected_compost_paths = expected_counts.get("compost_transport_paths_to_compost_sink")
        manure_source_paths = [
            link
            for link in state.transport_links
            if source_node is not None
            and link.origin == source_node
            and _canonical_product_key_for_state(state, link.product) == CANONICAL_MANURE_PRODUCT_ID
        ]
        compost_sink_paths = [
            link
            for link in state.transport_links
            if link.destination in compost_sink_nodes
            and _canonical_product_key_for_state(state, link.product) == CANONICAL_COMPOST_PRODUCT_ID
        ]
        rows.extend(
            [
                _metric_row(
                    "technology_count",
                    expected_technology_count,
                    len(state.technologies),
                    len(state.technologies) == expected_technology_count,
                ),
                _metric_row(
                    "source_outgoing_manure_paths",
                    expected_manure_paths,
                    len(manure_source_paths),
                    expected_manure_paths is None or len(manure_source_paths) == int(expected_manure_paths),
                    details=[link.destination for link in manure_source_paths],
                ),
                _metric_row(
                    "compost_paths_to_compost_sink",
                    expected_compost_paths,
                    len(compost_sink_paths),
                    expected_compost_paths is None or len(compost_sink_paths) == int(expected_compost_paths),
                    details=[link.destination for link in compost_sink_paths],
                ),
                _metric_row(
                    "technology_reachable_from_source",
                    True,
                    any(link.destination in tech_nodes for link in manure_source_paths),
                    any(link.destination in tech_nodes for link in manure_source_paths),
                    details={"technology_nodes": sorted(tech_nodes)},
                ),
                _metric_row(
                    "technology_reaches_compost_consumer",
                    True,
                    any(link.origin in tech_nodes for link in compost_sink_paths),
                    any(link.origin in tech_nodes for link in compost_sink_paths),
                    details={"compost_sink_nodes": sorted(compost_sink_nodes)},
                ),
            ]
        )
    rows.extend([
        _metric_row(
            "negative_bid_detection" if negative_bid_expected else "no_negative_bids",
            True,
            has_negative_bid if negative_bid_expected else all(bid.price >= 0 for bid in state.bids),
            has_negative_bid if negative_bid_expected else all(bid.price >= 0 for bid in state.bids),
            details=[bid.price for bid in state.bids],
        ),
        _metric_row(
            "transport_product_semantics",
            metrics.get(
                "transport_product_semantics",
                [CANONICAL_MANURE_PRODUCT_ID] * len(state.transport_links),
            ),
            [
                _canonical_product_key_for_state(state, link.product)
                for link in state.transport_links
            ],
            _multiset_matches_text(
                [
                    _canonical_product_key_for_state(state, link.product)
                    for link in state.transport_links
                ],
                metrics.get(
                    "transport_product_semantics",
                    [CANONICAL_MANURE_PRODUCT_ID] * len(state.transport_links),
                ),
            ),
            details=[link.product for link in state.transport_links],
        ),
    ])
    if negative_supplier_bid_expected:
        rows.append(
            _metric_row(
                "negative_supplier_bid_detection",
                True,
                has_negative_supplier_bid,
                has_negative_supplier_bid,
                details=[
                    bid.price
                    for bid in state.bids
                    if bid.owner_type == "supplier"
                ],
            )
        )
    return rows


def _route_economics_metric_rows(
    state: ProblemState,
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    expected_values = _expected_route_net_values(reference_solution)
    route_net_values = _route_net_values(state)
    expected_pathway_values = _expected_semantic_metrics(reference_solution).get(
        "technology_pathway_net_values_per_input"
    )
    pathway_values = _technology_pathway_net_values_per_input(state)
    rows = [
        _metric_row(
            "sorted_route_net_values",
            expected_values,
            route_net_values,
            _multiset_matches(route_net_values, expected_values, tolerance),
            details="consumer willingness-to-pay - transport cost - source cost",
        )
    ]
    if expected_pathway_values is not None:
        expected_combined = sorted([*expected_values, *[float(value) for value in expected_pathway_values]])
        actual_combined = sorted([*route_net_values, *pathway_values])
        rows.extend(
            [
                _metric_row(
                    "technology_pathway_net_values_per_input",
                    expected_pathway_values,
                    pathway_values,
                    _multiset_matches(pathway_values, expected_pathway_values, tolerance),
                    details="output value plus source contribution minus input transport, output transport, and technology operating cost per input ton",
                ),
                _metric_row(
                    "sorted_route_or_pathway_net_values",
                    expected_combined,
                    actual_combined,
                    _multiset_matches(actual_combined, expected_combined, tolerance),
                ),
            ]
        )
    return rows


def _route_association_metric_rows(
    state: ProblemState,
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    """Check route-specific economic associations when the reference supplies them."""

    association = _expected_semantic_metrics(reference_solution).get("route_association", {})
    if not isinstance(association, dict) or not association:
        return []

    rows: List[Dict[str, Any]] = []
    expected_manure_costs = [
        float(route["transport_cost"])
        for route in association.get("manure_routes", [])
        if route.get("transport_cost") is not None
    ]
    for route in association.get("manure_routes", []):
        origin = route.get("origin")
        destination = route.get("destination")
        product = route.get("product")
        link = _find_transport_link_by_semantic_route(state, origin, destination, product)
        consumer_bid = _consumer_price_at(state, link.destination, link.product) if link is not None else None
        source_price = _supplier_price_at(state, link.origin, link.product) if link is not None else None
        source_contribution = -source_price if source_price is not None else None
        transport_cost = float(getattr(link, "cost", 0.0) or 0.0) if link is not None else None
        computed_net_value = (
            float(source_contribution) + float(consumer_bid) - float(transport_cost)
            if source_contribution is not None and consumer_bid is not None and transport_cost is not None
            else None
        )
        expected_net_value = route.get("expected_route_net_value")
        expected_transport_cost = route.get("transport_cost")
        expected_consumer_bid = route.get("consumer_bid")
        expected_source_contribution = route.get("source_contribution")

        cost_matches = _optional_float_matches(transport_cost, expected_transport_cost, tolerance)
        bid_matches = _optional_float_matches(consumer_bid, expected_consumer_bid, tolerance)
        source_matches = _optional_float_matches(source_contribution, expected_source_contribution, tolerance)
        net_matches = _optional_float_matches(computed_net_value, expected_net_value, tolerance)
        passed = link is not None and cost_matches and bid_matches and source_matches and net_matches
        association_error = (
            link is not None
            and not cost_matches
            and transport_cost is not None
            and any(
                abs(float(transport_cost) - expected_cost) <= tolerance
                for expected_cost in expected_manure_costs
                if expected_transport_cost is None
                or abs(expected_cost - float(expected_transport_cost)) > tolerance
            )
        )
        rows.append(
            {
                "metric": f"route_association:{route.get('destination_role', destination)}",
                "pathway_type": "manure_route",
                "resolved_destination_role": route.get("destination_role", destination),
                "origin": origin,
                "destination": destination,
                "product": product,
                "consumer_bid_attached": consumer_bid,
                "expected_consumer_bid": expected_consumer_bid,
                "transport_cost_attached": transport_cost,
                "expected_transport_cost": expected_transport_cost,
                "source_contribution": source_contribution,
                "expected_source_contribution": expected_source_contribution,
                "computed_route_net_value": computed_net_value,
                "expected_route_net_value": expected_net_value,
                "pass": bool(passed),
                "association_error": association_error,
                "reason": (
                    "pass"
                    if passed
                    else "transport cost appears attached to a different destination"
                    if association_error
                    else "route-specific bid, cost, source contribution, or net value mismatch"
                ),
            }
        )

    compost_pathway = association.get("compost_pathway")
    if isinstance(compost_pathway, dict) and compost_pathway:
        rows.append(_compost_pathway_association_row(state, compost_pathway, tolerance))
    return rows


def _compost_pathway_association_row(
    state: ProblemState,
    pathway: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    source_origin = pathway.get("source_origin")
    technology_node = pathway.get("technology_node")
    consumer_destination = pathway.get("consumer_destination")
    input_product = pathway.get("technology_input_product") or pathway.get("input_product")
    output_product = pathway.get("technology_output_product") or pathway.get("output_product")

    input_link = _find_transport_link_by_semantic_route(
        state,
        source_origin,
        technology_node,
        input_product,
    )
    output_link = _find_transport_link_by_semantic_route(
        state,
        technology_node,
        consumer_destination,
        output_product,
    )
    technology = _find_technology_by_semantic_node(state, technology_node)
    source_price = (
        _supplier_price_at(state, input_link.origin, input_link.product)
        if input_link is not None
        else None
    )
    source_contribution = -source_price if source_price is not None else None
    compost_consumer_bid = (
        _consumer_price_at(state, output_link.destination, output_link.product)
        if output_link is not None
        else None
    )
    input_transport_cost = (
        float(getattr(input_link, "cost", 0.0) or 0.0)
        if input_link is not None
        else None
    )
    compost_transport_cost = (
        float(getattr(output_link, "cost", 0.0) or 0.0)
        if output_link is not None
        else None
    )
    yield_value = _technology_yield_for_product(state, technology, output_product)
    input_coefficient = _technology_yield_for_product(state, technology, input_product)
    technology_cost = float(getattr(technology, "cost", 0.0) or 0.0) if technology is not None else None
    expected_net_value = pathway.get("expected_pathway_net_value")

    computed_net_value = None
    if (
        source_contribution is not None
        and input_transport_cost is not None
        and compost_consumer_bid is not None
        and compost_transport_cost is not None
        and yield_value is not None
        and technology_cost is not None
    ):
        computed_net_value = (
            float(source_contribution)
            - float(input_transport_cost)
            + float(compost_consumer_bid) * float(yield_value)
            - float(compost_transport_cost) * float(yield_value)
            - float(technology_cost)
        )

    product_structure_pass = (
        input_coefficient is not None
        and input_coefficient < 0
        and yield_value is not None
        and yield_value > 0
    )
    expected_yield = pathway.get("yield")
    checks = [
        product_structure_pass,
        _optional_float_matches(compost_consumer_bid, pathway.get("compost_consumer_bid"), tolerance),
        _optional_float_matches(compost_transport_cost, pathway.get("compost_transport_cost"), tolerance),
        _optional_float_matches(yield_value, expected_yield, tolerance),
        _optional_float_matches(technology_cost, pathway.get("technology_cost"), tolerance),
        _optional_float_matches(computed_net_value, expected_net_value, tolerance),
    ]
    passed = all(checks)
    return {
        "metric": "route_association:compost_pathway",
        "pathway_type": "compost_pathway",
        "resolved_destination_role": pathway.get("destination_role", consumer_destination),
        "origin": source_origin,
        "destination": consumer_destination,
        "product": output_product,
        "compost_consumer_bid": compost_consumer_bid,
        "expected_compost_consumer_bid": pathway.get("compost_consumer_bid"),
        "compost_transport_cost": compost_transport_cost,
        "expected_compost_transport_cost": pathway.get("compost_transport_cost"),
        "technology_input_product": input_product,
        "technology_output_product": output_product,
        "input_transport_cost": input_transport_cost,
        "yield": yield_value,
        "expected_yield": expected_yield,
        "technology_cost": technology_cost,
        "expected_technology_cost": pathway.get("technology_cost"),
        "computed_pathway_net_value": computed_net_value,
        "expected_pathway_net_value": expected_net_value,
        "pass": bool(passed),
        "association_error": False,
        "reason": (
            "pass"
            if passed
            else "compost pathway bid, transport, yield, technology cost, or net value mismatch"
        ),
    }


def _formulation_completeness_metric_rows(
    state: ProblemState,
    reference_solution: Dict[str, Any],
    count_rows: Sequence[Dict[str, Any]],
    parameter_rows: Sequence[Dict[str, Any]],
    topology_rows: Sequence[Dict[str, Any]],
    technology_rows: Sequence[Dict[str, Any]],
    route_association_rows: Sequence[Dict[str, Any]],
    tolerance: float,
    semantic_plan: Optional[Dict[str, Any]] = None,
    prose_input: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build strict formulation-recovery checks separate from solve correctness."""

    rows: List[Dict[str, Any]] = []
    for row in count_rows:
        rows.append(
            _formulation_metric_row(
                f"entity_count:{row['metric']}",
                row.get("expected"),
                row.get("actual"),
                bool(row.get("pass", False)),
                "required products, participants, routes, and technologies must be recovered",
            )
        )
    for row in topology_rows:
        rows.append(
            _formulation_metric_row(
                f"topology:{row['metric']}",
                row.get("expected"),
                row.get("actual"),
                bool(row.get("pass", False)),
                "required network topology and product semantics must be recovered",
            )
        )
    for row in parameter_rows:
        reason = {
            "supplier_capacities": "all supplier capacities must be recovered",
            "consumer_capacities": "all consumer capacities must be recovered",
            "consumer_bid_prices": "all consumer bid prices must be recovered",
            "supplier_bid_prices": "all supplier bid prices must be recovered",
            "transport_costs": "transport-cost multiset is diagnostic; route association is checked separately",
            "transport_capacities": "all required transport capacities must be recovered",
        }.get(row["metric"], "required numeric parameters must be recovered")
        rows.append(
            _formulation_metric_row(
                f"parameter:{row['metric']}",
                row.get("expected"),
                row.get("actual"),
                bool(row.get("pass", False)),
                reason,
            )
        )
    rows.extend(
        _transport_link_formulation_rows(
            state=state,
            reference_solution=reference_solution,
            tolerance=tolerance,
            semantic_plan=semantic_plan,
            prose_input=prose_input,
        )
    )
    for row in technology_rows:
        rows.append(
            _formulation_metric_row(
                f"technology:{row['metric']}",
                row.get("expected"),
                row.get("actual"),
                bool(row.get("pass", False)),
                "technology yields, capacities, and costs must be recovered",
            )
        )
    for row in route_association_rows:
        rows.append(
            _formulation_metric_row(
                row["metric"],
                _route_association_expected_payload(row),
                _route_association_actual_payload(row),
                bool(row.get("pass", False)),
                row.get("reason") or "route-specific economics must be attached to the correct destination",
            )
        )
    return rows


def _transport_link_formulation_rows(
    state: ProblemState,
    reference_solution: Dict[str, Any],
    tolerance: float,
    semantic_plan: Optional[Dict[str, Any]],
    prose_input: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in _reference_transport_link_expectations(reference_solution):
        link = _find_transport_link_by_semantic_route(
            state,
            spec.get("origin"),
            spec.get("destination"),
            spec.get("product"),
        )
        route_label = _transport_expectation_label(spec)
        actual_route = _transport_link_payload(link) if link is not None else None
        expected_route = {
            "origin": spec.get("origin"),
            "destination": spec.get("destination"),
            "product": spec.get("product"),
        }
        rows.append(
            _formulation_metric_row(
                f"transport_link:{route_label}",
                expected_route,
                actual_route,
                link is not None,
                "each expected transport link must be recovered",
            )
        )
        expected_cost = spec.get("cost")
        actual_cost = float(getattr(link, "cost", 0.0) or 0.0) if link is not None else None
        rows.append(
            _formulation_metric_row(
                f"transport_cost:{route_label}",
                expected_cost,
                actual_cost,
                _optional_float_matches(actual_cost, expected_cost, tolerance),
                "route-specific transport costs must stay attached to the correct origin, destination, and product",
            )
        )
        expected_capacity = spec.get("capacity")
        actual_capacity = getattr(link, "capacity", None) if link is not None else None
        rows.append(
            _formulation_metric_row(
                f"transport_capacity:{route_label}",
                {
                    "route": expected_route,
                    "capacity": expected_capacity,
                },
                {
                    "route": actual_route,
                    "capacity": actual_capacity,
                    "diagnosis": _diagnose_transport_capacity_recovery(
                        spec=spec,
                        actual_link=link,
                        semantic_plan=semantic_plan,
                        prose_input=prose_input,
                    ),
                },
                _optional_float_matches(actual_capacity, expected_capacity, tolerance),
                "each required transport capacity must be recovered, even when currently nonbinding",
            )
        )
    return rows


def _solve_correctness_metric_rows(
    state: ProblemState,
    solve_result: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    checks = compare_solution_to_reference(state, solve_result, reference_solution, tolerance)
    components = checks.get("actual_components", {})
    residual_stats = _balance_residual_stats(state, solve_result, tolerance)
    expected_supply_contribution = reference_solution.get("supply_contribution")
    if expected_supply_contribution is None and reference_solution.get("supply_cost") is not None:
        expected_supply_contribution = -float(reference_solution.get("supply_cost"))
    rows = [
        _solution_metric_row("solve_success", True, checks.get("solve_success"), bool(checks.get("solve_success"))),
        _solution_metric_row(
            "objective_match",
            checks.get("expected_objective"),
            checks.get("actual_objective"),
            checks.get("objective_match") is True,
        ),
        _solution_metric_row(
            "accepted_supply_match",
            reference_solution.get("accepted_supply", {}),
            components.get("accepted_supply", {}),
            checks.get("accepted_supply_match") is True,
        ),
        _solution_metric_row(
            "accepted_demand_match",
            reference_solution.get("accepted_demands", {}),
            components.get("accepted_demands", {}),
            checks.get("accepted_demand_match") is True,
        ),
        _solution_metric_row(
            "active_flows_match",
            reference_solution.get("transport_flows", {}),
            components.get("transport_flows", {}),
            checks.get("transport_flow_match") is True,
        ),
        _solution_metric_row(
            "technology_activity_match",
            reference_solution.get("technology_activity", {}),
            components.get("technology_activity", {}),
            checks.get("technology_activity_match") is True,
        ),
        _solution_metric_row(
            "technology_outputs_match",
            reference_solution.get("technology_outputs", {}),
            components.get("technology_outputs", {}),
            checks.get("technology_outputs_match") is True,
        ),
        _solution_metric_row(
            "demand_revenue_match",
            reference_solution.get("demand_revenue"),
            components.get("demand_revenue"),
            checks.get("demand_revenue_match") is True,
        ),
        _solution_metric_row(
            "transport_cost_match",
            reference_solution.get("transport_cost"),
            components.get("transport_cost"),
            checks.get("transport_cost_match") is True,
        ),
        _solution_metric_row(
            "supply_contribution_match",
            expected_supply_contribution,
            components.get("supply_contribution"),
            checks.get("supply_contribution_match") is True,
        ),
        _solution_metric_row(
            "technology_cost_match",
            reference_solution.get("technology_cost", 0.0),
            components.get("technology_cost"),
            checks.get("technology_cost_match") is True,
        ),
        _solution_metric_row(
            "balance_residuals_pass",
            f"max <= {tolerance}",
            residual_stats["max_abs_balance_residual"],
            residual_stats["max_abs_balance_residual"] <= tolerance,
            details=residual_stats["residuals"],
        ),
    ]
    return rows


def _reasoning_readiness_metric_rows(
    formulation_completeness_pass: bool,
    route_association_pass: bool,
    technology_structure_pass: bool,
) -> List[Dict[str, Any]]:
    gate_payload = {
        "formulation_completeness_pass": formulation_completeness_pass,
        "route_association_pass": route_association_pass,
        "technology_structure_pass": technology_structure_pass,
    }
    ready = all(gate_payload.values())
    rows = []
    for capability in (
        "primal_lp_generation",
        "dual_lp_generation",
        "complementary_slackness_checks",
        "what_if_analysis",
    ):
        rows.append(
            {
                "metric": capability,
                "capability": capability,
                "expected": "safe",
                "actual": "safe" if ready else "not safe",
                "pass": ready,
                "requires": json.dumps(gate_payload, sort_keys=True),
                "reason": (
                    "full formulation, route associations, and technology structure recovered"
                    if ready
                    else "requires complete formulation recovery, correct route associations, and valid technology structure"
                ),
            }
        )
    return rows


def _classify_primary_failure(
    formulation_completeness_pass: bool,
    solve_correctness_pass: bool,
    reasoning_ready_pass: bool,
    route_association_rows: Sequence[Dict[str, Any]],
) -> str:
    if formulation_completeness_pass and solve_correctness_pass and reasoning_ready_pass:
        return "none"
    if any(bool(row.get("association_error")) for row in route_association_rows):
        return "route_cost_association_error"
    if not formulation_completeness_pass and solve_correctness_pass:
        return "incomplete_formulation_but_solution_equivalent"
    if not formulation_completeness_pass and not solve_correctness_pass:
        return "incomplete_formulation_and_wrong_solution"
    if not solve_correctness_pass:
        return "wrong_solution"
    if not reasoning_ready_pass:
        return "reasoning_not_ready"
    return "unknown"


def _formulation_metric_row(
    field: str,
    expected: Any,
    actual: Any,
    passed: bool,
    reason: str,
    severity: str = "blocking",
) -> Dict[str, Any]:
    rendered_expected = _render_metric_value(expected)
    rendered_actual = _render_metric_value(actual)
    return {
        "metric": field,
        "field": field,
        "expected": rendered_expected,
        "actual": rendered_actual,
        "expected_field": rendered_expected,
        "actual_field": rendered_actual,
        "pass": bool(passed),
        "severity": severity,
        "reason": reason,
    }


def _solution_metric_row(
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


def _route_association_expected_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("pathway_type") == "compost_pathway":
        return {
            "compost_consumer_bid": row.get("expected_compost_consumer_bid"),
            "compost_transport_cost": row.get("expected_compost_transport_cost"),
            "yield": row.get("expected_yield"),
            "technology_cost": row.get("expected_technology_cost"),
            "pathway_net_value": row.get("expected_pathway_net_value"),
        }
    return {
        "consumer_bid": row.get("expected_consumer_bid"),
        "transport_cost": row.get("expected_transport_cost"),
        "source_contribution": row.get("expected_source_contribution"),
        "route_net_value": row.get("expected_route_net_value"),
    }


def _route_association_actual_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("pathway_type") == "compost_pathway":
        return {
            "compost_consumer_bid": row.get("compost_consumer_bid"),
            "compost_transport_cost": row.get("compost_transport_cost"),
            "yield": row.get("yield"),
            "technology_cost": row.get("technology_cost"),
            "pathway_net_value": row.get("computed_pathway_net_value"),
        }
    return {
        "consumer_bid": row.get("consumer_bid_attached"),
        "transport_cost": row.get("transport_cost_attached"),
        "source_contribution": row.get("source_contribution"),
        "route_net_value": row.get("computed_route_net_value"),
    }


def _reference_transport_link_expectations(reference_solution: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = _expected_semantic_metrics(reference_solution).get("transport_links", [])
    if not isinstance(specs, list):
        return []
    return [dict(spec) for spec in specs if isinstance(spec, dict)]


def _find_transport_link_by_semantic_route(
    state: ProblemState,
    origin: Any,
    destination: Any,
    product: Any,
) -> Any:
    expected_origin = _canonical_node_key(origin)
    expected_destination = _canonical_node_key(destination)
    expected_product = _canonical_product_key(product)
    for link in state.transport_links:
        if (
            _canonical_node_key(link.origin) == expected_origin
            and _canonical_node_key(link.destination) == expected_destination
            and _canonical_product_key_for_state(state, link.product) == expected_product
        ):
            return link
    return None


def _find_semantic_plan_transport_link(
    semantic_plan: Optional[Dict[str, Any]],
    spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(semantic_plan, dict):
        return None
    expected_origin = _canonical_node_key(spec.get("origin"))
    expected_destination = _canonical_node_key(spec.get("destination"))
    expected_product = _canonical_product_key(spec.get("product"))
    for link in semantic_plan.get("transport_links", []) or []:
        if not isinstance(link, dict):
            continue
        if (
            _canonical_node_key(link.get("origin")) == expected_origin
            and _canonical_node_key(link.get("destination")) == expected_destination
            and _canonical_product_key(link.get("product")) == expected_product
        ):
            return link
    return None


def _find_technology_by_semantic_node(state: ProblemState, node: Any) -> Any:
    expected_node = _canonical_node_key(node)
    for technology in state.technologies:
        if _canonical_node_key(technology.node) == expected_node:
            return technology
    return None


def _technology_yield_for_product(
    state: ProblemState,
    technology: Any,
    product: Any,
) -> Optional[float]:
    if technology is None:
        return None
    expected_product = _canonical_product_key(product)
    for product_id, coefficient in technology.yield_coefficients.items():
        if _canonical_product_key_for_state(state, product_id) == expected_product:
            return float(coefficient)
    return None


def _transport_expectation_label(spec: Dict[str, Any]) -> str:
    return (
        f"{_canonical_node_key(spec.get('origin'))}"
        f"_to_{_canonical_node_key(spec.get('destination'))}"
        f":{_canonical_product_key(spec.get('product'))}"
    )


def _transport_link_payload(link: Any) -> Optional[Dict[str, Any]]:
    if link is None:
        return None
    return {
        "origin": link.origin,
        "destination": link.destination,
        "product": link.product,
        "capacity": link.capacity,
        "cost": float(getattr(link, "cost", 0.0) or 0.0),
    }


def _diagnose_transport_capacity_recovery(
    spec: Dict[str, Any],
    actual_link: Any,
    semantic_plan: Optional[Dict[str, Any]],
    prose_input: Optional[str],
) -> str:
    expected_capacity = spec.get("capacity")
    if actual_link is None:
        return "transport_link_missing_from_problem_state"
    if expected_capacity is None:
        return "no_capacity_required_by_reference"
    if getattr(actual_link, "capacity", None) is not None:
        return "capacity_recovered"
    plan_link = _find_semantic_plan_transport_link(semantic_plan, spec)
    if plan_link is not None and plan_link.get("capacity") is not None:
        return "state_builder_loss"
    if prose_input is not None and not _prose_mentions_capacity_for_route(prose_input, spec):
        return "prompt_omission"
    if semantic_plan is not None:
        return "llm_omission"
    return "not_diagnosable"


def _prose_mentions_capacity_for_route(prose_input: str, spec: Dict[str, Any]) -> bool:
    text = str(prose_input).lower()
    expected_capacity = spec.get("capacity")
    if expected_capacity is None:
        return True
    capacity_tokens = {
        str(expected_capacity).lower(),
        str(int(float(expected_capacity))).lower()
        if isinstance(expected_capacity, (int, float))
        else str(expected_capacity).lower(),
    }
    if not any(token in text for token in capacity_tokens):
        return False
    collective_phrases = (
        "all four transport links",
        "all transport links",
        "each transport link",
        "each route",
        "every route",
    )
    if any(phrase in text for phrase in collective_phrases):
        return True
    origin = str(spec.get("origin", "")).lower()
    destination = str(spec.get("destination", "")).lower()
    product = str(spec.get("product", "")).lower()
    route_tokens = [token for token in (origin, destination, product) if token]
    if not route_tokens:
        return False
    for token in route_tokens:
        index = text.find(token)
        while index >= 0:
            window = text[max(0, index - 120): index + 160]
            if "capacit" in window and any(capacity in window for capacity in capacity_tokens):
                return True
            index = text.find(token, index + len(token))
    return False


def _optional_float_matches(actual: Any, expected: Any, tolerance: float) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _technology_yield_metric_rows(
    state: ProblemState,
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    metrics = _expected_semantic_metrics(reference_solution)
    expected = metrics.get("technology", {})
    if not expected:
        return [
            _metric_row(
                "no_technology_yields_expected",
                True,
                len(state.technologies) == 0,
                len(state.technologies) == 0,
            )
        ]

    expected_input_products = expected.get("input_products", [])
    expected_output_products = expected.get("output_products", [])
    expected_input_coefficients = expected.get("input_coefficients", [])
    expected_output_coefficients = expected.get("output_coefficients", [])
    expected_capacities = expected.get("capacities", [])
    expected_operating_costs = expected.get("operating_costs", [])

    input_products: List[str] = []
    output_products: List[str] = []
    input_coefficients: List[float] = []
    output_coefficients: List[float] = []
    capacities: List[float] = []
    costs: List[float] = []
    for technology in state.technologies:
        if technology.capacity is not None:
            capacities.append(float(technology.capacity))
        costs.append(float(getattr(technology, "cost", 0.0) or 0.0))
        for product_id, coefficient in technology.yield_coefficients.items():
            product_key = _canonical_product_key_for_state(state, product_id)
            if coefficient < 0:
                input_products.append(product_key)
                input_coefficients.append(float(coefficient))
            elif coefficient > 0:
                output_products.append(product_key)
                output_coefficients.append(float(coefficient))

    rows = [
        _metric_row(
            "technology_count",
            expected.get("count", 1),
            len(state.technologies),
            len(state.technologies) == int(expected.get("count", 1)),
        ),
        _metric_row(
            "technology_input_products",
            expected_input_products,
            input_products,
            _multiset_matches_text(input_products, expected_input_products),
        ),
        _metric_row(
            "technology_output_products",
            expected_output_products,
            output_products,
            _multiset_matches_text(output_products, expected_output_products),
        ),
        _metric_row(
            "technology_input_coefficients",
            expected_input_coefficients,
            input_coefficients,
            _multiset_matches(input_coefficients, expected_input_coefficients, tolerance),
        ),
        _metric_row(
            "technology_output_coefficients",
            expected_output_coefficients,
            output_coefficients,
            _multiset_matches(output_coefficients, expected_output_coefficients, tolerance),
        ),
        _metric_row(
            "technology_capacities",
            expected_capacities,
            capacities,
            _multiset_matches(capacities, expected_capacities, tolerance),
        ),
        _metric_row(
            "technology_operating_costs",
            expected_operating_costs,
            costs,
            _multiset_matches(costs, expected_operating_costs, tolerance),
        ),
    ]
    return rows


def _bounded_transport_capacity_default(reference_solution: Dict[str, Any]) -> List[float]:
    metrics = _expected_semantic_metrics(reference_solution)
    consumer_capacities = metrics.get("consumer_capacities")
    if consumer_capacities is not None:
        return [float(value) for value in consumer_capacities]
    accepted_demands = reference_solution.get("accepted_demands", {})
    if accepted_demands:
        return [float(value) for value in accepted_demands.values()]
    return [500.0, 500.0]


def _expected_route_net_values(reference_solution: Dict[str, Any]) -> List[float]:
    metrics = _expected_semantic_metrics(reference_solution)
    if "route_net_values" in metrics:
        return sorted(float(value) for value in metrics["route_net_values"])
    route_net_values = reference_solution.get("route_net_values", {})
    if isinstance(route_net_values, dict):
        return sorted(float(value) for value in route_net_values.values())
    if route_net_values:
        return sorted(float(value) for value in route_net_values)
    return [0.4, 1.3]


def _expected_solver_aggregates(
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    metrics = _expected_semantic_metrics(reference_solution)
    metric_aggregates = metrics.get("solver_aggregates", {})
    accepted_supply = reference_solution.get("accepted_supply", {})
    accepted_demands = reference_solution.get("accepted_demands", {})
    transport_flows = reference_solution.get("transport_flows", {})

    expected = {
        "objective_value": reference_solution.get("objective_value", 850.0),
        "demand_revenue": reference_solution.get("demand_revenue", 1000.0),
        "transport_cost": reference_solution.get("transport_cost", 150.0),
        "supply_cost": reference_solution.get("supply_cost", 0.0),
        "supply_contribution": reference_solution.get(
            "supply_contribution",
            -float(reference_solution.get("supply_cost", 0.0) or 0.0),
        ),
        "technology_cost": reference_solution.get("technology_cost", 0.0),
        "total_accepted_supply": sum(float(value) for value in accepted_supply.values()),
        "total_accepted_demand": sum(float(value) for value in accepted_demands.values()),
        "total_transport_flow": sum(float(value) for value in transport_flows.values()),
        "active_transport_routes": sum(
            1 for value in transport_flows.values() if abs(float(value)) > tolerance
        ),
        "sorted_active_flow_values": sorted(
            float(value) for value in transport_flows.values() if abs(float(value)) > tolerance
        ),
        "sorted_accepted_demand_values": sorted(
            float(value) for value in accepted_demands.values() if abs(float(value)) > tolerance
        ),
    }
    expected.update(metric_aggregates)
    return expected


def _solver_aggregate_metric_rows(
    state: ProblemState,
    solve_result: Dict[str, Any],
    reference_solution: Dict[str, Any],
    tolerance: float,
) -> List[Dict[str, Any]]:
    aggregates = _solver_aggregates(state, solve_result)
    expected = _expected_solver_aggregates(reference_solution, tolerance)
    rows = [
        _numeric_metric_row(metric, expected[metric], aggregates.get(metric), tolerance)
        for metric in (
            "objective_value",
            "demand_revenue",
            "transport_cost",
            "supply_cost",
            "supply_contribution",
            "technology_cost",
            "total_accepted_supply",
            "total_accepted_demand",
            "total_transport_flow",
            "active_transport_routes",
        )
        if metric in expected
    ]
    rows.extend(
        [
            _metric_row(
                "sorted_active_flow_values",
                expected["sorted_active_flow_values"],
                aggregates.get("sorted_active_flow_values", []),
                _multiset_matches(
                    aggregates.get("sorted_active_flow_values", []),
                    expected["sorted_active_flow_values"],
                    tolerance,
                ),
            ),
            _metric_row(
                "sorted_accepted_demand_values",
                expected["sorted_accepted_demand_values"],
                aggregates.get("sorted_accepted_demand_values", []),
                _multiset_matches(
                    aggregates.get("sorted_accepted_demand_values", []),
                    expected["sorted_accepted_demand_values"],
                    tolerance,
                ),
            ),
        ]
    )
    for metric, expected_value in expected.items():
        if metric in {
            "objective_value",
            "demand_revenue",
            "transport_cost",
            "supply_cost",
            "supply_contribution",
            "technology_cost",
            "total_accepted_supply",
            "total_accepted_demand",
            "total_transport_flow",
            "active_transport_routes",
            "sorted_active_flow_values",
            "sorted_accepted_demand_values",
        }:
            continue
        actual_value = aggregates.get(metric)
        if isinstance(expected_value, (list, tuple)):
            rows.append(
                _metric_row(
                    metric,
                    expected_value,
                    actual_value or [],
                    _multiset_matches(actual_value or [], expected_value, tolerance),
                )
            )
        else:
            rows.append(_numeric_metric_row(metric, expected_value, actual_value, tolerance))
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


def _technology_pathway_net_values_per_input(state: ProblemState) -> List[float]:
    values: List[float] = []
    for technology in state.technologies:
        input_products = [
            (product_id, float(coefficient))
            for product_id, coefficient in technology.yield_coefficients.items()
            if coefficient < 0
        ]
        output_products = [
            (product_id, float(coefficient))
            for product_id, coefficient in technology.yield_coefficients.items()
            if coefficient > 0
        ]
        for input_product, input_coefficient in input_products:
            input_amount = abs(input_coefficient)
            input_links = [
                link
                for link in state.transport_links
                if link.destination == technology.node
                and _same_product(link.product, input_product)
            ]
            for output_product, output_coefficient in output_products:
                output_links = [
                    link
                    for link in state.transport_links
                    if link.origin == technology.node
                    and _same_product(link.product, output_product)
                ]
                for input_link in input_links:
                    source_cost = _supplier_price_at(state, input_link.origin, input_product)
                    if source_cost is None:
                        continue
                    for output_link in output_links:
                        consumer_price = _consumer_price_at(state, output_link.destination, output_product)
                        if consumer_price is None:
                            continue
                        value = (
                            float(consumer_price) * output_coefficient
                            - float(source_cost) * input_amount
                            - float(getattr(input_link, "cost", 0.0) or 0.0) * input_amount
                            - float(getattr(output_link, "cost", 0.0) or 0.0) * output_coefficient
                            - float(getattr(technology, "cost", 0.0) or 0.0)
                        )
                        values.append(value)
    return sorted(values)


def _solver_aggregates(state: ProblemState, solve_result: Dict[str, Any]) -> Dict[str, Any]:
    if not solve_result.get("success", False):
        return {
            "objective_value": solve_result.get("objective_value"),
            "demand_revenue": None,
            "transport_cost": None,
            "supply_cost": None,
            "supply_contribution": None,
            "technology_cost": None,
            "total_accepted_supply": None,
            "total_accepted_demand": None,
            "total_transport_flow": None,
            "active_transport_routes": None,
            "sorted_active_flow_values": [],
            "sorted_accepted_demand_values": [],
            "sorted_active_manure_flow_values": [],
            "sorted_active_compost_flow_values": [],
            "technology_activity": None,
            "total_manure_removed": None,
            "total_compost_produced": None,
        }

    solution = solve_result.get("solution", {}) if isinstance(solve_result, dict) else {}
    q_values = _solution_block(solution, "q")
    f_values = _solution_block(solution, "f")
    x_values = _solution_block(solution, "x")
    bids_by_id = {bid.id: bid for bid in state.bids}
    suppliers_by_id = {supplier.id: supplier for supplier in state.suppliers}

    demand_revenue = 0.0
    supply_cost = 0.0
    total_accepted_supply = 0.0
    total_accepted_demand = 0.0
    total_manure_removed = 0.0
    accepted_demand_values = []
    for bid_id, quantity in q_values.items():
        bid = bids_by_id.get(str(bid_id))
        if bid is None:
            continue
        if bid.owner_type == "supplier":
            total_accepted_supply += quantity
            supply_cost += float(bid.price) * quantity
            supplier = suppliers_by_id.get(bid.owner_id)
            product_id = getattr(supplier, "product", bid.product_id)
            if _canonical_product_key_for_state(state, product_id) == CANONICAL_MANURE_PRODUCT_ID:
                total_manure_removed += quantity
        elif bid.owner_type == "consumer":
            total_accepted_demand += quantity
            demand_revenue += float(bid.price) * quantity
            if abs(quantity) > TOLERANCE:
                accepted_demand_values.append(quantity)

    transport_activities = _transport_activity_rows(state, f_values)
    active_flows = [activity["flow"] for activity in transport_activities if abs(activity["flow"]) > TOLERANCE]
    active_manure_flows = [
        activity["flow"]
        for activity in transport_activities
        if abs(activity["flow"]) > TOLERANCE
        and _canonical_product_key_for_state(state, activity["product"]) == CANONICAL_MANURE_PRODUCT_ID
    ]
    active_compost_flows = [
        activity["flow"]
        for activity in transport_activities
        if abs(activity["flow"]) > TOLERANCE
        and _canonical_product_key_for_state(state, activity["product"]) == CANONICAL_COMPOST_PRODUCT_ID
    ]
    transport_cost = sum(activity["cost"] * activity["flow"] for activity in transport_activities)
    total_transport_flow = sum(activity["flow"] for activity in transport_activities)
    technology_cost = 0.0
    technology_activity = 0.0
    total_compost_produced = 0.0
    technologies_by_id = {technology.id: technology for technology in state.technologies}
    for raw_key, activity in x_values.items():
        technology = technologies_by_id.get(str(raw_key)) or _find_technology_for_key(state, str(raw_key))
        technology_activity += activity
        if technology is None:
            continue
        technology_cost += float(getattr(technology, "cost", 0.0) or 0.0) * activity
        for product_id, coefficient in technology.yield_coefficients.items():
            if (
                coefficient > 0
                and _canonical_product_key_for_state(state, product_id) == CANONICAL_COMPOST_PRODUCT_ID
            ):
                total_compost_produced += float(coefficient) * activity

    return {
        "objective_value": solve_result.get("objective_value"),
        "demand_revenue": demand_revenue,
        "transport_cost": transport_cost,
        "supply_cost": supply_cost,
        "supply_contribution": -supply_cost,
        "technology_cost": technology_cost,
        "total_accepted_supply": total_accepted_supply,
        "total_accepted_demand": total_accepted_demand,
        "total_transport_flow": total_transport_flow,
        "active_transport_routes": len(active_flows),
        "sorted_active_flow_values": sorted(active_flows),
        "sorted_accepted_demand_values": sorted(accepted_demand_values),
        "sorted_active_manure_flow_values": sorted(active_manure_flows),
        "sorted_active_compost_flow_values": sorted(active_compost_flows),
        "technology_activity": technology_activity,
        "total_manure_removed": total_manure_removed,
        "total_compost_produced": total_compost_produced,
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
    x_values = _solution_block(solution, "x")
    bids_by_id = {bid.id: bid for bid in state.bids}
    suppliers_by_id = {supplier.id: supplier for supplier in state.suppliers}
    consumers_by_id = {consumer.id: consumer for consumer in state.consumers}
    technologies_by_id = {technology.id: technology for technology in state.technologies}

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

    for raw_key, activity in x_values.items():
        technology = technologies_by_id.get(str(raw_key)) or _find_technology_for_key(state, str(raw_key))
        if technology is None:
            continue
        for product_id, coefficient in technology.yield_coefficients.items():
            add(technology.node, product_id, float(coefficient) * activity)

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


def _product_semantic_keys(state: ProblemState) -> List[str]:
    return [_canonical_product_key_from_parts(product.id, product.name) for product in state.products]


def _canonical_product_key_for_state(state: ProblemState, product_id: Any) -> str:
    for product in state.products:
        if str(product.id) == str(product_id):
            return _canonical_product_key_from_parts(product.id, product.name)
    return _canonical_product_key(product_id)


def _canonical_product_key_from_parts(product_id: Any, product_name: Any = None) -> str:
    id_key = _canonical_product_key(product_id)
    if id_key != str(product_id).replace(" ", ""):
        return id_key
    if product_name is not None:
        name_key = _canonical_product_key(product_name)
        if name_key != str(product_name).replace(" ", ""):
            return name_key
    return id_key


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


def _multiset_matches_text(actual: Sequence[Any], expected: Sequence[Any]) -> bool:
    return sorted(str(value) for value in actual) == sorted(str(value) for value in expected)


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
            "technology_yield_metrics",
            "route_economics_metrics",
            "route_association_metrics",
            "solver_aggregate_metrics",
            "balance_residual_metrics",
            "formulation_completeness_metrics",
            "solve_correctness_metrics",
            "reasoning_readiness_metrics",
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
            ("product", _canonical_product_key_from_parts(product.id, product.name)): _dump_model(product)
            for product in state.products
        }
    if collection == "suppliers":
        return {
            (
                "supplier",
                _canonical_node_key(supplier.node),
                _canonical_product_key_for_state(state, supplier.product),
            ): _dump_model(supplier)
            for supplier in state.suppliers
        }
    if collection == "consumers":
        return {
            (
                "consumer",
                _canonical_node_key(consumer.node),
                _canonical_product_key_for_state(state, consumer.product),
            ): _dump_model(consumer)
            for consumer in state.consumers
        }
    if collection == "transport_links":
        return {
            (
                "transport",
                _canonical_node_key(link.origin),
                _canonical_node_key(link.destination),
                _canonical_product_key_for_state(state, link.product),
            ): _dump_model(link)
            for link in state.transport_links
        }
    if collection == "technologies":
        return {
            (
                "technology",
                _canonical_node_key(technology.node),
                tuple(sorted(
                    _canonical_product_key_for_state(state, product_id)
                    for product_id, coefficient in technology.yield_coefficients.items()
                    if coefficient < 0
                )),
                tuple(sorted(
                    _canonical_product_key_for_state(state, product_id)
                    for product_id, coefficient in technology.yield_coefficients.items()
                    if coefficient > 0
                )),
            ): _dump_model(technology)
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
        product = _canonical_product_key_for_state(state, bid.product_id)
        if bid.owner_type == "supplier":
            supplier = suppliers_by_id.get(bid.owner_id)
            owner_key = _supplier_reference_key(supplier, bid.owner_id, bid.product_id, state=state)
        elif bid.owner_type == "consumer":
            consumer = consumers_by_id.get(bid.owner_id)
            owner_key = _consumer_reference_key(consumer, bid.owner_id, bid.product_id, state=state)
        elif bid.owner_type == "technology":
            technology = technologies_by_id.get(bid.owner_id)
            owner_key = (
                "technology",
                _canonical_node_key(getattr(technology, "node", bid.owner_id)),
                _technology_reference_key(technology, bid.owner_id),
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
        "technologies": ("capacity", "cost", "yield_coefficients"),
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


def _supplier_reference_key(
    supplier: Any,
    owner_id: str,
    product_id: Optional[str] = None,
    state: Optional[ProblemState] = None,
) -> str:
    candidates = [owner_id]
    if supplier is not None:
        candidates.append(getattr(supplier, "node", ""))
        product_id = product_id or getattr(supplier, "product", None)
    product_key = (
        _canonical_product_key_for_state(state, product_id)
        if state is not None and product_id is not None
        else _canonical_product_key(product_id)
    )
    product_ok = product_id is None or product_key == CANONICAL_MANURE_PRODUCT_ID
    if product_ok and any(_canonical_node_key(candidate) == "EauClaire" for candidate in candidates):
        return CANONICAL_SUPPLY_ID
    return str(owner_id)


def _consumer_reference_key(
    consumer: Any,
    owner_id: str,
    product_id: Optional[str] = None,
    state: Optional[ProblemState] = None,
) -> str:
    candidates = [owner_id]
    if consumer is not None:
        candidates.append(getattr(consumer, "node", ""))
        product_id = product_id or getattr(consumer, "product", None)
    product_key = (
        _canonical_product_key_for_state(state, product_id)
        if state is not None and product_id is not None
        else _canonical_product_key(product_id)
    )
    if product_key == CANONICAL_COMPOST_PRODUCT_ID:
        for candidate in candidates:
            key = _canonical_node_key(candidate)
            if key == CANONICAL_MADISON_COMPOST_DEMAND_ID:
                return CANONICAL_MADISON_COMPOST_DEMAND_ID
        return str(owner_id)
    product_ok = product_id is None or product_key == CANONICAL_MANURE_PRODUCT_ID
    if not product_ok:
        return str(owner_id)
    for candidate in candidates:
        key = _canonical_node_key(candidate)
        if key in {"Menomonie", "BlackRiverFalls"}:
            return key
    return str(owner_id)


def _technology_reference_key(technology: Any, raw_id: str) -> str:
    candidates = [raw_id]
    if technology is not None:
        candidates.append(getattr(technology, "id", ""))
        candidates.append(getattr(technology, "node", ""))
    if any(_canonical_node_key(candidate) == CANONICAL_COMPOSTER_ID for candidate in candidates):
        return CANONICAL_COMPOSTER_ID
    return str(raw_id)


def _find_technology_for_key(state: ProblemState, raw_key: str) -> Any:
    canonical_key = _canonical_node_key(raw_key)
    normalized_key = _normalize_token(raw_key)
    for technology in state.technologies:
        if _normalize_token(technology.id) == normalized_key:
            return technology
        if _canonical_node_key(technology.node) == canonical_key:
            return technology
    return None


def _route_reference_key(
    origin: str,
    destination: str,
    product_id: Optional[str] = None,
    state: Optional[ProblemState] = None,
) -> str:
    canonical_origin = _canonical_node_key(origin)
    canonical_destination = _canonical_node_key(destination)
    route_key = f"{canonical_origin}_to_{canonical_destination}"
    if product_id is not None:
        product_key = (
            _canonical_product_key_for_state(state, product_id)
            if state is not None
            else _canonical_product_key(product_id)
        )
        if product_key != CANONICAL_MANURE_PRODUCT_ID:
            return f"{route_key}:{product_key}"
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
        "source": "EauClaire",
        "src": "EauClaire",
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
        "sink2": "BlackRiverFalls",
        "c2": "BlackRiverFalls",
        "sf": "BlackRiverFalls",
        "soybean": "BlackRiverFalls",
        "soybeanfarm": "BlackRiverFalls",
        "soybeanfarmer": "BlackRiverFalls",
        "csoybean": "BlackRiverFalls",
        "consumersoybean": "BlackRiverFalls",
        "dc": CANONICAL_MADISON_COMPOST_DEMAND_ID,
        "mad": CANONICAL_MADISON_COMPOST_DEMAND_ID,
        "madison": CANONICAL_MADISON_COMPOST_DEMAND_ID,
        "madisoncompost": CANONICAL_MADISON_COMPOST_DEMAND_ID,
        "madisoncompostconsumer": CANONICAL_MADISON_COMPOST_DEMAND_ID,
        "compostconsumer": CANONICAL_MADISON_COMPOST_DEMAND_ID,
        "composter": CANONICAL_COMPOSTER_ID,
        "k1": CANONICAL_COMPOSTER_ID,
        "knode": CANONICAL_COMPOSTER_ID,
        "technology": CANONICAL_COMPOSTER_ID,
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
        "compost": CANONICAL_COMPOST_PRODUCT_ID,
        "comp": CANONICAL_COMPOST_PRODUCT_ID,
        "p2": CANONICAL_COMPOST_PRODUCT_ID,
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


def _compute_generic_balance_checks(
    state: ProblemState,
    solve_result: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    if not solve_result.get("success", False):
        return {}

    solution = solve_result.get("solution", {}) if isinstance(solve_result, dict) else {}
    q_values = _solution_block(solution, "q")
    f_values = _solution_block(solution, "f")
    x_values = _solution_block(solution, "x")
    bids_by_id = {bid.id: bid for bid in state.bids}
    suppliers_by_id = {supplier.id: supplier for supplier in state.suppliers}
    consumers_by_id = {consumer.id: consumer for consumer in state.consumers}
    technologies_by_id = {technology.id: technology for technology in state.technologies}

    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def key(node: str, product: str) -> Tuple[str, str]:
        return (_canonical_node_key(node), _canonical_product_key_for_state(state, product))

    def row(node: str, product: str) -> Dict[str, Any]:
        node_key, product_key = key(node, product)
        return rows.setdefault(
            (node_key, product_key),
            {
                "supply": 0.0,
                "accepted_demand": 0.0,
                "incoming_flow": 0.0,
                "outgoing_flow": 0.0,
                "technology_net": 0.0,
                "residual": 0.0,
                "holds": True,
            },
        )

    for bid_id, quantity in q_values.items():
        bid = bids_by_id.get(str(bid_id))
        if bid is None:
            continue
        if bid.owner_type == "supplier":
            supplier = suppliers_by_id.get(bid.owner_id)
            if supplier is not None:
                row(supplier.node, supplier.product)["supply"] += quantity
        elif bid.owner_type == "consumer":
            consumer = consumers_by_id.get(bid.owner_id)
            if consumer is not None:
                row(consumer.node, consumer.product)["accepted_demand"] += quantity

    for activity in _transport_activity_rows(state, f_values):
        product = activity["product"]
        if product is None:
            continue
        row(activity["origin"], product)["outgoing_flow"] += activity["flow"]
        row(activity["destination"], product)["incoming_flow"] += activity["flow"]

    for raw_key, activity in x_values.items():
        technology = technologies_by_id.get(str(raw_key)) or _find_technology_for_key(state, str(raw_key))
        if technology is None:
            continue
        for product_id, coefficient in technology.yield_coefficients.items():
            row(technology.node, product_id)["technology_net"] += float(coefficient) * activity

    rendered: Dict[str, Dict[str, Any]] = {}
    for (node_key, product_key), values in sorted(rows.items()):
        residual = (
            values["supply"]
            + values["incoming_flow"]
            + values["technology_net"]
            - values["accepted_demand"]
            - values["outgoing_flow"]
        )
        values["residual"] = residual
        values["holds"] = abs(residual) <= TOLERANCE
        rendered[f"{node_key}:{product_key}"] = values
    return rendered


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
        _component_diagnostic_rows(
            component="technology_activity",
            actual=components.get("technology_activity", {}),
            expected=reference_solution.get("technology_activity", {}),
            raw_rows=raw_rows,
            tolerance=tolerance,
        )
    )
    rows.extend(
        _component_diagnostic_rows(
            component="technology_outputs",
            actual=components.get("technology_outputs", {}),
            expected=reference_solution.get("technology_outputs", {}),
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
        product_label = node.split(":", 1)[1] if ":" in node else CANONICAL_MANURE_PRODUCT_ID
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
                "raw_product": product_label,
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
    "DEFAULT_Q1_BENCHMARK_DIR",
    "DEFAULT_Q2_BENCHMARK_DIR",
    "DEFAULT_Q3_BENCHMARK_DIR",
    "DEFAULT_Q4_BENCHMARK_DIR",
    "MIDTERM_REASONING_PROMPTS",
    "MIDTERM_Q2_REASONING_PROMPTS",
    "MIDTERM_Q3_REASONING_PROMPTS",
    "MIDTERM_Q4_REASONING_PROMPTS",
    "MidtermBenchmarkConfig",
    "build_supplier_removal_incentive_diagnostics",
    "build_midterm_manure_cases",
    "build_midterm_manure_q2_cases",
    "build_midterm_manure_q3_cases",
    "build_midterm_manure_q4_cases",
    "build_midterm_manure_expected_plan",
    "build_midterm_manure_q2_expected_plan",
    "build_midterm_manure_q3_expected_plan",
    "build_midterm_manure_q4_expected_plan",
    "build_midterm_output_tables",
    "compare_problem_states_for_midterm",
    "compare_solution_to_reference",
    "evaluate_primary_semantic_metrics",
    "evaluate_midterm_prompt_case",
    "extract_midterm_solution_components",
    "gemini_is_configured",
    "load_benchmark_files",
    "run_midterm_manure_q1_benchmark",
    "run_midterm_manure_q2_benchmark",
    "run_midterm_manure_q3_benchmark",
    "run_midterm_manure_q4_benchmark",
    "run_midterm_reasoning_prompt_battery",
    "write_midterm_outputs",
]
