"""Structured formal math context builder."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .dual_generator import (
    build_dual_scaffold,
    build_primal_representation,
    infer_negative_bid_notes,
)
from .schema import FormalMathContext, ProblemState, TheoremCheck
from .theorem_checker import get_theorem_check_map
from .domain.sampat2019 import (
    CANONICAL_NOTATION,
    DOMAIN_SOURCE,
    SECTION23_CONCEPTS,
    get_section_metadata,
    get_theorem_metadata,
    infer_benchmark_case,
)


def _contains_any(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


def plan_formal_math_request(user_message: str) -> Dict[str, Any]:
    """Infer a small semantic plan for flexible formal-math generation."""

    message = user_message.lower()
    theorem_match = re.search(r"theorem\s+(\d+)", message)
    theorem_id = f"theorem_{theorem_match.group(1)}" if theorem_match else None

    mentions_dual = "dual" in message
    mentions_primal = "primal" in message
    mentions_negative_bids = _contains_any(
        message,
        ["negative bid", "negative bids", "negative price", "negative prices", "section 2.3"],
    )
    mentions_strong_duality = "strong duality" in message
    mentions_complementary_slackness = "complementary slackness" in message
    mentions_prices = _contains_any(
        message,
        ["node-product prices", "node product prices", "nodal prices", "dual variables", "prices", "price"],
    )
    wants_explanation = _contains_any(
        message,
        ["explain", "why", "meaning", "economic", "interpret", "interpretation", "role", "compare", "relationship", "relates", "incentives"],
    )
    wants_proof = _contains_any(message, ["prove", "proof", "proof structure"]) or (
        theorem_match is not None and _contains_any(message, ["show", "holds"])
    )
    wants_verification = _contains_any(
        message,
        ["verify", "verification", "check", "holds", "applies", "applicability", "assumptions", "valid"],
    )
    wants_formulation = _contains_any(
        message,
        ["write the dual", "dual problem", "formulate", "formulation", "show me the dual", "give me the dual"],
    )
    wants_only = _contains_any(message, ["only", "no prose", "just"])
    wants_concise = _contains_any(message, ["shorter", "brief", "concise"])

    math_topics: List[str] = []
    if theorem_id:
        math_topics.append(theorem_id)
    if mentions_primal:
        math_topics.append("primal")
    if mentions_dual:
        math_topics.append("dual")
    if mentions_strong_duality or theorem_id == "theorem_1":
        math_topics.append("strong_duality")
    if mentions_complementary_slackness:
        math_topics.append("complementary_slackness")
    if mentions_prices:
        math_topics.append("node_product_prices")
    if mentions_negative_bids:
        math_topics.append("negative_bids")
    if wants_explanation and mentions_dual:
        math_topics.append("economic_interpretation")
    math_topics = list(dict.fromkeys(math_topics))

    task_modes: List[str] = []
    if wants_formulation:
        task_modes.append("formulation")
    if wants_proof:
        task_modes.append("proof")
    if wants_verification:
        task_modes.append("verification")
    if wants_explanation or not task_modes:
        task_modes.append("explanation")
    if mentions_primal and mentions_dual and ("compare" in message or "relationship" in message):
        task_modes.append("comparison")
    task_modes = list(dict.fromkeys(task_modes))

    requested_dual_formulation = mentions_dual and wants_formulation
    requested_primal_formulation = mentions_primal and wants_formulation
    include_dual_formulation = requested_dual_formulation
    include_primal_formulation = (
        requested_primal_formulation or theorem_id == "theorem_1" or "comparison" in task_modes
    )
    include_proof_structure = "proof" in task_modes
    include_economic_interpretation = (
        "economic_interpretation" in math_topics
        or "node_product_prices" in math_topics
        or "complementary_slackness" in math_topics
    )

    if theorem_id and include_proof_structure:
        primary_goal = "theorem_proof"
    elif requested_dual_formulation and include_economic_interpretation:
        primary_goal = "mixed_dual_interpretation"
    elif requested_dual_formulation:
        primary_goal = "dual_formulation"
    elif requested_primal_formulation:
        primary_goal = "primal_formulation"
    elif theorem_id:
        primary_goal = "theorem_explanation"
    elif mentions_negative_bids:
        primary_goal = "section23_explanation"
    else:
        primary_goal = "conceptual_explanation"

    response_contract = {
        "include_dual_formulation": include_dual_formulation,
        "include_primal_formulation": include_primal_formulation,
        "include_proof_structure": include_proof_structure,
        "include_economic_interpretation": include_economic_interpretation,
        "explain_assumptions": theorem_id is not None,
        "prefer_latex": "latex" in message or include_dual_formulation or include_proof_structure,
        "prefer_concise": wants_concise or wants_only,
        "avoid_full_dual_formulation": mentions_dual and not include_dual_formulation,
    }

    target_section = None
    if mentions_negative_bids:
        target_section = "2.3"
    elif mentions_dual or theorem_id:
        target_section = "2.2"

    return {
        "primary_goal": primary_goal,
        "task_modes": task_modes,
        "math_topics": math_topics,
        "theorem_id": theorem_id,
        "target_section": target_section,
        "response_contract": response_contract,
        "is_supported_request": bool(math_topics or theorem_id or mentions_negative_bids),
    }


def identify_formal_math_request(user_message: str) -> Dict[str, Optional[str]]:
    """Classify a user request into a supported formal-math request type."""

    plan = plan_formal_math_request(user_message)
    theorem_id = plan["theorem_id"]
    response_contract = plan["response_contract"]

    if plan["primary_goal"] == "dual_formulation":
        return {
            "request_type": "dual",
            "theorem_id": theorem_id,
            "target_section": plan["target_section"],
        }
    if plan["primary_goal"] == "primal_formulation":
        return {
            "request_type": "primal",
            "theorem_id": theorem_id,
            "target_section": plan["target_section"],
        }
    if plan["primary_goal"] == "theorem_proof":
        return {
            "request_type": "theorem_proof",
            "theorem_id": theorem_id,
            "target_section": plan["target_section"],
        }
    if theorem_id and (
        "explanation" in plan["task_modes"]
        or "verification" in plan["task_modes"]
        or response_contract["explain_assumptions"]
    ):
        return {
            "request_type": "theorem_explanation",
            "theorem_id": theorem_id,
            "target_section": plan["target_section"],
        }
    if plan["primary_goal"] == "section23_explanation":
        return {
            "request_type": "section23_explanation",
            "theorem_id": None,
            "target_section": plan["target_section"],
        }
    return {
        "request_type": "general_math_explanation",
        "theorem_id": theorem_id,
        "target_section": plan["target_section"],
    }


def _resolve_theorem_check(
    theorem_checks: List[TheoremCheck],
    theorem_id: Optional[str],
) -> Optional[TheoremCheck]:
    if not theorem_id:
        return None

    check_map = get_theorem_check_map(theorem_checks)
    return check_map.get(theorem_id)


def build_formal_math_context(
    state: ProblemState,
    user_message: str,
    pedagogical_mode: str = "guided",
) -> FormalMathContext:
    """Build a structured context for theorem/proof/dual exposition."""

    semantic_plan = plan_formal_math_request(user_message)
    request_info = identify_formal_math_request(user_message)
    theorem_checks = state.theorem_checks or []
    if not theorem_checks:
        from .theorem_checker import check_theorems

        theorem_checks = check_theorems(state)

    theorem_check = _resolve_theorem_check(theorem_checks, request_info["theorem_id"])
    primal_representation = build_primal_representation(state)
    dual_representation = (
        build_dual_scaffold(primal_representation)
        if (
            request_info["request_type"] == "dual"
            or "dual" in semantic_plan["math_topics"]
            or request_info["request_type"] == "theorem_proof"
        )
        else None
    )

    has_negative_bids = any(bid.price < 0 for bid in state.bids)
    benchmark_case = (
        state.benchmark.case_family
        if state.benchmark and state.benchmark.case_family
        else infer_benchmark_case(bool(state.technologies), has_negative_bids)
    )

    theorem_metadata = (
        get_theorem_metadata(request_info["theorem_id"])
        if request_info["theorem_id"]
        else None
    )
    section_metadata = (
        get_section_metadata(request_info["target_section"])
        if request_info["target_section"]
        else None
    )

    source_notes: List[str] = []
    source_notes.append(DOMAIN_SOURCE)
    if theorem_metadata:
        source_notes.extend(theorem_metadata.get("source_notes", []))
    elif request_info["theorem_id"]:
        source_notes.append(
            f"Theorem identifier {request_info['theorem_id']} is outside the supported Sampat Sections 2.1-2.3 registry."
        )
    if section_metadata:
        source_notes.append(section_metadata["title"])
    source_notes.extend(infer_negative_bid_notes(state))

    supporting_equations = [
        "Primal objective: maximize accepted consumer value minus accepted supplier cost minus transport and technology costs.",
        "Node-product balance constraints define the clearing conditions.",
        "Capacity upper bounds induce nonnegative dual multipliers.",
    ]
    if request_info["target_section"] == "2.3":
        supporting_equations.append(SECTION23_CONCEPTS["negative_prices"])

    assumptions_verified = list(theorem_check.assumptions_verified) if theorem_check else []
    assumptions_missing = list(theorem_check.assumptions_missing) if theorem_check else []
    applicable = theorem_check.applies if theorem_check else None

    return FormalMathContext(
        request_type=request_info["request_type"] or "general_math_explanation",
        domain_source=DOMAIN_SOURCE,
        target_section=request_info["target_section"],
        theorem_id=request_info["theorem_id"],
        applicable=applicable,
        assumptions_verified=assumptions_verified,
        assumptions_missing=assumptions_missing,
        notation_profile={
            "canonical": CANONICAL_NOTATION,
            "nodes": state.node_ids(),
            "products": state.product_ids(),
            "bids": [bid.id for bid in state.bids],
            "technologies": [tech.id for tech in state.technologies],
            "supported_theorem_ids": ["theorem_1"],
            "theorem_title": theorem_metadata["title"] if theorem_metadata else None,
        },
        primal_formulation=primal_representation,
        dual_formulation=dual_representation,
        objective=primal_representation.get("objective"),
        constraints=primal_representation.get("constraints", []),
        variables=primal_representation.get("variables", []),
        dual_variables=(dual_representation or {}).get("dual_variables", []),
        profit_definitions=[
            {
                "name": "coordinated_surplus",
                "description": "Consumer value minus supplier cost minus transport cost minus technology cost.",
            }
        ],
        lagrangian_components=[
            {
                "name": "node_balance",
                "description": "Free dual multipliers for node-product balance equalities.",
            },
            {
                "name": "capacity_bounds",
                "description": "Nonnegative multipliers for supported upper-bound constraints.",
            },
        ],
        benchmark_case=benchmark_case,
        supporting_equations=supporting_equations,
        source_notes=source_notes,
        semantic_plan=semantic_plan,
        problem_state_snapshot=state.dict(),
        latex_mode="align" if semantic_plan["response_contract"]["prefer_latex"] else "plain",
        pedagogical_mode=pedagogical_mode,
        user_request=user_message,
    )
