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
from .validator import validate_state
from .domain.sampat2019 import (
    CANONICAL_NOTATION,
    DOMAIN_SOURCE,
    SECTION23_CONCEPTS,
    get_section_metadata,
    get_theorem_metadata,
    infer_benchmark_case,
)


def identify_formal_math_request(user_message: str) -> Dict[str, Optional[str]]:
    """Classify a user request into a supported formal-math request type."""

    message = user_message.lower()
    theorem_match = re.search(r"theorem\s+(\d+)", message)
    theorem_id = f"theorem_{theorem_match.group(1)}" if theorem_match else None

    if "dual" in message:
        return {
            "request_type": "dual",
            "theorem_id": theorem_id,
            "target_section": "2.2",
        }
    if theorem_match and any(token in message for token in ("show", "prove", "proof", "holds")):
        return {
            "request_type": "theorem_proof",
            "theorem_id": theorem_id,
            "target_section": "2.2",
        }
    if theorem_match and any(token in message for token in ("apply", "applies", "applicability", "why")):
        return {
            "request_type": "theorem_explanation",
            "theorem_id": theorem_id,
            "target_section": "2.2",
        }
    if "negative bid" in message or "negative price" in message or "section 2.3" in message:
        return {
            "request_type": "section23_explanation",
            "theorem_id": None,
            "target_section": "2.3",
        }
    if "theorem" in message:
        return {
            "request_type": "theorem_explanation",
            "theorem_id": theorem_id,
            "target_section": "2.2",
        }
    return {
        "request_type": "general_math_explanation",
        "theorem_id": theorem_id,
        "target_section": None,
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

    request_info = identify_formal_math_request(user_message)
    theorem_checks = state.theorem_checks or []
    if not theorem_checks:
        from .theorem_checker import check_theorems

        theorem_checks = check_theorems(state)

    theorem_check = _resolve_theorem_check(theorem_checks, request_info["theorem_id"])
    validation = validate_state(state)
    primal_representation = build_primal_representation(state)
    dual_representation = (
        build_dual_scaffold(primal_representation)
        if request_info["request_type"] == "dual"
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
        latex_mode="align" if "latex" in user_message.lower() or request_info["request_type"] == "dual" else "plain",
        pedagogical_mode=pedagogical_mode,
        user_request=user_message,
    )
