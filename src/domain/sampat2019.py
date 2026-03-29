"""Hand-curated Sampat et al. (2019) metadata for Sections 2.1-2.3.

This module intentionally provides a narrow structured registry rather than
general document retrieval. The data is designed to support deterministic
routing and bounded mathematical exposition for:

- primal/dual formulation requests
- theorem applicability and proof-style requests
- Section 2.3 interpretations involving negative bids and prices
"""

from __future__ import annotations

from typing import Any, Dict, Optional


DOMAIN_SOURCE = (
    "Sampat et al. (2019), Sections 2.1-2.3, with supporting information "
    "for benchmark cases A-C."
)


CANONICAL_NOTATION: Dict[str, Dict[str, str]] = {
    "variables": {
        "q_b": "Accepted quantity for bid b",
        "f_ij": "Transport flow on arc (i,j)",
        "x_k": "Technology activity level for technology k",
    },
    "duals": {
        "pi_np": "Dual variable for node-product balance at node n and product p",
        "mu_b": "Dual variable for supplier bid upper bound",
        "nu_b": "Dual variable for consumer bid upper bound",
        "tau_ij": "Dual variable for transport capacity",
    },
    "sets": {
        "N": "Nodes",
        "P": "Products",
        "B_s": "Supplier bids",
        "B_c": "Consumer bids",
        "T": "Transport arcs",
        "K": "Technologies",
    },
    "economics": {
        "negative_bid": "A bid with price below zero, capturing disposal, remediation, storage, or VOS-like incentives.",
        "negative_price": "An equilibrium shadow price indicating the system is willing to pay for additional withdrawal or remediation of a commodity.",
    },
    "rendering": {
        "primal_sense": "maximize coordinated surplus",
        "dual_sense": "minimize capacity-weighted dual costs",
        "preferred_environment": "aligned",
    },
}


SECTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "2.1": {
        "title": "Primal coordinated clearing formulation",
        "themes": [
            "network welfare maximization",
            "node-product balance",
            "bid acceptance variables",
        ],
    },
    "2.2": {
        "title": "Dual interpretation and theorem-style economic interpretation",
        "themes": [
            "Lagrangian structure",
            "shadow prices",
            "competitive interpretation",
        ],
    },
    "2.3": {
        "title": "Negative bids and economic interpretation",
        "themes": [
            "negative bids",
            "disposal and remediation",
            "storage and VOS",
            "negative prices",
        ],
    },
}


THEOREM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "theorem_1": {
        "title": "Theorem 1",
        "target_section": "2.2",
        "supported_request_types": [
            "theorem_proof",
            "theorem_explanation",
            "theorem_applicability",
        ],
        "assumptions": [
            "validated_linear_problem_state",
            "basic_supply_demand_structure",
            "finite_bid_data",
            "technology_yields_specified_when_present",
        ],
        "proof_skeleton": [
            "State the deterministic assumptions verified by the checker.",
            "Write the primal model and identify linear feasibility conditions.",
            "Introduce the Lagrangian and dual multipliers for each supported constraint family.",
            "Use linear programming duality and complementary-slackness interpretation.",
            "Conclude only within the verified scope of the current ProblemState.",
        ],
        "statement_template": (
            "For the validated linear coordinated-clearing instance, the primal welfare-maximization "
            "formulation and the supported dual price system are consistent within the scoped Sections 2.1-2.2 interpretation."
        ),
        "proof_style": "concise OR / mathematical programming proof",
        "source_notes": [
            "Scoped to the Sections 2.1-2.2 exposition supported by this repository.",
            "Applicability is determined by deterministic structural checks, not by the LLM.",
        ],
    }
}


SECTION23_CONCEPTS: Dict[str, str] = {
    "negative_bids": (
        "Negative bids represent activities for which an agent is willing to pay "
        "the market to accept or remove material, such as disposal, remediation, "
        "or storage obligations."
    ),
    "disposal": "Disposal can be modeled as a negative-value activity when clearing the material reduces system burden.",
    "remediation": "Remediation enters as a costly but beneficial removal activity and can justify negative bids.",
    "storage": "Storage can appear as a temporally shifted sink/source with negative-bid interpretation when holding inventory is costly.",
    "vos": "Value-of-service interpretations can rationalize willingness to pay for reliability, removal, or treatment.",
    "negative_prices": (
        "Negative nodal prices signal that the system benefits from additional withdrawal "
        "or pays to induce removal at that location-product pair."
    ),
}


def get_section_metadata(section_id: str) -> Optional[Dict[str, Any]]:
    """Return structured metadata for a supported section."""

    return SECTION_REGISTRY.get(section_id)


def get_theorem_metadata(theorem_id: str) -> Optional[Dict[str, Any]]:
    """Return theorem metadata for supported theorem identifiers."""

    return THEOREM_REGISTRY.get(theorem_id)


def infer_benchmark_case(has_technology: bool, has_negative_bids: bool) -> str:
    """Infer the benchmark family name from structural features."""

    if has_technology:
        return "Case C"
    if has_negative_bids:
        return "Case B"
    return "Case A"
