"""Theorem applicability checker for coordinated supply chain problems.

This module implements a lightweight rule-based evaluation workflow that
examines the current :class:`~src.schema.ProblemState` and determines whether
certain theoretical conditions or benchmark family assumptions apply.  The
checks are deliberately simple and deterministic so they can be used in both
unit tests and classroom explanations.

The results are captured as a list of :class:`~src.schema.TheoremCheck`
objects.  Each check has a human-readable name, a boolean ``applies`` flag, and
an explanatory message.  The ``check_theorems`` function populates the
``problem_state.theorem_checks`` list as a side-effect and returns the list so
callers may inspect the results programmatically.

The current set of checks covers:

* basic structural conditions (is there at least one supply and one demand
  entity?)
* benchmark compatibility checks for the three case families from the Sampat
  supporting information: Case A (no transformation), Case B (negative bids),
  and Case C (transformation technology).

Additional theorems may be added in the future as the system expands.
"""

from typing import Dict, List

from .schema import ProblemState, TheoremCheck
from .validator import validate_state


def get_theorem_check_map(theorem_checks: List[TheoremCheck]) -> Dict[str, TheoremCheck]:
    """Index theorem checks by theorem identifier when available."""

    return {
        check.theorem_id: check
        for check in theorem_checks
        if check.theorem_id
    }


def check_theorems(state: ProblemState) -> List[TheoremCheck]:
    """Evaluate a collection of theorem/assumption checks on ``state``.

    The function examines the state for various structural and benchmark
    conditions.  It always returns a fresh list of ``TheoremCheck`` objects and
    updates ``state.theorem_checks`` to the same list.

    Args:
        state: The current ProblemState to evaluate.

    Returns:
        A list of ``TheoremCheck`` records summarizing each check.
    """
    checks: List[TheoremCheck] = []
    validation = validate_state(state)

    # basic supply-demand structure
    has_supply = bool(state.suppliers or state.technologies)
    has_demand = bool(state.consumers or state.technologies)
    if has_supply and has_demand:
        explanation = "At least one supply-side and one demand-side entity present."
        applies = True
    else:
        explanation = "Missing supply or demand entity: "
        if not has_supply:
            explanation += "no suppliers/technologies"
        if not has_demand:
            if not has_supply:
                explanation += "; "
            explanation += "no consumers/technologies"
        applies = False
    checks.append(
        TheoremCheck(
            theorem_name="Basic supply-demand structure",
            theorem_id="basic_supply_demand_structure",
            applies=applies,
            explanation=explanation,
            assumptions_verified=(
                ["basic_supply_demand_structure"] if applies else []
            ),
            assumptions_missing=(
                [] if applies else ["basic_supply_demand_structure"]
            ),
        )
    )

    # Case A compatibility: no transformation technologies
    no_tech = not state.technologies
    checks.append(
        TheoremCheck(
            theorem_name="Case A compatibility (no transformation)",
            theorem_id="case_a_compatibility",
            applies=no_tech,
            explanation="no technologies present" if no_tech else "technologies present (transformation detected)",
            assumptions_verified=["no_transformation_technology"] if no_tech else [],
            assumptions_missing=[] if no_tech else ["no_transformation_technology"],
        )
    )

    # Case B compatibility: presence of at least one negative bid price
    negative_count = sum(1 for b in state.bids if b.price < 0)
    checks.append(
        TheoremCheck(
            theorem_name="Case B compatibility (negative bidding)",
            theorem_id="case_b_compatibility",
            applies=negative_count > 0,
            explanation=(
                f"found {negative_count} negative bid(s)" if negative_count > 0
                else "no negative bids found"
            ),
            assumptions_verified=["negative_bid_present"] if negative_count > 0 else [],
            assumptions_missing=[] if negative_count > 0 else ["negative_bid_present"],
        )
    )

    # Case C compatibility: at least one technology with both input (neg)
    # and output (pos) yields and a positive capacity
    case_c_applies = False
    case_c_explanation = ""
    for tech in state.technologies:
        pos = any(v > 0 for v in tech.yield_coefficients.values())
        neg = any(v < 0 for v in tech.yield_coefficients.values())
        cap_ok = tech.capacity is not None and tech.capacity > 0
        if pos and neg and cap_ok:
            case_c_applies = True
            case_c_explanation = (
                f"technology {tech.id} has transformation yields and positive capacity"
            )
            break
        if pos and neg and not cap_ok:
            case_c_explanation = (
                f"technology {tech.id} has transformation yields but missing positive capacity"
            )
            break
    if not case_c_explanation:
        case_c_explanation = "no technology with both input (neg) and output (pos) yields found"
    checks.append(
        TheoremCheck(
            theorem_name="Case C compatibility (transformation)",
            theorem_id="case_c_compatibility",
            applies=case_c_applies,
            explanation=case_c_explanation,
            assumptions_verified=["transformation_structure_present"] if case_c_applies else [],
            assumptions_missing=[] if case_c_applies else ["transformation_structure_present"],
        )
    )

    theorem_1_missing = []
    theorem_1_verified = []
    if validation["solver_ready"]:
        theorem_1_verified.append("validated_linear_problem_state")
    else:
        theorem_1_missing.append("validated_linear_problem_state")

    if has_supply and has_demand:
        theorem_1_verified.append("basic_supply_demand_structure")
    else:
        theorem_1_missing.append("basic_supply_demand_structure")

    if all(bid.price == bid.price for bid in state.bids):
        theorem_1_verified.append("finite_bid_data")
    else:
        theorem_1_missing.append("finite_bid_data")

    technologies_complete = len(validation["incomplete_technologies"]) == 0
    if technologies_complete:
        theorem_1_verified.append("technology_yields_specified_when_present")
    else:
        theorem_1_missing.append("technology_yields_specified_when_present")

    theorem_1_applies = len(theorem_1_missing) == 0
    theorem_1_explanation = (
        "validated linear structure supports the scoped Sections 2.1-2.2 theorem exposition"
        if theorem_1_applies
        else "deterministic prerequisites for the supported Theorem 1 exposition are incomplete"
    )
    checks.append(
        TheoremCheck(
            theorem_name="Theorem 1 applicability",
            theorem_id="theorem_1",
            target_section="2.2",
            applies=theorem_1_applies,
            explanation=theorem_1_explanation,
            assumptions_verified=theorem_1_verified,
            assumptions_missing=theorem_1_missing,
            metadata={
                "validation_solver_ready": validation["solver_ready"],
                "benchmark_compatibility": validation["benchmark_compatibility"],
            },
        )
    )

    # update state
    state.theorem_checks = checks
    return checks


__all__ = ["check_theorems", "get_theorem_check_map"]
