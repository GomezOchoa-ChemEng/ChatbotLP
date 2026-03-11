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

from typing import List

from .schema import ProblemState, TheoremCheck


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
            applies=applies,
            explanation=explanation,
        )
    )

    # Case A compatibility: no transformation technologies
    no_tech = not state.technologies
    checks.append(
        TheoremCheck(
            theorem_name="Case A compatibility (no transformation)",
            applies=no_tech,
            explanation="no technologies present" if no_tech else "technologies present (transformation detected)",
        )
    )

    # Case B compatibility: presence of at least one negative bid price
    negative_count = sum(1 for b in state.bids if b.price < 0)
    checks.append(
        TheoremCheck(
            theorem_name="Case B compatibility (negative bidding)",
            applies=negative_count > 0,
            explanation=(
                f"found {negative_count} negative bid(s)" if negative_count > 0
                else "no negative bids found"
            ),
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
            applies=case_c_applies,
            explanation=case_c_explanation,
        )
    )

    # update state
    state.theorem_checks = checks
    return checks


__all__ = ["check_theorems"]
