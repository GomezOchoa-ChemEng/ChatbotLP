"""Lightweight validation for formal math requests and generated output."""

from __future__ import annotations

import re
from typing import List

from .domain.sampat2019 import get_theorem_metadata
from .schema import FormalMathContext


def validate_formal_math_context(context: FormalMathContext) -> List[str]:
    """Return validation issues for a formal math context."""

    issues: List[str] = []

    if context.request_type == "theorem_proof":
        if not context.theorem_id:
            issues.append("Proof generation requires an explicit supported theorem identifier.")

    if context.request_type == "dual":
        if not context.dual_variables:
            issues.append("Dual generation requires dual variables derived from known constraints.")
        if not context.primal_formulation:
            issues.append("Dual generation requires a primal formulation scaffold.")

    required_symbols = []
    if context.request_type == "dual":
        required_symbols.extend(["q_b", "pi_np"])
    if context.request_type in {"theorem_proof", "theorem_explanation"}:
        required_symbols.append("q_b")

    canonical = context.notation_profile.get("canonical", {})
    variable_symbols = canonical.get("variables", {})
    dual_symbols = canonical.get("duals", {})
    for symbol in required_symbols:
        if symbol not in variable_symbols and symbol not in dual_symbols:
            issues.append(f"Required notation symbol missing from notation profile: {symbol}")

    return issues


def validate_generated_math_response(
    context: FormalMathContext,
    response_text: str,
) -> List[str]:
    """Return post-generation issues for a generated mathematical response."""

    issues: List[str] = []
    theorem_ids = re.findall(r"theorem[_ ]\d+", response_text.lower())
    if context.theorem_id:
        allowed_token = context.theorem_id.replace("_", " ")
        for token in theorem_ids:
            if token not in {context.theorem_id, allowed_token}:
                issues.append(f"Unexpected theorem reference in response: {token}")

    if context.assumptions_missing:
        missing_disclosure = "missing assumptions" in response_text.lower() or "assumptions are not fully verified" in response_text.lower()
        if context.request_type == "theorem_proof" and not missing_disclosure:
            issues.append("Response did not clearly disclose missing assumptions.")

    forbidden_wrappers = [
        "\\documentclass",
        "\\usepackage",
        "\\begin{document}",
        "\\end{document}",
    ]
    for wrapper in forbidden_wrappers:
        if wrapper in response_text:
            issues.append("Response must be a render-ready LaTeX fragment, not a full document.")
            break

    if context.request_type == "dual":
        for dual_variable in context.dual_variables:
            if dual_variable["symbol"] not in response_text:
                issues.append(
                    f"Dual response omitted expected dual symbol {dual_variable['symbol']}"
                )
        if "\\min" not in response_text or "\\text{s.t.}" not in response_text:
            issues.append("Dual response is missing standard optimization LaTeX structure.")
        if "$$" not in response_text and "\\[" not in response_text:
            issues.append("Dual response should expose a notebook-friendly display-math block.")

    if context.request_type == "theorem_proof":
        if context.applicable is True:
            lowered = response_text.lower()
            has_proof_marker = "\\begin{proof}" in response_text or "**proof.**" in lowered or "\\textbf{proof.}" in lowered
            if not has_proof_marker:
                issues.append("Proof response is missing a clear proof label.")
            if "$$" not in response_text and "\\[" not in response_text:
                issues.append("Proof response should expose at least one notebook-friendly display-math block.")
        else:
            lowered = response_text.lower()
            if "cannot certify" not in lowered and "out of scope" not in lowered:
                issues.append("Non-applicable proof response did not clearly explain the failure mode.")

    return issues
