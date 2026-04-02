"""Lightweight validation for formal math requests and generated output."""

from __future__ import annotations

import re
from typing import List

from .domain.sampat2019 import get_theorem_metadata
from .schema import FormalMathContext


def _extract_wrapped_display_blocks(response_text: str) -> List[str]:
    return re.findall(r"\$\$\s*(.*?)\s*\$\$", response_text, flags=re.DOTALL)


def _strip_display_blocks(response_text: str) -> str:
    return re.sub(r"\$\$.*?\$\$", "", response_text, flags=re.DOTALL).strip()


def _extract_aligned_rows(block_text: str) -> List[str]:
    content = re.sub(r"\\begin\{aligned\}", "", block_text)
    content = re.sub(r"\\end\{aligned\}", "", content)
    return [row.strip() for row in re.split(r"\\\\", content) if row.strip()]


def _contains_relation(text: str) -> bool:
    relation_tokens = ("\\ge", "\\le", "\\geq", "\\leq", "=")
    return any(token in text for token in relation_tokens)


def _relation_count(text: str) -> int:
    relation_tokens = ("\\ge", "\\le", "\\geq", "\\leq", "=")
    return sum(text.count(token) for token in relation_tokens)


def _is_sign_restriction_row(row: str) -> bool:
    compact = row.replace("&", "").strip()
    if not compact:
        return False
    if any(token in compact for token in ("(D)", "\\min", "\\max", "\\text{s.t.}")):
        return False
    sign_tokens = ("\\in \\mathbb{R}", "\\ge 0", "\\le 0", "\\geq 0", "\\leq 0")
    return any(token in compact for token in sign_tokens)


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
        explanatory_sentence = "The dual problem is formulated as follows:"
        display_blocks = _extract_wrapped_display_blocks(response_text)
        surrounding_text = _strip_display_blocks(response_text)
        raw_align_leakage = re.search(r"\\begin\{align\*?\}", response_text) is not None

        for dual_variable in context.dual_variables:
            if dual_variable["symbol"] not in response_text:
                issues.append(
                    f"Dual response omitted expected dual symbol {dual_variable['symbol']}"
                )
        if surrounding_text != explanatory_sentence:
            issues.append("Dual response must include the explanatory sentence exactly once.")
        if len(display_blocks) != 2 or response_text.count("$$") != 4:
            issues.append("Dual response must contain exactly two wrapped display-math blocks.")
        if raw_align_leakage:
            issues.append("Dual response contains raw align-environment leakage.")
        if len(display_blocks) == 2:
            first_block, second_block = display_blocks
            first_rows = _extract_aligned_rows(first_block)
            second_rows = _extract_aligned_rows(second_block)

            if first_block.count("\\begin{aligned}") != 1 or first_block.count("\\end{aligned}") != 1:
                issues.append("Dual response first block must be a single aligned block.")
            if "(D)" not in first_block or "\\min" not in first_block or "\\text{s.t.}" not in first_block:
                issues.append("Dual response first block must contain the dual label, objective, and s.t.")
            if "sign restrictions" in first_block.lower():
                issues.append("Dual response first block must contain only the objective and inequalities.")
            if not first_rows or "(D)" not in first_rows[0] or "\\min" not in first_rows[0]:
                issues.append("Dual response first block must begin with the dual objective.")
            if len(first_rows) < 3 or first_rows[1].replace("&", "").strip() != "\\text{s.t.}":
                issues.append("Dual response first block must place s.t. on its own line.")
            inequality_rows = [row for row in first_rows[2:] if row.replace("&", "").strip()]
            if not inequality_rows:
                issues.append("Dual response first block must place the inequalities under s.t.")
            if any(not _contains_relation(row) for row in inequality_rows):
                issues.append("Dual response first block must contain one inequality per line.")
            if any(_relation_count(row) != 1 for row in inequality_rows):
                issues.append("Dual response first block must not horizontally pack multiple inequalities.")
            if any(
                "\\text{" in row and "\\text{s.t.}" not in row
                for row in first_rows
            ):
                issues.append("Dual response must not include inline labels inside the dual blocks.")

            if any(token in second_block for token in ("(D)", "\\min", "\\max", "\\text{s.t.}")):
                issues.append("Dual response second block must contain sign restrictions only.")
            if not second_rows or any(not _is_sign_restriction_row(row) for row in second_rows):
                issues.append("Dual response second block must contain sign restrictions only.")
            if any("\\text{" in row for row in second_rows):
                issues.append("Dual response must not include inline labels inside the dual blocks.")

    if context.request_type == "theorem_proof":
        if context.applicable is True:
            lowered = response_text.lower()
            has_proof_marker = "\\begin{proof}" in response_text or "**proof.**" in lowered or "\\textbf{proof.}" in lowered
            if not has_proof_marker:
                issues.append("Proof response is missing a clear proof label.")
            if "$$" not in response_text and "\\[" not in response_text:
                issues.append("Proof response should expose at least one notebook-friendly display-math block.")
            if context.theorem_id == "theorem_1":
                compact = re.sub(r"\s+", "", response_text)
                aligned_count = response_text.count("\\begin{aligned}")
                forbidden_phrases = [
                    "max coordinated surplus over",
                    "dual objective induced by",
                    "coordinated surplus over",
                    "supported structured context",
                    "concise or / mathematical programming proof",
                    "grounded in the verified structured context",
                ]
                if "strong duality" not in lowered:
                    issues.append("Theorem 1 response must explicitly state strong duality.")
                if "z_P^*=z_D^*" not in compact:
                    issues.append("Theorem 1 response must explicitly include z_P^* = z_D^*.")
                if "**primal problem.**" not in lowered or "**dual problem.**" not in lowered:
                    issues.append("Theorem 1 response must include clean primal and dual problem headings.")
                if "\\mathcal{L}" not in response_text and "lagrangian" not in lowered:
                    issues.append("Theorem 1 response must explicitly define the Lagrangian.")
                if aligned_count < 3:
                    issues.append("Theorem 1 response must be equation-driven and include aligned primal, dual, and derivation blocks.")
                if lowered.count("dual problem") > 1 or response_text.count("(D)\\qquad \\min") > 1:
                    issues.append("Theorem 1 response contains duplicate dual blocks.")
                if "primal has an optimal solution" in lowered or "primal optimum exists" in lowered:
                    issues.append("Theorem 1 response drifted to primal-optimum existence instead of strong duality.")
                if any(phrase in lowered for phrase in forbidden_phrases):
                    issues.append("Theorem 1 response contains placeholder phrasing instead of explicit mathematical programming content.")
                if re.search(r"(q_b\s+q_b\s+q_b)|(\(p\)\(p\))", lowered):
                    issues.append("Theorem 1 response contains duplicated mathematical fragments.")
        else:
            lowered = response_text.lower()
            if "cannot certify" not in lowered and "out of scope" not in lowered:
                issues.append("Non-applicable proof response did not clearly explain the failure mode.")

    return issues
