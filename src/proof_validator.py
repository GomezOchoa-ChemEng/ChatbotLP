"""Lightweight validation for formal math requests and generated output."""

from __future__ import annotations

import re
from typing import Dict, List

from .domain.sampat2019 import get_theorem_metadata
from .schema import FormalMathContext


GROUNDING_WARNING = "This response may not be fully grounded in the deterministic model."
ValidationResult = Dict[str, List[str]]


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


def empty_validation_result() -> ValidationResult:
    return {"fatal": [], "warnings": []}


def _finalize_validation(
    warnings: List[str],
    fatal: List[str] | None = None,
) -> ValidationResult:
    return {
        "fatal": list(dict.fromkeys(fatal or [])),
        "warnings": list(dict.fromkeys(warnings)),
    }


def validate_formal_math_context(context: FormalMathContext) -> ValidationResult:
    """Return structured validation results for a formal math context."""

    warnings: List[str] = []

    if context.request_type == "theorem_proof":
        if not context.theorem_id:
            warnings.append("Proof generation requires an explicit supported theorem identifier.")

    if context.request_type == "dual":
        if not context.dual_variables:
            warnings.append("Dual generation requires dual variables derived from known constraints.")
        if not context.primal_formulation:
            warnings.append("Dual generation requires a primal formulation scaffold.")
    if context.request_type == "primal":
        if not context.primal_formulation:
            warnings.append("Primal generation requires a primal formulation scaffold.")

    required_symbols = []
    if context.request_type == "dual":
        required_symbols.extend(["q_b", "pi_np"])
    if context.request_type == "primal":
        required_symbols.append("q_b")
    if context.request_type in {"theorem_proof", "theorem_explanation"}:
        required_symbols.append("q_b")

    canonical = context.notation_profile.get("canonical", {})
    variable_symbols = canonical.get("variables", {})
    dual_symbols = canonical.get("duals", {})
    for symbol in required_symbols:
        if symbol not in variable_symbols and symbol not in dual_symbols:
            warnings.append(f"Required notation symbol missing from notation profile: {symbol}")

    fatal = warnings if not context_is_structurally_usable(context) else []
    remaining_warnings = [] if fatal else warnings
    return _finalize_validation(remaining_warnings, fatal)


def context_is_structurally_usable(context: FormalMathContext) -> bool:
    """Return whether the formal context is usable for generation.

    Minor grounding gaps should not block generation. This check is intentionally
    narrow and only catches requests that cannot be rendered in any meaningful
    way from the available context.
    """

    if context.request_type == "dual":
        return bool(context.primal_formulation and context.dual_variables)
    if context.request_type == "primal":
        return bool(context.primal_formulation)
    if context.request_type == "theorem_proof":
        return bool(context.theorem_id)
    return True


def response_is_structurally_usable(
    context: FormalMathContext,
    response_text: str,
) -> bool:
    """Return whether generated output is usable enough to show the user."""

    text = (response_text or "").strip()
    if not text:
        return False

    if text.count("$$") % 2 != 0:
        return False

    latex_envs = (
        "aligned",
        "align",
        "align*",
        "equation",
        "equation*",
        "gather",
        "gather*",
    )
    for env in latex_envs:
        if text.count(f"\\begin{{{env}}}") != text.count(f"\\end{{{env}}}"):
            return False

    if context.request_type in {"dual", "primal"}:
        has_math_marker = any(token in text for token in ("$$", "\\begin{", "\\["))
        if not has_math_marker:
            return False

    return True


def validate_generated_math_response(
    context: FormalMathContext,
    response_text: str,
) -> ValidationResult:
    """Return structured post-generation validation results for a math response."""

    warnings: List[str] = []
    theorem_ids = re.findall(r"theorem[_ ]\d+", response_text.lower())
    if context.theorem_id:
        allowed_token = context.theorem_id.replace("_", " ")
        for token in theorem_ids:
            if token not in {context.theorem_id, allowed_token}:
                warnings.append(f"Unexpected theorem reference in response: {token}")

    if context.assumptions_missing:
        missing_disclosure = "missing assumptions" in response_text.lower() or "assumptions are not fully verified" in response_text.lower()
        if context.request_type == "theorem_proof" and not missing_disclosure:
            warnings.append("Response did not clearly disclose missing assumptions.")

    forbidden_wrappers = [
        "\\documentclass",
        "\\usepackage",
        "\\begin{document}",
        "\\end{document}",
    ]
    for wrapper in forbidden_wrappers:
        if wrapper in response_text:
            warnings.append("Response must be a render-ready LaTeX fragment, not a full document.")
            break

    if context.request_type == "dual":
        display_blocks = _extract_wrapped_display_blocks(response_text)
        if "The dual problem is formulated as follows:" not in response_text:
            warnings.append("Dual response should introduce the dual formulation explicitly.")
        if "(D)" not in response_text or "\\min" not in response_text:
            warnings.append("Dual response should include a recognizable dual objective.")
        if "\\text{s.t.}" not in response_text and "subject to" not in response_text.lower():
            warnings.append("Dual response should mark the dual constraints.")
        for dual_variable in context.dual_variables:
            if dual_variable["symbol"] not in response_text:
                warnings.append(
                    f"Dual response omitted expected dual symbol {dual_variable['symbol']}"
                )
        if len(display_blocks) >= 1:
            first_rows = _extract_aligned_rows(display_blocks[0])
            inequality_rows = [
                row for row in first_rows
                if _contains_relation(row) and "\\text{s.t.}" not in row
            ]
            if any(_relation_count(row) > 1 for row in inequality_rows):
                warnings.append("Dual response packs multiple inequalities into a single row.")
        else:
            warnings.append("Dual response should include at least one display-math block.")

    if context.request_type == "primal":
        lowered = response_text.lower()
        if "the primal problem is formulated as follows:" not in lowered:
            warnings.append("Primal response should introduce the primal formulation explicitly.")
        if "(p)" not in lowered or "\\max" not in response_text:
            warnings.append("Primal response should contain the primal label and objective.")
        if "\\text{s.t.}" not in response_text:
            warnings.append("Primal response should include the constraint label s.t.")
        if "q_b" not in response_text and "q_{" not in response_text:
            warnings.append("Primal response should include primal decision variables.")

    if context.request_type == "theorem_proof":
        if context.applicable is True:
            lowered = response_text.lower()
            has_proof_marker = "\\begin{proof}" in response_text or "**proof.**" in lowered or "\\textbf{proof.}" in lowered
            if not has_proof_marker:
                warnings.append("Proof response should include a clear proof label.")
            if "$$" not in response_text and "\\[" not in response_text:
                warnings.append("Proof response should expose at least one notebook-friendly display-math block.")
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
                    warnings.append("Theorem 1 response should explicitly state strong duality.")
                if "z_P^*=z_D^*" not in compact:
                    warnings.append("Theorem 1 response should explicitly include z_P^* = z_D^*.")
                if "**primal problem.**" not in lowered or "**dual problem.**" not in lowered:
                    warnings.append("Theorem 1 response should include clean primal and dual problem headings.")
                if "\\mathcal{L}" not in response_text and "lagrangian" not in lowered:
                    warnings.append("Theorem 1 response should explicitly define the Lagrangian.")
                if aligned_count < 3:
                    warnings.append("Theorem 1 response should be equation-driven and include aligned primal, dual, and derivation blocks.")
                if "primal has an optimal solution" in lowered or "primal optimum exists" in lowered:
                    warnings.append("Theorem 1 response drifted to primal-optimum existence instead of strong duality.")
                if any(phrase in lowered for phrase in forbidden_phrases):
                    warnings.append("Theorem 1 response contains placeholder phrasing instead of explicit mathematical programming content.")
                if re.search(r"(q_b\s+q_b\s+q_b)|(\(p\)\(p\))", lowered):
                    warnings.append("Theorem 1 response contains duplicated mathematical fragments.")
        else:
            lowered = response_text.lower()
            if "cannot certify" not in lowered and "out of scope" not in lowered:
                warnings.append("Non-applicable proof response did not clearly explain the failure mode.")

    if context.request_type == "general_math_explanation":
        lowered = response_text.lower()
        plan = context.semantic_plan
        topics = set(plan.get("math_topics", []))
        response_contract = plan.get("response_contract", {})

        if response_contract.get("avoid_full_dual_formulation"):
            if "the dual problem is formulated as follows:" in lowered:
                warnings.append("Explanation response drifted into a full dual formulation.")

        if "strong_duality" in topics and "strong duality" not in lowered:
            warnings.append("Strong-duality explanation should explicitly mention strong duality.")

        if "complementary_slackness" in topics and "complementary slackness" not in lowered:
            warnings.append("Complementary-slackness explanation should explicitly mention complementary slackness.")

        if "node_product_prices" in topics and "\\pi" not in response_text and "price" not in lowered:
            warnings.append("Node-price explanation should mention the node-product price multipliers.")

        if response_contract.get("include_economic_interpretation"):
            economic_tokens = ("economic", "price", "scarcity", "shadow", "marginal", "incentive")
            if not any(token in lowered for token in economic_tokens):
                warnings.append("Interpretive response should include an economic explanation.")

        expected_layers = (
            ("intuition", ("intuition", "intuitively")),
            ("mathematical meaning", ("mathematical", "equation", "constraint", "dual", "primal")),
            ("economic interpretation", ("economic", "price", "scarcity", "shadow", "coordinated")),
        )
        for layer_name, tokens in expected_layers:
            if not any(token in lowered for token in tokens):
                warnings.append(f"Explanation response should include a {layer_name} layer.")

    fatal = warnings if not response_is_structurally_usable(context, response_text) else []
    remaining_warnings = [] if fatal else warnings
    return _finalize_validation(remaining_warnings, fatal)
