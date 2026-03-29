"""Generation of bounded mathematical exposition from formal context."""

from __future__ import annotations

from typing import Dict, List, Optional

from .domain.sampat2019 import SECTION23_CONCEPTS, get_theorem_metadata
from .proof_validator import (
    validate_formal_math_context,
    validate_generated_math_response,
)
from .schema import FormalMathContext


class MathResponseGenerator:
    """Generate theorem, proof, and dual responses from structured context."""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def generate_dual_latex(self, context: FormalMathContext) -> str:
        return self._generate("dual", context)

    def generate_theorem_proof_latex(self, context: FormalMathContext) -> str:
        return self._generate("theorem_proof", context)

    def generate_theorem_explanation(self, context: FormalMathContext) -> str:
        return self._generate("theorem_explanation", context)

    def generate_section23_explanation(self, context: FormalMathContext) -> str:
        return self._generate("section23_explanation", context)

    def generate(self, context: FormalMathContext) -> str:
        return self._generate(context.request_type, context)

    def _generate(self, response_kind: str, context: FormalMathContext) -> str:
        issues = validate_formal_math_context(context)
        if issues:
            return "Formal math request could not be completed:\n- " + "\n- ".join(issues)

        llm_response = self._generate_with_llm(response_kind, context)
        if llm_response:
            output_issues = validate_generated_math_response(context, llm_response)
            if not output_issues:
                return llm_response

        deterministic_response = self._generate_without_llm(response_kind, context)
        output_issues = validate_generated_math_response(context, deterministic_response)
        if output_issues:
            deterministic_response += "\n\nValidation notes:\n- " + "\n- ".join(output_issues)
        return deterministic_response

    def _generate_with_llm(
        self,
        response_kind: str,
        context: FormalMathContext,
    ) -> Optional[str]:
        if not self.use_llm:
            return None

        try:
            from .llm_adapter import LLMProviderRegistry

            provider = LLMProviderRegistry.get_instance()
            explanation_generator = provider.get_explanation_generator()
            prompt_context = {
                "type": "formal_math",
                "formal_math_request": response_kind,
                "prompt_constraints": [
                    "use only supplied notation",
                    "do not invent symbols",
                    "do not claim results not grounded in context",
                    "if assumptions are missing, say so clearly",
                    "prefer theorem/proof structure for proof requests",
                    "prefer align-ready LaTeX for formulations",
                    "style should be concise operations research exposition",
                    "for duals, write an optimization model with objective, constraints, and sign restrictions",
                    "for theorem_1, prefer a polished theorem statement followed by a proof environment",
                    "if the request is out of scope, say so plainly and do not improvise",
                ],
                "formal_math_context": context.dict(),
            }
            response = explanation_generator.generate("full", prompt_context)
            if response and response.strip():
                return response
        except Exception:
            return None
        return None

    def _generate_without_llm(
        self,
        response_kind: str,
        context: FormalMathContext,
    ) -> str:
        if response_kind == "dual":
            return self._deterministic_dual(context)
        if response_kind == "theorem_proof":
            return self._deterministic_theorem_proof(context)
        if response_kind == "theorem_explanation":
            return self._deterministic_theorem_explanation(context)
        if response_kind == "section23_explanation":
            return self._deterministic_section23_explanation(context)
        return self._deterministic_general_math_explanation(context)

    def _format_scalar(self, value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"

    def _format_linear_expression(self, terms: List[Dict[str, object]]) -> str:
        if not terms:
            return "0"

        rendered: List[str] = []
        for index, term in enumerate(terms):
            coefficient = float(term["coefficient"])
            symbol = str(term.get("symbol", term.get("dual_symbol", "0")))
            magnitude = abs(coefficient)
            coeff_text = "" if abs(magnitude - 1.0) < 1e-9 else f"{self._format_scalar(magnitude)} "
            piece = f"{coeff_text}{symbol}"

            if index == 0:
                rendered.append(piece if coefficient >= 0 else f"-{piece}")
            else:
                sign = "+" if coefficient >= 0 else "-"
                rendered.append(f" {sign} {piece}")
        return "".join(rendered)

    def _objective_coefficient_text(self, condition: Dict[str, object]) -> str:
        coefficient = float(condition.get("objective_coefficient", 0.0))
        return self._format_scalar(coefficient)

    def _dual_condition_rhs(self, condition: Dict[str, object]) -> str:
        return self._objective_coefficient_text(condition)

    def _dual_condition_label(self, condition: Dict[str, object]) -> str:
        variable_class = condition.get("variable_class")
        if variable_class == "supplier_bid":
            return f"{condition['primal_variable']} \\; (\\text{{supplier bid}})"
        if variable_class == "consumer_bid":
            return f"{condition['primal_variable']} \\; (\\text{{consumer bid}})"
        if variable_class == "transport_flow":
            return f"{condition['primal_variable']} \\; (\\text{{transport flow}})"
        if variable_class == "technology_activity":
            return f"{condition['primal_variable']} \\; (\\text{{technology activity}})"
        return str(condition["primal_variable"])

    def _deterministic_dual(self, context: FormalMathContext) -> str:
        dual = context.dual_formulation or {}
        objective_terms = dual.get("objective_terms", [])
        stationarity_conditions = dual.get("stationarity_conditions", [])
        objective = " + ".join(
            f"{self._format_scalar(float(term['coefficient']))} {term['symbol']}"
            for term in objective_terms
        ) or "0"

        stationarity_lines = []
        for condition in stationarity_conditions:
            lhs = self._format_linear_expression(condition.get("dual_expression_terms", []))
            rhs = self._dual_condition_rhs(condition)
            stationarity_lines.append(
                f"{lhs} \\ge {rhs}, && {self._dual_condition_label(condition)}"
            )

        balance_lines = [
            f"{dual_var['symbol']} \\in \\mathbb{{R}}, && {dual_var['constraint_name']}"
            for dual_var in context.dual_variables
            if dual_var.get("constraint_type") == "balance"
        ]
        capacity_lines = [
            f"{dual_var['symbol']} \\ge 0, && {dual_var['constraint_name']}"
            for dual_var in context.dual_variables
            if dual_var.get("constraint_type") == "upper_bound"
        ]

        notes = (
            "This dual is rendered from the deterministic primal scaffold currently supported in ChatbotLP. "
            "It keeps the Sampat-style price interpretation narrow: node-product balances carry free multipliers, "
            "and explicit capacity bounds carry nonnegative multipliers."
        )

        return "\n".join(
            [
                "\\[",
                "\\begin{aligned}",
                "\\min \\quad & " + objective + " \\\\",
                "\\text{s.t.} \\quad & " + " \\\n& ".join(stationarity_lines or ["0 \\ge 0"]) + " \\\\",
                "\\text{sign restrictions} \\quad & "
                + " \\\n& ".join(balance_lines + capacity_lines or ["\\text{none}"]),
                "\\end{aligned}",
                "\\]",
                "",
                notes,
            ]
        )

    def _deterministic_theorem_proof(self, context: FormalMathContext) -> str:
        theorem_metadata = get_theorem_metadata(context.theorem_id or "")
        if theorem_metadata is None:
            return (
                f"Theorem request `{context.theorem_id}` is out of scope for the current exposition layer. "
                "Only theorem identifiers explicitly curated for Sampat et al. (2019) Sections 2.1-2.3 are supported."
            )

        skeleton = theorem_metadata.get("proof_skeleton", [])
        if context.applicable is not True or context.assumptions_missing:
            return (
                f"Theorem proof request for `{context.theorem_id}` cannot certify that the theorem holds for the current ProblemState.\n\n"
                "Missing assumptions:\n- "
                + "\n- ".join(context.assumptions_missing)
                + (
                    "\n\nThe request remains within scope, but the deterministic checker has not verified the prerequisites needed for a theorem-style proof."
                    if theorem_metadata
                    else "\n\nThe requested theorem is outside the supported scope."
                )
            )

        theorem_number = context.theorem_id.split("_")[-1] if context.theorem_id else ""
        statement = theorem_metadata.get("statement_template", "Supported theorem statement.")
        verified_assumptions = ", ".join(context.assumptions_verified) or "none"
        benchmark_case = context.benchmark_case or "supported benchmark family"
        proof_style = theorem_metadata.get("proof_style", "proof")

        lines = [
            "\\[",
            "\\begin{aligned}",
            f"\\textbf{{Theorem {theorem_number}.}}\\;& {statement}",
            "\\end{aligned}",
            "\\]",
            "",
            f"\\textit{{{proof_style}, grounded in the verified structured context.}}",
            "",
            f"Verified assumptions: {verified_assumptions}.",
            f"Benchmark interpretation: {benchmark_case}.",
            "",
            "\\begin{itemize}",
            *[f"\\item {item}" for item in context.assumptions_verified],
            "\\end{itemize}",
            "",
            "\\begin{proof}",
            "The current `ProblemState` induces a linear coordinated-clearing model with nonnegative primal variables "
            "$q_b$, $f_{ij}$, and $x_k$, together with node-product balance equalities and explicit upper-bound constraints "
            "for supported bid and transport quantities.",
            "By construction, the validator and theorem checker have already confirmed that the instance is solver-ready within the "
            "implemented scope, so the primal model is a well-posed linear program.",
            "Associate a free multiplier $\\pi_{np}$ with each node-product balance constraint and nonnegative multipliers "
            "$\\mu_b$, $\\nu_b$, and $\\tau_{ij}$ with the supported supplier, consumer, and transport upper bounds, respectively.",
            "Forming the Lagrangian and collecting coefficients of each nonnegative primal variable yields the dual feasibility "
            "conditions appearing in the dual formulation rendered by this layer.",
            "Because the primal is a linear maximization problem with affine constraints and the required feasibility structure has been "
            "verified deterministically, the standard linear-programming duality argument applies on the supported domain.",
            "Therefore the coordinated-clearing formulation and its associated price system are consistent for the present instance, "
            "which is exactly the scoped conclusion asserted here for Theorem "
            + theorem_number
            + ".",
        ]
        if skeleton:
            lines.append("")
            lines.append("Proof outline used:")
            lines.extend(f"- {step}" for step in skeleton)
        lines.append("\\end{proof}")
        return "\n".join(lines)

    def _deterministic_theorem_explanation(self, context: FormalMathContext) -> str:
        theorem_metadata = get_theorem_metadata(context.theorem_id or "")
        if theorem_metadata is None and context.theorem_id:
            return (
                f"Theorem request `{context.theorem_id}` is out of scope. "
                "The current registry is intentionally limited to Sampat et al. (2019) Sections 2.1-2.3 and the supported theorem identifiers in that scope."
            )

        theorem_title = theorem_metadata.get("title", context.theorem_id or "theorem")
        status = "applies" if context.applicable else "does not yet apply"
        lines = [
            f"{theorem_title} {status} to the current ProblemState within the supported Sections 2.1-2.3 scope.",
        ]
        if context.assumptions_verified:
            lines.append("Verified assumptions:")
            lines.extend(f"- {item}" for item in context.assumptions_verified)
        if context.assumptions_missing:
            lines.append("Missing assumptions:")
            lines.extend(f"- {item}" for item in context.assumptions_missing)
        lines.append(
            "Interpretation: the theorem is evaluated deterministically from validation and theorem-check metadata, "
            "while any polished exposition is downstream of that authoritative check."
        )
        return "\n".join(lines)

    def _deterministic_section23_explanation(self, context: FormalMathContext) -> str:
        negative_bid_count = sum(
            1
            for variable in context.variables
            if variable.get("name", "").startswith("q_")
        )
        lines = [
            "Section 2.3 explanation grounded in the supported Sampat registry:",
            f"- {SECTION23_CONCEPTS['negative_bids']}",
            f"- {SECTION23_CONCEPTS['disposal']}",
            f"- {SECTION23_CONCEPTS['remediation']}",
            f"- {SECTION23_CONCEPTS['storage']}",
            f"- {SECTION23_CONCEPTS['vos']}",
            f"- {SECTION23_CONCEPTS['negative_prices']}",
            (
                "In the current state, negative-bid interpretation is relevant."
                if any("negative bid" in note.lower() for note in context.source_notes)
                else "In the current state, Section 2.3 can be explained conceptually even if no negative bid is currently present."
            ),
            f"The current formulation tracks {negative_bid_count} acceptance variable(s) in the primal scaffold.",
        ]
        return "\n".join(lines)

    def _deterministic_general_math_explanation(self, context: FormalMathContext) -> str:
        return (
            "The current request was routed to the formal math layer, but it is outside the narrow supported templates "
            "for duals, theorem_1 exposition, and Section 2.3 negative-bid interpretation."
        )


def generate_math_response(
    context: FormalMathContext,
    use_llm: bool = False,
) -> str:
    """Convenience wrapper for formal math generation."""

    generator = MathResponseGenerator(use_llm=use_llm)
    return generator.generate(context)
