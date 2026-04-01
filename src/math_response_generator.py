"""Generation of bounded mathematical exposition from formal context."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .dual_generator import build_dual_scaffold
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

    @staticmethod
    def infer_render_mode(context: FormalMathContext) -> str:
        """Return the preferred notebook rendering mode for a formal-math response."""

        if context.request_type in {"dual", "theorem_proof"}:
            return "markdown_latex"
        return "markdown"

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
            llm_response = strip_full_latex_document(llm_response)
            output_issues = validate_generated_math_response(context, llm_response)
            if not output_issues:
                return llm_response

        deterministic_response = strip_full_latex_document(
            self._generate_without_llm(response_kind, context)
        )
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
                    "you are writing a mathematical programming proof, not an explanation",
                    "write in the style of a textbook or research note",
                    "do not invent symbols",
                    "do not claim results not grounded in context",
                    "if assumptions are missing, say so clearly",
                    "return a render-ready LaTeX fragment, not a standalone LaTeX document",
                    "do not include documentclass, usepackage, begin{document}, or end{document}",
                    "prefer notebook-friendly markdown plus LaTeX fragments for proof requests",
                    "prefer align-ready LaTeX for formulations",
                    "use display equations wherever possible",
                    "avoid descriptive placeholders; use explicit algebraic expressions",
                    "minimize prose; prefer equations",
                    "style should be concise operations research exposition",
                    "for duals, write an optimization model with objective, constraints, and sign restrictions",
                    "for theorem_1, treat the result as a curated strong-duality theorem, not a primal-optimum existence claim",
                    "for theorem_1, explicitly include both the primal problem and the dual problem in clean display blocks",
                    "for theorem_1, explicitly conclude strong duality and the equality z_P^* = z_D^*",
                    "for theorem_1, do not replace the theorem with only a claim that the primal has an optimal solution",
                    "for theorem_1, prefer a polished theorem statement followed by a clear Proof. label rather than theorem or proof environments",
                    "avoid duplicated inline math and do not repeat the same objective or constraint in both plain text and raw LaTeX on the same line",
                    "if the request is out of scope, say so plainly and do not improvise",
                    "do not expose internal metadata labels such as validated_linear_problem_state, assumptions_verified, ProblemState, or raw field names",
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

    def _format_objective_expression(self, terms: List[Dict[str, object]]) -> str:
        expression_terms = [
            {
                "coefficient": float(term["coefficient"]),
                "symbol": str(term["symbol"]),
            }
            for term in terms
        ]
        return self._format_linear_expression(expression_terms)

    def _constraint_relation(self, sense: str) -> str:
        return {"=": "=", "<=": "\\le", ">=": "\\ge"}.get(sense, sense)

    def _format_constraint_line(self, constraint: Dict[str, object]) -> str:
        lhs = self._format_linear_expression(constraint.get("lhs_terms", []))
        rhs = self._format_scalar(float(constraint.get("rhs", 0.0)))
        relation = self._constraint_relation(str(constraint.get("sense", "=")))
        return f"{lhs} {relation} {rhs}"

    def _missing_assumption_text(self, context: FormalMathContext) -> List[str]:
        theorem_metadata = get_theorem_metadata(context.theorem_id or "") or {}
        mathematical_assumptions = theorem_metadata.get("mathematical_assumptions", {})
        return [
            mathematical_assumptions.get(item, item.replace("_", " "))
            for item in context.assumptions_missing
        ]

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
                "**Dual Problem.**",
                "",
                "$$",
                "\\begin{aligned}",
                "\\min \\quad & " + objective + " \\\\",
                "\\text{s.t.} \\quad & " + " \\\n& ".join(stationarity_lines or ["0 \\ge 0"]) + " \\\\",
                "\\text{sign restrictions} \\quad & "
                + " \\\n& ".join(balance_lines + capacity_lines or ["\\text{none}"]),
                "\\end{aligned}",
                "$$",
                "",
                notes,
            ]
        )

    def _theorem_1_primal_block(self, context: FormalMathContext) -> List[str]:
        primal = context.primal_formulation or {}
        objective = self._format_objective_expression(primal.get("objective", {}).get("terms", []))
        constraints = primal.get("constraints", [])
        balance_lines = [
            self._format_constraint_line(constraint)
            for constraint in constraints
            if constraint.get("type") == "balance"
        ]
        upper_bound_lines = [
            self._format_constraint_line(constraint)
            for constraint in constraints
            if constraint.get("type") == "upper_bound"
        ]
        remaining_constraint_lines = balance_lines[1:] + upper_bound_lines
        return [
            "**Primal Problem.**",
            "$$",
            "\\begin{aligned}",
            "(P)\\qquad \\max \\quad & " + objective + " \\\\",
            "\\text{s.t.} \\quad & " + (balance_lines[0] if balance_lines else "0 = 0") + " \\\\",
            *(
                ["& " + " \\\\\n& ".join(remaining_constraint_lines) + " \\\\"]
                if remaining_constraint_lines
                else []
            ),
            "& q_b,\\ f_{ij},\\ x_k \\ge 0.",
            "\\end{aligned}",
            "$$",
        ]

    def _theorem_1_dual_block(self, dual: Dict[str, object], context: FormalMathContext) -> List[str]:
        objective = self._format_objective_expression(dual.get("objective_terms", []))
        stationarity_conditions = dual.get("stationarity_conditions", [])
        stationarity_lines = [
            (
                f"{self._format_linear_expression(condition.get('dual_expression_terms', []))} "
                f"\\ge {self._dual_condition_rhs(condition)}"
            )
            for condition in stationarity_conditions
        ]
        balance_lines = [
            f"{dual_var['symbol']} \\in \\mathbb{{R}}"
            for dual_var in context.dual_variables
            if dual_var.get("constraint_type") == "balance"
        ]
        capacity_lines = [
            f"{dual_var['symbol']} \\ge 0"
            for dual_var in context.dual_variables
            if dual_var.get("constraint_type") == "upper_bound"
        ]
        sign_lines = balance_lines + capacity_lines
        remaining_stationarity_lines = stationarity_lines[1:]
        remaining_sign_lines = sign_lines[1:]
        return [
            "**Dual Problem.**",
            "$$",
            "\\begin{aligned}",
            "(D)\\qquad \\min \\quad & " + objective + " \\\\",
            "\\text{s.t.} \\quad & " + (stationarity_lines[0] if stationarity_lines else "0 \\ge 0") + " \\\\",
            *(
                ["& " + " \\\\\n& ".join(remaining_stationarity_lines) + " \\\\"]
                if remaining_stationarity_lines
                else []
            ),
            "\\text{sign restrictions} \\quad & " + (sign_lines[0] if sign_lines else "0 \\in \\mathbb{R}") + " \\\\",
            *(
                ["& " + " \\\\\n& ".join(remaining_sign_lines)]
                if remaining_sign_lines
                else []
            ),
            "\\end{aligned}",
            "$$",
        ]

    def _theorem_1_lagrangian_block(self, context: FormalMathContext) -> List[str]:
        primal = context.primal_formulation or {}
        objective = self._format_objective_expression(primal.get("objective", {}).get("terms", []))
        balance_terms = [
            f"{constraint['dual_symbol']}\\left({self._format_linear_expression(constraint.get('lhs_terms', []))}\\right)"
            for constraint in context.constraints
            if constraint.get("type") == "balance"
        ]
        upper_bound_terms = [
            f"{constraint['dual_symbol']}\\left({self._format_scalar(float(constraint.get('rhs', 0.0)))} - "
            f"{self._format_linear_expression(constraint.get('lhs_terms', []))}\\right)"
            for constraint in context.constraints
            if constraint.get("type") == "upper_bound"
        ]
        lagrangian_terms = [objective] + balance_terms + upper_bound_terms
        lagrangian_terms = [term for term in lagrangian_terms if term]
        if not lagrangian_terms:
            lagrangian_terms = ["0"]
        remaining_terms = lagrangian_terms[1:]
        return [
            "$$",
            "\\begin{aligned}",
            "\\mathcal{L}(q,f,x;\\pi,\\mu,\\nu,\\tau)",
            "&= " + lagrangian_terms[0] + " \\\\",
            *(
                ["&\\quad " + " \\\\\n&\\quad ".join(f"+ {term}" for term in remaining_terms)]
                if remaining_terms
                else []
            ),
            "\\end{aligned}",
            "$$",
        ]

    def _theorem_1_stationarity_block(self, dual: Dict[str, object]) -> List[str]:
        stationarity_lines = []
        for condition in dual.get("stationarity_conditions", []):
            lhs = self._format_linear_expression(condition.get("dual_expression_terms", []))
            rhs = self._dual_condition_rhs(condition)
            stationarity_lines.append(
                f"{lhs} \\ge {rhs}, \\qquad {condition['primal_variable']}"
            )
        return [
            "$$",
            "\\begin{aligned}",
            "\\text{Coefficient conditions:} \\qquad & " + (stationarity_lines[0] if stationarity_lines else "0 \\ge 0") + " \\\\",
            *(
                ["& " + " \\\\\n& ".join(stationarity_lines[1:])]
                if stationarity_lines[1:]
                else []
            ),
            "\\end{aligned}",
            "$$",
        ]

    def _deterministic_theorem_proof(self, context: FormalMathContext) -> str:
        theorem_metadata = get_theorem_metadata(context.theorem_id or "")
        if theorem_metadata is None:
            return (
                f"Theorem request `{context.theorem_id}` is out of scope for the current exposition layer. "
                "Only theorem identifiers explicitly curated for Sampat et al. (2019) Sections 2.1-2.3 are supported."
            )

        missing_assumptions = self._missing_assumption_text(context)
        if context.applicable is not True or context.assumptions_missing:
            return (
                f"Theorem {context.theorem_id.split('_')[-1] if context.theorem_id else ''} cannot certify that the claimed conclusion holds for the current instance.\n\n"
                "Missing assumptions:\n- "
                + "\n- ".join(missing_assumptions or ["required assumptions are incomplete"])
                + "\n\nWithin the present deterministic scope, the appropriate conclusion is that the theorem has not yet been verified for this coordinated clearing model."
            )

        theorem_number = context.theorem_id.split("_")[-1] if context.theorem_id else ""
        statement = theorem_metadata.get("statement_template", "Supported theorem statement.")
        dual = context.dual_formulation or build_dual_scaffold(context.primal_formulation or {})
        context.dual_variables = dual.get("dual_variables", context.dual_variables)

        lines = [
            f"**Theorem {theorem_number}**",
            statement,
            "",
        ]
        lines.extend(self._theorem_1_primal_block(context))
        lines.append("")
        lines.extend(self._theorem_1_dual_block(dual, context))
        lines.extend(
            [
                "",
                "**Proof.**",
                "Define the Lagrangian:",
            ]
        )
        lines.extend(self._theorem_1_lagrangian_block(context))
        lines.extend(self._theorem_1_stationarity_block(dual))
        lines.extend(
            [
            "",
                "The displayed inequalities are exactly the dual feasibility conditions obtained from the Lagrangian coefficients.",
                "Because $(P)$ is feasible and has a finite optimal value, the strong duality theorem of linear programming applies.",
                "$$",
                "z_P^* = z_D^*.",
                "$$",
                "Complementary slackness implies that the active reduced-cost relations are supported by the node-product prices $\\pi_{np}$.",
            ]
        )
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


def strip_full_latex_document(text: str) -> str:
    """Normalize generated math into a notebook-friendly Markdown + LaTeX fragment."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:latex|tex|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\begin\{document\}", "", cleaned)
    cleaned = re.sub(r"\\end\{document\}", "", cleaned)
    cleaned = re.sub(r"\\begin\{proof\}", "**Proof.**", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\end\{proof\}", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\begin\{theorem\*?\}", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\end\{theorem\*?\}", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\\[(.*?)\\\]", r"$$\n\1\n$$", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\$\$\s*", "$$\n", cleaned)
    cleaned = re.sub(r"\s*\$\$", "\n$$", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
