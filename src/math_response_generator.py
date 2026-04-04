"""Generation of bounded mathematical exposition from formal context."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .dual_generator import build_dual_model, build_dual_scaffold
from .domain.sampat2019 import SECTION23_CONCEPTS, get_theorem_metadata
from .model_builder import build_model_from_state
from .proof_validator import (
    validate_formal_math_context,
    validate_generated_math_response,
)
from .schema import FormalMathContext, ProblemState
from .solver import solve_model


class MathResponseGenerator:
    """Generate theorem, proof, and dual responses from structured context."""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    @staticmethod
    def infer_render_mode(context: FormalMathContext) -> str:
        """Return the preferred notebook rendering mode for a formal-math response."""

        response_contract = context.semantic_plan.get("response_contract", {})
        if context.request_type in {"primal", "dual", "theorem_proof"} or response_contract.get("prefer_latex"):
            return "markdown_latex"
        return "markdown"

    def generate_primal_latex(self, context: FormalMathContext) -> str:
        return self._generate("primal", context)

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
        if output_issues and response_kind != "dual":
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
            prompt_constraints = self._build_prompt_constraints(response_kind, context)
            prompt_context = {
                "type": "formal_math",
                "formal_math_request": response_kind,
                "prompt_constraints": prompt_constraints,
                "formal_math_context": self._build_llm_context(context),
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
        if response_kind == "primal":
            return self._deterministic_primal(context)
        if response_kind == "dual":
            return self._deterministic_dual(context)
        if response_kind == "theorem_proof":
            return self._deterministic_theorem_proof(context)
        if response_kind == "theorem_explanation":
            return self._deterministic_theorem_explanation(context)
        if response_kind == "section23_explanation":
            return self._deterministic_section23_explanation(context)
        return self._deterministic_general_math_explanation(context)

    def _build_llm_context(self, context: FormalMathContext) -> Dict[str, object]:
        dual = context.dual_formulation or {}
        response_contract = context.semantic_plan.get("response_contract", {})
        return {
            "user_request": context.user_request,
            "request_type": context.request_type,
            "semantic_plan": context.semantic_plan,
            "target_section": context.target_section,
            "theorem_id": context.theorem_id,
            "applicable": context.applicable,
            "assumptions_verified": context.assumptions_verified,
            "assumptions_missing": context.assumptions_missing,
            "benchmark_case": context.benchmark_case,
            "notation_profile": context.notation_profile,
            "supporting_equations": context.supporting_equations,
            "source_notes": context.source_notes,
            "response_contract": response_contract,
            "primal_summary": {
                "objective_terms": (context.primal_formulation or {}).get("objective", {}).get("terms", []),
                "constraint_count": len(context.constraints),
                "variable_count": len(context.variables),
            },
            "dual_summary": {
                "objective_terms": dual.get("objective_terms", []),
                "stationarity_conditions": dual.get("stationarity_conditions", []),
                "dual_variables": context.dual_variables,
            },
            "profit_definitions": context.profit_definitions,
            "lagrangian_components": context.lagrangian_components,
        }

    def _build_prompt_constraints(
        self,
        response_kind: str,
        context: FormalMathContext,
    ) -> List[str]:
        plan = context.semantic_plan
        response_contract = plan.get("response_contract", {})
        constraints = [
            "use only supplied notation",
            "write in the style of a textbook or research note",
            "do not invent symbols",
            "do not claim results not grounded in context",
            "if assumptions are missing, say so clearly",
            "return a render-ready LaTeX fragment, not a standalone LaTeX document",
            "do not include documentclass, usepackage, begin{document}, or end{document}",
            "avoid duplicated inline math and do not repeat the same expression in prose and raw LaTeX on the same line",
            "if the request is out of scope, say so plainly and do not improvise",
            "do not expose internal metadata labels such as validated_linear_problem_state, assumptions_verified, ProblemState, or raw field names",
            f"follow the semantic plan primary goal: {plan.get('primary_goal', 'unknown')}",
            f"follow the task modes exactly: {', '.join(plan.get('task_modes', [])) or 'none'}",
            f"focus on these math topics: {', '.join(plan.get('math_topics', [])) or 'none'}",
        ]

        if response_contract.get("prefer_latex"):
            constraints.append("use display equations where they improve clarity")
        if response_contract.get("prefer_concise"):
            constraints.append("be concise and avoid unnecessary exposition")

        if response_kind == "primal":
            constraints.extend(
                [
                    "for primal-only requests, write the optimization model with objective, constraints, and nonnegativity conditions",
                    "include the exact sentence 'The primal problem is formulated as follows:' exactly once",
                    "do not include theorem or proof material unless explicitly requested",
                ]
            )
        elif response_kind == "dual":
            constraints.extend(
                [
                    "for duals, write an optimization model with objective, constraints, and sign restrictions",
                    "include the exact sentence 'The dual problem is formulated as follows:' exactly once",
                    "place the dual formulation in exactly two wrapped display-math blocks: the first must be a single aligned block with (D), the objective on its own line, s.t. on its own line, and one inequality per line; the second must contain only sign restrictions",
                    "do not include inline labels, constraint names, validation notes, or any prose before, between, or after the two display blocks beyond that exact sentence",
                ]
            )
        elif response_kind == "theorem_proof":
            constraints.extend(
                [
                    "for theorem_1, treat the result as a curated strong-duality theorem, not a primal-optimum existence claim",
                    "explicitly include both the primal problem and the dual problem in clean display blocks",
                    "explicitly conclude strong duality and the equality z_P^* = z_D^*",
                    "prefer a polished theorem statement followed by a clear Proof. label rather than theorem or proof environments",
                ]
            )
        else:
            constraints.extend(
                [
                    "answer the user's actual question directly",
                    "for explanation requests, do not restate a full optimization model unless the response contract explicitly requires it",
                    "separate formulation, proof, and interpretation when more than one is requested",
                    "for explanation requests, add interpretive value beyond the deterministic scaffold instead of merely rephrasing it",
                ]
            )
            if response_contract.get("avoid_full_dual_formulation"):
                constraints.append("do not output the full dual formulation")
            if response_contract.get("include_economic_interpretation"):
                constraints.append("include economic interpretation tied to the supplied constraints and dual variables")
            if "complementary_slackness" in plan.get("math_topics", []):
                constraints.append("explain complementary slackness as a verification and interpretation concept, not as a request to reformulate the dual")
            if "strong_duality" in plan.get("math_topics", []):
                constraints.append("explain strong duality directly and avoid drifting into a full dual derivation unless explicitly requested")

        return constraints

    def _rebuild_state(self, context: FormalMathContext) -> ProblemState:
        snapshot = context.problem_state_snapshot or {}
        if not snapshot:
            raise ValueError("ProblemState snapshot is unavailable for solver-backed verification.")
        return ProblemState.from_dict(snapshot)

    def _extract_primal_solution_map(
        self,
        context: FormalMathContext,
        solution: Dict[str, Any],
    ) -> Dict[str, float]:
        mapped: Dict[str, float] = {}
        q_values = solution.get("q", {})
        f_values = solution.get("f", {})
        x_values = solution.get("x", {})
        for variable in context.variables:
            symbol = str(variable["symbol"])
            variable_class = variable.get("variable_class")
            if variable_class in {"supplier_bid", "consumer_bid"}:
                mapped[symbol] = float(q_values.get(variable.get("bid_id"), 0.0) or 0.0)
            elif variable_class == "transport_flow":
                arc = tuple(variable.get("arc", ()))
                mapped[symbol] = float(f_values.get(str(arc), 0.0) or 0.0)
            elif variable_class == "technology_activity":
                mapped[symbol] = float(x_values.get(variable.get("technology_id"), 0.0) or 0.0)
            else:
                mapped[symbol] = 0.0
        return mapped

    def _extract_dual_solution_map(self, solution: Dict[str, Any]) -> Dict[str, float]:
        y_values = solution.get("y", {})
        return {str(symbol): float(value or 0.0) for symbol, value in y_values.items()}

    def _evaluate_linear_expression(
        self,
        terms: List[Dict[str, object]],
        values: Dict[str, float],
    ) -> float:
        return sum(float(term["coefficient"]) * values.get(str(term["symbol"]), 0.0) for term in terms)

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

    def _deterministic_dual(self, context: FormalMathContext) -> str:
        dual = context.dual_formulation or {}
        objective_terms = dual.get("objective_terms", [])
        stationarity_conditions = dual.get("stationarity_conditions", [])
        objective = self._format_objective_expression(objective_terms)

        stationarity_lines = []
        for condition in stationarity_conditions:
            lhs = self._format_linear_expression(condition.get("dual_expression_terms", []))
            rhs = self._dual_condition_rhs(condition)
            stationarity_lines.append(f"{lhs} \\ge {rhs}")

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
        sign_lines = balance_lines + capacity_lines or ["\\text{none}"]

        return "\n".join(
            [
                "The dual problem is formulated as follows:",
                "",
                "$$",
                "\\begin{aligned}",
                "(D)\\qquad \\min \\quad & " + objective + " \\\\",
                "\\text{s.t.} \\\\",
                "& " + " \\\\\n& ".join(stationarity_lines or ["0 \\ge 0"]),
                "\\end{aligned}",
                "$$",
                "$$",
                "\\begin{aligned}",
                "& " + " \\\\\n& ".join(sign_lines),
                "\\end{aligned}",
                "$$",
            ]
        )

    def _deterministic_primal(self, context: FormalMathContext) -> str:
        lines = ["The primal problem is formulated as follows:", ""]
        lines.extend(self._theorem_1_primal_block(context))
        return "\n".join(lines)

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

    def _dual_variable_meaning_lines(self, context: FormalMathContext) -> List[str]:
        lines: List[str] = []
        balance_symbols = [
            dual_var["symbol"]
            for dual_var in context.dual_variables
            if dual_var.get("constraint_type") == "balance"
        ]
        capacity_symbols = [
            dual_var["symbol"]
            for dual_var in context.dual_variables
            if dual_var.get("constraint_type") == "upper_bound"
        ]
        if balance_symbols:
            joined = ", ".join(balance_symbols)
            lines.append(
                f"The free balance multipliers {joined} act as node-product prices that support material balance at each node-product pair."
            )
        if capacity_symbols:
            joined = ", ".join(capacity_symbols)
            lines.append(
                f"The nonnegative capacity multipliers {joined} measure marginal scarcity for the corresponding bid, transport, or technology capacity limits."
            )
        if not lines:
            lines.append("No dual variable interpretation is available because the dual scaffold is incomplete.")
        return lines

    def _solver_backed_complementary_slackness(self, context: FormalMathContext) -> str:
        state = self._rebuild_state(context)
        primal_model = build_model_from_state(state)
        primal_result = solve_model(primal_model, fallback_solver="glpk")
        if not primal_result.success:
            return (
                "Complementary slackness could not be verified because the primal model did not solve successfully.\n\n"
                f"Solver status: {primal_result.status}. {primal_result.message}"
            )

        dual_representation = context.dual_formulation or build_dual_scaffold(context.primal_formulation or {})
        dual_model = build_dual_model(dual_representation)
        dual_result = solve_model(dual_model, solver_name="glpk", fallback_solver="glpk")
        if not dual_result.success:
            return (
                "Complementary slackness could not be verified because the dual model did not solve successfully.\n\n"
                f"Solver status: {dual_result.status}. {dual_result.message}"
            )

        primal_values = self._extract_primal_solution_map(context, primal_result.solution)
        dual_values = self._extract_dual_solution_map(dual_result.solution)
        tolerance = 1e-6

        upper_bound_checks: List[str] = []
        for constraint in context.constraints:
            if constraint.get("type") != "upper_bound":
                continue
            lhs = self._evaluate_linear_expression(constraint.get("lhs_terms", []), primal_values)
            rhs = float(constraint.get("rhs", 0.0))
            slack = rhs - lhs
            dual_symbol = str(constraint.get("dual_symbol", "0"))
            dual_value = dual_values.get(dual_symbol, 0.0)
            product = slack * dual_value
            status = "verified" if abs(product) <= tolerance else "not verified"
            upper_bound_checks.append(
                f"- ${dual_symbol}\\,({self._format_scalar(rhs)} - {self._format_linear_expression(constraint.get('lhs_terms', []))}) = {product:.3g}$ ({status})"
            )

        stationarity_checks: List[str] = []
        for condition in dual_representation.get("stationarity_conditions", []):
            variable_symbol = str(condition["primal_variable"])
            primal_value = primal_values.get(variable_symbol, 0.0)
            lhs = sum(
                float(term["coefficient"]) * dual_values.get(str(term["dual_symbol"]), 0.0)
                for term in condition.get("dual_expression_terms", [])
            )
            rhs = float(condition.get("objective_coefficient", 0.0))
            surplus = lhs - rhs
            product = primal_value * surplus
            status = "verified" if abs(product) <= tolerance else "not verified"
            stationarity_checks.append(
                f"- ${variable_symbol}\\,({self._format_linear_expression(condition.get('dual_expression_terms', []))} - {self._format_scalar(rhs)}) = {product:.3g}$ ({status})"
            )

        all_products = upper_bound_checks + stationarity_checks
        summary = (
            "All checked complementary-slackness products are within tolerance."
            if all("(verified)" in item for item in all_products)
            else "Some complementary-slackness products are not within tolerance."
        )

        return "\n".join(
            [
                "Complementary slackness was checked using solver-backed primal and dual solutions.",
                f"Primal status: {primal_result.status}; dual status: {dual_result.status}.",
                f"Primal objective: {primal_result.objective_value}; dual objective: {dual_result.objective_value}.",
                summary,
                "",
                "**Upper-Bound Conditions.**",
                *upper_bound_checks,
                "",
                "**Reduced-Cost Conditions.**",
                *stationarity_checks,
                "",
                "Economic interpretation: positive shadow values appear only on binding capacities, while positive primal activities require zero reduced-cost surplus in the associated dual inequality.",
            ]
        )

    def _deterministic_general_math_explanation(self, context: FormalMathContext) -> str:
        plan = context.semantic_plan
        response_contract = plan.get("response_contract", {})
        topics = set(plan.get("math_topics", []))
        lines: List[str] = []

        if response_contract.get("include_dual_formulation"):
            lines.append(self._deterministic_dual(context))
            lines.append("")

        if "strong_duality" in topics:
            lines.append("Strong duality is the key relationship between the primal and dual problems in the current Sampat-style linear model.")
            lines.append(
                "In linear-programming terms, the central conclusion is that the optimal primal and dual objective values agree whenever the scoped feasibility and attainment conditions hold:"
            )
            lines.append("$$")
            lines.append("z_P^* = z_D^*.")
            lines.append("$$")
            if context.theorem_id == "theorem_1" and context.applicable is True:
                lines.append(
                    "Within the supported Theorem 1 scope, the verified linear structure makes that equality a grounded conclusion rather than a generic slogan."
                )
            else:
                lines.append(
                    "In this layer, strong duality should be explained through verified assumptions rather than asserted abstractly beyond the checked scope."
                )
            if response_contract.get("include_proof_structure"):
                lines.append(
                    "Proof outline: write the Lagrangian, derive the dual feasibility conditions from the coefficient terms, and then invoke the strong-duality theorem for the validated linear program."
                )

        if "complementary_slackness" in topics:
            lines.append(self._solver_backed_complementary_slackness(context))

        if "node_product_prices" in topics:
            lines.append(
                "The node-product price multipliers $\\pi_{np}$ are the balance-constraint dual variables, so they measure the marginal value of one additional unit of product $p$ at node $n$."
            )
            lines.append(
                "They coordinate supplier acceptance, consumer acceptance, transport, and transformation decisions by making profitable activities satisfy the corresponding reduced-cost conditions."
            )

        if response_contract.get("include_economic_interpretation"):
            lines.extend(self._dual_variable_meaning_lines(context))

        if "negative_bids" in topics:
            lines.append(self._deterministic_section23_explanation(context))

        if not lines:
            return (
                "The current request was routed to the formal math layer, but it remains outside the currently grounded explanation patterns "
                "for duality, theorem_1 exposition, and Section 2.3 interpretation."
            )

        return "\n\n".join(line for line in lines if line is not None)


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
    cleaned = cleaned.replace("\u2265", "\\ge").replace("\u2264", "\\le")
    cleaned = cleaned.replace("\u2208", "\\in").replace("\u211d", "\\mathbb{R}")
    cleaned = re.sub(r"\$\$\s*", "$$\n", cleaned)
    cleaned = re.sub(r"\s*\$\$", "\n$$", cleaned)
    cleaned = re.sub(
        r"(\$\$\s*\\begin\{aligned\}.*?\\end\{aligned\}\s*\$\$)(?:\s*\1)+",
        r"\1",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
