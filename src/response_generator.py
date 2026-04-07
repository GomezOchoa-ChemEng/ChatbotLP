"""Response generator for user-facing explanations in the supply chain chatbot.

This module generates clear, classroom-friendly explanations based on structured
outputs from the problem state, validator, solver, theorem checker, and scenario
engine. It supports three progressive disclosure modes:

- Hint Mode: Provides subtle guidance without revealing solutions.
- Guided Mode: Offers step-by-step assistance with partial results.
- Full Solution Mode: Delivers complete answers and detailed explanations.

The design is modular to allow easy integration of an LLM for more natural
language generation in the future.
"""

import logging
from typing import Dict, Any

from .proof_validator import GROUNDING_WARNING


LOGGER = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses based on structured context data."""

    def generate_response(self, mode: str, context: Dict[str, Any]) -> str:
        """Generate a response string based on the specified mode and context."""
        if mode == "hint":
            return self._generate_hint(context)
        elif mode == "guided":
            return self._generate_guided(context)
        elif mode in {"full", "exploration"}:
            return self._generate_full(context)
        else:
            return "Invalid mode specified. Choose 'hint', 'guided', 'full', or 'exploration'."

    def _get_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context.get("validation_result", context.get("validation", {})) or {}

    def _get_solve_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        solve_result = context.get("solve_result")
        if isinstance(solve_result, dict):
            return solve_result
        return {}

    def _get_scenario_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context.get("scenario_result", context.get("scenario_results", {})) or {}

    def _get_scenario_summary(self, context: Dict[str, Any]) -> str:
        scenario_results = self._get_scenario_results(context)
        return str(scenario_results.get("summary", "") or "")

    def _generate_hint(self, context: Dict[str, Any]) -> str:
        scenario_summary = self._get_scenario_summary(context)
        if scenario_summary:
            return scenario_summary

        hints = []

        validation = self._get_validation(context)
        issues = validation.get("issues", [])
        missing = validation.get("missing_parameters", [])
        if issues or missing:
            hints.append("Consider reviewing your problem setup for missing or invalid parameters.")

        solve_result = self._get_solve_result(context)
        solve_status = solve_result.get("status")
        if solve_status == "solver_unavailable":
            hints.append("Think about solver availability for optimization.")
        elif solve_status == "infeasible":
            hints.append("Your constraints might be too restrictive.")

        theorem_checks = context.get("theorem_checks", [])
        failed_theorems = []
        for tc in theorem_checks:
            if isinstance(tc, dict):
                if not tc.get("applies", False):
                    failed_theorems.append(tc)
            else:
                if not getattr(tc, "applies", False):
                    failed_theorems.append(tc)

        if failed_theorems:
            hints.append("Reflect on the theoretical assumptions for your case.")

        scenario_results = self._get_scenario_results(context)
        if scenario_results:
            hints.append("What-if scenarios can help explore changes.")

        if not hints:
            hints.append("Your setup looks good so far. Try proceeding to validation or solving.")

        return "Hints: " + " ".join(hints)

    def _generate_guided(self, context: Dict[str, Any]) -> str:
        scenario_summary = self._get_scenario_summary(context)
        if scenario_summary:
            return scenario_summary

        response_parts = []

        problem_state = context.get("problem_state")
        if problem_state:
            response_parts.append(f"Problem: {problem_state.problem_title}")
            response_parts.append(
                f"Entities: {len(problem_state.nodes)} nodes, "
                f"{len(problem_state.products)} products, "
                f"{len(problem_state.suppliers)} suppliers, "
                f"{len(problem_state.consumers)} consumers."
            )

        validation = self._get_validation(context)
        issues = validation.get("issues", [])
        missing = validation.get("missing_parameters", [])
        if issues:
            response_parts.append("Issues found: " + "; ".join(issues))
        if missing:
            response_parts.append("Missing parameters: " + "; ".join(missing))

        if validation:
            if validation.get("solver_ready"):
                response_parts.append("The model is ready to solve.")
            else:
                response_parts.append("Address the issues before solving.")

        theorem_checks = context.get("theorem_checks", [])
        if theorem_checks:
            applicable = []
            not_applicable = []

            for tc in theorem_checks:
                if isinstance(tc, dict):
                    theorem_name = tc.get("theorem_name", "unknown theorem")
                    applies = tc.get("applies", False)
                else:
                    theorem_name = getattr(tc, "theorem_name", "unknown theorem")
                    applies = getattr(tc, "applies", False)

                if applies:
                    applicable.append(theorem_name)
                else:
                    not_applicable.append(theorem_name)

            if applicable:
                response_parts.append("Applicable theorems: " + ", ".join(applicable))
            if not_applicable:
                response_parts.append("Non-applicable theorems: " + ", ".join(not_applicable))

        solve_result = self._get_solve_result(context)
        if solve_result:
            solve_status = solve_result.get("status")
            objective_value = solve_result.get("objective_value")

            response_parts.append(f"Solver status: {solve_status}")
            if objective_value is not None:
                try:
                    response_parts.append(f"Objective value: {float(objective_value):.2f}")
                except (TypeError, ValueError):
                    response_parts.append(f"Objective value: {objective_value}")

            if solve_status == "solver_unavailable":
                response_parts.append("No solver available. Results not computed.")

        scenario_results = self._get_scenario_results(context)
        if scenario_results:
            diff = scenario_results.get("difference", {})
            delta = diff.get("objective_delta")
            if delta is not None:
                try:
                    response_parts.append(f"Scenario objective change: {float(delta):.2f}")
                except (TypeError, ValueError):
                    response_parts.append(f"Scenario objective change: {delta}")
            else:
                response_parts.append("Scenario run completed, but no objective comparison available.")

        return "Guided Response:\n" + "\n".join(response_parts)

    def _generate_full(self, context: Dict[str, Any]) -> str:
        scenario_summary = self._get_scenario_summary(context)
        if scenario_summary:
            return scenario_summary

        response_parts = []

        problem_state = context.get("problem_state")
        if problem_state:
            response_parts.append(f"Problem Title: {problem_state.problem_title}")
            response_parts.append("Nodes: " + ", ".join(n.id for n in problem_state.nodes))
            response_parts.append("Products: " + ", ".join(p.id for p in problem_state.products))
            response_parts.append("Suppliers: " + ", ".join(s.id for s in problem_state.suppliers))
            response_parts.append("Consumers: " + ", ".join(c.id for c in problem_state.consumers))
            response_parts.append("Bids: " + ", ".join(f"{b.id} ({b.price})" for b in problem_state.bids))

        validation = self._get_validation(context)
        response_parts.append("Validation Issues: " + "; ".join(validation.get("issues", [])))
        response_parts.append("Missing Parameters: " + "; ".join(validation.get("missing_parameters", [])))
        response_parts.append(f"Solver Ready: {validation.get('solver_ready', False)}")

        theorem_checks = context.get("theorem_checks", [])
        for tc in theorem_checks:
            if isinstance(tc, dict):
                theorem_name = tc.get("theorem_name", "unknown theorem")
                applies = tc.get("applies", False)
                explanation = tc.get("explanation", "")
            else:
                theorem_name = getattr(tc, "theorem_name", "unknown theorem")
                applies = getattr(tc, "applies", False)
                explanation = getattr(tc, "explanation", "")

            status_text = "Applies" if applies else "Does not apply"
            response_parts.append(f"Theorem '{theorem_name}': {status_text} - {explanation}")

        solve_result = self._get_solve_result(context)
        if solve_result:
            response_parts.append(f"Solver Status: {solve_result.get('status')}")
            response_parts.append(f"Message: {solve_result.get('message')}")
            response_parts.append(f"Objective Value: {solve_result.get('objective_value')}")

            solver_time = solve_result.get("solver_time")
            if solver_time is not None:
                try:
                    response_parts.append(f"Solver Time: {float(solver_time):.2f} seconds")
                except (TypeError, ValueError):
                    response_parts.append(f"Solver Time: {solver_time}")

            response_parts.append("Solution Variables:")
            solution = solve_result.get("solution", {})
            if isinstance(solution, dict) and solution:
                for var, val in solution.items():
                    response_parts.append(f"  {var}: {val}")
            else:
                response_parts.append("  No solution variables reported.")

        scenario_results = self._get_scenario_results(context)
        if scenario_results:
            base_res = scenario_results.get("base")
            scen_res = scenario_results.get("scenario")
            diff = scenario_results.get("difference", {})

            if isinstance(base_res, dict):
                response_parts.append(f"Base Objective: {base_res.get('objective_value')}")
            elif base_res is not None:
                response_parts.append(f"Base Objective: {getattr(base_res, 'objective_value', None)}")

            if isinstance(scen_res, dict):
                response_parts.append(f"Scenario Objective: {scen_res.get('objective_value')}")
            elif scen_res is not None:
                response_parts.append(f"Scenario Objective: {getattr(scen_res, 'objective_value', None)}")

            delta = diff.get("objective_delta")
            if delta is not None:
                try:
                    response_parts.append(f"Objective Delta: {float(delta):.2f}")
                except (TypeError, ValueError):
                    response_parts.append(f"Objective Delta: {delta}")

        return "Full Solution:\n" + "\n".join(response_parts)


def _infer_grounding_mode(context: Dict[str, Any]) -> str:
    response_mode = str(context.get("response_mode", "") or "")
    if "solver" in response_mode or context.get("type") == "scenario":
        return "solver"
    if "theorem" in response_mode or context.get("type") == "theorem_check":
        return "theorem"
    if "model" in response_mode or context.get("type") in {"problem_formulation", "solve"}:
        return "model"
    return "paper"


def _build_response_metadata(
    *,
    response_source: str,
    fallback_triggered: bool,
    grounding_warning_applied: bool,
    validation_warnings: list[str],
    mode_used: str,
    grounding_mode: str,
) -> Dict[str, Any]:
    return {
        "response_source": response_source,
        "fallback_triggered": fallback_triggered,
        "grounding_warning_applied": grounding_warning_applied,
        "validation_warnings": list(dict.fromkeys(validation_warnings)),
        "mode_used": mode_used,
        "grounding_mode": grounding_mode,
    }


def _format_exploration_response(
    llm_interpretation: str,
    reference_response: str,
    validation_warnings: list[str],
) -> str:
    sections = [
        "LLM Interpretation",
        llm_interpretation.strip(),
        "",
        "Model-grounded reference",
        reference_response.strip(),
    ]
    if validation_warnings:
        sections.extend(
            [
                "",
                "Grounding note",
                GROUNDING_WARNING,
            ]
        )
    return "\n".join(sections)


def generate_response(
    mode: str,
    context: Dict[str, Any],
    use_llm: bool = False,
    include_reference: bool = False,
) -> str:
    response, _ = generate_response_with_metadata(
        mode=mode,
        context=context,
        use_llm=use_llm,
        include_reference=include_reference,
    )
    return response


def generate_response_with_metadata(
    mode: str,
    context: Dict[str, Any],
    use_llm: bool = False,
    include_reference: bool = False,
) -> tuple[str, Dict[str, Any]]:
    """Convenience function to generate a response."""
    generator = ResponseGenerator()
    reference_response = generator.generate_response(
        "full" if mode == "exploration" else mode,
        context,
    )
    grounding_mode = _infer_grounding_mode(context)
    validation = context.get("validation_result", {}) or {}
    validation_warnings = list(validation.get("warnings", []))
    if use_llm:
        try:
            from .llm_adapter import LLMProviderRegistry

            provider = LLMProviderRegistry.get_instance()
            llm_gen = provider.get_explanation_generator()
            llm_mode = "hint" if mode == "hint" else "full" if mode in {"full", "exploration"} else "guided"
            llm_response = llm_gen.generate(llm_mode, context)
            if llm_response and llm_response.strip():
                LOGGER.debug("LLM output used for response type %s", context.get("type", "general"))
                if include_reference:
                    response_text = _format_exploration_response(
                        llm_interpretation=llm_response,
                        reference_response=reference_response,
                        validation_warnings=validation_warnings,
                    )
                else:
                    response_text = llm_response
                return response_text, _build_response_metadata(
                    response_source="llm",
                    fallback_triggered=False,
                    grounding_warning_applied=bool(validation_warnings),
                    validation_warnings=validation_warnings,
                    mode_used=mode,
                    grounding_mode=grounding_mode,
                )
            LOGGER.debug("Fallback triggered because LLM returned an empty response")
        except Exception as exc:
            LOGGER.debug("Fallback triggered because LLM generation failed: %s", exc)

    return reference_response, _build_response_metadata(
        response_source="deterministic",
        fallback_triggered=use_llm,
        grounding_warning_applied=False,
        validation_warnings=validation_warnings,
        mode_used=mode,
        grounding_mode=grounding_mode,
    )


__all__ = ["ResponseGenerator", "generate_response", "generate_response_with_metadata"]
