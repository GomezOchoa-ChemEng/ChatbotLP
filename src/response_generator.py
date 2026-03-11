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

from typing import Dict, Any, Optional


class ResponseGenerator:
    """Generates responses based on structured context data."""

    def generate_response(self, mode: str, context: Dict[str, Any]) -> str:
        """Generate a response string based on the specified mode and context.

        Args:
            mode: One of "hint", "guided", or "full".
            context: Dictionary containing structured outputs, e.g.:
                - "problem_state": ProblemState instance
                - "validation": dict from validate_state
                - "solve_result": SolveResult instance or None
                - "theorem_checks": list of TheoremCheck
                - "scenario_results": dict from run_scenario or None

        Returns:
            A formatted string suitable for user display.
        """
        if mode == "hint":
            return self._generate_hint(context)
        elif mode == "guided":
            return self._generate_guided(context)
        elif mode == "full":
            return self._generate_full(context)
        else:
            return "Invalid mode specified. Choose 'hint', 'guided', or 'full'."

    def _generate_hint(self, context: Dict[str, Any]) -> str:
        """Generate a hint-mode response with subtle guidance."""
        hints = []

        # Check validation issues
        validation = context.get("validation", {})
        issues = validation.get("issues", [])
        missing = validation.get("missing_parameters", [])
        if issues or missing:
            hints.append("Consider reviewing your problem setup for missing or invalid parameters.")

        # Check solver status
        solve_result = context.get("solve_result")
        if solve_result and solve_result.status == "solver_unavailable":
            hints.append("Think about solver availability for optimization.")
        elif solve_result and solve_result.status == "infeasible":
            hints.append("Your constraints might be too restrictive.")

        # Check theorems
        theorem_checks = context.get("theorem_checks", [])
        failed_theorems = [tc for tc in theorem_checks if not tc.applies]
        if failed_theorems:
            hints.append("Reflect on the theoretical assumptions for your case.")

        # Scenario hint
        scenario_results = context.get("scenario_results")
        if scenario_results:
            hints.append("What-if scenarios can help explore changes.")

        if not hints:
            hints.append("Your setup looks good so far. Try proceeding to validation or solving.")

        return "Hints: " + " ".join(hints)

    def _generate_guided(self, context: Dict[str, Any]) -> str:
        """Generate a guided-mode response with step-by-step assistance."""
        response_parts = []

        # Summarize problem state
        problem_state = context.get("problem_state")
        if problem_state:
            response_parts.append(f"Problem: {problem_state.problem_title}")
            response_parts.append(f"Entities: {len(problem_state.nodes)} nodes, {len(problem_state.products)} products, {len(problem_state.suppliers)} suppliers, {len(problem_state.consumers)} consumers.")

        # Validation feedback
        validation = context.get("validation", {})
        issues = validation.get("issues", [])
        missing = validation.get("missing_parameters", [])
        if issues:
            response_parts.append("Issues found: " + "; ".join(issues))
        if missing:
            response_parts.append("Missing parameters: " + "; ".join(missing))
        if validation.get("solver_ready"):
            response_parts.append("The model is ready to solve.")
        else:
            response_parts.append("Address the issues before solving.")

        # Theorem checks
        theorem_checks = context.get("theorem_checks", [])
        if theorem_checks:
            applicable = [tc.theorem_name for tc in theorem_checks if tc.applies]
            not_applicable = [tc.theorem_name for tc in theorem_checks if not tc.applies]
            if applicable:
                response_parts.append("Applicable theorems: " + ", ".join(applicable))
            if not_applicable:
                response_parts.append("Non-applicable theorems: " + ", ".join(not_applicable))

        # Solver results
        solve_result = context.get("solve_result")
        if solve_result:
            response_parts.append(f"Solver status: {solve_result.status}")
            if solve_result.objective_value is not None:
                response_parts.append(f"Objective value: {solve_result.objective_value:.2f}")
            if solve_result.status == "solver_unavailable":
                response_parts.append("No solver available. Results not computed.")

        # Scenario summary
        scenario_results = context.get("scenario_results")
        if scenario_results:
            diff = scenario_results.get("difference", {})
            delta = diff.get("objective_delta")
            if delta is not None:
                response_parts.append(f"Scenario objective change: {delta:.2f}")
            else:
                response_parts.append("Scenario run completed, but no objective comparison available.")

        return "Guided Response:\n" + "\n".join(response_parts)

    def _generate_full(self, context: Dict[str, Any]) -> str:
        """Generate a full-mode response with complete details."""
        response_parts = []

        # Full problem state
        problem_state = context.get("problem_state")
        if problem_state:
            response_parts.append(f"Problem Title: {problem_state.problem_title}")
            response_parts.append("Nodes: " + ", ".join(n.id for n in problem_state.nodes))
            response_parts.append("Products: " + ", ".join(p.id for p in problem_state.products))
            response_parts.append("Suppliers: " + ", ".join(s.id for s in problem_state.suppliers))
            response_parts.append("Consumers: " + ", ".join(c.id for c in problem_state.consumers))
            response_parts.append("Bids: " + ", ".join(f"{b.id} ({b.price})" for b in problem_state.bids))

        # Full validation
        validation = context.get("validation", {})
        response_parts.append("Validation Issues: " + "; ".join(validation.get("issues", [])))
        response_parts.append("Missing Parameters: " + "; ".join(validation.get("missing_parameters", [])))
        response_parts.append(f"Solver Ready: {validation.get('solver_ready', False)}")

        # Full theorem checks
        theorem_checks = context.get("theorem_checks", [])
        for tc in theorem_checks:
            response_parts.append(f"Theorem '{tc.theorem_name}': {'Applies' if tc.applies else 'Does not apply'} - {tc.explanation}")

        # Full solver results
        solve_result = context.get("solve_result")
        if solve_result:
            response_parts.append(f"Solver Status: {solve_result.status}")
            response_parts.append(f"Message: {solve_result.message}")
            response_parts.append(f"Objective Value: {solve_result.objective_value}")
            response_parts.append(f"Solver Time: {solve_result.solver_time:.2f} seconds")
            response_parts.append("Solution Variables:")
            for var, val in solve_result.solution.items():
                if isinstance(val, dict):
                    response_parts.append(f"  {var}: {val}")
                else:
                    response_parts.append(f"  {var}: {val}")

        # Full scenario results
        scenario_results = context.get("scenario_results")
        if scenario_results:
            base_res = scenario_results.get("base")
            scen_res = scenario_results.get("scenario")
            diff = scenario_results.get("difference", {})
            if base_res:
                response_parts.append(f"Base Objective: {base_res.objective_value}")
            if scen_res:
                response_parts.append(f"Scenario Objective: {scen_res.objective_value}")
            delta = diff.get("objective_delta")
            if delta is not None:
                response_parts.append(f"Objective Delta: {delta:.2f}")

        return "Full Solution:\n" + "\n".join(response_parts)


def generate_response(mode: str, context: Dict[str, Any]) -> str:
    """Convenience function to generate a response."""
    generator = ResponseGenerator()
    return generator.generate_response(mode, context)


__all__ = ["ResponseGenerator", "generate_response"]
