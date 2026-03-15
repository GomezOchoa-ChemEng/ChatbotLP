"""Gemini-based explanation provider for the coordinated supply chain chatbot.

This module provides a standalone Gemini implementation for generating
domain-grounded explanations for the coordinated supply chain optimization
chatbot. It implements the ExplanationGenerator interface using Google's
Gemini API.

The explanations are explicitly grounded in the coordinated supply chain
framework of Sampat et al. (2019) and its supporting information.
Designed for Google Colab usage with free-tier friendly models.
"""

from typing import Dict, Any
import json

from src.llm_interfaces import ExplanationGenerator


class GeminiExplanationProvider(ExplanationGenerator):
    """Gemini-based explanation generator using Google Gemini API."""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """Initialize the Gemini generator.

        Args:
            model_name: The Gemini model to use. Defaults to gemini-2.5-flash-lite.

        Raises:
            ImportError: If google-generativeai is not installed.
            ValueError: If GEMINI_API_KEY environment variable is not set.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Gemini explanations. "
                "Install with: pip install google-generativeai"
            )

        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable must be set to use Gemini explanations"
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def generate(self, mode: str, context: Dict[str, Any]) -> str:
        """Generate an explanation using Gemini.

        Args:
            mode: Response disclosure level ("hint", "guided", or "full").
            context: Problem state and analysis results.

        Returns:
            A natural language explanation appropriate for the mode.

        Raises:
            ValueError: If mode is invalid.
            RuntimeError: If Gemini API call fails.
        """
        if mode not in ("hint", "guided", "full"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'hint', 'guided', or 'full'"
            )

        prompt = self._build_prompt(mode, context)

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    def _safe_json(self, obj: Any) -> str:
        """Serialize objects safely for prompt injection."""
        if obj is None:
            return "None"

        if hasattr(obj, "to_dict"):
            try:
                return json.dumps(obj.to_dict(), indent=2, default=str)
            except Exception:
                pass

        if hasattr(obj, "model_dump"):
            try:
                return json.dumps(obj.model_dump(), indent=2, default=str)
            except Exception:
                pass

        if hasattr(obj, "dict"):
            try:
                return json.dumps(obj.dict(), indent=2, default=str)
            except Exception:
                pass

        try:
            return json.dumps(obj, indent=2, default=str)
        except Exception:
            return str(obj)

    def _build_prompt(self, mode: str, context: Dict[str, Any]) -> str:
        """Build a domain-grounded prompt for Gemini."""
        user_message = context.get("user_message", "")
        intent = context.get("intent", "")
        problem_state = context.get("problem_state", None)
        solve_result = context.get("solve_result", None)
        validation_result = context.get("validation_result", None)
        theorem_checks = context.get("theorem_checks", None)
        scenario_result = context.get("scenario_result", None)

        problem_state_text = self._safe_json(problem_state)
        solve_result_text = self._safe_json(solve_result)
        validation_result_text = self._safe_json(validation_result)
        theorem_checks_text = self._safe_json(theorem_checks)
        scenario_result_text = self._safe_json(scenario_result)

        mode_instruction = {
            "hint": (
                "Provide a short, domain-specific hint. Do not give the full solution. "
                "Use the actual model data and coordinated supply chain terminology."
            ),
            "guided": (
                "Provide a guided explanation of the actual model instance. "
                "Explain the roles of suppliers, consumers, bids, nodes, products, "
                "transport links, and technologies if present. Stay tied to the "
                "provided problem data."
            ),
            "full": (
                "Provide a full domain-specific explanation of the actual model instance. "
                "Explain the economic interpretation, coordinated market-clearing logic, "
                "and, if available, the meaning of the solution results."
            ),
        }[mode]

        return f"""
You are an expert assistant in coordinated supply chain optimization and market-clearing
models for operations research and chemical engineering applications.

The chatbot you support is NOT a general supply chain tutor.
It is specifically intended to explain and analyze coordinated supply chain problems
using the framework introduced in:

- Sampat et al. (2019), coordinated supply chain management using market-clearing optimization
- The supporting information to that work, including benchmark cases:
  - Case A: no transformation
  - Case B: negative bids
  - Case C: transformation technologies

You must explain the ACTUAL problem instance provided below using the terminology
and concepts of the Sampat framework.

STRICT RULES:
- Do NOT invent unrelated toy examples such as pencils, water bottles, school supplies,
  warehouses, generic retail inventory stories, or classroom analogies unless the user
  explicitly provided them.
- Do NOT replace the supplied model with a simpler example.
- Do NOT fabricate missing parameters.
- If information is missing, say so explicitly.
- Use the actual entities and data from the provided problem state.
- Use domain-appropriate terms such as:
  coordinated supply chain, bids, suppliers, consumers, transport links,
  technologies, products, nodes, capacities, economic surplus, market clearing,
  negative bids, transformation yields.
- If the model contains bids, explain their economic role.
- If the model contains transport links, explain their network role.
- If the model contains technologies, explain their transformation role.
- If solve results are present, interpret them directly.
- If theorem checks are present, mention them only if relevant.
- Focus on the provided instance, not on generic teaching examples.

MODE INSTRUCTION:
{mode_instruction}

USER MESSAGE:
{user_message}

DETECTED INTENT:
{intent}

PROBLEM STATE (JSON):
{problem_state_text}

VALIDATION RESULT:
{validation_result_text}

SOLVE RESULT:
{solve_result_text}

THEOREM CHECKS:
{theorem_checks_text}

SCENARIO RESULT:
{scenario_result_text}

Write a response with this structure:

1. What this specific coordinated supply chain model represents
2. The key entities in this instance
3. The economic/optimization interpretation under the Sampat framework
4. If appropriate, the next step for the student or user

Be precise, domain-specific, and faithful to the supplied problem instance.
"""

def create_gemini_provider(
    model_name: str = "gemini-2.5-flash-lite",
) -> GeminiExplanationProvider:
    """Create a Gemini explanation provider.

    Args:
        model_name: The Gemini model to use.

    Returns:
        A configured GeminiExplanationProvider instance.

    Raises:
        ImportError: If google-generativeai is not available.
        ValueError: If GEMINI_API_KEY is not set.
    """
    return GeminiExplanationProvider(model_name)