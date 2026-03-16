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
                "Provide a short, domain-specific hint tied to the supplied coordinated supply chain model. "
                "Do not give a full solution. Do not invent a different scenario."
            ),
            "guided": (
                "Provide a guided explanation of the supplied coordinated supply chain model using the actual data in problem_state. "
                "Explain the roles of suppliers, consumers, bids, nodes, products, transport links, and technologies if present. "
                "Do NOT turn the response into a generic teaching exercise. "
                "Do NOT ask the user unrelated questions. "
                "Stay tied to the actual entity IDs and actual problem data."
            ),
            "full": (
                "Provide a full explanation of the supplied coordinated supply chain model using the actual data in problem_state. "
                "Explain the economic interpretation, coordinated market-clearing logic, and, if available, the meaning of the solution results. "
                "Stay tied to the actual entity IDs and actual problem data."
            ),
        }[mode]

        return f"""
You are an expert assistant in coordinated supply chain optimization, market-clearing
models, and operations research.

The chatbot you support is NOT a general educational tutor and NOT a classroom supply example generator.
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
- Do NOT invent unrelated toy examples.
- Do NOT introduce classroom, school, pencils, water bottles, inventory stories,
  warehouses, or retail examples unless the user explicitly asked for them.
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
4. If appropriate, the next analytical step

Be precise, domain-specific, and faithful to the supplied problem instance.
"""

def create_gemini_provider(
    model_name: str = "gemini-2.5-flash-lite",
) -> GeminiExplanationProvider:
    return GeminiExplanationProvider(model_name)