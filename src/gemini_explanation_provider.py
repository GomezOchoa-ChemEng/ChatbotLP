"""Gemini-based explanation provider for supply chain chatbot.

This module provides a standalone Gemini implementation for generating
educational explanations in the supply chain optimization chatbot.
It implements the ExplanationGenerator interface using Google's Gemini API.

Designed for Google Colab usage with free-tier friendly models.
Fails gracefully if API key is not available.
"""

from typing import Dict, Any

# Import the shared interface defined elsewhere in the project
from src.llm_interfaces import ExplanationGenerator


class GeminiExplanationProvider(ExplanationGenerator):
    """Gemini-based explanation generator using Google Gemini API.

    This implementation uses Google's Gemini model to generate natural language
    explanations for supply chain optimization problems. It supports the three
    disclosure modes (hint, guided, full) and is designed for classroom use.

    The constructor will raise an error if the GEMINI_API_KEY environment
    variable is missing; callers can catch this and fall back to the rule-based
    generator if desired. API call failures raise a RuntimeError so clients can
    perform similar graceful degradation.
    """

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
            raise ValueError(f"Invalid mode: {mode}. Must be 'hint', 'guided', or 'full'")

        # Build prompt based on mode and context
        prompt = self._build_prompt(mode, context)

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    def _build_prompt(self, mode: str, context: Dict[str, Any]) -> str:
        """Build a prompt for Gemini based on mode and context."""
        base_prompt = (
            "You are a helpful AI assistant for a classroom supply chain optimization chatbot. "
            "Generate a response in the following mode:\n\n"
        )

        if mode == "hint":
            prompt = (
                base_prompt +
                "HINT MODE: Provide minimal guidance without revealing solutions. "
                "Suggest next steps and ask questions to guide the student toward understanding. "
                "Keep it brief and encouraging.\n\n"
            )
        elif mode == "guided":
            prompt = (
                base_prompt +
                "GUIDED MODE: Provide step-by-step assistance with partial results. "
                "Show some intermediate calculations or logic, but leave key insights for the student to discover. "
                "Include prompts for the student to continue.\n\n"
            )
        else:  # full
            prompt = (
                base_prompt +
                "FULL MODE: Provide complete solutions with detailed explanations. "
                "Show all calculations, reasoning, and results clearly. "
                "Explain concepts pedagogically for learning.\n\n"
            )

        # Add context information
        prompt += "Context:\n"
        if "user_message" in context:
            prompt += f"User message: {context['user_message']}\n"
        if "intent" in context:
            prompt += f"Detected intent: {context['intent']}\n"
        if "problem_state" in context:
            prompt += f"Current problem state: {context['problem_state']}\n"
        if "solve_result" in context and context["solve_result"]:
            prompt += f"Solution results: {context['solve_result']}\n"
        if "validation_result" in context:
            prompt += f"Validation issues: {context['validation_result']}\n"
        if "theorem_checks" in context:
            prompt += f"Theorem checks: {context['theorem_checks']}\n"

        prompt += "\nGenerate a helpful, educational response for the student."
        return prompt


# Convenience function for easy usage
def create_gemini_provider(model_name: str = "gemini-2.5-flash-lite") -> GeminiExplanationProvider:
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