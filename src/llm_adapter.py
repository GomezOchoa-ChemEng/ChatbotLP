"""LLM adapter implementations and provider registry.

This module provides:
1. Mock implementations of the LLM interfaces (placeholder behavior)
2. A provider registry for managing LLM implementations
3. Adapters to integrate with the existing rule-based system

Mock implementations return placeholder results and do not connect to any
external service. They are designed to:
- Allow testing the integration without external dependencies
- Serve as templates for real LLM provider implementations
- Enable graceful fallback if a provider is unavailable

The design uses a registry pattern to allow runtime switching between
rule-based and LLM-based implementations.
"""

import json
import os
from typing import Callable, Dict, Any, Optional
from .llm_interfaces import (
    IntentClassifier,
    SupplyChainParser,
    ExplanationGenerator,
    LLMProvider,
)


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


def _safe_json(obj: Any) -> str:
    """Serialize context objects for prompt construction."""

    if obj is None:
        return "null"

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

    if hasattr(obj, "to_dict"):
        try:
            return json.dumps(obj.to_dict(), indent=2, default=str)
        except Exception:
            pass

    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


class MockIntentClassifier(IntentClassifier):
    """Mock intent classifier that returns a default intent.

    This implementation does not analyze the text at all and always returns
    "problem_formulation". It serves as a placeholder for real LLM-based
    intent classification.

    In production, a real implementation would:
    - Send the text to an LLM API
    - Parse the response into one of the six valid intents
    - Include confidence scores or fallback logic
    """

    def classify(self, text: str) -> str:
        """Always returns 'problem_formulation'."""
        if not text or not text.strip():
            raise ValueError("Intent classifier received empty text")
        return "problem_formulation"


class MockSupplyChainParser(SupplyChainParser):
    """Mock parser that returns empty entity lists.

    This implementation does not extract any entities from the text and always
    returns an empty result. It serves as a placeholder for rule-based or
    LLM-based parsing.

    In production, a real implementation would:
    - Use regex patterns (rule-based) or LLM-based extraction
    - Parse the text to identify nodes, products, suppliers, etc.
    - Return structured dictionaries with entity attributes
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """Always returns empty entity lists."""
        if not text or not text.strip():
            raise ValueError("Parser received empty text")

        return {
            "nodes": [],
            "products": [],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }


class MockExplanationGenerator(ExplanationGenerator):
    """Mock explanation generator that returns placeholder text.

    This implementation returns generic placeholder text without analyzing
    the context. It serves as a template for real explanation generation.

    In production, a real implementation would:
    - Analyze the problem state and analysis results
    - Generate mode-appropriate responses (hint/guided/full)
    - Use templates or LLM-based generation for natural language
    - Include relevant data and suggestions from the context
    """

    def generate(self, mode: str, context: Dict[str, Any]) -> str:
        """Returns a placeholder message."""
        if mode not in ("hint", "guided", "full"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'hint', 'guided', or 'full'")

        mode_descriptions = {
            "hint": "minimal guidance without revealing solutions",
            "guided": "step-by-step assistance with partial results",
            "full": "complete solution with detailed explanations",
        }

        return (
            f"[Mock LLM response in {mode} mode "
            f"({mode_descriptions[mode]}) - placeholder implementation]"
        )


class GeminiExplanationGenerator(ExplanationGenerator):
    """Gemini-backed explanation generator using the official Google GenAI SDK."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        client: Any = None,
    ):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._client = client

        if self._client is None:
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError(
                    "GEMINI_API_KEY not found. In Colab, set it using os.environ or userdata.get(...)."
                )
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "google-genai package is required for Gemini explanations. "
                    "Install with: pip install google-genai"
                ) from exc

            self._client = genai.Client()

    def generate(self, mode: str, context: Dict[str, Any]) -> str:
        """Generate a Gemini response from structured context."""

        if mode not in ("hint", "guided", "full"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'hint', 'guided', or 'full'")

        prompt = self._build_prompt(mode, context)
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

        response_text = getattr(response, "text", None)
        if not response_text or not str(response_text).strip():
            raise RuntimeError("Gemini API call returned an empty text response")
        return str(response_text).strip()

    def _build_prompt(self, mode: str, context: Dict[str, Any]) -> str:
        """Build a Gemini prompt from the current structured context."""

        context_type = context.get("type", "general")
        if context_type == "formal_math":
            return self._build_formal_math_prompt(mode, context)
        return self._build_general_prompt(mode, context)

    def _build_general_prompt(self, mode: str, context: Dict[str, Any]) -> str:
        user_message = context.get("user_message", "")
        intent = context.get("intent", "")
        problem_state = _safe_json(context.get("problem_state"))
        validation_result = _safe_json(context.get("validation_result"))
        solve_result = _safe_json(context.get("solve_result"))
        theorem_checks = _safe_json(context.get("theorem_checks"))
        scenario_result = _safe_json(context.get("scenario_result"))

        mode_instruction = {
            "hint": "Provide a short hint tied to the supplied instance. Do not give a full solution.",
            "guided": "Provide a guided explanation tied to the supplied instance and current structured state.",
            "full": "Provide a complete explanation tied to the supplied instance and current structured state.",
        }[mode]

        return f"""
You are an assistant for a deterministic coordinated supply chain optimization chatbot.

Stay grounded in the supplied structured context.
Do not fabricate data, entity IDs, theorem applicability, or solver results.
If information is missing, state that clearly.

MODE:
{mode_instruction}

USER MESSAGE:
{user_message}

INTENT:
{intent}

PROBLEM STATE:
{problem_state}

VALIDATION RESULT:
{validation_result}

SOLVE RESULT:
{solve_result}

THEOREM CHECKS:
{theorem_checks}

SCENARIO RESULT:
{scenario_result}
""".strip()

    def _build_formal_math_prompt(self, mode: str, context: Dict[str, Any]) -> str:
        formal_math_request = context.get("formal_math_request", "")
        prompt_constraints = context.get("prompt_constraints", [])
        formal_math_context = _safe_json(context.get("formal_math_context"))

        return f"""
You are writing a bounded mathematical exposition for ChatbotLP using the Sampat et al. (2019) Sections 2.1-2.3 support implemented in the repository.

The deterministic system remains the mathematical authority.
Use only the supplied notation and structured context.
Do not invent symbols, assumptions, theorem applicability, or model structure.
If assumptions are missing, say so plainly.
If the theorem or request is outside the supported scope, say so plainly.
When LaTeX is appropriate, write polished plain-text LaTeX suitable for direct display.
Return render-ready LaTeX fragments for notebooks, not standalone LaTeX documents.
Do not include documentclass, usepackage, begin{{document}}, end{{document}}, or theorem/proof environments.
For dual requests, write a clean optimization model with objective, constraints, and sign restrictions.
For theorem-proof requests, prefer a bold theorem heading, display-math blocks where helpful, and a clear Proof. label.

MODE:
{mode}

FORMAL MATH REQUEST:
{formal_math_request}

PROMPT CONSTRAINTS:
{_safe_json(prompt_constraints)}

FORMAL MATH CONTEXT:
{formal_math_context}
""".strip()


class RuleBasedIntentClassifierAdapter(IntentClassifier):
    """Adapter to use the existing rule-based IntentRouter with the LLM interface.

    This adapter wraps the IntentRouter from chatbot_engine so that it can be
    used as an IntentClassifier implementation. This allows the rule-based
    system to be registered in the LLM provider registry.
    """

    def __init__(self, intent_router):
        """Initialize with an IntentRouter instance.

        Args:
            intent_router: An IntentRouter instance from chatbot_engine.
        """
        self.intent_router = intent_router

    def classify(self, text: str) -> str:
        """Use the rule-based router to classify intent."""
        if not text or not text.strip():
            raise ValueError("Intent classifier received empty text")
        return self.intent_router.detect_intent(text)


class RuleBasedParserAdapter(SupplyChainParser):
    """Adapter to use the existing rule-based parser with the LLM interface.

    This adapter wraps parse_supply_chain_text from the parser module so that
    it can be used as a SupplyChainParser implementation.
    """

    def __init__(self, parse_function):
        """Initialize with the parse_supply_chain_text function.

        Args:
            parse_function: The parse_supply_chain_text function from parser.py.
        """
        self.parse_function = parse_function

    def parse(self, text: str) -> Dict[str, Any]:
        """Use the rule-based parser to extract entities."""
        if not text or not text.strip():
            raise ValueError("Parser received empty text")
        return self.parse_function(text)


class RuleBasedExplanationGeneratorAdapter(ExplanationGenerator):
    """Adapter to use the existing rule-based generator with the LLM interface.

    This adapter wraps generate_response from the response_generator module so
    that it can be used as an ExplanationGenerator implementation.
    """

    def __init__(self, generate_function):
        """Initialize with the generate_response function.

        Args:
            generate_function: The generate_response function from response_generator.py,
                             or a callable with signature (mode, context) -> str.
        """
        self.generate_function = generate_function

    def generate(self, mode: str, context: Dict[str, Any]) -> str:
        """Use the rule-based generator to create a response."""
        if mode not in ("hint", "guided", "full"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'hint', 'guided', or 'full'")
        return self.generate_function(mode, context)


class MockLLMProvider(LLMProvider):
    """Provider that returns mock implementations of all three interfaces.

    Used for testing and as a placeholder for real LLM providers.
    """

    def get_intent_classifier(self) -> IntentClassifier:
        """Return a MockIntentClassifier."""
        return MockIntentClassifier()

    def get_parser(self) -> SupplyChainParser:
        """Return a MockSupplyChainParser."""
        return MockSupplyChainParser()

    def get_explanation_generator(self) -> ExplanationGenerator:
        """Return a MockExplanationGenerator."""
        return MockExplanationGenerator()


class GeminiLLMProvider(LLMProvider):
    """Provider with Gemini-backed explanations and explicit classifier/parser choices."""

    def __init__(
        self,
        intent_router: Any = None,
        parse_function: Optional[Callable[[str], Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        client: Any = None,
    ):
        self.intent_router = intent_router
        self.parse_function = parse_function
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.client = client
        self._generator: Optional[GeminiExplanationGenerator] = None

    def get_intent_classifier(self) -> IntentClassifier:
        """Prefer the supplied rule-based router; otherwise use the mock classifier."""
        if self.intent_router is not None:
            return RuleBasedIntentClassifierAdapter(self.intent_router)
        return MockIntentClassifier()

    def get_parser(self) -> SupplyChainParser:
        """Prefer the supplied rule-based parser; otherwise use the mock parser."""
        if self.parse_function is not None:
            return RuleBasedParserAdapter(self.parse_function)
        return MockSupplyChainParser()

    def get_explanation_generator(self) -> ExplanationGenerator:
        """Return a lazily initialized Gemini explanation generator."""
        if self._generator is None:
            self._generator = GeminiExplanationGenerator(
                model_name=self.model_name,
                client=self.client,
            )
        return self._generator


class RuleBasedProvider(LLMProvider):
    """Provider that returns adapters wrapping the existing rule-based system.

    This is the default provider, which maintains backward compatibility with
    the current implementation while using the new LLM interface abstraction.
    """

    def __init__(
        self,
        intent_router=None,
        parse_function=None,
        generate_function=None,
    ):
        """Initialize with rule-based components.

        Args:
            intent_router: IntentRouter instance (from chatbot_engine).
            parse_function: parse_supply_chain_text function (from parser).
            generate_function: generate_response function or callable.
        """
        self.intent_router = intent_router
        self.parse_function = parse_function
        self.generate_function = generate_function

    def get_intent_classifier(self) -> IntentClassifier:
        """Return an adapter wrapping the rule-based IntentRouter."""
        if self.intent_router is None:
            raise RuntimeError(
                "RuleBasedProvider was not initialized with an intent_router"
            )
        return RuleBasedIntentClassifierAdapter(self.intent_router)

    def get_parser(self) -> SupplyChainParser:
        """Return an adapter wrapping the rule-based parser."""
        if self.parse_function is None:
            raise RuntimeError(
                "RuleBasedProvider was not initialized with a parse_function"
            )
        return RuleBasedParserAdapter(self.parse_function)

    def get_explanation_generator(self) -> ExplanationGenerator:
        """Return an adapter wrapping the rule-based generator."""
        if self.generate_function is None:
            raise RuntimeError(
                "RuleBasedProvider was not initialized with a generate_function"
            )
        return RuleBasedExplanationGeneratorAdapter(self.generate_function)


class LLMProviderRegistry:
    """Registry for managing LLM provider implementations.

    This singleton registry allows the chatbot to switch between providers
    (rule-based, mock, real LLM, etc.) at runtime without modifying core code.

    The default provider is RuleBasedProvider, which uses the existing
    deterministic parser, intent router, and response generator.
    """

    _instance: Optional["LLMProviderRegistry"] = None
    _provider: Optional[LLMProvider] = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LLMProviderRegistry":
        """Get the singleton registry instance."""
        return cls()

    def set_provider(self, provider: LLMProvider) -> None:
        """Register an LLM provider as the active implementation.

        Args:
            provider: An LLMProvider instance to use for classification,
                     parsing, and explanation generation.

        Example::

            from llm_adapter import MockLLMProvider, LLMProviderRegistry

            registry = LLMProviderRegistry.get_instance()
            registry.set_provider(MockLLMProvider())
        """
        self._provider = provider

    def get_provider(self) -> LLMProvider:
        """Retrieve the active LLM provider.

        Returns:
            The currently registered LLMProvider instance. If no provider
            has been set explicitly, returns a RuleBasedProvider.

        Raises:
            RuntimeError: If the provider has not been initialized with
                         required components.
        """
        if self._provider is None:
            # Return a default rule-based provider (will fail if components not set)
            return RuleBasedProvider()
        return self._provider

    def reset(self) -> None:
        """Reset the registry to its default state.

        Useful for testing to ensure no state leaks between test cases.
        """
        self._provider = None

    def get_intent_classifier(self) -> IntentClassifier:
        """Convenience method to get the intent classifier from the active provider."""
        return self.get_provider().get_intent_classifier()

    def get_parser(self) -> SupplyChainParser:
        """Convenience method to get the parser from the active provider."""
        return self.get_provider().get_parser()

    def get_explanation_generator(self) -> ExplanationGenerator:
        """Convenience method to get the explanation generator from the active provider."""
        return self.get_provider().get_explanation_generator()


def get_active_provider_debug_info() -> str:
    """Return a short debug string showing the active provider and model."""

    provider = LLMProviderRegistry.get_instance().get_provider()
    provider_name = provider.__class__.__name__
    model_name = getattr(provider, "model_name", None)
    if model_name:
        return f"Active provider: {provider_name}; active model: {model_name}"
    return f"Active provider: {provider_name}; active model: deterministic fallback / n/a"


def print_active_provider_debug_info() -> str:
    """Print and return the active provider debug string."""

    message = get_active_provider_debug_info()
    print(message)
    return message


def configure_gemini_provider(
    model_name: Optional[str] = None,
    client: Any = None,
) -> GeminiLLMProvider:
    """Register Gemini as the active provider with rule-based routing/parsing.

    This helper does not auto-enable Gemini. Call it explicitly when you want
    `use_llm=True` requests to use Gemini for explanation generation.
    """

    from .chatbot_engine import IntentRouter
    from .parser import parse_supply_chain_text

    provider = GeminiLLMProvider(
        intent_router=IntentRouter(),
        parse_function=parse_supply_chain_text,
        model_name=model_name,
        client=client,
    )
    registry = LLMProviderRegistry.get_instance()
    registry.set_provider(provider)
    return provider
