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

from typing import Dict, Any, Optional
from .llm_interfaces import (
    IntentClassifier,
    SupplyChainParser,
    ExplanationGenerator,
    LLMProvider,
)


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
