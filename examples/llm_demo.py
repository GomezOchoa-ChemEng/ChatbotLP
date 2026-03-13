"""Demonstration scripts for LLM integration layer.

This script shows three different ways to use the LLM integration layer:
1. Using the default rule-based system (no LLM setup needed)
2. Using mock LLM implementations (for testing)
3. Switching between providers at runtime

Run this script with: python examples/llm_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_adapter import (
    MockLLMProvider,
    RuleBasedProvider,
    LLMProviderRegistry,
)
from src.chatbot_engine import IntentRouter
from src.parser import parse_supply_chain_text
from src.response_generator import generate_response


def demo_1_default_rule_based():
    """Demonstration 1: Default rule-based system (no LLM layer)."""
    print("\n" + "=" * 70)
    print("DEMO 1: Default Rule-Based System")
    print("=" * 70)
    print(
        "\nThe default chatbot works as before without any LLM configuration."
        "\nNo need to use the LLM layer at all if you don't want to."
    )

    # The IntentRouter and parser work exactly as before
    intent_router = IntentRouter()
    user_messages = [
        "Add a supplier at node N1",
        "Check the problem for issues",
        "Solve the model",
        "What about Case A?",
    ]

    for user_text in user_messages:
        intent = intent_router.detect_intent(user_text)
        print(f"\n  User: '{user_text}'")
        print(f"  Intent (rule-based): {intent}")


def demo_2_mock_provider():
    """Demonstration 2: Using mock LLM provider for testing."""
    print("\n" + "=" * 70)
    print("DEMO 2: Mock LLM Provider (For Testing)")
    print("=" * 70)
    print(
        "\nMock provider returns placeholder results without calling any API."
        "\nUseful for testing the integration without external dependencies."
    )

    # Set up the registry with mock provider
    registry = LLMProviderRegistry.get_instance()
    registry.set_provider(MockLLMProvider())

    # Now use implementations from the registry
    classifier = registry.get_intent_classifier()
    parser = registry.get_parser()
    generator = registry.get_explanation_generator()

    # All return placeholder values
    test_text = "Define node N1 and product P1"
    print(f"\n  User: '{test_text}'")

    intent = classifier.classify(test_text)
    print(f"  Intent (mock): {intent}")

    entities = parser.parse(test_text)
    print(f"  Parsed entities (mock): {len(entities['nodes'])} nodes detected")

    response = generator.generate("hint", {})
    print(f"  Response (mock): {response[:60]}...")


def demo_3_rule_based_via_registry():
    """Demonstration 3: Rule-based provider via registry (optional, explicit control)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Rule-Based Provider Via Registry")
    print("=" * 70)
    print(
        "\nOptionally use the LLM layer to manage rule-based implementations."
        "\nThis gives you the flexibility to switch providers if needed later."
    )

    # Create rule-based components
    intent_router = IntentRouter()

    # Wrap in a provider
    provider = RuleBasedProvider(
        intent_router=intent_router,
        parse_function=parse_supply_chain_text,
        generate_function=generate_response,
    )

    # Register the provider
    registry = LLMProviderRegistry.get_instance()
    registry.reset()  # Clear previous demo's mock provider
    registry.set_provider(provider)

    # Use implementations from the registry
    classifier = registry.get_intent_classifier()

    test_cases = [
        "Add node N1",
        "Validate the problem",
        "Show me Case B",
        "Explain the solution",
    ]

    print(f"\n  Testing classification via rule-based provider:")
    for user_text in test_cases:
        intent = classifier.classify(user_text)
        print(f"    '{user_text}' → {intent}")


def demo_4_provider_switching():
    """Demonstration 4: Switching providers at runtime."""
    print("\n" + "=" * 70)
    print("DEMO 4: Provider Switching at Runtime")
    print("=" * 70)
    print(
        "\nThe main advantage of the provider pattern:"
        "\nSwitch between implementations without changing your code."
    )

    registry = LLMProviderRegistry.get_instance()
    test_text = "Solve the optimization model"

    # Start with mock provider
    print(f"\n  Test: '{test_text}'")

    registry.set_provider(MockLLMProvider())
    classifier1 = registry.get_intent_classifier()
    result1 = classifier1.classify(test_text)
    print(f"    With MockProvider: {result1}")

    # Switch to rule-based provider
    intent_router = IntentRouter()
    provider = RuleBasedProvider(
        intent_router=intent_router,
        parse_function=parse_supply_chain_text,
        generate_function=generate_response,
    )
    registry.set_provider(provider)
    classifier2 = registry.get_intent_classifier()
    result2 = classifier2.classify(test_text)
    print(f"    With RuleBasedProvider: {result2}")

    print("\n  Note: Same code, different implementations!")


def demo_5_real_llm_template():
    """Demonstration 5: Template for real LLM implementation."""
    print("\n" + "=" * 70)
    print("DEMO 5: Real LLM Integration (Code Template)")
    print("=" * 70)
    print(
        "\nWhen you're ready to use a real LLM (GPT-4, Claude, etc.),"
        "\nfollow this template to implement your provider:"
    )

    template = '''
from src.llm_interfaces import IntentClassifier, LLMProvider

class MyLLMIntentClassifier(IntentClassifier):
    """Your custom LLM-based intent classifier."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        # Initialize your LLM client here
    
    def classify(self, text: str) -> str:
        """Call your LLM API to classify intent."""
        # 1. Validate input
        if not text or not text.strip():
            raise ValueError("Intent classifier received empty text")
        
        # 2. Call your LLM
        intent = self.call_llm_for_intent(text)
        
        # 3. Validate and return
        valid_intents = [
            "problem_formulation", "validation", "solve",
            "theorem_check", "scenario", "explanation"
        ]
        return intent if intent in valid_intents else "problem_formulation"

class MyLLMProvider(LLMProvider):
    """Your custom provider wrapping all three components."""
    
    def get_intent_classifier(self):
        return MyLLMIntentClassifier(self.api_key)
    
    # ... similarly for parser and generator

# Usage:
from src.llm_adapter import LLMProviderRegistry

provider = MyLLMProvider(api_key="your-api-key-here")
registry = LLMProviderRegistry.get_instance()
registry.set_provider(provider)

# Now all intent classification uses your LLM!
classifier = registry.get_intent_classifier()
intent = classifier.classify("User message here")
    '''
    print(template)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("LLM INTEGRATION LAYER DEMONSTRATIONS")
    print("=" * 70)

    demo_1_default_rule_based()
    demo_2_mock_provider()
    demo_3_rule_based_via_registry()
    demo_4_provider_switching()
    # demonstrate using use_llm flag directly
    print("\n-- Demo: Using use_llm flag with a provider registered --")
    from src.llm_adapter import RuleBasedProvider
    from unittest.mock import Mock
    reg = LLMProviderRegistry.get_instance()
    reg.reset()
    # register simple provider that tags intents
    provider = RuleBasedProvider(
        intent_router=Mock(detect_intent=Mock(return_value="problem_formulation")),
        parse_function=lambda t: {"nodes":[{"id":"LMML","name":"LMML"}],"products":[],"suppliers":[],"consumers":[],"transport_links":[],"technologies":[],"bids":[]},
        generate_function=lambda mode, ctx: "LLM explanation",
    )
    reg.set_provider(provider)
    state = ProblemState()
    res = run_chatbot_session(state, "anything", use_llm=True)
    print("  After LLM parse, nodes:", [n.id for n in res['state'].nodes])
    print("  After LLM response: ", res['response'])
    demo_5_real_llm_template()

    print("\n" + "=" * 70)
    print("Demonstrations Complete")
    print("=" * 70)
    print(
        "\nKey Takeaways:"
        "\n1. Default: Use rule-based system (no LLM layer needed)"
        "\n2. Testing: Use MockLLMProvider (placeholder implementations)"
        "\n3. Real LLM: Implement your provider and register it"
        "\n4. Switching: Change providers anytime without code changes"
        "\n5. Modular: Each component (intent/parser/generator) is independent"
    )
    print("\nFor detailed documentation, see: docs/llm_integration_guide.md\n")


if __name__ == "__main__":
    main()
