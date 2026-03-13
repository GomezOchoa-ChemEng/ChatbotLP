"""Unit tests for LLM interfaces and adapters.

Tests cover:
- Mock implementations (placeholder behavior)
- Rule-based adapters wrapping existing functionality
- Provider registry pattern
- Error cases and validation
"""

import unittest
from unittest.mock import Mock, MagicMock

from src.llm_interfaces import (
    IntentClassifier,
    SupplyChainParser,
    ExplanationGenerator,
    LLMProvider,
)
from src.llm_adapter import (
    MockIntentClassifier,
    MockSupplyChainParser,
    MockExplanationGenerator,
    RuleBasedIntentClassifierAdapter,
    RuleBasedParserAdapter,
    RuleBasedExplanationGeneratorAdapter,
    MockLLMProvider,
    RuleBasedProvider,
    LLMProviderRegistry,
)


class TestMockIntentClassifier(unittest.TestCase):
    """Test MockIntentClassifier placeholder behavior."""

    def setUp(self):
        self.classifier = MockIntentClassifier()

    def test_classify_returns_problem_formulation(self):
        """Should always return problem_formulation."""
        result = self.classifier.classify("any text")
        self.assertEqual(result, "problem_formulation")

    def test_classify_with_different_inputs(self):
        """Should return problem_formulation regardless of input."""
        inputs = [
            "solve this problem",
            "check validation",
            "what is a theorem?",
            "xyz abc 123",
        ]
        for text in inputs:
            result = self.classifier.classify(text)
            self.assertEqual(result, "problem_formulation")

    def test_classify_raises_on_empty_text(self):
        """Should raise ValueError on empty text."""
        with self.assertRaises(ValueError):
            self.classifier.classify("")

    def test_classify_raises_on_none(self):
        """Should raise ValueError on None."""
        with self.assertRaises(ValueError):
            self.classifier.classify(None)


class TestMockSupplyChainParser(unittest.TestCase):
    """Test MockSupplyChainParser placeholder behavior."""

    def setUp(self):
        self.parser = MockSupplyChainParser()

    def test_parse_returns_empty_entities(self):
        """Should return dict with empty lists for all entity types."""
        result = self.parser.parse("any text")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["products"], [])
        self.assertEqual(result["suppliers"], [])
        self.assertEqual(result["consumers"], [])
        self.assertEqual(result["transport_links"], [])
        self.assertEqual(result["technologies"], [])
        self.assertEqual(result["bids"], [])

    def test_parse_returns_all_required_keys(self):
        """Should include all required entity type keys."""
        result = self.parser.parse("test")
        required_keys = {
            "nodes",
            "products",
            "suppliers",
            "consumers",
            "transport_links",
            "technologies",
            "bids",
        }
        self.assertEqual(set(result.keys()), required_keys)

    def test_parse_raises_on_empty_text(self):
        """Should raise ValueError on empty text."""
        with self.assertRaises(ValueError):
            self.parser.parse("")

    def test_parse_raises_on_none(self):
        """Should raise ValueError on None."""
        with self.assertRaises(ValueError):
            self.parser.parse(None)


class TestMockExplanationGenerator(unittest.TestCase):
    """Test MockExplanationGenerator placeholder behavior."""

    def setUp(self):
        self.generator = MockExplanationGenerator()

    def test_generate_valid_modes(self):
        """Should accept all valid modes."""
        for mode in ["hint", "guided", "full"]:
            result = self.generator.generate(mode, {})
            self.assertIn(mode, result)
            self.assertIn("Mock LLM", result)

    def test_generate_includes_mode_description(self):
        """Should include mode descriptions in result."""
        result_hint = self.generator.generate("hint", {})
        self.assertIn("minimal guidance", result_hint)

        result_guided = self.generator.generate("guided", {})
        self.assertIn("step-by-step", result_guided)

        result_full = self.generator.generate("full", {})
        self.assertIn("complete solution", result_full)

    def test_generate_raises_on_invalid_mode(self):
        """Should raise ValueError on invalid mode."""
        with self.assertRaises(ValueError):
            self.generator.generate("invalid_mode", {})

    def test_generate_context_not_used(self):
        """Mock generator should not actually use context."""
        # Should work with empty, partial, or rich context
        for context in [{}, {"some": "data"}, {"problem_state": None}]:
            result = self.generator.generate("hint", context)
            self.assertIsInstance(result, str)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for the convenience helper functions that expose LLM integration."""

    def setUp(self):
        self.registry = LLMProviderRegistry.get_instance()
        self.registry.reset()

    def tearDown(self):
        self.registry.reset()

    def test_generate_response_use_llm(self):
        # register a provider with a custom generator
        provider = RuleBasedProvider(
            intent_router=Mock(),
            parse_function=lambda t: {"nodes": [], "products": [], "suppliers": [], "consumers": [], "transport_links": [], "technologies": [], "bids": []},
            generate_function=lambda mode, ctx: "LLM-based explanation",
        )
        self.registry.set_provider(provider)
        from src.response_generator import generate_response

        resp = generate_response("hint", {}, use_llm=True)
        self.assertEqual(resp, "LLM-based explanation")

        # fallback when generator raises
        def broken(mode, ctx):
            raise RuntimeError("boom")
        provider = RuleBasedProvider(
            intent_router=Mock(),
            parse_function=lambda t: {},
            generate_function=broken,
        )
        self.registry.set_provider(provider)
        resp2 = generate_response("hint", {}, use_llm=True)
        # should not raise and should return default hint string
        self.assertIn("Hints:", resp2)

    def test_parse_supply_chain_text_use_llm(self):
        # provider with simple parse result
        provider = RuleBasedProvider(
            intent_router=Mock(),
            parse_function=lambda t: {"nodes": [{"id": "foo", "name": "foo"}], "products": [], "suppliers": [], "consumers": [], "transport_links": [], "technologies": [], "bids": []},
            generate_function=lambda mode, ctx: "",
        )
        self.registry.set_provider(provider)
        from src.parser import parse_supply_chain_text

        entities = parse_supply_chain_text("whatever", use_llm=True)
        self.assertEqual(entities["nodes"][0]["id"], "foo")

        self.registry.reset()


class TestRuleBasedIntentClassifierAdapter(unittest.TestCase):
    """Test adapter wrapping rule-based IntentRouter."""

    def test_adapter_delegates_to_router(self):
        """Should delegate classification to the wrapped router."""
        mock_router = Mock()
        mock_router.detect_intent.return_value = "solve"

        adapter = RuleBasedIntentClassifierAdapter(mock_router)
        result = adapter.classify("solve the model")

        self.assertEqual(result, "solve")
        mock_router.detect_intent.assert_called_once_with("solve the model")

    def test_adapter_preserves_router_intent(self):
        """Should preserve the exact intent returned by router."""
        mock_router = Mock()
        for intent in [
            "problem_formulation",
            "validation",
            "solve",
            "theorem_check",
            "scenario",
            "explanation",
        ]:
            mock_router.detect_intent.return_value = intent
            adapter = RuleBasedIntentClassifierAdapter(mock_router)
            result = adapter.classify("test")
            self.assertEqual(result, intent)

    def test_adapter_raises_on_empty_text(self):
        """Should raise ValueError on empty text."""
        adapter = RuleBasedIntentClassifierAdapter(Mock())
        with self.assertRaises(ValueError):
            adapter.classify("")


class TestRuleBasedParserAdapter(unittest.TestCase):
    """Test adapter wrapping rule-based parser."""

    def test_adapter_delegates_to_parse_function(self):
        """Should delegate parsing to the wrapped function."""
        mock_parse = Mock(
            return_value={
                "nodes": [{"id": "N1"}],
                "products": [],
                "suppliers": [],
                "consumers": [],
                "transport_links": [],
                "technologies": [],
                "bids": [],
            }
        )

        adapter = RuleBasedParserAdapter(mock_parse)
        result = adapter.parse("Node N1")

        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["nodes"][0]["id"], "N1")
        mock_parse.assert_called_once_with("Node N1")

    def test_adapter_preserves_parser_output(self):
        """Should preserve exact output from parser function."""
        expected_result = {
            "nodes": [{"id": "N1"}, {"id": "N2"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "location_node": "N1"}],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }
        mock_parse = Mock(return_value=expected_result)

        adapter = RuleBasedParserAdapter(mock_parse)
        result = adapter.parse("complex problem")

        self.assertEqual(result, expected_result)

    def test_adapter_raises_on_empty_text(self):
        """Should raise ValueError on empty text."""
        adapter = RuleBasedParserAdapter(Mock())
        with self.assertRaises(ValueError):
            adapter.parse("")


class TestRuleBasedExplanationGeneratorAdapter(unittest.TestCase):
    """Test adapter wrapping rule-based explanation generator."""

    def test_adapter_delegates_to_generate_function(self):
        """Should delegate generation to the wrapped function."""
        mock_generate = Mock(return_value="This is a hint.")
        adapter = RuleBasedExplanationGeneratorAdapter(mock_generate)

        result = adapter.generate("hint", {"key": "value"})

        self.assertEqual(result, "This is a hint.")
        mock_generate.assert_called_once_with("hint", {"key": "value"})

    def test_adapter_supports_all_modes(self):
        """Should accept all valid modes."""
        mock_generate = Mock(return_value="response")
        adapter = RuleBasedExplanationGeneratorAdapter(mock_generate)

        for mode in ["hint", "guided", "full"]:
            result = adapter.generate(mode, {})
            self.assertEqual(result, "response")

    def test_adapter_raises_on_invalid_mode(self):
        """Should raise ValueError on invalid mode."""
        mock_generate = Mock()
        adapter = RuleBasedExplanationGeneratorAdapter(mock_generate)

        with self.assertRaises(ValueError):
            adapter.generate("invalid", {})

        # Wrapped function should not be called if mode is invalid
        mock_generate.assert_not_called()

    def test_adapter_passes_context_unchanged(self):
        """Should pass context dict unchanged to wrapped function."""
        context = {
            "problem_state": Mock(),
            "validation_result": {"issues": []},
            "solve_result": Mock(),
        }
        mock_generate = Mock(return_value="response")
        adapter = RuleBasedExplanationGeneratorAdapter(mock_generate)

        adapter.generate("full", context)

        # Check that the exact context was passed
        call_args = mock_generate.call_args
        self.assertIs(call_args[0][1], context)


class TestMockLLMProvider(unittest.TestCase):
    """Test provider returning mock implementations."""

    def setUp(self):
        self.provider = MockLLMProvider()

    def test_provides_intent_classifier(self):
        """Should provide a MockIntentClassifier."""
        classifier = self.provider.get_intent_classifier()
        self.assertIsInstance(classifier, MockIntentClassifier)

    def test_provides_parser(self):
        """Should provide a MockSupplyChainParser."""
        parser = self.provider.get_parser()
        self.assertIsInstance(parser, MockSupplyChainParser)

    def test_provides_explanation_generator(self):
        """Should provide a MockExplanationGenerator."""
        generator = self.provider.get_explanation_generator()
        self.assertIsInstance(generator, MockExplanationGenerator)

    def test_provides_independent_instances(self):
        """Should provide new instances each call."""
        classifier1 = self.provider.get_intent_classifier()
        classifier2 = self.provider.get_intent_classifier()
        self.assertIsNot(classifier1, classifier2)


class TestRuleBasedProvider(unittest.TestCase):
    """Test provider wrapping rule-based implementations."""

    def test_provides_adapters_when_initialized(self):
        """Should provide adapters when initialized with components."""
        mock_router = Mock()
        mock_parse = Mock()
        mock_generate = Mock()

        provider = RuleBasedProvider(
            intent_router=mock_router,
            parse_function=mock_parse,
            generate_function=mock_generate,
        )

        self.assertIsInstance(
            provider.get_intent_classifier(), RuleBasedIntentClassifierAdapter
        )
        self.assertIsInstance(provider.get_parser(), RuleBasedParserAdapter)
        self.assertIsInstance(
            provider.get_explanation_generator(), RuleBasedExplanationGeneratorAdapter
        )

    def test_raises_when_intent_router_not_provided(self):
        """Should raise RuntimeError if intent_router not initialized."""
        provider = RuleBasedProvider(intent_router=None)
        with self.assertRaises(RuntimeError):
            provider.get_intent_classifier()

    def test_raises_when_parse_function_not_provided(self):
        """Should raise RuntimeError if parse_function not initialized."""
        provider = RuleBasedProvider(parse_function=None)
        with self.assertRaises(RuntimeError):
            provider.get_parser()

    def test_raises_when_generate_function_not_provided(self):
        """Should raise RuntimeError if generate_function not initialized."""
        provider = RuleBasedProvider(generate_function=None)
        with self.assertRaises(RuntimeError):
            provider.get_explanation_generator()


class TestLLMProviderRegistry(unittest.TestCase):
    """Test provider registry singleton and switching."""

    def setUp(self):
        """Reset registry before each test."""
        self.registry = LLMProviderRegistry.get_instance()
        self.registry.reset()

    def tearDown(self):
        """Reset registry after each test."""
        self.registry.reset()

    def test_singleton_pattern(self):
        """Multiple calls should return same instance."""
        reg1 = LLMProviderRegistry.get_instance()
        reg2 = LLMProviderRegistry.get_instance()
        self.assertIs(reg1, reg2)

    def test_set_and_get_provider(self):
        """Should store and retrieve registered provider."""
        provider = MockLLMProvider()
        self.registry.set_provider(provider)
        self.assertIs(self.registry.get_provider(), provider)

    def test_convenience_methods(self):
        """Should provide convenience methods for all three components."""
        provider = MockLLMProvider()
        self.registry.set_provider(provider)

        classifier = self.registry.get_intent_classifier()
        self.assertIsInstance(classifier, IntentClassifier)

        parser = self.registry.get_parser()
        self.assertIsInstance(parser, SupplyChainParser)

        generator = self.registry.get_explanation_generator()
        self.assertIsInstance(generator, ExplanationGenerator)

    def test_reset_clears_provider(self):
        """Reset should clear the registered provider."""
        provider = MockLLMProvider()
        self.registry.set_provider(provider)
        self.assertIs(self.registry.get_provider(), provider)

        self.registry.reset()

        # After reset, get_provider() should return a default (or raise if not initialized)
        # For this test, we just verify reset happened
        self.assertIsNone(self.registry._provider)

    def test_switching_providers_at_runtime(self):
        """Should allow switching between providers."""
        mock_provider = MockLLMProvider()
        self.registry.set_provider(mock_provider)

        classifier1 = self.registry.get_intent_classifier()
        self.assertIsInstance(classifier1, MockIntentClassifier)

        # Switch to a different provider
        rule_based_provider = RuleBasedProvider(
            intent_router=Mock(detect_intent=Mock(return_value="solve")),
            parse_function=Mock(),
            generate_function=Mock(),
        )
        self.registry.set_provider(rule_based_provider)

        classifier2 = self.registry.get_intent_classifier()
        self.assertIsInstance(classifier2, RuleBasedIntentClassifierAdapter)
        self.assertIsNot(classifier1, classifier2)

    def test_convenience_methods_use_active_provider(self):
        """Convenience methods should use the active provider."""
        provider1 = MockLLMProvider()
        self.registry.set_provider(provider1)
        classifier1 = self.registry.get_intent_classifier()

        provider2 = MockLLMProvider()
        self.registry.set_provider(provider2)
        classifier2 = self.registry.get_intent_classifier()

        # Different providers should provide different instances
        self.assertIsNot(classifier1, classifier2)


class TestInterfaceAbstractness(unittest.TestCase):
    """Test that interfaces are properly abstract."""

    def test_intent_classifier_cannot_be_instantiated(self):
        """Should not be able to instantiate abstract IntentClassifier."""
        with self.assertRaises(TypeError):
            IntentClassifier()

    def test_supply_chain_parser_cannot_be_instantiated(self):
        """Should not be able to instantiate abstract SupplyChainParser."""
        with self.assertRaises(TypeError):
            SupplyChainParser()

    def test_explanation_generator_cannot_be_instantiated(self):
        """Should not be able to instantiate abstract ExplanationGenerator."""
        with self.assertRaises(TypeError):
            ExplanationGenerator()

    def test_llm_provider_cannot_be_instantiated(self):
        """Should not be able to instantiate abstract LLMProvider."""
        with self.assertRaises(TypeError):
            LLMProvider()


class TestEndToEndIntegration(unittest.TestCase):
    """Integration tests for the full LLM adapter system."""

    def setUp(self):
        self.registry = LLMProviderRegistry.get_instance()
        self.registry.reset()

    def tearDown(self):
        self.registry.reset()

    def test_mock_provider_full_workflow(self):
        """Test using mock provider through registry."""
        self.registry.set_provider(MockLLMProvider())

        # Simulate a user interaction
        user_text = "Define node N1 and product P1"

        classifier = self.registry.get_intent_classifier()
        intent = classifier.classify(user_text)
        self.assertEqual(intent, "problem_formulation")

        parser = self.registry.get_parser()
        entities = parser.parse(user_text)
        self.assertEqual(entities["nodes"], [])  # Mock returns empty

        generator = self.registry.get_explanation_generator()
        response = generator.generate("hint", {"intent": intent})
        self.assertIn("placeholder", response)

    def test_rule_based_provider_full_workflow(self):
        """Test using rule-based provider through registry."""
        mock_router = Mock(detect_intent=Mock(return_value="validation"))
        mock_parse = Mock(return_value={"nodes": [{"id": "N1"}], "products": []})
        mock_generate = Mock(return_value="Check these issues...")

        provider = RuleBasedProvider(
            intent_router=mock_router,
            parse_function=mock_parse,
            generate_function=mock_generate,
        )
        self.registry.set_provider(provider)

        user_text = "Check the problem"

        classifier = self.registry.get_intent_classifier()
        intent = classifier.classify(user_text)
        self.assertEqual(intent, "validation")

        parser = self.registry.get_parser()
        entities = parser.parse(user_text)
        self.assertEqual(len(entities["nodes"]), 1)

        generator = self.registry.get_explanation_generator()
        response = generator.generate("full", {})
        self.assertEqual(response, "Check these issues...")


if __name__ == "__main__":
    unittest.main()
