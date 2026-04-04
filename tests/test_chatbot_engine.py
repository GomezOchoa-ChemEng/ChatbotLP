import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.schema import (
    ProblemState,
    Node,
    Product,
    Supplier,
    Consumer,
    Bid,
    TransportLink,
    Technology,
)
from src.chatbot_engine import IntentRouter, run_chatbot_session
from src.solver import SolveResult


def make_minimal_state():
    """Create a minimal valid problem state."""
    s = ProblemState(problem_title="Test Problem")
    s.add_node(Node(id="n1"))
    s.add_product(Product(id="p1"))
    s.add_supplier(Supplier(id="sup1", node="n1", product="p1", capacity=10))
    s.add_consumer(Consumer(id="con1", node="n1", product="p1", capacity=5))
    s.add_bid(
        Bid(
            id="b1",
            owner_id="sup1",
            owner_type="supplier",
            product_id="p1",
            price=1.0,
            quantity=5,
        )
    )
    s.add_bid(
        Bid(
            id="b2",
            owner_id="con1",
            owner_type="consumer",
            product_id="p1",
            price=2.0,
            quantity=5,
        )
    )
    return s


def make_case_c_state():
    s = make_minimal_state()
    s.add_product(Product(id="p2"))
    s.add_transport(TransportLink(id="t1", origin="n1", destination="n1", product="p1", capacity=100))
    s.add_technology(
        Technology(
            id="tech1",
            node="n1",
            capacity=100,
            yield_coefficients={"p1": -1.0, "p2": 0.8},
        )
    )
    return s


class TestIntentDetection:
    """Test the rule-based intent router."""

    def test_problem_formulation_intent(self):
        router = IntentRouter()
        assert router.detect_intent("Add a supplier in node A") == "problem_formulation"
        assert router.detect_intent("Define product P") == "problem_formulation"
        assert router.detect_intent("Node 1 and Node 2") == "problem_formulation"

    def test_validation_intent(self):
        router = IntentRouter()
        assert router.detect_intent("Validate the problem") == "validation"
        assert router.detect_intent("Check for issues") == "validation"
        assert router.detect_intent("Is the model ready?") == "validation"

    def test_solve_intent(self):
        router = IntentRouter()
        assert router.detect_intent("Solve the model") == "solve"
        assert router.detect_intent("Find the optimal solution") == "solve"
        assert router.detect_intent("What is the objective value?") == "solve"

    def test_theorem_check_intent(self):
        router = IntentRouter()
        assert router.detect_intent("Check theorem applicability") == "theorem_check"
        assert router.detect_intent("Case A compatibility") == "theorem_check"
        assert router.detect_intent("Does Case B apply?") == "theorem_check"

    def test_scenario_intent(self):
        router = IntentRouter()
        assert router.detect_intent("Run a what-if scenario") == "scenario"
        assert router.detect_intent("What if I change the capacity?") == "scenario"
        assert router.detect_intent("Modify the price parameter") == "scenario"
        assert (
            router.detect_intent(
                "How would the optimal flows and prices change if the transformation technology were unavailable?"
            )
            == "scenario"
        )

    def test_explanation_intent(self):
        router = IntentRouter()
        assert router.detect_intent("Explain the results") == "explanation"
        assert router.detect_intent("Help me understand") == "explanation"
        assert router.detect_intent("Give me a hint") == "explanation"

    def test_default_intent(self):
        router = IntentRouter()
        # Text with no specific keywords defaults to explanation
        assert router.detect_intent("Hello there") == "explanation"


class TestProblemFormulation:
    """Test problem formulation workflow."""

    def test_add_entities_from_text(self):
        state = ProblemState()
        result = run_chatbot_session(state, "Node A. Product P.")
        assert result["intent"] == "problem_formulation"
        assert result["success"]
        assert result["state"].nodes
        assert result["state"].products

    def test_formulation_with_insufficient_text(self):
        state = ProblemState()
        result = run_chatbot_session(state, "xyz abc def")
        # Should default to explanation since no specific keywords match
        assert result["intent"] == "explanation"
        assert result["success"]


class TestValidation:
    """Test validation workflow."""

    def test_validate_minimal_state(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Validate the problem")
        assert result["intent"] == "validation"
        assert result["success"]
        # Response should be a non-empty string
        assert len(result["response"]) > 0

    def test_validate_empty_state(self):
        state = ProblemState()
        result = run_chatbot_session(state, "Check status")
        assert result["intent"] == "validation"
        assert result["success"]


class TestSolve:
    """Test solve workflow."""

    def test_solve_minimal_state(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Solve the model")
        assert result["intent"] == "solve"
        # May succeed or fail depending on solver availability; both are OK
        assert "response" in result


class TestTheoremCheck:
    """Test theorem checking workflow."""

    def test_theorem_check_basic(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Check theorem applicability")
        assert result["intent"] == "theorem_check"
        # Should produce a response even if success varies
        assert len(result["response"]) > 0


class TestScenario:
    """Test scenario analysis workflow."""

    def test_scenario_with_capacity_change(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "What if I increase capacity to 20")
        assert result["intent"] == "scenario"
        assert result["success"]

    def test_scenario_with_price_change(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Change price to 5.0")
        assert result["intent"] == "scenario"
        assert result["success"]

    def test_scenario_with_invalid_params(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Make a scenario")  # no specific param
        # Intent is detected, but change_spec is empty so unsuccessful
        assert result["intent"] == "scenario"
        assert not result["success"]

    def test_scenario_response_mentions_baseline_and_modified(self, monkeypatch):
        state = make_minimal_state()
        base = SolveResult(
            model=None,
            status="optimal",
            message="",
            objective_value=5.0,
            solver_time=0.0,
            solution={"q": {"b1": 5.0, "b2": 5.0}, "f": {}, "x": {}},
            success=True,
        )
        modified = SolveResult(
            model=None,
            status="optimal",
            message="",
            objective_value=7.0,
            solver_time=0.0,
            solution={"q": {"b1": 7.0, "b2": 5.0}, "f": {}, "x": {}},
            success=True,
        )

        monkeypatch.setattr(
            "src.chatbot_engine.run_scenario",
            lambda *args, **kwargs: {
                "base": base,
                "scenario": modified,
                "difference": {
                    "objective_delta": 2.0,
                    "accepted_bid_changes": {"b1": {"before": 5.0, "after": 7.0, "delta": 2.0}},
                    "flow_changes": {},
                    "technology_activity_changes": {},
                    "price_changes": {},
                    "binding_constraint_changes": {},
                    "unchanged_dimensions": ["flow_changes", "technology_activity_changes", "price_changes"],
                },
            },
        )

        result = run_chatbot_session(state, "What happens if supplier capacity increases from 100 to 150?")
        assert result["intent"] == "scenario"
        assert result["response_mode"] == "solver_grounded_verification"
        assert "Baseline" in result["response"]
        assert "Modified scenario" in result["response"]

    def test_case_c_unavailable_technology_routes_to_scenario(self, monkeypatch):
        state = make_case_c_state()
        monkeypatch.setattr(
            "src.chatbot_engine.run_scenario",
            lambda *args, **kwargs: {
                "base": SolveResult(None, "optimal", "", 10.0, 0.0, {"q": {}, "f": {}, "x": {"tech1": 3.0}}, True),
                "scenario": SolveResult(None, "optimal", "", 8.0, 0.0, {"q": {}, "f": {}, "x": {"tech1": 0.0}}, True),
                "difference": {
                    "objective_delta": -2.0,
                    "accepted_bid_changes": {},
                    "flow_changes": {},
                    "technology_activity_changes": {"tech1": {"before": 3.0, "after": 0.0, "delta": -3.0}},
                    "price_changes": {},
                    "binding_constraint_changes": {},
                    "unchanged_dimensions": ["flow_changes", "price_changes"],
                },
            },
        )
        result = run_chatbot_session(
            state,
            "How would the optimal flows and prices change if the transformation technology were unavailable?",
        )
        assert result["intent"] == "scenario"
        assert result["scenario_extraction"]["parameter_type"] == "technology_availability"
        assert result["success"]


class TestChatbotEngineLLMIntegration:
    """Focus tests on optional LLM integration paths."""

    def test_intent_classification_with_llm(self):
        from src.llm_adapter import LLMProviderRegistry, RuleBasedProvider
        from unittest.mock import Mock
        registry = LLMProviderRegistry.get_instance()
        registry.reset()

        # provider that always returns "solve"
        provider = RuleBasedProvider(
            intent_router=Mock(detect_intent=Mock(return_value="solve")),
            parse_function=lambda t: {},
            generate_function=lambda mode, ctx: "",
        )
        registry.set_provider(provider)
        state = make_minimal_state()
        result = run_chatbot_session(state, "random text", use_llm=True)
        assert result["intent"] == "solve"

        registry.reset()

    def test_parser_with_llm_in_chatbot(self):
        from src.llm_adapter import LLMProviderRegistry, RuleBasedProvider
        from unittest.mock import Mock
        registry = LLMProviderRegistry.get_instance()
        registry.reset()

        # provider that returns node LLM1 and classifies as problem_formulation
        provider = RuleBasedProvider(
            intent_router=Mock(detect_intent=Mock(return_value="problem_formulation")),
            parse_function=lambda t: {"nodes":[{"id":"LLM1","name":"LLM1"}],"products":[],"suppliers":[],"consumers":[],"transport_links":[],"technologies":[],"bids":[]},
            generate_function=lambda mode, ctx: "",
        )
        registry.set_provider(provider)
        state = ProblemState()
        result = run_chatbot_session(state, "ignored text", use_llm=True)
        assert any(n.id == "LLM1" for n in result["state"].nodes)

        registry.reset()

    def test_response_with_llm_in_chatbot(self):
        from src.llm_adapter import LLMProviderRegistry, RuleBasedProvider
        from unittest.mock import Mock
        registry = LLMProviderRegistry.get_instance()
        registry.reset()

        provider = RuleBasedProvider(
            intent_router=Mock(detect_intent=Mock(return_value="validation")),
            parse_function=lambda t: {},
            generate_function=lambda mode, ctx: "LLM response",
        )
        registry.set_provider(provider)
        state = make_minimal_state()
        result = run_chatbot_session(state, "Validate please", use_llm=True)
        assert result["response"] == "LLM response"

        registry.reset()


class TestExplanation:
    """Test explanation/help workflow."""

    def test_help_request(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Help me understand the problem")
        assert result["intent"] == "explanation"
        assert result["success"]

    def test_default_explanation(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Random text")  # default to explanation
        assert result["intent"] == "explanation"
        assert result["success"]

    def test_sampat_reasoning_engine_handles_price_interpretation(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "What do node-product prices represent?")
        assert result["success"]
        assert result["render_mode"] == "markdown"
        assert result["sampat_reasoning_package"].response_mode == "paper_grounded_explanation"
        assert "shadow values" in result["response"].lower() or "marginal" in result["response"].lower()

    def test_sampat_reasoning_engine_handles_case_comparison(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Compare Case A and Case B.")
        assert result["success"]
        assert result["sampat_reasoning_package"].plan.object == "benchmark_case"
        assert "Case A" in result["response"]
        assert "Case B" in result["response"]

    def test_section_23_interpretation_is_not_a_dual_dump(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Explain how Section 2.3 changes the interpretation of bids and prices.")
        assert result["success"]
        assert result["sampat_reasoning_package"].response_mode == "paper_grounded_explanation"
        assert "The dual problem is formulated as follows:" not in result["response"]


class TestModes:
    """Test different response modes."""

    def test_hint_mode(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Help", mode="hint")
        assert result["success"]
        # Response should be generated in hint mode
        assert len(result["response"]) > 0

    def test_guided_mode(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Help", mode="guided")
        assert result["success"]
        assert len(result["response"]) > 0

    def test_full_mode(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "Help", mode="full")
        assert result["success"]
        assert len(result["response"]) > 0


class TestStateUpdates:
    """Test that state is properly updated through the workflow."""

    def test_state_persists_through_session(self):
        state = ProblemState()
        initial_nodes = len(state.nodes)
        result = run_chatbot_session(state, "Node X")
        assert len(result["state"].nodes) > initial_nodes

    def test_multiple_sessions_accumulate(self):
        state = ProblemState()
        result1 = run_chatbot_session(state, "Node A")
        state = result1["state"]
        result2 = run_chatbot_session(state, "Product P")
        state = result2["state"]
        assert len(state.nodes) >= 1
        assert len(state.products) >= 1


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_malformed_request(self):
        state = make_minimal_state()
        result = run_chatbot_session(state, "")
        assert "response" in result
        # Should handle gracefully even on empty input

    def test_none_values_handled(self):
        state = make_minimal_state()
        # Ensure None doesn't break anything
        result = run_chatbot_session(state, "Validate")
        assert result["success"]
