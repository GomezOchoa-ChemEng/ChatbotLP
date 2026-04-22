"""Integration tests for LLM-based problem interpreter.

These tests require a real GEMINI_API_KEY to be set and will be skipped
if the key is not available. They test the actual LLM integration.
"""

import json
import os
import pytest

from src.llm_problem_interpreter import interpret_problem_from_text, build_state_from_semantic_plan
from src.schema import ProblemState


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set, skipping integration tests"
)
class TestLLMIntegration:
    """Integration tests that call the real LLM."""

    def test_interpret_case_a_description(self):
        """Test interpreting a Case A-like description with real LLM."""
        # Set required environment variables
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-3-flash-preview"

        description = """
        There are two nodes, N1 and N2. N1 has a supplier S1 that can supply up to 100 units of product P1.
        N2 has a consumer C1 that can consume up to 50 units of P1.
        There is a transport link from N1 to N2 with capacity 100.
        Supplier S1 bids at price 10 for 100 units.
        Consumer C1 bids at price 20 for 50 units.
        """

        # This should work with real LLM
        plan = interpret_problem_from_text(description.strip())

        # Verify the structure
        assert "problem_title" in plan
        assert isinstance(plan["nodes"], list)
        assert isinstance(plan["products"], list)
        assert isinstance(plan["suppliers"], list)
        assert isinstance(plan["consumers"], list)
        assert isinstance(plan["transport_links"], list)
        assert isinstance(plan["bids"], list)
        assert isinstance(plan["technologies"], list)

        # Should be able to build a state from the plan
        state = build_state_from_semantic_plan(plan)
        assert isinstance(state, ProblemState)

    def test_interpret_simple_description(self):
        """Test interpreting a simple supply chain description with real LLM."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-3-flash-preview"

        description = """
        There is a supplier that can provide 50 units of a product at $10 each.
        There is a consumer that wants 30 units at $15 each.
        They are connected by a transport link with capacity 100.
        """

        plan = interpret_problem_from_text(description.strip())
        state = build_state_from_semantic_plan(plan)

        # Basic checks
        assert isinstance(state, ProblemState)
        assert len(state.suppliers) >= 1
        assert len(state.consumers) >= 1
        assert len(state.transport_links) >= 1
        """Test interpreting a paraphrased version of Case A."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-3-flash-preview"

        description = """
        In this supply chain network, we have two locations: a production site and a demand location.
        The production site has a supplier that can provide up to 100 units of a certain good.
        The demand location has a buyer that needs up to 50 units of the same good.
        Goods can be transported between the two locations with unlimited capacity.
        The supplier offers 100 units at $10 each.
        The buyer is willing to pay $20 for each of the 50 units they need.
        """

        plan = interpret_problem_from_text(description.strip())
        state = build_state_from_semantic_plan(plan)

        # Should have the right number of entities
        assert len(state.nodes) >= 2  # At least 2 nodes
        assert len(state.products) >= 1  # At least 1 product
        assert len(state.suppliers) >= 1  # At least 1 supplier
        assert len(state.consumers) >= 1  # At least 1 consumer
        assert len(state.bids) >= 2  # At least 2 bids

    def test_interpret_incomplete_description(self):
        """Test that incomplete descriptions still produce valid plans."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-3-flash-preview"

        description = """
        There is a supplier that can supply 50 units.
        There is a consumer that wants 30 units.
        """

        plan = interpret_problem_from_text(description.strip())
        state = build_state_from_semantic_plan(plan)

        # Should still create a valid state, even if minimal
        assert isinstance(state, ProblemState)
        assert "problem_title" in plan