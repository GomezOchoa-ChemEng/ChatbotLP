"""Tests for LLM-based problem interpreter.

Tests the LLM-first architecture for converting natural language
supply chain descriptions into structured ProblemState objects.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch

from src.llm_problem_interpreter import (
    interpret_problem_from_text,
    build_state_from_semantic_plan,
    _validate_semantic_plan,
    _build_interpretation_prompt,
)
from src.schema import ProblemState


class TestSemanticPlanValidation:
    """Test validation of semantic plans."""

    def test_valid_semantic_plan(self):
        """Test validation of a complete, valid semantic plan."""
        plan = {
            "problem_title": "Test Problem",
            "nodes": [{"id": "N1", "name": "Node 1"}],
            "products": [{"id": "P1", "name": "Product 1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
            "consumers": [{"id": "C1", "node": "N1", "product": "P1", "capacity": 50.0}],
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N1", "product": "P1", "capacity": 100.0}],
            "bids": [{"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0}],
            "technologies": [],
        }
        # Should not raise
        _validate_semantic_plan(plan)

    def test_missing_required_field(self):
        """Test validation fails with missing required field."""
        plan = {
            # Missing "problem_title"
            "nodes": [],
            "products": [],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "bids": [],
            "technologies": [],
        }
        with pytest.raises(ValueError, match="Missing required key: problem_title"):
            _validate_semantic_plan(plan)

    def test_duplicate_ids(self):
        """Test validation fails with duplicate IDs."""
        plan = {
            "problem_title": "Test Problem",
            "nodes": [{"id": "N1"}, {"id": "N1"}],  # Duplicate
            "products": [],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "bids": [],
            "technologies": [],
        }
        with pytest.raises(ValueError, match="Duplicate ID found: N1"):
            _validate_semantic_plan(plan)


class TestStateConstruction:
    """Test building ProblemState from semantic plans."""

    def test_build_case_a_like_state(self):
        """Test building a state similar to Case A."""
        plan = {
            "problem_title": "Case A - No Transformation",
            "nodes": [
                {"id": "N1", "name": "Supplier Node"},
                {"id": "N2", "name": "Consumer Node"}
            ],
            "products": [{"id": "P1", "name": "Product A"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 100.0}],
            "bids": [
                {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0, "quantity": 100.0},
                {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0}
            ],
            "technologies": [],
        }

        state = build_state_from_semantic_plan(plan)

        assert state.problem_title == "Case A - No Transformation"
        assert len(state.nodes) == 2
        assert len(state.products) == 1
        assert len(state.suppliers) == 1
        assert len(state.consumers) == 1
        assert len(state.transport_links) == 1
        assert len(state.bids) == 2
        assert len(state.technologies) == 0

        # Check specific values
        assert state.suppliers[0].capacity == 100.0
        assert state.bids[0].price == 10.0
        assert state.bids[1].price == 20.0

    def test_build_state_with_technology(self):
        """Test building a state with transformation technology."""
        plan = {
            "problem_title": "Case C - Transformation",
            "nodes": [{"id": "N1"}],
            "products": [{"id": "P1"}, {"id": "P2"}],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "bids": [],
            "technologies": [{
                "id": "K1",
                "node": "N1",
                "capacity": 100.0,
                "yield_coefficients": {"P1": -1.0, "P2": 1.0}
            }],
        }

        state = build_state_from_semantic_plan(plan)

        assert len(state.technologies) == 1
        tech = state.technologies[0]
        assert tech.id == "K1"
        assert tech.capacity == 100.0
        assert tech.yield_coefficients["P1"] == -1.0
        assert tech.yield_coefficients["P2"] == 1.0


class TestInterpretationPrompt:
    """Test the LLM interpretation prompt building."""

    def test_prompt_structure(self):
        """Test that the interpretation prompt has the expected structure."""
        text = "Test description"
        prompt = _build_interpretation_prompt(text)

        assert "You are an expert at extracting structured supply chain optimization problems" in prompt
        assert "Output ONLY valid JSON" in prompt
        assert "Test description" in prompt
        assert "problem_title" in prompt
        assert "nodes" in prompt
        assert "suppliers" in prompt


class TestLLMInterpretation:
    """Test the full LLM interpretation pipeline."""

    @patch('src.llm_problem_interpreter.LLMProviderRegistry')
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"})
    def test_interpret_problem_from_text_success(self, mock_registry):
        """Test successful interpretation with mocked LLM."""
        # Mock the LLM response
        mock_provider = Mock()
        mock_generator = Mock()
        mock_generator.generate.return_value = json.dumps({
            "problem_title": "Test Problem",
            "nodes": [{"id": "N1"}],
            "products": [{"id": "P1"}],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "bids": [],
            "technologies": [],
        })
        mock_provider.get_explanation_generator.return_value = mock_generator
        mock_registry.get_instance.return_value.get_provider.return_value = mock_provider

        result = interpret_problem_from_text("Test description")

        assert result["problem_title"] == "Test Problem"
        assert len(result["nodes"]) == 1
        assert result["problem_state"].problem_title == "Test Problem"
        assert result["market_instance"].problem_title == "Test Problem"

    @patch('src.llm_problem_interpreter.LLMProviderRegistry')
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"})
    def test_interpret_problem_from_text_invalid_json(self, mock_registry):
        """Test handling of invalid JSON from LLM."""
        # Mock invalid JSON response
        mock_provider = Mock()
        mock_generator = Mock()
        mock_generator.generate.return_value = "Invalid JSON response"
        mock_provider.get_explanation_generator.return_value = mock_generator
        mock_registry.get_instance.return_value.get_provider.return_value = mock_provider

        with pytest.raises(ValueError, match="LLM output is not valid JSON"):
            interpret_problem_from_text("Test description")

    def test_interpret_empty_text(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Cannot interpret empty problem description"):
            interpret_problem_from_text("")

    def test_interpret_whitespace_text(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Cannot interpret empty problem description"):
            interpret_problem_from_text("   \n\t   ")

    def test_interpret_without_env_vars(self):
        """Test that missing environment variables raise RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="LLM interpretation requires explicit configuration"):
                interpret_problem_from_text("Test description")

    @patch.dict(os.environ, {"LLM_PROVIDER": "other", "GEMINI_API_KEY": "test-key"})
    def test_interpret_wrong_provider(self):
        """Test that wrong provider raises RuntimeError."""
        with pytest.raises(RuntimeError, match="LLM interpretation requires explicit configuration"):
            interpret_problem_from_text("Test description")

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True)
    def test_interpret_missing_api_key(self):
        """Test that missing API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY environment variable is required"):
            interpret_problem_from_text("Test description")


class TestIntegrationWithValidation:
    """Test integration with the validation system."""

    def test_valid_interpreted_state_passes_validation(self):
        """Test that a properly constructed state passes validation."""
        from src.validator import validate_state

        plan = {
            "problem_title": "Valid Test Problem",
            "nodes": [
                {"id": "N1", "name": "Node 1"},
                {"id": "N2", "name": "Node 2"}
            ],
            "products": [{"id": "P1", "name": "Product 1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 100.0}],
            "bids": [
                {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0, "quantity": 100.0},
                {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0}
            ],
            "technologies": [],
        }

        state = build_state_from_semantic_plan(plan)
        validation = validate_state(state)

        assert validation["solver_ready"] is True
        assert len(validation["missing_parameters"]) == 0
        assert len(validation["invalid_references"]) == 0

    def test_invalid_interpreted_state_fails_validation(self):
        """Test that an invalid state fails validation."""
        from src.validator import validate_state

        plan = {
            "problem_title": "Invalid Test Problem",
            "nodes": [{"id": "N1"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1"}],  # Missing capacity
            "consumers": [],
            "transport_links": [],
            "bids": [],
            "technologies": [],
        }

        state = build_state_from_semantic_plan(plan)
        validation = validate_state(state)

        assert validation["solver_ready"] is False
        assert len(validation["missing_parameters"]) > 0
