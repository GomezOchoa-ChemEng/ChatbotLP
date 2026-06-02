"""Tests for LLM-based problem interpreter.

Tests the LLM-first architecture for converting natural language
supply chain descriptions into structured ProblemState objects.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch

from src.llm_problem_interpreter import (
    build_problem_artifacts_from_semantic_plan,
    interpret_problem_from_text,
    build_state_from_semantic_plan,
    _validate_semantic_plan,
    _build_interpretation_prompt,
    normalize_semantic_plan,
)
from src.schema import ProblemState
from src.llm_adapter import LLMConfigurationError


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
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 100.0, "cost": 0.0}],
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
                "cost": 2.0,
                "yield_coefficients": {"P1": -1.0, "P2": 1.0}
            }],
        }

        state = build_state_from_semantic_plan(plan)

        assert len(state.technologies) == 1
        tech = state.technologies[0]
        assert tech.id == "K1"
        assert tech.capacity == 100.0
        assert tech.cost == 2.0
        assert tech.yield_coefficients["P1"] == -1.0
        assert tech.yield_coefficients["P2"] == 1.0

    def test_builder_assigns_missing_ids_and_resolves_names(self):
        """Test that lightweight normalization can backfill IDs and resolve named references."""
        plan = {
            "problem_title": "Paraphrased Case A",
            "nodes": [{"name": "Production Site"}, {"name": "Demand Site"}],
            "products": [{"name": "Ammonia"}],
            "suppliers": [{"node": "Production Site", "product": "Ammonia", "capacity": 100.0}],
            "consumers": [{"node": "Demand Site", "product": "Ammonia", "capacity": 60.0}],
            "transport_links": [{"origin": "Production Site", "destination": "Demand Site", "product": "Ammonia", "capacity": 100.0}],
            "bids": [
                {"owner_id": "S1", "owner_type": "supplier", "product_id": "Ammonia", "price": 8.0, "quantity": 100.0},
                {"owner_id": "C1", "owner_type": "consumer", "product_id": "Ammonia", "price": 14.0, "quantity": 60.0},
            ],
            "technologies": [],
        }

        normalized = normalize_semantic_plan(plan)
        state = build_state_from_semantic_plan(normalized)

        assert state.nodes[0].id == "Production_Site"
        assert state.nodes[1].id == "Demand_Site"
        assert state.products[0].id == "Ammonia"
        assert state.suppliers[0].node == "Production_Site"
        assert state.suppliers[0].product == "Ammonia"
        assert state.consumers[0].node == "Demand_Site"
        assert state.bids[0].owner_id == "S1"
        assert state.bids[0].product_id == "Ammonia"

    def test_build_problem_artifacts_exposes_state_summary(self):
        plan = {
            "problem_title": "Case A Summary",
            "nodes": [{"id": "N1"}, {"id": "N2"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 100.0}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 50.0}],
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 100.0}],
            "bids": [
                {"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 10.0, "quantity": 100.0},
                {"id": "B2", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 20.0, "quantity": 50.0},
            ],
            "technologies": [],
        }

        artifacts = build_problem_artifacts_from_semantic_plan(plan)

        assert artifacts["state_summary"]["counts"]["nodes"] == 2
        assert artifacts["state_summary"]["counts"]["bids"] == 2
        assert artifacts["market_instance"].problem_title == "Case A Summary"

    def test_builder_preserves_collective_transport_capacities_on_all_links(self):
        plan = {
            "problem_title": "Collective Route Capacity",
            "nodes": [{"id": "S"}, {"id": "A"}, {"id": "B"}, {"id": "K"}, {"id": "D"}],
            "products": [{"id": "Manure"}, {"id": "Compost"}],
            "suppliers": [],
            "consumers": [],
            "transport_links": [
                {"id": "T1", "origin": "S", "destination": "A", "product": "Manure", "capacity": 1000.0, "cost": 0.1},
                {"id": "T2", "origin": "S", "destination": "B", "product": "Manure", "capacity": 1000.0, "cost": 0.2},
                {"id": "T3", "origin": "S", "destination": "K", "product": "Manure", "capacity": 1000.0, "cost": 0.0},
                {"id": "T4", "origin": "K", "destination": "D", "product": "Compost", "capacity": 1000.0, "cost": 1.0},
            ],
            "bids": [],
            "technologies": [],
        }

        state = build_state_from_semantic_plan(plan)

        assert [link.capacity for link in state.transport_links] == [1000.0] * 4

    def test_builder_preserves_product_specific_transport_capacity_statements(self):
        plan = {
            "problem_title": "Product Specific Route Capacity",
            "nodes": [{"id": "S"}, {"id": "A"}, {"id": "K"}, {"id": "D"}],
            "products": [{"id": "Manure"}, {"id": "Compost"}],
            "suppliers": [],
            "consumers": [],
            "transport_links": [
                {"id": "Manure_A", "origin": "S", "destination": "A", "product": "Manure", "capacity": 1000.0, "cost": 0.1},
                {"id": "Manure_K", "origin": "S", "destination": "K", "product": "Manure", "capacity": 1000.0, "cost": 0.0},
                {"id": "Compost_D", "origin": "K", "destination": "D", "product": "Compost", "capacity": 1000.0, "cost": 1.0},
            ],
            "bids": [],
            "technologies": [],
        }

        state = build_state_from_semantic_plan(plan)
        capacities_by_product = {
            (link.product, link.destination): link.capacity
            for link in state.transport_links
        }

        assert capacities_by_product[("Manure", "A")] == 1000.0
        assert capacities_by_product[("Manure", "K")] == 1000.0
        assert capacities_by_product[("Compost", "D")] == 1000.0

    def test_builder_preserves_missing_route_cost_and_explicit_zero_distinctly(self):
        plan = {
            "problem_title": "Missing versus explicit route cost",
            "nodes": [{"id": "S"}, {"id": "A"}, {"id": "B"}],
            "products": [{"id": "P"}],
            "suppliers": [],
            "consumers": [],
            "transport_links": [
                {"id": "missing", "origin": "S", "destination": "A", "product": "P", "capacity": 10.0},
                {"id": "free", "origin": "S", "destination": "B", "product": "P", "capacity": 10.0, "cost": 0.0},
            ],
            "bids": [],
            "technologies": [],
        }

        state = build_state_from_semantic_plan(plan)

        assert state.transport_links[0].cost is None
        assert state.transport_links[1].cost == 0.0

    def test_builder_preserves_missing_technology_cost(self):
        plan = {
            "problem_title": "Missing technology cost",
            "nodes": [{"id": "K"}],
            "products": [{"id": "P1"}, {"id": "P2"}],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "technologies": [
                {
                    "id": "K1",
                    "node": "K",
                    "capacity": 10.0,
                    "yield_coefficients": {"P1": -1.0, "P2": 0.8},
                },
                {
                    "id": "K2",
                    "node": "K",
                    "capacity": 10.0,
                    "cost": 0.0,
                    "yield_coefficients": {"P1": -1.0, "P2": 0.8},
                },
            ],
            "bids": [],
        }

        state = build_state_from_semantic_plan(plan)

        assert state.technologies[0].cost is None
        assert state.technologies[1].cost == 0.0


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
        assert "missing_information" in prompt
        assert "Preserve every explicit numeric value exactly as written" in prompt

    def test_prompt_instructs_route_specific_cost_and_capacity_recovery(self):
        """The interpreter prompt should guard against unordered route costs and omitted collective capacities."""
        prompt = _build_interpretation_prompt("All four transport links have capacity 1000.")
        instruction_block = prompt.split("Problem description:", 1)[0]

        assert "Extract each transport link as one complete structured record" in prompt
        assert "Do not treat transport costs as unordered numerical values" in prompt
        assert "Do not assign route costs by position" in prompt
        assert "preserve those origin-destination-cost bindings exactly" in prompt
        assert "expand them to every affected transport link" in prompt
        assert "all transport links have capacity 1000" in prompt
        assert "Do not leave transport_links[].capacity null" in prompt
        for domain_name in ("manure", "compost", "menomonie", "black river falls", "madison"):
            assert domain_name not in instruction_block.lower()


class TestLLMInterpretation:
    """Test the full LLM interpretation pipeline."""

    @patch("src.llm_problem_interpreter.ensure_gemini_provider")
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"})
    def test_interpret_problem_from_text_success(self, mock_ensure_gemini_provider):
        """Test successful interpretation with mocked LLM."""
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
        mock_ensure_gemini_provider.return_value = mock_provider

        result = interpret_problem_from_text("Test description")

        assert result["problem_title"] == "Test Problem"
        assert len(result["nodes"]) == 1
        assert result["problem_state"].problem_title == "Test Problem"
        assert result["market_instance"].problem_title == "Test Problem"

    @patch("src.llm_problem_interpreter.ensure_gemini_provider")
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"})
    def test_interpret_problem_from_text_incomplete_case_a(self, mock_ensure_gemini_provider):
        """Test incomplete prose preserves missing data rather than inventing it."""
        mock_provider = Mock()
        mock_generator = Mock()
        mock_generator.generate.return_value = json.dumps({
            "problem_title": "Incomplete Case A",
            "problem_type": "case_a",
            "nodes": [{"id": "N1"}],
            "products": [{"id": "P1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": None}],
            "consumers": [{"id": "C1", "node": "N1", "product": "P1", "capacity": None}],
            "transport_links": [],
            "bids": [],
            "technologies": [],
            "missing_information": ["supplier capacity", "consumer bid price"],
            "ambiguities": [],
        })
        mock_provider.get_explanation_generator.return_value = mock_generator
        mock_ensure_gemini_provider.return_value = mock_provider

        result = interpret_problem_from_text("Supplier and consumer exist but capacities were not stated.")

        assert result["semantic_plan"]["missing_information"] == ["supplier capacity", "consumer bid price"]
        assert result["problem_state"].suppliers[0].capacity is None
        assert result["problem_state"].consumers[0].capacity is None

    @patch("src.llm_problem_interpreter.ensure_gemini_provider")
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"})
    def test_interpret_problem_from_text_invalid_json(self, mock_ensure_gemini_provider):
        """Test handling of invalid JSON from LLM."""
        mock_provider = Mock()
        mock_generator = Mock()
        mock_generator.generate.return_value = "Invalid JSON response"
        mock_provider.get_explanation_generator.return_value = mock_generator
        mock_ensure_gemini_provider.return_value = mock_provider

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
        """Test that missing environment variables raise a configuration error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                LLMConfigurationError,
                match="LLM_PROVIDER is not set to 'gemini'",
            ):
                interpret_problem_from_text("Test description")

    @patch.dict(os.environ, {"LLM_PROVIDER": "other", "GEMINI_API_KEY": "test-key"})
    def test_interpret_wrong_provider(self):
        """Test that wrong provider raises a configuration error."""
        with pytest.raises(
            LLMConfigurationError,
            match="LLM_PROVIDER is not set to 'gemini'",
        ):
            interpret_problem_from_text("Test description")

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True)
    def test_interpret_missing_api_key(self):
        """Test that missing API key raises a configuration error."""
        with pytest.raises(
            LLMConfigurationError,
            match="GEMINI_API_KEY is not set",
        ):
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
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 100.0, "cost": 0.0}],
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
