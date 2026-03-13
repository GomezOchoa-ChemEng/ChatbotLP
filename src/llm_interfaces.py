"""Abstract interfaces for LLM-based supply chain chatbot components.

This module defines clean, provider-agnostic interfaces for three key chatbot
capabilities that can be implemented using local rule-based logic, a real LLM
provider (OpenAI, Anthropic, etc.), or hybrid approaches.

The interfaces are designed to:
- Match the signatures of existing rule-based implementations
- Allow pluggable implementations without modifying the chatbot core
- Support fallback to deterministic rule-based versions
- Facilitate testing and mocking

Implementations should be stateless and deterministic for reproducibility
in classroom settings. If an LLM provider is used, caching and deterministic
prompting strategies are recommended.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class IntentClassifier(ABC):
    """Abstract interface for classifying user intent from natural language.

    Intent classification is the first step in routing user requests to the
    appropriate handlers (problem formulation, validation, solving, etc.).
    """

    @abstractmethod
    def classify(self, text: str) -> str:
        """Classify the intent of a user message.

        Args:
            text: The user input message.

        Returns:
            A string representing the detected intent. Should be one of:
            - "problem_formulation": User is describing or adding to a problem
            - "validation": User wants to check the problem state for issues
            - "solve": User requests solving the optimization model
            - "theorem_check": User asks about theoretical assumptions
            - "scenario": User requests what-if analysis
            - "explanation": User asks for help or guidance

        Raises:
            ValueError: If text is empty or None.
        """
        pass


class SupplyChainParser(ABC):
    """Abstract interface for parsing supply chain descriptions.

    This parser extracts structured entities (nodes, products, suppliers,
    consumers, technologies, bids, transport links) from unstructured
    natural language descriptions.
    """

    @abstractmethod
    def parse(self, text: str) -> Dict[str, Any]:
        """Extract supply chain entities from natural language text.

        Args:
            text: Natural language description of a supply chain problem.

        Returns:
            A dictionary with the following keys (all values are lists of dicts
            unless otherwise noted):

            - "nodes": List of nodes, each as {"id": str, "name": str}
            - "products": List of products, each as {"id": str, "name": str}
            - "suppliers": List with keys:
                - "id": Unique supplier ID
                - "location_node": Node ID where supplier is located
                - "product_id": Product ID supplied
                - "capacity": Supply capacity (float or None if not specified)
            - "consumers": List with keys:
                - "id": Unique consumer ID
                - "location_node": Node ID where consumer is located
                - "product_id": Product ID demanded
                - "demand": Demand quantity (float or None if not specified)
            - "transport_links": List with keys:
                - "from_node": Source node ID
                - "to_node": Destination node ID
                - "product_id": Product ID transported
                - "cost": Transport cost (float or None if not specified)
            - "technologies": List with keys:
                - "id": Unique technology ID
                - "node_id": Node where transformation occurs
                - "input_product_id": Input product ID
                - "output_product_id": Output product ID
                - "yield": Efficiency factor (float or None if not specified)
            - "bids": List with keys:
                - "id": Unique bid ID
                - "supplier_id": Supplier ID making the bid
                - "amount": Bid quantity (float or None if not specified)
                - "cost": Bid cost (float or None if not specified)

            If no entities of a given type are found, that key should map to
            an empty list, not be omitted.

        Raises:
            ValueError: If text is empty or None.
        """
        pass


class ExplanationGenerator(ABC):
    """Abstract interface for generating user-facing explanations.

    This generator produces responses tailored to classroom use, supporting
    three progressive disclosure modes for pedagogical effectiveness.
    """

    @abstractmethod
    def generate(self, mode: str, context: Dict[str, Any]) -> str:
        """Generate an explanation or response based on problem state and results.

        Args:
            mode: Response disclosure level, one of:
                - "hint": Minimal guidance; suggest next steps without revealing answers
                - "guided": Step-by-step assistance with partial results and prompts
                - "full": Complete solution with detailed explanations and all results

            context: A dictionary containing problem state and analysis results.
                Common keys:
                - "problem_state": ProblemState instance (the current problem)
                - "user_message": str (the original user message)
                - "intent": str (detected or explicit intent)
                - "validation_result": dict from validate_state() (issues, status)
                - "solve_result": SolveResult instance or None (optimization results)
                - "theorem_checks": list of TheoremCheck (theory checks)
                - "scenario_result": dict with scenario analysis outputs
                - "previous_responses": list of str (chat history for context)

        Returns:
            A human-readable string response suitable for the specified mode.
            Should be clear, concise, and pedagogically appropriate.

        Raises:
            ValueError: If mode is not one of the three valid options.
            KeyError: If required context keys are missing for the requested mode.
        """
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM provider implementations.

    An LLM provider encapsulates all three interfaces (intent classification,
    parsing, and explanation generation) and can be swapped as a unit.
    """

    @abstractmethod
    def get_intent_classifier(self) -> IntentClassifier:
        """Return an IntentClassifier implementation."""
        pass

    @abstractmethod
    def get_parser(self) -> SupplyChainParser:
        """Return a SupplyChainParser implementation."""
        pass

    @abstractmethod
    def get_explanation_generator(self) -> ExplanationGenerator:
        """Return an ExplanationGenerator implementation."""
        pass
