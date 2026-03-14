#!/usr/bin/env python3
"""
Google Colab-friendly demo for the supply chain chatbot with Gemini explanations.

This script demonstrates:
1. Setting up GEMINI_API_KEY for Colab or local use
2. Creating a minimal supply chain problem
3. Initializing GeminiExplanationProvider
4. Registering it with the LLM provider registry
5. Running a chatbot session with LLM explanations
6. Demonstrating fallback behavior when Gemini is unavailable

Notes:
- Install dependencies before running:
    pip install -r requirements.txt
    pip install google-generativeai
- In Colab, prefer storing GEMINI_API_KEY in Secrets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add repo root so `from src...` works
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import (
    ProblemState,
    Node,
    Product,
    Supplier,
    Consumer,
    TransportLink,
    Bid,
)
from src.chatbot_engine import run_chatbot_session, IntentRouter
from src.parser import parse_supply_chain_text
from src.response_generator import generate_response
from src.llm_adapter import RuleBasedProvider, LLMProviderRegistry
from src.gemini_explanation_provider import GeminiExplanationProvider


def setup_gemini_api_key() -> None:
    """Set GEMINI_API_KEY from Colab secrets if available, otherwise prompt."""
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"].strip():
        return

    # Try Colab secrets first
    try:
        from google.colab import userdata  # type: ignore

        secret_key = userdata.get("GEMINI_API_KEY")
        if secret_key:
            os.environ["GEMINI_API_KEY"] = secret_key
            print("Loaded GEMINI_API_KEY from Colab secrets.")
            return
        print("No GEMINI_API_KEY found in Colab secrets.")
    except Exception:
        pass

    # Fallback to manual input
    try:
        from getpass import getpass

        manual_key = getpass(
            "Enter GEMINI_API_KEY (leave blank to continue in fallback mode): "
        ).strip()
        if manual_key:
            os.environ["GEMINI_API_KEY"] = manual_key
            print("GEMINI_API_KEY set from manual input.")
        else:
            print("No API key provided. The demo will use fallback mode.")
    except Exception:
        print("Could not prompt for GEMINI_API_KEY. The demo will use fallback mode.")


def create_minimal_supply_chain_problem() -> ProblemState:
    """Create a minimal supply chain problem compatible with the current schema."""
    state = ProblemState(problem_title="Minimal Supply Chain Demo")

    state.add_node(Node(id="N1", name="Supplier Location"))
    state.add_node(Node(id="N2", name="Consumer Location"))

    state.add_product(Product(id="P1", name="Product A"))

    state.add_supplier(
        Supplier(
            id="S1",
            node="N1",
            product="P1",
            capacity=100.0,
        )
    )

    state.add_consumer(
        Consumer(
            id="C1",
            node="N2",
            product="P1",
            capacity=50.0,
        )
    )

    state.add_transport(
        TransportLink(
            id="T1",
            origin="N1",
            destination="N2",
            product="P1",
            capacity=100.0,
        )
    )

    state.add_bid(
        Bid(
            id="B1",
            owner_id="S1",
            owner_type="supplier",
            product_id="P1",
            price=10.0,
            quantity=100.0,
        )
    )

    state.add_bid(
        Bid(
            id="B2",
            owner_id="C1",
            owner_type="consumer",
            product_id="P1",
            price=20.0,
            quantity=50.0,
        )
    )

    return state


def register_gemini_provider():
    """Register Gemini as the explanation generator while keeping rule-based intent/parser."""
    gemini_provider = GeminiExplanationProvider()

    class GeminiLLMProvider(RuleBasedProvider):
        def get_explanation_generator(self):
            return gemini_provider

    registry = LLMProviderRegistry.get_instance()
    registry.set_provider(
        GeminiLLMProvider(
            intent_router=IntentRouter(),
            parse_function=parse_supply_chain_text,
            generate_function=generate_response,
        )
    )
    return gemini_provider


def demo_with_gemini() -> None:
    """Run the chatbot with Gemini explanations enabled."""
    print("\n" + "=" * 60)
    print("DEMO: Supply Chain Chatbot with Gemini Explanations")
    print("=" * 60)

    setup_gemini_api_key()
    state = create_minimal_supply_chain_problem()

    print(
        f"\nCreated problem with {len(state.nodes)} nodes, "
        f"{len(state.suppliers)} supplier(s), {len(state.consumers)} consumer(s)"
    )

    try:
        register_gemini_provider()
        print("GeminiExplanationProvider initialized and registered.")
    except Exception as e:
        print(f"Could not initialize Gemini provider: {e}")
        print("The system will fall back to the rule-based explanation generator.")

    result = run_chatbot_session(
        state=state,
        user_message="Explain the current supply chain problem",
        mode="guided",
        use_llm=True,
    )

    print(f"\nIntent detected: {result.get('intent')}")
    print(f"Success: {result.get('success')}")
    print(f"\nResponse:\n{result.get('response')}")


def demo_fallback_without_api_key() -> None:
    """Demonstrate automatic fallback when GEMINI_API_KEY is missing."""
    print("\n" + "=" * 60)
    print("DEMO: Fallback to Rule-Based Explanations")
    print("=" * 60)

    original_key = os.environ.pop("GEMINI_API_KEY", None)
    LLMProviderRegistry.get_instance().reset()

    state = create_minimal_supply_chain_problem()

    print("Running without GEMINI_API_KEY. use_llm=True should trigger fallback.")

    result = run_chatbot_session(
        state=state,
        user_message="Explain the current supply chain problem",
        mode="guided",
        use_llm=True,
    )

    print(f"\nIntent detected: {result.get('intent')}")
    print(f"Success: {result.get('success')}")
    print(f"\nFallback response:\n{result.get('response')}")

    if original_key is not None:
        os.environ["GEMINI_API_KEY"] = original_key
        print("\nGEMINI_API_KEY restored.")


if __name__ == "__main__":
    print("Supply Chain Chatbot Colab Demo")
    print("This script demonstrates Gemini integration with graceful fallback.")

    demo_with_gemini()
    demo_fallback_without_api_key()

    print("\n" + "=" * 60)
    print("Demo completed.")
    print("In Colab, you can paste this logic into notebook cells directly.")
    print("=" * 60)