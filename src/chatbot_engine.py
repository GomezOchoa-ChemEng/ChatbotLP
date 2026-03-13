"""Chatbot orchestration engine for coordinated supply chain optimization.

This module coordinates the full chatbot workflow by routing user requests
to the appropriate processing modules. It combines lightweight rule-based
intent detection with modular component invocation to provide an interactive
experience suitable for classroom use.

The design is intentionally simple and rule-based so that intent routing
can be easily replaced with an LLM-based approach in the future.

Main entry point is ``run_chatbot_session``, which takes a user message
and the current problem state and returns an updated state plus a
user-facing response.
"""

import re
from typing import Dict, Any, Tuple, Optional

from .schema import ProblemState, ScenarioRecord
from .parser import parse_supply_chain_text
from .state_manager import StateManager
from .validator import validate_state
from .model_builder import build_model_from_state
from .solver import solve_model
from .theorem_checker import check_theorems
from .scenario_engine import clone_state, apply_parameter_change, run_scenario
from .response_generator import generate_response


class IntentRouter:
    """Rule-based intent router for chatbot requests.

    Uses simple keyword matching to classify user intent. Can be easily
    replaced with an LLM-based router later.
    """

    def __init__(self):
        # Define keyword patterns for each intent.
        # Ordered by specificity: more specific patterns checked first.
        # Use word boundaries and more specific context to avoid overlap.
        self.intent_patterns = [
            ("solve", re.compile(r"\b(solve|solution|objective|optimize)\b", re.IGNORECASE)),
            ("scenario", re.compile(r"\b(scenario|what.if|what if|modify|change|parameter)\b", re.IGNORECASE)),
            ("theorem_check", re.compile(r"\b(theorem|case [abc]|assumption|applicability)\b", re.IGNORECASE)),
            ("validation", re.compile(r"\b(validate|check\s+(issues|status|parameters)|ready|issues)\b", re.IGNORECASE)),
            ("explanation", re.compile(r"\b(explain|help|hint|guide|how|why|tutorial)\b", re.IGNORECASE)),
            ("problem_formulation", re.compile(
                r"\b(add|define|node|product|supplier|consumer|technology|formulate|bid)\b",
                re.IGNORECASE
            )),
        ]

    def detect_intent(self, text: str) -> str:
        """Detect the primary intent from user text.

        Returns the name of the detected intent, or 'explanation' as default.
        Patterns are checked in order of specificity; first match wins.
        """
        # Apply patterns in order; first match wins
        for intent, pattern in self.intent_patterns:
            if pattern.search(text):
                return intent
        return "explanation"  # default fallback


def incorporate_parsed_entities(state: ProblemState, parsed_entities: Dict[str, list]) -> None:
    """Add parsed entities from text to the given ProblemState.

    This helper modifies the state in-place by adding nodes, products, and
    entities extracted from natural language.
    """
    from .schema import Node, Product, Supplier, Consumer, TransportLink, Technology, Bid

    # Add nodes
    for node_data in parsed_entities.get("nodes", []):
        state.add_node(Node(**node_data))

    # Add products
    for prod_data in parsed_entities.get("products", []):
        state.add_product(Product(**prod_data))

    # Add suppliers
    for supp_data in parsed_entities.get("suppliers", []):
        state.add_supplier(Supplier(**supp_data))

    # Add consumers
    for cons_data in parsed_entities.get("consumers", []):
        state.add_consumer(Consumer(**cons_data))

    # Add transport links
    for trans_data in parsed_entities.get("transport_links", []):
        state.add_transport(TransportLink(**trans_data))

    # Add technologies
    for tech_data in parsed_entities.get("technologies", []):
        state.add_technology(Technology(**tech_data))

    # Add bids
    for bid_data in parsed_entities.get("bids", []):
        state.add_bid(Bid(**bid_data))


def run_chatbot_session(
    state: ProblemState,
    user_message: str,
    mode: str = "guided",
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Execute a single chatbot session turn.

    Args:
        state: The current ProblemState.
        user_message: The user's natural language input.
        mode: Response mode ("hint", "guided", or "full").

    Returns:
        A dictionary containing:
            - "state": updated ProblemState
            - "response": user-facing explanation
            - "intent": detected intent
            - "success": whether processing succeeded
    """
    router = IntentRouter()
    # determine intent potentially using LLM classifier
    if use_llm:
        try:
            from .llm_adapter import LLMProviderRegistry

            classifier = LLMProviderRegistry.get_instance().get_intent_classifier()
            intent = classifier.classify(user_message)
        except Exception:
            intent = router.detect_intent(user_message)
    else:
        intent = router.detect_intent(user_message)
    result = {
        "state": state,
        "response": "",
        "intent": intent,
        "success": False,
    }

    try:
        if intent == "problem_formulation":
            # Parse text and incorporate entities into state
            parsed = parse_supply_chain_text(user_message, use_llm=use_llm)
            if any(parsed.values()):  # Only apply if something was parsed
                incorporate_parsed_entities(state, parsed)
                result["response"] = f"Added entities to the problem. Current state has {len(state.nodes)} node(s), {len(state.products)} product(s), {len(state.suppliers)} supplier(s), {len(state.consumers)} consumer(s)."
                result["success"] = True
            else:
                result["response"] = "Could not extract entities from the description. Try describing nodes, products, suppliers, consumers, and bids."
                result["success"] = False

        elif intent == "validation":
            # Validate the current state
            diag = validate_state(state)
            context = {
                "issues": diag.get("issues", []),
                "missing_parameters": diag.get("missing_parameters", []),
                "invalid_references": diag.get("invalid_references", []),
                "solver_ready": diag.get("solver_ready", False),
                "benchmark_compatibility": diag.get("benchmark_compatibility", {}),
            }
            response = generate_response(mode, {"type": "validation", **context}, use_llm=use_llm)
            result["response"] = response
            result["success"] = True

        elif intent == "solve":
            # Build and solve the model
            model = build_model_from_state(state)
            solve_result = solve_model(model)
            context = {
                "model_status": "built",
                "solver_status": solve_result.status,
                "objective_value": solve_result.objective_value,
                "success": solve_result.success,
                "message": solve_result.message,
                "solution": solve_result.solution,
            }
            response = generate_response(mode, {"type": "solve", **context}, use_llm=use_llm)
            result["response"] = response
            result["success"] = solve_result.success

        elif intent == "theorem_check":
            # Check theorem applicability
            checks = check_theorems(state)
            context = {
                "theorem_checks": [
                    {
                        "name": c.theorem_name,
                        "applies": c.applies,
                        "explanation": c.explanation,
                    }
                    for c in checks
                ],
            }
            response = generate_response(mode, {"type": "theorem_check", **context}, use_llm=use_llm)
            result["response"] = response
            result["success"] = True

        elif intent == "scenario":
            # Parse change spec from message (simple heuristic)
            # Look for patterns like "increase capacity to X" or "change price"
            change_spec = {}
            capacity_match = re.search(r"(?:capacity|cap)\s+to\s+(\d+)", user_message, re.IGNORECASE)
            if capacity_match:
                new_cap = int(capacity_match.group(1))
                # Apply to first supplier (simple heuristic)
                if state.suppliers:
                    change_spec["suppliers"] = [{"id": state.suppliers[0].id, "capacity": new_cap}]
            price_match = re.search(r"(?:price|cost)\s+to\s+([\d.-]+)", user_message, re.IGNORECASE)
            if price_match:
                new_price = float(price_match.group(1))
                # Apply to first bid (simple heuristic)
                if state.bids:
                    change_spec["bids"] = [{"id": state.bids[0].id, "price": new_price}]
            
            if change_spec:
                change_spec["name"] = "user_scenario"
                change_spec["description"] = user_message
                scen_results = run_scenario(state, change_spec, solve=False)
                context = {
                    "change_spec": change_spec,
                    "base_count": {"suppliers": len(state.suppliers), "bids": len(state.bids)},
                }
                response = generate_response(mode, {"type": "scenario", **context}, use_llm=use_llm)
                result["response"] = response
                result["success"] = True
            else:
                result["response"] = "Could not parse scenario parameters. Try: 'increase capacity to 20' or 'change price to 5.0'"
                result["success"] = False

        else:
            # Default: explanation/help mode
            context = {
                "current_state_summary": {
                    "nodes": len(state.nodes),
                    "products": len(state.products),
                    "suppliers": len(state.suppliers),
                    "consumers": len(state.consumers),
                    "bids": len(state.bids),
                },
            }
            response = generate_response(mode, {"type": "explanation", **context}, use_llm=use_llm)
            result["response"] = response
            result["success"] = True

    except Exception as e:
        result["response"] = f"Error processing request: {str(e)}"
        result["success"] = False

    return result


__all__ = ["IntentRouter", "run_chatbot_session"]
