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
from typing import Dict, Any

from .schema import ProblemState
from .parser import parse_supply_chain_text
from .validator import validate_state
from .model_builder import build_model_from_state
from .solver import solve_model
from .theorem_checker import check_theorems
from .scenario_engine import run_scenario
from .response_generator import generate_response


class IntentRouter:
    """Rule-based intent router for chatbot requests.

    Uses simple keyword matching to classify user intent. Can be easily
    replaced with an LLM-based router later.
    """

    def __init__(self):
        self.intent_patterns = [
            ("solve", re.compile(r"\b(solve|solution|objective|optimize)\b", re.IGNORECASE)),
            ("scenario", re.compile(r"\b(scenario|what.if|what if|modify|change|parameter)\b", re.IGNORECASE)),
            ("theorem_check", re.compile(r"\b(theorem|case [abc]|assumption|applicability)\b", re.IGNORECASE)),
            ("validation", re.compile(r"\b(validate|check\s+(issues|status|parameters)|ready|issues)\b", re.IGNORECASE)),
            ("explanation", re.compile(r"\b(explain|help|hint|guide|how|why|tutorial)\b", re.IGNORECASE)),
            (
                "problem_formulation",
                re.compile(
                    r"\b(add|define|node|product|supplier|consumer|technology|formulate|bid)\b",
                    re.IGNORECASE,
                ),
            ),
        ]

    def detect_intent(self, text: str) -> str:
        """Detect the primary intent from user text."""
        for intent, pattern in self.intent_patterns:
            if pattern.search(text):
                return intent
        return "explanation"


def incorporate_parsed_entities(state: ProblemState, parsed_entities: Dict[str, list]) -> None:
    """Add parsed entities from text to the given ProblemState."""
    from .schema import Node, Product, Supplier, Consumer, TransportLink, Technology, Bid

    for node_data in parsed_entities.get("nodes", []):
        state.add_node(Node(**node_data))

    for prod_data in parsed_entities.get("products", []):
        state.add_product(Product(**prod_data))

    for supp_data in parsed_entities.get("suppliers", []):
        state.add_supplier(Supplier(**supp_data))

    for cons_data in parsed_entities.get("consumers", []):
        state.add_consumer(Consumer(**cons_data))

    for trans_data in parsed_entities.get("transport_links", []):
        state.add_transport(TransportLink(**trans_data))

    for tech_data in parsed_entities.get("technologies", []):
        state.add_technology(Technology(**tech_data))

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
        use_llm: Whether to use the registered LLM provider for supported tasks.

    Returns:
        A dictionary containing:
            - "state": updated ProblemState
            - "response": user-facing explanation
            - "intent": detected intent
            - "success": whether processing succeeded
    """
    router = IntentRouter()

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
            parsed = parse_supply_chain_text(user_message, use_llm=use_llm)
            if any(parsed.values()):
                incorporate_parsed_entities(state, parsed)

                context = {
                    "type": "problem_formulation",
                    "user_message": user_message,
                    "intent": intent,
                    "problem_state": state,
                }
                result["response"] = generate_response(mode, context, use_llm=use_llm)
                result["success"] = True
            else:
                result["response"] = (
                    "Could not extract entities from the description. "
                    "Try describing nodes, products, suppliers, consumers, and bids."
                )
                result["success"] = False

        elif intent == "validation":
            diag = validate_state(state)
            context = {
                "type": "validation",
                "user_message": user_message,
                "intent": intent,
                "problem_state": state,
                "validation_result": diag,
            }
            result["response"] = generate_response(mode, context, use_llm=use_llm)
            result["success"] = True

        elif intent == "solve":
            model = build_model_from_state(state)
            solve_result = solve_model(model)

            context = {
                "type": "solve",
                "user_message": user_message,
                "intent": intent,
                "problem_state": state,
                "solve_result": solve_result,
            }

            result["response"] = generate_response(mode, context, use_llm=use_llm)
            result["success"] = bool(solve_result.get("success", False))

        elif intent == "theorem_check":
            checks = check_theorems(state)
            context = {
                "type": "theorem_check",
                "user_message": user_message,
                "intent": intent,
                "problem_state": state,
                "theorem_checks": [
                    {
                        "theorem_name": c.theorem_name,
                        "applies": c.applies,
                        "explanation": c.explanation,
                    }
                    for c in checks
                ],
            }
            result["response"] = generate_response(mode, context, use_llm=use_llm)
            result["success"] = True

        elif intent == "scenario":
            change_spec = {}

            capacity_match = re.search(
                r"(?:capacity|cap)\s+to\s+(\d+)",
                user_message,
                re.IGNORECASE,
            )
            if capacity_match:
                new_cap = int(capacity_match.group(1))
                if state.suppliers:
                    change_spec["suppliers"] = [
                        {"id": state.suppliers[0].id, "capacity": new_cap}
                    ]

            price_match = re.search(
                r"(?:price|cost)\s+to\s+([\d.-]+)",
                user_message,
                re.IGNORECASE,
            )
            if price_match:
                new_price = float(price_match.group(1))
                if state.bids:
                    change_spec["bids"] = [
                        {"id": state.bids[0].id, "price": new_price}
                    ]

            if change_spec:
                change_spec["name"] = "user_scenario"
                change_spec["description"] = user_message
                scen_results = run_scenario(state, change_spec, solve=False)

                context = {
                    "type": "scenario",
                    "user_message": user_message,
                    "intent": intent,
                    "problem_state": state,
                    "scenario_result": scen_results,
                }
                result["response"] = generate_response(mode, context, use_llm=use_llm)
                result["success"] = True
            else:
                result["response"] = (
                    "Could not parse scenario parameters. "
                    "Try: 'increase capacity to 20' or 'change price to 5.0'"
                )
                result["success"] = False

        else:
            context = {
                "type": "explanation",
                "user_message": user_message,
                "intent": intent,
                "problem_state": state,
                "validation_result": {
                    "missing_parameters": state.missing_parameters,
                    "solver_ready": state.solver_ready(),
                },
            }
            result["response"] = generate_response(mode, context, use_llm=use_llm)
            result["success"] = True

    except Exception as e:
        result["response"] = f"Error processing request: {str(e)}"
        result["success"] = False

    return result


__all__ = ["IntentRouter", "run_chatbot_session"]