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

import logging
import os
import re
from typing import Dict, Any

from .schema import ProblemState
from .llm_problem_interpreter import interpret_problem_from_text
from .parser import parse_supply_chain_text
from .validator import validate_state
from .model_builder import build_market_instance, build_model_from_market_instance, build_model_from_state
from .solver import solve_model
from .solver_results import SolverResults
from .theorem_checker import check_theorems
from .scenario_engine import (
    extract_scenario_request,
    run_scenario,
    summarize_scenario_results,
)
from .response_generator import generate_response_with_metadata
from .formal_context_builder import (
    build_formal_math_context,
    identify_formal_math_request,
)
from .math_response_generator import generate_math_response
from .math_response_generator import MathResponseGenerator
from .proof_validator import context_is_structurally_usable, validate_formal_math_context
from .sampat_reasoning_engine import SampatReasoningEngine
from .domain.sampat2019 import get_theorem_metadata
from .llm_adapter import LLMConfigurationError


LOGGER = logging.getLogger(__name__)


def _default_response_metadata(mode: str, grounding_mode: str = "paper") -> Dict[str, Any]:
    return {
        "response_source": "deterministic",
        "fallback_triggered": False,
        "raw_llm_output_present": False,
        "llm_output_length": 0,
        "fallback_reason": None,
        "llm_exception_type": None,
        "grounding_warning_applied": False,
        "validation_warnings": [],
        "validation_fatal": [],
        "mode_used": mode,
        "grounding_mode": grounding_mode,
    }


class IntentRouter:
    """Rule-based intent router for chatbot requests.

    Uses simple keyword matching to classify user intent. Can be easily
    replaced with an LLM-based router later.
    """

    def __init__(self):
        self.intent_patterns = [
            (
                "formal_math",
                re.compile(
                    r"\b(dual|proof|prove|latex|theorem\s+\d+|section\s+2\.3|negative\s+bid|negative\s+price)\b",
                    re.IGNORECASE,
                ),
            ),
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
        if self._looks_like_solver_grounded_scenario(text):
            return "scenario"
        if self._looks_like_problem_description(text):
            return "problem_formulation"
        if self._looks_like_explanation_or_comparison(text):
            formal_request = identify_formal_math_request(text)
            lowered = text.lower()
            math_interpretation_tokens = [
                "dual variable",
                "dual variables",
                "dual problem",
                "primal",
                "section 2.3",
                "strong duality",
                "complementary slackness",
            ]
            if formal_request["request_type"] != "general_math_explanation" or any(
                token in lowered for token in math_interpretation_tokens
            ):
                return "formal_math"
            return "explanation"
        for intent, pattern in self.intent_patterns:
            if pattern.search(text):
                return intent
        return "explanation"

    def _looks_like_problem_description(self, text: str) -> bool:
        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "explain",
                "why",
                "how do",
                "how does",
                "compare",
                "proof",
                "theorem",
                "validate",
                "check status",
                "solve the model",
                "objective value",
            )
        ):
            return False

        entity_markers = sum(
            token in lowered
            for token in (
                "supplier",
                "consumer",
                "buyer",
                "seller",
                "node",
                "location",
                "product",
                "transport",
                "link",
                "capacity",
                "price",
                "bid",
                "units",
            )
        )
        structure_markers = sum(
            phrase in lowered
            for phrase in (
                "there is",
                "there are",
                "at node",
                "from ",
                "to ",
                "willing to pay",
                "can supply",
                "can consume",
                "up to",
            )
        )
        numeric_markers = bool(re.search(r"-?\d+(?:\.\d+)?", text))
        return (entity_markers >= 3 and numeric_markers) or (entity_markers >= 2 and structure_markers >= 2)

    def _looks_like_solver_grounded_scenario(self, text: str) -> bool:
        lowered = text.lower()
        scenario_markers = [
            "what happens if",
            "how would",
            "if ",
            "increase",
            "decrease",
            "drop from",
            "change from",
            "changes from",
            "set ",
            "unavailable",
            "disable",
            "remove",
            "reduce",
        ]
        parameter_markers = [
            "capacity",
            "bid",
            "price",
            "cost",
            "technology",
            "transformation",
            "transport",
            "supplier",
            "consumer",
        ]
        outcome_markers = [
            "solution",
            "optimal",
            "flows",
            "prices",
            "objective",
            "binding",
            "accepted",
            "compare",
        ]
        return (
            any(marker in lowered for marker in scenario_markers)
            and any(marker in lowered for marker in parameter_markers)
        ) or (
            "what if" in lowered
            and any(marker in lowered for marker in parameter_markers)
        ) or (
            any(marker in lowered for marker in ["what happens if", "how would"])
            and any(marker in lowered for marker in parameter_markers)
        ) or (
            any(marker in lowered for marker in scenario_markers)
            and any(marker in lowered for marker in parameter_markers)
            and any(marker in lowered for marker in outcome_markers)
        )

    def _looks_like_explanation_or_comparison(self, text: str) -> bool:
        lowered = text.lower()
        explanation_tokens = [
            "explain",
            "how do",
            "how does",
            "why",
            "meaning",
            "interpret",
            "interpretation",
            "role",
            "compare",
        ]
        theorem_only_tokens = ["theorem", "proof", "prove", "applicability", "assumption"]
        return any(token in lowered for token in explanation_tokens) and not any(
            token in lowered for token in theorem_only_tokens
        )


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


def normalize_solve_result(solve_result: Any) -> Dict[str, Any]:
    """Normalize solver output into a dictionary.

    Supports either:
    - a dict returned directly by solve_model(), or
    - an object-like result with attributes such as success, status,
      objective_value, message, solver_time, solution, and
      termination_condition.
    """
    if isinstance(solve_result, dict):
        return solve_result

    return {
        "success": bool(getattr(solve_result, "success", False)),
        "status": getattr(solve_result, "status", None),
        "objective_value": getattr(solve_result, "objective_value", None),
        "message": getattr(solve_result, "message", None),
        "solver_time": getattr(solve_result, "solver_time", None),
        "solution": getattr(solve_result, "solution", {}),
        "termination_condition": getattr(solve_result, "termination_condition", None),
        "raw_result": solve_result,
    }


def _summarize_validation_gaps(validation: Dict[str, Any]) -> str:
    missing = list(validation.get("missing_parameters", []))
    invalid = list(validation.get("invalid_references", []))
    incomplete = list(validation.get("incomplete_technologies", []))

    issues = missing + invalid
    if incomplete:
        issues.extend(f"technology:{tech_id} is incomplete" for tech_id in incomplete)

    if not issues:
        return "The prose description was interpreted, but it is not solver-ready yet."

    preview = "\n".join(f"- {issue}" for issue in issues[:6])
    suffix = "\n- ..." if len(issues) > 6 else ""
    return (
        "I interpreted the prose into a structured problem, but some information is still missing or inconsistent:\n"
        f"{preview}{suffix}\n"
        "Add the missing capacities, bid details, or references and I can try solving it."
    )


def _gemini_runtime_was_explicitly_requested() -> bool:
    provider_name = os.getenv("LLM_PROVIDER", "").strip().lower()
    return provider_name == "gemini" or bool(os.getenv("GEMINI_API_KEY"))


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
        mode: Response mode ("hint", "guided", "full", or "exploration").
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
        "response_metadata": _default_response_metadata(mode),
    }
    include_reference = mode == "exploration"

    try:
        reasoning_engine = SampatReasoningEngine()
        if reasoning_engine.should_handle(user_message, intent):
            reasoning_package = reasoning_engine.build_reasoning_package(
                user_query=user_message,
                state=state,
                pedagogical_mode=mode,
            )
            response_text, render_mode, response_metadata = reasoning_engine.render_response(
                package=reasoning_package,
                state=state,
                pedagogical_mode=mode,
                use_llm=use_llm,
            )
            result["response"] = response_text
            result["sampat_reasoning_package"] = reasoning_package
            result["render_mode"] = render_mode
            result["response_metadata"] = response_metadata

            if reasoning_package.recommended_path == "math_response_generator":
                formal_context = build_formal_math_context(
                    state=state,
                    user_message=user_message,
                    pedagogical_mode=mode,
                )
                result["formal_math_context"] = formal_context
                request_type = identify_formal_math_request(user_message)["request_type"]
                validation = validate_formal_math_context(formal_context)
                theorem_supported = (
                    True
                    if not formal_context.theorem_id
                    else get_theorem_metadata(formal_context.theorem_id) is not None
                )
                if validation["fatal"] or validation["warnings"]:
                    LOGGER.debug("Formal math context validation for reasoning path: %s", validation)
                result["success"] = (
                    formal_context.semantic_plan.get(
                        "is_supported_request",
                        request_type != "general_math_explanation",
                    )
                    and context_is_structurally_usable(formal_context)
                    and theorem_supported
                )
            else:
                result["success"] = reasoning_package.can_answer

            return result

        if intent == "problem_formulation":
            # Try LLM-based interpretation first if use_llm is True
            if use_llm:
                try:
                    interpretation_result = interpret_problem_from_text(user_message)
                    semantic_plan = interpretation_result["semantic_plan"]
                    narrative_interpretation = interpretation_result["narrative_interpretation"]
                    new_state = interpretation_result["problem_state"]
                    market_instance = interpretation_result["market_instance"]
                    state_summary = interpretation_result.get("state_summary")

                    validation = validate_state(new_state)
                    result["state"] = new_state
                    result["semantic_plan"] = semantic_plan
                    result["narrative_interpretation"] = narrative_interpretation
                    result["validation_result"] = validation
                    result["market_instance"] = market_instance
                    result["state_summary"] = state_summary

                    context = {
                        "type": "problem_formulation",
                        "user_message": user_message,
                        "intent": intent,
                        "problem_state": new_state,
                        "market_instance": market_instance,
                        "semantic_plan": semantic_plan,
                        "narrative_interpretation": narrative_interpretation,
                        "validation_result": validation,
                        "state_summary": state_summary,
                    }

                    if validation["solver_ready"]:
                        model = build_model_from_market_instance(market_instance)
                        raw_solve_result = solve_model(model)
                        solve_result = normalize_solve_result(raw_solve_result)
                        solver_results = SolverResults.from_solve_result(raw_solve_result, new_state)

                        context["solve_result"] = solve_result
                        context["solver_results"] = solver_results
                        result["solve_result"] = solve_result
                        result["solver_results"] = solver_results
                        result["plotting_data"] = solver_results.plotting_data

                        result["response"], result["response_metadata"] = generate_response_with_metadata(
                            mode,
                            context,
                            use_llm=use_llm,
                            include_reference=include_reference,
                        )
                        result["success"] = True
                    else:
                        result["response"] = _summarize_validation_gaps(validation)
                        result["success"] = False

                except LLMConfigurationError as e:
                    if _gemini_runtime_was_explicitly_requested():
                        result["response"] = f"LLM interpretation configuration error: {e}"
                        result["response_metadata"]["fallback_triggered"] = False
                        result["response_metadata"]["fallback_reason"] = "llm_configuration_error"
                        result["success"] = False
                    else:
                        parsed = parse_supply_chain_text(user_message, use_llm=True)
                        if any(parsed.values()):
                            incorporate_parsed_entities(state, parsed)
                            context = {
                                "type": "problem_formulation",
                                "user_message": user_message,
                                "intent": intent,
                                "problem_state": state,
                            }
                            result["response"], result["response_metadata"] = generate_response_with_metadata(
                                mode,
                                context,
                                use_llm=use_llm,
                                include_reference=include_reference,
                            )
                            result["success"] = True
                        else:
                            result["response"] = f"LLM interpretation configuration error: {e}"
                            result["response_metadata"]["fallback_triggered"] = False
                            result["response_metadata"]["fallback_reason"] = "llm_configuration_error"
                            result["success"] = False
                except Exception as e:
                    result["response"] = f"LLM interpretation failed: {e}"
                    result["response_metadata"]["fallback_triggered"] = False
                    result["response_metadata"]["fallback_reason"] = "llm_interpretation_error"
                    result["success"] = False
            else:
                # Use rule-based parsing when LLM is disabled
                parsed = parse_supply_chain_text(user_message, use_llm=False)
                if any(parsed.values()):
                    incorporate_parsed_entities(state, parsed)

                    context = {
                        "type": "problem_formulation",
                        "user_message": user_message,
                        "intent": intent,
                        "problem_state": state,
                    }
                    result["response"], result["response_metadata"] = generate_response_with_metadata(
                        mode,
                        context,
                        use_llm=use_llm,
                        include_reference=include_reference,
                    )
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
            result["response"], result["response_metadata"] = generate_response_with_metadata(
                mode,
                context,
                use_llm=use_llm,
                include_reference=include_reference,
            )
            result["success"] = True

        elif intent == "solve":
            diag = validate_state(state)
            market_instance = build_market_instance(state)
            model = build_model_from_market_instance(market_instance)
            raw_solve_result = solve_model(model)
            solve_result = normalize_solve_result(raw_solve_result)

            # Create structured solver results
            solver_results = SolverResults.from_solve_result(raw_solve_result, state)

            solve_diag = dict(diag)
            if solve_result.get("status") == "optimal":
                solve_diag["solver_ready"] = True

            context = {
                "type": "solve",
                "user_message": user_message,
                "intent": intent,
                "problem_state": state,
                "market_instance": market_instance,
                "validation_result": solve_diag,
                "solve_result": solve_result,
                "solver_results": solver_results,
            }

            result["response"], result["response_metadata"] = generate_response_with_metadata(
                mode,
                context,
                use_llm=use_llm,
                include_reference=include_reference,
            )
            result["market_instance"] = market_instance
            result["solver_results"] = solver_results
            result["plotting_data"] = solver_results.plotting_data
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
            result["response"], result["response_metadata"] = generate_response_with_metadata(
                mode,
                context,
                use_llm=use_llm,
                include_reference=include_reference,
            )
            result["success"] = True

        elif intent == "formal_math":
            check_theorems(state)
            formal_context = build_formal_math_context(
                state=state,
                user_message=user_message,
                pedagogical_mode=mode,
            )
            math_generator = MathResponseGenerator(use_llm=use_llm)
            result["response"], result["response_metadata"] = math_generator.generate_with_metadata(formal_context)
            result["formal_math_context"] = formal_context
            result["render_mode"] = MathResponseGenerator.infer_render_mode(formal_context)
            request_type = identify_formal_math_request(user_message)["request_type"]
            validation = validate_formal_math_context(formal_context)
            theorem_supported = (
                True
                if not formal_context.theorem_id
                else get_theorem_metadata(formal_context.theorem_id) is not None
            )
            if validation["fatal"] or validation["warnings"]:
                LOGGER.debug("Formal math context validation for direct path: %s", validation)
            result["success"] = (
                formal_context.semantic_plan.get("is_supported_request", request_type != "general_math_explanation")
                and context_is_structurally_usable(formal_context)
                and theorem_supported
            )

        elif intent == "scenario":
            extraction = extract_scenario_request(state, user_message)
            result["response_mode"] = "solver_grounded_verification"
            result["scenario_extraction"] = extraction

            if extraction.get("missing"):
                missing_text = ", ".join(extraction["missing"])
                result["response"] = (
                    "Could not fully ground the scenario request.\n\n"
                    f"Missing: {missing_text}.\n"
                    "Try specifying the entity or the new value explicitly."
                )
                result["success"] = False
            else:
                scen_results = run_scenario(state, extraction["change_spec"], solve=True)
                scen_results["summary"] = summarize_scenario_results(extraction, scen_results)
                scen_results["requested_dimensions"] = extraction.get("requested_dimensions", [])

                context = {
                    "type": "scenario",
                    "user_message": user_message,
                    "intent": intent,
                    "problem_state": state,
                    "scenario_result": scen_results,
                    "scenario_extraction": extraction,
                    "response_mode": "solver_grounded_verification",
                }
                result["response"], result["response_metadata"] = generate_response_with_metadata(
                    mode,
                    context,
                    use_llm=use_llm,
                    include_reference=include_reference,
                )
                result["scenario_result"] = scen_results
                result["success"] = True

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
            result["response"], result["response_metadata"] = generate_response_with_metadata(
                mode,
                context,
                use_llm=use_llm,
                include_reference=include_reference,
            )
            result["success"] = True

    except Exception as e:
        import traceback

        result["response"] = (
            f"Error processing request: {str(e)}\n\n"
            f"TRACEBACK:\n{traceback.format_exc()}"
        )
        result["success"] = False

    return result


def ask_chatbot(
    state: ProblemState,
    prompt: str,
    use_llm: bool = False,
    mode: str = "guided",
) -> Dict[str, Any]:
    """Compatibility helper for notebook-style chatbot prompts."""

    return run_chatbot_session(
        state=state,
        user_message=prompt,
        mode=mode,
        use_llm=use_llm,
    )


__all__ = ["IntentRouter", "ask_chatbot", "run_chatbot_session", "normalize_solve_result"]
