"""LLM-first interpretation for natural-language market descriptions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .llm_adapter import LLMProviderRegistry
from .model_builder import build_market_instance
from .schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology, TransportLink


SEMANTIC_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "problem_title": {"type": "string"},
        "nodes": {"type": "array"},
        "products": {"type": "array"},
        "suppliers": {"type": "array"},
        "consumers": {"type": "array"},
        "transport_links": {"type": "array"},
        "bids": {"type": "array"},
        "technologies": {"type": "array"},
    },
    "required": ["problem_title"],
}


def interpret_problem_from_text(text: str) -> Dict[str, Any]:
    """Interpret a natural-language description into state plus machine instance."""

    if not text or not text.strip():
        raise ValueError("Cannot interpret empty problem description")

    llm_provider = os.getenv("LLM_PROVIDER")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not llm_provider or llm_provider.lower() != "gemini":
        raise RuntimeError(
            "LLM interpretation requires explicit configuration. "
            "Set LLM_PROVIDER=gemini and GEMINI_API_KEY=<your-key> environment variables."
        )
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required for LLM interpretation.")

    registry = LLMProviderRegistry.get_instance()
    provider = registry.get_provider()
    llm_generator = provider.get_explanation_generator()
    context = {
        "type": "llm_interpretation",
        "user_message": text,
        "interpretation_request": "Extract structured supply chain problem from natural language",
        "expected_format": "Valid JSON only, no explanations",
    }

    try:
        original_build_prompt = getattr(llm_generator, "_build_prompt", None)
        if original_build_prompt is not None:
            llm_generator._build_prompt = lambda mode, ctx: _build_interpretation_prompt(text)
        response_text = llm_generator.generate("full", context)
        if original_build_prompt is not None:
            llm_generator._build_prompt = original_build_prompt

        if not response_text or not str(response_text).strip():
            raise RuntimeError("LLM returned empty response")

        semantic_plan = json.loads(str(response_text).strip())
        _validate_semantic_plan(semantic_plan)
        problem_state = build_state_from_semantic_plan(semantic_plan)
        market_instance = build_market_instance(problem_state)
        narrative_interpretation = _generate_narrative_interpretation(semantic_plan)

        result = dict(semantic_plan)
        result.update(
            {
                "semantic_plan": semantic_plan,
                "problem_state": problem_state,
                "market_instance": market_instance,
                "narrative_interpretation": narrative_interpretation,
            }
        )
        return result
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc}") from exc
    except Exception:
        raise


def _generate_narrative_interpretation(semantic_plan: Dict[str, Any]) -> str:
    title = semantic_plan.get("problem_title", "Supply Chain Problem")
    nodes = semantic_plan.get("nodes", [])
    products = semantic_plan.get("products", [])
    suppliers = semantic_plan.get("suppliers", [])
    consumers = semantic_plan.get("consumers", [])
    transport_links = semantic_plan.get("transport_links", [])
    technologies = semantic_plan.get("technologies", [])
    bids = semantic_plan.get("bids", [])

    lines = [f"I interpreted this as '{title}'."]
    if nodes:
        lines.append(f"Nodes: {', '.join(node.get('name', node['id']) for node in nodes)}.")
    if products:
        lines.append(f"Products: {', '.join(product.get('name', product['id']) for product in products)}.")
    if suppliers or consumers:
        lines.append(f"Participants: {len(suppliers)} supplier(s) and {len(consumers)} consumer(s).")
    if transport_links:
        lines.append(f"Transport links: {len(transport_links)}.")
    if technologies:
        lines.append(f"Technologies: {len(technologies)} transformation option(s).")
    if bids:
        negative_bid_count = sum(1 for bid in bids if float(bid.get("price", 0)) < 0)
        if negative_bid_count:
            lines.append(f"Negative bids detected: {negative_bid_count}.")
    lines.append("This interpretation can now be converted into a solver-grounded market instance.")
    return " ".join(lines)


def _build_interpretation_prompt(text: str) -> str:
    return f"""You are an expert at extracting structured supply chain optimization problems from natural language descriptions.

Output ONLY valid JSON. No prose, no markdown, no code fences.

Requirements:
- Capture the user meaning faithfully.
- Allow negative bid prices.
- Allow transformation technologies with positive and negative yield coefficients.
- Keep the structure lightweight and solver-ready.
- Use null for unknown capacities or quantities.

Required JSON shape:
{{
  "problem_title": "string",
  "nodes": [{{"id": "N1", "name": "optional"}}],
  "products": [{{"id": "P1", "name": "optional"}}],
  "suppliers": [{{"id": "S1", "node": "N1", "product": "P1", "capacity": 10.0}}],
  "consumers": [{{"id": "C1", "node": "N2", "product": "P1", "capacity": 10.0}}],
  "transport_links": [{{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 10.0}}],
  "bids": [{{"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 1.0, "quantity": 10.0}}],
  "technologies": [{{"id": "K1", "node": "N1", "capacity": 5.0, "yield_coefficients": {{"P1": -1.0, "P2": 0.8}}}}]
}}

Problem description:
{text}
"""


def _validate_semantic_plan(plan: Dict[str, Any]) -> None:
    for key in SEMANTIC_PLAN_SCHEMA["required"]:
        if key not in plan:
            raise ValueError(f"Missing required key: {key}")

    array_keys = ["nodes", "products", "suppliers", "consumers", "transport_links", "bids", "technologies"]
    for key in array_keys:
        if key in plan and not isinstance(plan[key], list):
            raise ValueError(f"Expected list for {key}, got {type(plan[key])}")

    all_ids = set()
    for key in array_keys:
        for entity in plan.get(key, []):
            entity_id = entity.get("id")
            if entity_id is None:
                continue
            if entity_id in all_ids:
                raise ValueError(f"Duplicate ID found: {entity_id}")
            all_ids.add(entity_id)


def build_state_from_semantic_plan(plan: Dict[str, Any]) -> ProblemState:
    state = ProblemState(problem_title=plan.get("problem_title", "LLM-Interpreted Problem"))

    for node_data in plan.get("nodes", []):
        state.add_node(Node(id=node_data["id"], name=node_data.get("name")))
    for product_data in plan.get("products", []):
        state.add_product(Product(id=product_data["id"], name=product_data.get("name")))
    for supplier_data in plan.get("suppliers", []):
        state.add_supplier(
            Supplier(
                id=supplier_data["id"],
                node=supplier_data["node"],
                product=supplier_data["product"],
                capacity=supplier_data.get("capacity"),
            )
        )
    for consumer_data in plan.get("consumers", []):
        state.add_consumer(
            Consumer(
                id=consumer_data["id"],
                node=consumer_data["node"],
                product=consumer_data["product"],
                capacity=consumer_data.get("capacity"),
            )
        )
    for transport_data in plan.get("transport_links", []):
        state.add_transport(
            TransportLink(
                id=transport_data["id"],
                origin=transport_data["origin"],
                destination=transport_data["destination"],
                product=transport_data["product"],
                capacity=transport_data.get("capacity"),
            )
        )
    for technology_data in plan.get("technologies", []):
        state.add_technology(
            Technology(
                id=technology_data["id"],
                node=technology_data["node"],
                capacity=technology_data.get("capacity"),
                yield_coefficients=technology_data.get("yield_coefficients", {}),
            )
        )
    for bid_data in plan.get("bids", []):
        state.add_bid(
            Bid(
                id=bid_data["id"],
                owner_id=bid_data["owner_id"],
                owner_type=bid_data["owner_type"],
                product_id=bid_data["product_id"],
                price=bid_data["price"],
                quantity=bid_data.get("quantity"),
            )
        )

    return state


__all__ = [
    "SEMANTIC_PLAN_SCHEMA",
    "build_state_from_semantic_plan",
    "interpret_problem_from_text",
]
