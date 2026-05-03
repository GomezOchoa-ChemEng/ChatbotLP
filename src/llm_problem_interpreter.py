"""LLM-first interpretation for natural-language market descriptions."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict

from .llm_adapter import ensure_gemini_provider
from .model_builder import build_market_instance
from .schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology, TransportLink


SEMANTIC_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "problem_title": {"type": "string"},
        "problem_type": {"type": "string"},
        "nodes": {"type": "array"},
        "products": {"type": "array"},
        "suppliers": {"type": "array"},
        "consumers": {"type": "array"},
        "transport_links": {"type": "array"},
        "bids": {"type": "array"},
        "technologies": {"type": "array"},
        "missing_information": {"type": "array"},
        "ambiguities": {"type": "array"},
    },
    "required": ["problem_title"],
}


ENTITY_KEYS = ("nodes", "products", "suppliers", "consumers", "transport_links", "bids", "technologies")
ENTITY_PREFIXES = {
    "nodes": "N",
    "products": "P",
    "suppliers": "S",
    "consumers": "C",
    "transport_links": "T",
    "bids": "B",
    "technologies": "K",
}


def interpret_problem_from_text(text: str) -> Dict[str, Any]:
    """Interpret a natural-language description into state plus machine instance."""

    if not text or not text.strip():
        raise ValueError("Cannot interpret empty problem description")

    provider = ensure_gemini_provider()
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
        try:
            response_text = llm_generator.generate("full", context)
        finally:
            if original_build_prompt is not None:
                llm_generator._build_prompt = original_build_prompt

        if not response_text or not str(response_text).strip():
            raise RuntimeError("LLM returned empty response")

        semantic_plan = normalize_semantic_plan(json.loads(str(response_text).strip()))
        _validate_semantic_plan(semantic_plan)
        return build_problem_artifacts_from_semantic_plan(semantic_plan)
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
    missing_information = semantic_plan.get("missing_information", [])
    ambiguities = semantic_plan.get("ambiguities", [])

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
    if missing_information:
        lines.append(f"Still missing: {', '.join(str(item) for item in missing_information[:3])}.")
    if ambiguities:
        lines.append(f"Ambiguities noted: {', '.join(str(item) for item in ambiguities[:2])}.")
    lines.append("This interpretation can now be converted into a solver-grounded market instance.")
    return " ".join(lines)


def _build_interpretation_prompt(text: str) -> str:
    return f"""You are an expert at extracting structured supply chain optimization problems from natural language descriptions.

Output ONLY valid JSON. No prose, no markdown, no code fences.

Requirements:
- Capture the user meaning faithfully.
- Prioritize a Case A style coordinated clearing interpretation unless the user clearly specifies negative bids or transformation.
- Preserve every explicit numeric value exactly as written, including negative signs.
- Do not invent extra nodes, products, links, bids, or technologies just to make the model complete.
- If something important is missing or ambiguous, keep the missing field as null and record it in missing_information or ambiguities.
- Put per-unit transportation costs on transport_links[].cost. Do not encode transport costs as supplier or consumer bids.
- Allow negative bid prices.
- Allow transformation technologies with positive and negative yield coefficients.
- Keep the structure lightweight and solver-ready.
- Use null for unknown capacities or quantities.

Required JSON shape:
{{
  "problem_title": "string",
  "problem_type": "case_a|case_b|case_c|mixed|unknown",
  "nodes": [{{"id": "N1", "name": "optional"}}],
  "products": [{{"id": "P1", "name": "optional"}}],
  "suppliers": [{{"id": "S1", "node": "N1", "product": "P1", "capacity": 10.0}}],
  "consumers": [{{"id": "C1", "node": "N2", "product": "P1", "capacity": 10.0}}],
  "transport_links": [{{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": 10.0, "cost": 0.0}}],
  "bids": [{{"id": "B1", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 1.0, "quantity": 10.0}}],
  "technologies": [{{"id": "K1", "node": "N1", "capacity": 5.0, "yield_coefficients": {{"P1": -1.0, "P2": 0.8}}}}],
  "missing_information": ["short string"],
  "ambiguities": ["short string"]
}}

Problem description:
{text}
"""


def _validate_semantic_plan(plan: Dict[str, Any]) -> None:
    for key in SEMANTIC_PLAN_SCHEMA["required"]:
        if key not in plan:
            raise ValueError(f"Missing required key: {key}")

    for key in ENTITY_KEYS + ("missing_information", "ambiguities"):
        if key in plan and not isinstance(plan[key], list):
            raise ValueError(f"Expected list for {key}, got {type(plan[key])}")

    all_ids = set()
    for key in ENTITY_KEYS:
        for entity in plan.get(key, []):
            entity_id = entity.get("id")
            if entity_id is None:
                continue
            if entity_id in all_ids:
                raise ValueError(f"Duplicate ID found: {entity_id}")
            all_ids.add(entity_id)


def build_state_from_semantic_plan(plan: Dict[str, Any]) -> ProblemState:
    normalized_plan = normalize_semantic_plan(plan)
    state = ProblemState(problem_title=normalized_plan.get("problem_title", "LLM-Interpreted Problem"))

    node_aliases: Dict[str, str] = {}
    product_aliases: Dict[str, str] = {}
    supplier_aliases: Dict[str, str] = {}
    consumer_aliases: Dict[str, str] = {}
    transport_aliases: Dict[str, str] = {}
    technology_aliases: Dict[str, str] = {}

    for node_data in normalized_plan.get("nodes", []):
        node = Node(id=node_data["id"], name=node_data.get("name"))
        state.add_node(node)
        _register_alias(node_aliases, node.id, node.name)
    for product_data in normalized_plan.get("products", []):
        product = Product(id=product_data["id"], name=product_data.get("name"))
        state.add_product(product)
        _register_alias(product_aliases, product.id, product.name)
    for supplier_data in normalized_plan.get("suppliers", []):
        supplier = Supplier(
            id=supplier_data["id"],
            node=_resolve_reference(supplier_data.get("node"), node_aliases),
            product=_resolve_reference(supplier_data.get("product"), product_aliases),
            capacity=supplier_data.get("capacity"),
        )
        state.add_supplier(
            supplier
        )
        _register_alias(supplier_aliases, supplier.id, supplier_data.get("name"))
    for consumer_data in normalized_plan.get("consumers", []):
        consumer = Consumer(
            id=consumer_data["id"],
            node=_resolve_reference(consumer_data.get("node"), node_aliases),
            product=_resolve_reference(consumer_data.get("product"), product_aliases),
            capacity=consumer_data.get("capacity"),
        )
        state.add_consumer(
            consumer
        )
        _register_alias(consumer_aliases, consumer.id, consumer_data.get("name"))
    for transport_data in normalized_plan.get("transport_links", []):
        transport = TransportLink(
            id=transport_data["id"],
            origin=_resolve_reference(transport_data.get("origin"), node_aliases),
            destination=_resolve_reference(transport_data.get("destination"), node_aliases),
            product=_resolve_reference(transport_data.get("product"), product_aliases),
            capacity=transport_data.get("capacity"),
            cost=transport_data.get("cost", 0.0) or 0.0,
        )
        state.add_transport(
            transport
        )
        _register_alias(transport_aliases, transport.id, transport_data.get("name"))
    for technology_data in normalized_plan.get("technologies", []):
        technology = Technology(
            id=technology_data["id"],
            node=_resolve_reference(technology_data.get("node"), node_aliases),
            capacity=technology_data.get("capacity"),
            yield_coefficients={
                _resolve_reference(product_id, product_aliases): coefficient
                for product_id, coefficient in technology_data.get("yield_coefficients", {}).items()
            },
        )
        state.add_technology(
            technology
        )
        _register_alias(technology_aliases, technology.id, technology_data.get("name"))
    owner_aliases = {
        "supplier": supplier_aliases,
        "consumer": consumer_aliases,
        "transport": transport_aliases,
        "technology": technology_aliases,
    }
    for bid_data in normalized_plan.get("bids", []):
        owner_type = bid_data["owner_type"]
        state.add_bid(
            Bid(
                id=bid_data["id"],
                owner_id=_resolve_reference(bid_data.get("owner_id"), owner_aliases.get(owner_type, {})),
                owner_type=owner_type,
                product_id=_resolve_reference(bid_data.get("product_id"), product_aliases),
                price=bid_data["price"],
                quantity=bid_data.get("quantity"),
            )
        )

    return state


def build_problem_artifacts_from_semantic_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    semantic_plan = normalize_semantic_plan(plan)
    problem_state = build_state_from_semantic_plan(semantic_plan)
    market_instance = build_market_instance(problem_state)
    return {
        **semantic_plan,
        "semantic_plan": semantic_plan,
        "problem_state": problem_state,
        "market_instance": market_instance,
        "narrative_interpretation": _generate_narrative_interpretation(semantic_plan),
        "state_summary": summarize_problem_state(problem_state),
    }


def normalize_semantic_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(plan)
    normalized.setdefault("problem_title", "LLM-Interpreted Problem")
    normalized.setdefault("problem_type", "unknown")
    normalized.setdefault("missing_information", [])
    normalized.setdefault("ambiguities", [])

    for key in ENTITY_KEYS:
        items = normalized.get(key) or []
        if not isinstance(items, list):
            continue
        normalized[key] = items
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            item.setdefault("id", _generate_entity_id(key, index, item))
            if key == "technologies":
                item.setdefault("yield_coefficients", {})

    return normalized


def summarize_problem_state(state: ProblemState) -> Dict[str, Any]:
    return {
        "problem_title": state.problem_title,
        "counts": {
            "nodes": len(state.nodes),
            "products": len(state.products),
            "suppliers": len(state.suppliers),
            "consumers": len(state.consumers),
            "transport_links": len(state.transport_links),
            "bids": len(state.bids),
            "technologies": len(state.technologies),
        },
        "node_ids": state.node_ids(),
        "product_ids": state.product_ids(),
    }


def _generate_entity_id(entity_key: str, index: int, item: Dict[str, Any]) -> str:
    explicit_name = item.get("name")
    if explicit_name:
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", str(explicit_name)).strip("_")
        if sanitized:
            return sanitized
    return f"{ENTITY_PREFIXES[entity_key]}{index}"


def _register_alias(alias_map: Dict[str, str], entity_id: str, name: Any) -> None:
    alias_map[str(entity_id)] = str(entity_id)
    alias_map[str(entity_id).lower()] = str(entity_id)
    if name:
        alias_map[str(name)] = str(entity_id)
        alias_map[str(name).lower()] = str(entity_id)


def _resolve_reference(value: Any, alias_map: Dict[str, str]) -> Any:
    if value is None:
        return None
    return alias_map.get(value, alias_map.get(str(value).lower(), value))


__all__ = [
    "SEMANTIC_PLAN_SCHEMA",
    "build_problem_artifacts_from_semantic_plan",
    "build_state_from_semantic_plan",
    "interpret_problem_from_text",
    "normalize_semantic_plan",
    "summarize_problem_state",
]
