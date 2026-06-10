"""Backend-neutral network graph specifications for supply chain instances.

The builders in this module consume the existing ``ProblemState`` and
``SolverResults`` objects without changing the optimization architecture.  They
produce a small intermediate representation that can be rendered by Graphviz,
Mermaid, or another backend later.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology, TransportLink
from .solver_results import SolverResults


NodeType = Literal["source", "sink", "technology", "intermediate"]
EdgeType = Literal["transport"]
GraphType = Literal["problem", "solution"]


class GraphNodeSpec(BaseModel):
    """Backend-independent node description for a supply chain graph."""

    id: str
    label: str
    node_type: NodeType
    products: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    display_attributes: Dict[str, str] = Field(default_factory=dict)
    solution: Dict[str, Any] = Field(default_factory=dict)
    active: Optional[bool] = None


class GraphEdgeSpec(BaseModel):
    """Backend-independent directed edge description for a supply chain graph."""

    id: str
    source: str
    target: str
    edge_type: EdgeType = "transport"
    product_id: Optional[str] = None
    product_label: Optional[str] = None
    cost: Optional[float] = None
    capacity: Optional[float] = None
    flow: Optional[float] = None
    utilization: Optional[float] = None
    active: Optional[bool] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    display_attributes: Dict[str, str] = Field(default_factory=dict)


class NetworkGraphSpec(BaseModel):
    """Backend-neutral graph representation for problem or solution views."""

    title: str
    graph_type: GraphType
    nodes: List[GraphNodeSpec] = Field(default_factory=list)
    edges: List[GraphEdgeSpec] = Field(default_factory=list)
    product_ids: List[str] = Field(default_factory=list)
    product_labels: Dict[str, str] = Field(default_factory=dict)
    objective_value: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    display_metadata: Dict[str, str] = Field(default_factory=dict)


def build_problem_graph_spec(state: ProblemState) -> NetworkGraphSpec:
    """Build a directed problem graph specification from ``ProblemState``.

    Missing numeric values remain ``None`` in raw fields and are displayed as
    ``"missing"`` in ``display_attributes``.  Explicit zero values therefore
    remain distinguishable from unknown values.
    """

    node_map = {node.id: node for node in state.nodes}
    product_labels = _product_labels(state.products)
    suppliers_by_node = _group_by_node(state.suppliers)
    consumers_by_node = _group_by_node(state.consumers)
    technologies_by_node = _group_by_node(state.technologies)
    bids_by_owner = _group_bids_by_owner(state.bids)

    referenced_node_ids = set(node_map)
    for link in state.transport_links:
        referenced_node_ids.add(link.origin)
        referenced_node_ids.add(link.destination)
    for supplier in state.suppliers:
        referenced_node_ids.add(supplier.node)
    for consumer in state.consumers:
        referenced_node_ids.add(consumer.node)
    for technology in state.technologies:
        referenced_node_ids.add(technology.node)

    nodes = [
        _build_node_spec(
            node_id=node_id,
            node=node_map.get(node_id),
            suppliers=suppliers_by_node.get(node_id, []),
            consumers=consumers_by_node.get(node_id, []),
            technologies=technologies_by_node.get(node_id, []),
            bids_by_owner=bids_by_owner,
            product_labels=product_labels,
        )
        for node_id in sorted(referenced_node_ids)
    ]

    edges = [
        _build_transport_edge_spec(link, product_labels)
        for link in state.transport_links
    ]

    return NetworkGraphSpec(
        title=state.problem_title,
        graph_type="problem",
        nodes=nodes,
        edges=edges,
        product_ids=state.product_ids(),
        product_labels=product_labels,
        metadata={
            "node_count": len(nodes),
            "transport_edge_count": len(edges),
            "technology_count": len(state.technologies),
            "supplier_count": len(state.suppliers),
            "consumer_count": len(state.consumers),
        },
    )


def build_solution_graph_spec(
    state: ProblemState,
    results: SolverResults,
    activity_tolerance: float = 1e-9,
) -> NetworkGraphSpec:
    """Build a solution graph spec by overlaying ``SolverResults`` on a problem graph."""

    spec = build_problem_graph_spec(state)
    spec.graph_type = "solution"
    spec.objective_value = results.objective_value
    spec.display_metadata["objective_value"] = _format_value(results.objective_value)

    nodes_by_id = {node.id: node for node in spec.nodes}
    edges_by_id = {edge.id: edge for edge in spec.edges}

    _overlay_bid_allocations(state, results, nodes_by_id)
    _overlay_transport_flows(state, results, edges_by_id, activity_tolerance)
    _overlay_technology_activity(state, results, nodes_by_id, spec.product_labels, activity_tolerance)

    spec.metadata.update(
        {
            "solver_name": results.solver_name,
            "solver_status": results.solver_status,
            "termination_condition": results.termination_condition,
            "success": results.success,
            "active_transport_edge_count": sum(1 for edge in spec.edges if edge.active is True),
            "active_technology_count": sum(1 for node in spec.nodes if node.active is True),
        }
    )
    return spec


def _build_node_spec(
    node_id: str,
    node: Optional[Node],
    suppliers: List[Supplier],
    consumers: List[Consumer],
    technologies: List[Technology],
    bids_by_owner: Dict[Tuple[str, str], List[Bid]],
    product_labels: Dict[str, str],
) -> GraphNodeSpec:
    node_type = _infer_node_type(bool(suppliers), bool(consumers), bool(technologies))
    products = _node_product_labels(suppliers, consumers, technologies, product_labels)
    attributes: Dict[str, Any] = {
        "node_id": node_id,
        "node_name": node.name if node is not None else None,
        "supplier_ids": [supplier.id for supplier in suppliers],
        "consumer_ids": [consumer.id for consumer in consumers],
        "technology_ids": [technology.id for technology in technologies],
        "missing_node_reference": node is None,
    }
    display_attributes: Dict[str, str] = {
        "role": node_type,
    }

    for supplier in suppliers:
        prefix = f"supplier:{supplier.id}"
        display_attributes[f"{prefix}:product"] = _label_for_product(supplier.product, product_labels)
        display_attributes[f"{prefix}:capacity"] = _format_value(supplier.capacity)
        for bid in bids_by_owner.get(("supplier", supplier.id), []):
            display_attributes[f"supplier_bid:{bid.id}:price"] = _format_value(bid.price)
            display_attributes[f"supplier_bid:{bid.id}:quantity"] = _format_value(bid.quantity)

    for consumer in consumers:
        prefix = f"consumer:{consumer.id}"
        display_attributes[f"{prefix}:product"] = _label_for_product(consumer.product, product_labels)
        display_attributes[f"{prefix}:capacity"] = _format_value(consumer.capacity)
        for bid in bids_by_owner.get(("consumer", consumer.id), []):
            display_attributes[f"consumer_bid:{bid.id}:price"] = _format_value(bid.price)
            display_attributes[f"consumer_bid:{bid.id}:quantity"] = _format_value(bid.quantity)

    for technology in technologies:
        prefix = f"technology:{technology.id}"
        display_attributes[f"{prefix}:capacity"] = _format_value(technology.capacity)
        display_attributes[f"{prefix}:cost"] = _format_value(technology.cost)
        display_attributes[f"{prefix}:fixed_cost"] = _format_value(getattr(technology, "fixed_cost", None))
        display_attributes[f"{prefix}:yields"] = _format_yields(technology.yield_coefficients, product_labels)

    return GraphNodeSpec(
        id=node_id,
        label=node.name if node is not None and node.name else node_id,
        node_type=node_type,
        products=products,
        attributes=attributes,
        display_attributes=display_attributes,
    )


def _build_transport_edge_spec(
    link: TransportLink,
    product_labels: Dict[str, str],
) -> GraphEdgeSpec:
    product_label = _label_for_product(link.product, product_labels)
    display_attributes = {
        "product": product_label,
        "cost": _format_value(link.cost),
        "capacity": _format_value(link.capacity),
    }
    return GraphEdgeSpec(
        id=link.id,
        source=link.origin,
        target=link.destination,
        product_id=link.product,
        product_label=product_label,
        cost=link.cost,
        capacity=link.capacity,
        attributes={
            "transport_id": link.id,
            "origin": link.origin,
            "destination": link.destination,
        },
        display_attributes=display_attributes,
    )


def _overlay_bid_allocations(
    state: ProblemState,
    results: SolverResults,
    nodes_by_id: Dict[str, GraphNodeSpec],
) -> None:
    supplier_nodes = {supplier.id: supplier.node for supplier in state.suppliers}
    consumer_nodes = {consumer.id: consumer.node for consumer in state.consumers}

    supply_totals: Dict[str, float] = defaultdict(float)
    demand_totals: Dict[str, float] = defaultdict(float)

    for bid in state.bids:
        quantity = _lookup_result_value(results.bid_allocations, bid.id)
        if bid.owner_type == "supplier":
            node_id = supplier_nodes.get(bid.owner_id)
            label = "accepted_supply"
            if node_id is not None and quantity is not None:
                supply_totals[node_id] += quantity
        elif bid.owner_type == "consumer":
            node_id = consumer_nodes.get(bid.owner_id)
            label = "accepted_demand"
            if node_id is not None and quantity is not None:
                demand_totals[node_id] += quantity
        else:
            continue

        if node_id is None or node_id not in nodes_by_id:
            continue
        node = nodes_by_id[node_id]
        bid_allocations = node.solution.setdefault("bid_allocations", {})
        bid_allocations[bid.id] = quantity
        node.display_attributes[f"solution:{label}:{bid.id}"] = _format_value(quantity)

    for node_id, total in supply_totals.items():
        nodes_by_id[node_id].solution["accepted_supply_total"] = total
        nodes_by_id[node_id].display_attributes["solution:accepted_supply_total"] = _format_value(total)
    for node_id, total in demand_totals.items():
        nodes_by_id[node_id].solution["accepted_demand_total"] = total
        nodes_by_id[node_id].display_attributes["solution:accepted_demand_total"] = _format_value(total)


def _overlay_transport_flows(
    state: ProblemState,
    results: SolverResults,
    edges_by_id: Dict[str, GraphEdgeSpec],
    activity_tolerance: float,
) -> None:
    for link in state.transport_links:
        edge = edges_by_id.get(link.id)
        if edge is None:
            continue
        flow = _flow_value_for_link(link, results.transport_flows)
        edge.flow = flow
        edge.display_attributes["flow"] = _format_value(flow)
        if flow is None:
            edge.active = None
            continue
        edge.active = abs(flow) > activity_tolerance
        if link.capacity is not None and link.capacity > 0:
            edge.utilization = flow / link.capacity
            edge.display_attributes["utilization"] = _format_percent(edge.utilization)


def _overlay_technology_activity(
    state: ProblemState,
    results: SolverResults,
    nodes_by_id: Dict[str, GraphNodeSpec],
    product_labels: Dict[str, str],
    activity_tolerance: float,
) -> None:
    for technology in state.technologies:
        node = nodes_by_id.get(technology.node)
        if node is None:
            continue
        activity = _lookup_result_value(results.technology_extents, technology.id)
        installation = _lookup_result_value(getattr(results, "technology_installations", {}), technology.id)
        node.solution.setdefault("technology_activity", {})[technology.id] = activity
        node.display_attributes[f"solution:technology_activity:{technology.id}"] = _format_value(activity)
        if installation is not None:
            node.solution.setdefault("technology_installation", {})[technology.id] = installation
            node.display_attributes[f"solution:technology_selected:{technology.id}"] = (
                "yes" if installation >= 0.5 else "no"
            )
        if activity is not None:
            node.active = abs(activity) > activity_tolerance
        if installation is not None:
            node.active = installation >= 0.5

        outputs: Dict[str, Optional[float]] = {}
        inputs: Dict[str, Optional[float]] = {}
        for product_id, coefficient in technology.yield_coefficients.items():
            product_label = _label_for_product(product_id, product_labels)
            if coefficient is None or activity is None:
                quantity = None
            else:
                quantity = float(coefficient) * activity
            if coefficient is not None and coefficient > 0:
                outputs[product_id] = quantity
                node.display_attributes[f"solution:technology_output:{technology.id}:{product_label}"] = (
                    _format_value(quantity)
                )
            elif coefficient is not None and coefficient < 0:
                inputs[product_id] = None if quantity is None else -quantity
                node.display_attributes[f"solution:technology_input:{technology.id}:{product_label}"] = (
                    _format_value(inputs[product_id])
                )
        node.solution["technology_outputs"] = outputs
        node.solution["technology_inputs"] = inputs


def _infer_node_type(has_supplier: bool, has_consumer: bool, has_technology: bool) -> NodeType:
    if has_technology:
        return "technology"
    if has_supplier and not has_consumer:
        return "source"
    if has_consumer and not has_supplier:
        return "sink"
    return "intermediate"


def _product_labels(products: Iterable[Product]) -> Dict[str, str]:
    return {product.id: product.name or product.id for product in products}


def _label_for_product(product_id: Optional[str], product_labels: Dict[str, str]) -> str:
    if product_id is None:
        return "missing"
    return product_labels.get(product_id, product_id)


def _node_product_labels(
    suppliers: Iterable[Supplier],
    consumers: Iterable[Consumer],
    technologies: Iterable[Technology],
    product_labels: Dict[str, str],
) -> List[str]:
    product_ids = []
    for supplier in suppliers:
        product_ids.append(supplier.product)
    for consumer in consumers:
        product_ids.append(consumer.product)
    for technology in technologies:
        product_ids.extend(technology.yield_coefficients.keys())
    labels = [_label_for_product(product_id, product_labels) for product_id in product_ids]
    return list(dict.fromkeys(labels))


def _group_by_node(items: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        grouped[item.node].append(item)
    return grouped


def _group_bids_by_owner(bids: Iterable[Bid]) -> Dict[Tuple[str, str], List[Bid]]:
    grouped: Dict[Tuple[str, str], List[Bid]] = defaultdict(list)
    for bid in bids:
        grouped[(bid.owner_type, bid.owner_id)].append(bid)
    return grouped


def _format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return str(value)
    return str(value)


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    return f"{value:.1%}"


def _format_yields(
    yield_coefficients: Dict[str, Optional[float]],
    product_labels: Dict[str, str],
) -> str:
    if not yield_coefficients:
        return "missing"
    pieces = [
        f"{_label_for_product(product_id, product_labels)}={_format_value(coefficient)}"
        for product_id, coefficient in yield_coefficients.items()
    ]
    return ", ".join(pieces)


def _lookup_result_value(values: Dict[str, float], key: str) -> Optional[float]:
    if key in values:
        return float(values[key])
    string_key = str(key)
    if string_key in values:
        return float(values[string_key])
    return None


def _flow_value_for_link(link: TransportLink, flows: Dict[str, float]) -> Optional[float]:
    if link.id in flows:
        return float(flows[link.id])

    candidate_keys = {
        str((link.origin, link.destination)),
        f"({link.origin}, {link.destination})",
        f"{link.origin}->{link.destination}",
        f"{link.origin}_to_{link.destination}",
        f"{link.origin}_to_{link.destination}:{link.product}",
    }
    for key in candidate_keys:
        if key in flows:
            return float(flows[key])

    for raw_key, value in flows.items():
        parsed = _parse_arc_key(raw_key)
        if parsed == (link.origin, link.destination):
            return float(value)
    return None


def _parse_arc_key(raw_key: str) -> Optional[Tuple[str, str]]:
    text = str(raw_key).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if ":" in text:
        text = text.split(":", 1)[0]
    if "->" in text:
        parts = text.split("->", 1)
    elif "_to_" in text:
        parts = text.split("_to_", 1)
    elif "," in text:
        parts = text.split(",", 1)
    else:
        return None
    if len(parts) != 2:
        return None
    return _clean_arc_part(parts[0]), _clean_arc_part(parts[1])


def _clean_arc_part(part: str) -> str:
    return part.strip().strip("'\"")


__all__ = [
    "GraphEdgeSpec",
    "GraphNodeSpec",
    "NetworkGraphSpec",
    "build_problem_graph_spec",
    "build_solution_graph_spec",
]
