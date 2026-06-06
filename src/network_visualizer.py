"""Render network graph specifications with optional backends."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, List

from .network_graph import GraphEdgeSpec, GraphNodeSpec, NetworkGraphSpec


NODE_SHAPES = {
    "source": "box",
    "sink": "doublecircle",
    "technology": "diamond",
    "intermediate": "ellipse",
}

NODE_COLORS = {
    "source": "#dcecc9",
    "sink": "#d9e8fb",
    "technology": "#ffe8bd",
    "intermediate": "#f6f6f6",
}

GRAPHVIZ_INSTALL_MESSAGE = (
    "Graphviz is not installed. Install it with: brew install graphviz and pip install graphviz."
)


def render_graphviz(spec: NetworkGraphSpec, output_path: str, format: str = "svg") -> str:
    """Render a directed network graph to a file and return the file path.

    The Python ``graphviz`` package and the system Graphviz executable are
    optional runtime dependencies.  If either is unavailable, this function
    raises ``RuntimeError`` with installation guidance instead of failing with a
    backend-specific traceback.
    """

    try:
        from graphviz import Digraph
        from graphviz.backend import ExecutableNotFound
    except ImportError as exc:
        raise RuntimeError(GRAPHVIZ_INSTALL_MESSAGE) from exc

    output = Path(output_path)
    if not output.suffix:
        output = output.with_suffix(f".{format}")
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = Digraph(
        name=_safe_graph_id(spec.title),
        graph_attr={
            "rankdir": "LR",
            "label": _graph_title(spec),
            "labelloc": "t",
            "fontsize": "18",
            "fontname": "Helvetica",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "10",
            "style": "rounded,filled",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "arrowsize": "0.8",
        },
    )

    for node in spec.nodes:
        graph.node(
            node.id,
            label=_node_label(node),
            shape=NODE_SHAPES.get(node.node_type, "ellipse"),
            fillcolor=_node_fillcolor(node),
            color=_node_border_color(node),
            penwidth="2.0" if node.active else "1.0",
        )

    for edge in spec.edges:
        attrs = _edge_style(edge)
        graph.edge(edge.source, edge.target, label=_edge_label(edge), **attrs)

    try:
        rendered = graph.pipe(format=format)
    except ExecutableNotFound as exc:
        raise RuntimeError(GRAPHVIZ_INSTALL_MESSAGE) from exc
    except Exception as exc:
        raise RuntimeError(f"Graphviz rendering failed: {exc}") from exc

    output.write_bytes(rendered)
    return str(output)


def render_mermaid(spec: NetworkGraphSpec) -> str:
    """Render a network graph specification as a Mermaid flowchart string."""

    id_map = _mermaid_id_map(spec)
    lines = [
        "---",
        f"title: {_escape_mermaid_text(_graph_title(spec))}",
        "---",
        "flowchart LR",
    ]

    for node in spec.nodes:
        node_id = id_map[node.id]
        label = _escape_mermaid_text(_compact_label(_node_label(node)))
        if node.node_type == "source":
            lines.append(f"    {node_id}[\"{label}\"]")
        elif node.node_type == "sink":
            lines.append(f"    {node_id}((\"{label}\"))")
        elif node.node_type == "technology":
            lines.append(f"    {node_id}{{\"{label}\"}}")
        else:
            lines.append(f"    {node_id}(\"{label}\")")

    for edge in spec.edges:
        label = _escape_mermaid_text(_compact_label(_edge_label(edge)))
        lines.append(f"    {id_map[edge.source]} -->|\"{label}\"| {id_map[edge.target]}")

    lines.extend(
        [
            "    classDef source fill:#dcecc9,stroke:#5b7f31",
            "    classDef sink fill:#d9e8fb,stroke:#4b70a8",
            "    classDef technology fill:#ffe8bd,stroke:#9a6b00",
            "    classDef intermediate fill:#f6f6f6,stroke:#777777",
            "    classDef active stroke:#2f7f72,stroke-width:3px",
            "    classDef inactive stroke:#b8b8b8,stroke-dasharray: 4 3",
        ]
    )

    for node in spec.nodes:
        lines.append(f"    class {id_map[node.id]} {node.node_type}")
    for index, edge in enumerate(spec.edges):
        if edge.active is True:
            lines.append(f"    linkStyle {index} stroke:#2f7f72,stroke-width:3px")
        elif edge.active is False:
            lines.append(f"    linkStyle {index} stroke:#b8b8b8,stroke-dasharray: 4 3")

    return "\n".join(lines)


def _graph_title(spec: NetworkGraphSpec) -> str:
    pieces = [spec.title, spec.graph_type.capitalize()]
    if spec.objective_value is not None:
        pieces.append(f"objective={spec.objective_value}")
    return " - ".join(pieces)


def _node_label(node: GraphNodeSpec) -> str:
    lines = [node.label, f"type: {node.node_type}"]
    if node.products:
        lines.append(f"products: {', '.join(node.products)}")
    lines.extend(_display_lines(node.display_attributes))
    return "\n".join(dict.fromkeys(lines))


def _edge_label(edge: GraphEdgeSpec) -> str:
    ordered_keys = ["product", "cost", "capacity", "flow", "utilization"]
    lines: List[str] = []
    for key in ordered_keys:
        if key in edge.display_attributes:
            lines.append(f"{key}: {edge.display_attributes[key]}")
    for key, value in edge.display_attributes.items():
        if key not in ordered_keys:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _display_lines(display_attributes: Dict[str, str]) -> List[str]:
    lines = []
    for key, value in display_attributes.items():
        if key == "role":
            continue
        label = _humanize_key(key)
        lines.append(f"{label}: {value}")
    return lines


def _humanize_key(key: str) -> str:
    return key.replace("solution:", "").replace(":", " ").replace("_", " ")


def _node_fillcolor(node: GraphNodeSpec) -> str:
    if node.active is True:
        return "#dff3e6"
    return NODE_COLORS.get(node.node_type, "#f6f6f6")


def _node_border_color(node: GraphNodeSpec) -> str:
    if node.active is True:
        return "#2f7f72"
    return {
        "source": "#5b7f31",
        "sink": "#4b70a8",
        "technology": "#9a6b00",
        "intermediate": "#777777",
    }.get(node.node_type, "#777777")


def _edge_style(edge: GraphEdgeSpec) -> Dict[str, str]:
    if edge.active is True:
        return {"color": "#2f7f72", "penwidth": "2.4"}
    if edge.active is False:
        return {"color": "#b8b8b8", "style": "dashed", "penwidth": "1.4"}
    return {"color": "#666666", "penwidth": "1.2"}


def _safe_graph_id(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return safe or "network_graph"


def _mermaid_id_map(spec: NetworkGraphSpec) -> Dict[str, str]:
    mapped: Dict[str, str] = {}
    used = set()
    for index, node in enumerate(spec.nodes, start=1):
        base = "N_" + re.sub(r"[^A-Za-z0-9_]+", "_", node.id).strip("_")
        if base == "N_":
            base = f"N_{index}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        mapped[node.id] = candidate
        used.add(candidate)
    return mapped


def _compact_label(label: str) -> str:
    return label.replace("\n", "<br/>")


def _escape_mermaid_text(text: str) -> str:
    return html.escape(text, quote=True)


__all__ = ["render_graphviz", "render_mermaid"]
