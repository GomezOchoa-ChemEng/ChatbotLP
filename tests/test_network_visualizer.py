import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path.cwd()))

from src.chatbot_engine import IntentRouter
from src.llm_problem_interpreter import build_state_from_semantic_plan
from src.midterm_benchmark import (
    build_midterm_manure_expected_plan,
    build_midterm_manure_q4_expected_plan,
)
from src.network_graph import build_problem_graph_spec, build_solution_graph_spec
from src.network_visualizer import GRAPHVIZ_INSTALL_MESSAGE, render_graphviz, render_mermaid
from src.schema import TransportLink
from src.solver_results import SolverResults


def _q4_solution_results() -> SolverResults:
    return SolverResults(
        solver_name="mock",
        solver_status="optimal",
        termination_condition="optimal",
        solver_time_seconds=0.0,
        success=True,
        objective_value=5800.0,
        bid_allocations={
            "B_DF_DM": 1000.0,
            "B_CF_DM": 0.0,
            "B_SF_DM": 500.0,
            "B_DC_Compost": 50.0,
        },
        transport_flows={
            "DF_to_CF": 0.0,
            "DF_to_SF": 500.0,
            "DF_to_Composter": 500.0,
            "Composter_to_DC": 50.0,
        },
        technology_extents={"Composter": 500.0},
    )


def test_q1_problem_graph_has_source_sink_nodes_and_two_transport_edges():
    state = build_state_from_semantic_plan(build_midterm_manure_expected_plan("canonical"))

    spec = build_problem_graph_spec(state)

    assert len(spec.edges) == 2
    assert {edge.edge_type for edge in spec.edges} == {"transport"}
    node_types = {node.id: node.node_type for node in spec.nodes}
    assert node_types["EauClaire"] == "source"
    assert node_types["Menomonie"] == "sink"
    assert node_types["BlackRiverFalls"] == "sink"


def test_q4_problem_graph_has_two_products_one_technology_and_four_transport_edges():
    state = build_state_from_semantic_plan(build_midterm_manure_q4_expected_plan("canonical"))

    spec = build_problem_graph_spec(state)

    assert set(spec.product_ids) == {"DM", "Compost"}
    assert len([node for node in spec.nodes if node.node_type == "technology"]) == 1
    assert len(spec.edges) == 4
    assert {edge.product_id for edge in spec.edges} == {"DM", "Compost"}


def test_q4_solution_graph_marks_active_flows_and_technology_outputs():
    state = build_state_from_semantic_plan(build_midterm_manure_q4_expected_plan("canonical"))
    results = _q4_solution_results()

    spec = build_solution_graph_spec(state, results)

    edges = {edge.id: edge for edge in spec.edges}
    assert edges["DF_to_CF"].flow == 0.0
    assert edges["DF_to_CF"].active is False
    assert edges["DF_to_SF"].flow == 500.0
    assert edges["DF_to_SF"].active is True
    assert edges["DF_to_Composter"].flow == 500.0
    assert edges["DF_to_Composter"].active is True
    assert edges["Composter_to_DC"].flow == 50.0
    assert edges["Composter_to_DC"].active is True

    composter = {node.id: node for node in spec.nodes}["Composter"]
    assert composter.active is True
    assert composter.solution["technology_activity"]["Composter"] == 500.0
    assert composter.solution["technology_outputs"]["Compost"] == 50.0
    assert spec.objective_value == 5800.0


def test_missing_route_cost_is_displayed_as_missing_not_zero():
    state = build_state_from_semantic_plan(build_midterm_manure_expected_plan("canonical"))
    state.transport_links[0] = TransportLink(
        id=state.transport_links[0].id,
        origin=state.transport_links[0].origin,
        destination=state.transport_links[0].destination,
        product=state.transport_links[0].product,
        capacity=state.transport_links[0].capacity,
        cost=None,
    )

    spec = build_problem_graph_spec(state)
    edge = {edge.id: edge for edge in spec.edges}["EauClaire_to_Menomonie"]

    assert edge.cost is None
    assert edge.display_attributes["cost"] == "missing"
    assert edge.display_attributes["cost"] != "0.0"


def test_explicit_zero_route_cost_is_displayed_as_zero_point_zero():
    state = build_state_from_semantic_plan(build_midterm_manure_q4_expected_plan("canonical"))

    spec = build_problem_graph_spec(state)
    edge = {edge.id: edge for edge in spec.edges}["DF_to_Composter"]

    assert edge.cost == 0.0
    assert edge.display_attributes["cost"] == "0.0"


def test_graph_construction_does_not_mutate_problem_state():
    state = build_state_from_semantic_plan(build_midterm_manure_q4_expected_plan("canonical"))
    before = state.to_dict()

    build_problem_graph_spec(state)
    build_solution_graph_spec(state, _q4_solution_results())

    assert state.to_dict() == before


def test_visualization_intent_detection_for_network_prompts():
    router = IntentRouter()

    assert router.detect_intent("show the network") == "visualization"
    assert router.detect_intent("draw the problem") == "visualization"
    assert router.detect_intent("visualize the solution") == "visualization"
    assert router.detect_intent("show active flows") == "visualization"


def test_mermaid_renderer_does_not_require_graphviz():
    state = build_state_from_semantic_plan(build_midterm_manure_expected_plan("canonical"))
    spec = build_problem_graph_spec(state)

    mermaid = render_mermaid(spec)

    assert "flowchart LR" in mermaid
    assert "EauClaire" in mermaid
    assert "Menomonie" in mermaid


def test_graphviz_missing_python_package_message_is_clear(tmp_path):
    state = build_state_from_semantic_plan(build_midterm_manure_expected_plan("canonical"))
    spec = build_problem_graph_spec(state)

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "graphviz" or name.startswith("graphviz."):
            raise ImportError("missing graphviz")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="Graphviz is not installed"):
            render_graphviz(spec, str(tmp_path / "missing.svg"))

    with patch("builtins.__import__", side_effect=fake_import):
        try:
            render_graphviz(spec, str(tmp_path / "missing.svg"))
        except RuntimeError as exc:
            assert str(exc) == GRAPHVIZ_INSTALL_MESSAGE


def test_graphviz_renderer_creates_svg_file(tmp_path):
    pytest.importorskip("graphviz")
    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot executable is not installed")

    state = build_state_from_semantic_plan(build_midterm_manure_expected_plan("canonical"))
    spec = build_problem_graph_spec(state)

    output_path = render_graphviz(spec, str(tmp_path / "q1_network.svg"))

    svg_path = Path(output_path)
    assert svg_path.exists()
    assert "<svg" in svg_path.read_text(encoding="utf-8")
