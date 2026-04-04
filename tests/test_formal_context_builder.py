import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.formal_context_builder import (
    build_formal_math_context,
    identify_formal_math_request,
    plan_formal_math_request,
)
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier


def make_state() -> ProblemState:
    state = ProblemState(problem_title="Math Context Test")
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p1"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=10))
    state.add_consumer(Consumer(id="c1", node="n1", product="p1", capacity=8))
    state.add_bid(Bid(id="bs", owner_id="s1", owner_type="supplier", product_id="p1", price=1.0, quantity=8))
    state.add_bid(Bid(id="bc", owner_id="c1", owner_type="consumer", product_id="p1", price=3.0, quantity=8))
    return state


def test_identify_dual_request():
    request = identify_formal_math_request("Give me the dual problem in LaTeX.")
    assert request["request_type"] == "dual"
    assert request["target_section"] == "2.2"


def test_identify_primal_request():
    request = identify_formal_math_request(
        "Formulate the current coordinated clearing problem as the primal linear program in LaTeX."
    )
    assert request["request_type"] == "primal"


def test_build_formal_math_context_for_theorem():
    state = make_state()
    context = build_formal_math_context(state, "Show me that Theorem 1 holds.")
    assert context.request_type == "theorem_proof"
    assert context.theorem_id == "theorem_1"
    assert context.target_section == "2.2"
    assert context.primal_formulation is not None
    assert context.variables
    assert "canonical" in context.notation_profile


def test_build_formal_math_context_for_section23():
    state = make_state()
    state.bids[0].price = -2.0
    context = build_formal_math_context(state, "Explain the role of negative bids in Section 2.3.")
    assert context.request_type == "section23_explanation"
    assert context.target_section == "2.3"
    assert any("negative bid" in note.lower() for note in context.source_notes)


def test_semantic_plan_keeps_dual_explanation_open_ended():
    plan = plan_formal_math_request(
        "Explain the economic meaning of the dual variables in the current model."
    )

    assert plan["primary_goal"] == "conceptual_explanation"
    assert "dual" in plan["math_topics"]
    assert "economic_interpretation" in plan["math_topics"]
    assert plan["response_contract"]["avoid_full_dual_formulation"] is True
    assert plan["response_contract"]["include_economic_interpretation"] is True


def test_identify_strong_duality_explanation_is_not_collapsed_to_dual_formulation():
    request = identify_formal_math_request(
        "Explain strong duality for the current model and outline the proof structure."
    )

    assert request["request_type"] == "general_math_explanation"


def test_build_formal_math_context_adds_semantic_plan():
    state = make_state()
    context = build_formal_math_context(
        state,
        "First write the dual problem in LaTeX, and then explain how that dual relates to prices and incentives in the model.",
    )

    assert context.semantic_plan["primary_goal"] == "mixed_dual_interpretation"
    assert context.semantic_plan["response_contract"]["include_dual_formulation"] is True
    assert context.semantic_plan["response_contract"]["include_economic_interpretation"] is True
