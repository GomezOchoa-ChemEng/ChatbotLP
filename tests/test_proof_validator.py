import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.formal_context_builder import build_formal_math_context
from src.proof_validator import (
    validate_formal_math_context,
    validate_generated_math_response,
)
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier


def make_state() -> ProblemState:
    state = ProblemState()
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p1"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=6))
    state.add_consumer(Consumer(id="c1", node="n1", product="p1", capacity=6))
    state.add_bid(Bid(id="bs", owner_id="s1", owner_type="supplier", product_id="p1", price=1.0, quantity=6))
    state.add_bid(Bid(id="bc", owner_id="c1", owner_type="consumer", product_id="p1", price=2.0, quantity=6))
    return state


def test_validate_formal_math_context_for_dual():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    issues = validate_formal_math_context(context)
    assert issues == []


def test_validate_generated_math_response_flags_missing_dual_symbol():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    issues = validate_generated_math_response(context, "dual text without symbols")
    assert issues


def test_validate_generated_math_response_flags_missing_proof_environment():
    context = build_formal_math_context(make_state(), "Show me that Theorem 1 holds.")
    issues = validate_generated_math_response(context, "Theorem 1 applies.")
    assert issues
