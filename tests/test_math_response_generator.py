import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path.cwd()))

from src.formal_context_builder import build_formal_math_context
from src.math_response_generator import generate_math_response
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier
from src.llm_adapter import LLMProviderRegistry, GeminiLLMProvider


def make_state() -> ProblemState:
    state = ProblemState(problem_title="Math Response Test")
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p1"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=12))
    state.add_consumer(Consumer(id="c1", node="n1", product="p1", capacity=10))
    state.add_bid(Bid(id="bs", owner_id="s1", owner_type="supplier", product_id="p1", price=1.0, quantity=10))
    state.add_bid(Bid(id="bc", owner_id="c1", owner_type="consumer", product_id="p1", price=4.0, quantity=10))
    return state


def test_dual_generation_without_llm():
    state = make_state()
    context = build_formal_math_context(state, "Give me the dual problem in LaTeX.")
    response = generate_math_response(context, use_llm=False)
    assert "\\begin{aligned}" in response
    assert "\\pi_{n1,p1}" in response
    assert "\\min \\quad" in response
    assert "\\text{s.t.}" in response
    assert "\\mu_{bs}" in response
    assert "\\nu_{bc}" in response


def test_theorem_generation_without_llm():
    state = make_state()
    context = build_formal_math_context(state, "Explain why Theorem 1 applies in this case.")
    response = generate_math_response(context, use_llm=False)
    assert "Theorem 1" in response
    assert "Verified assumptions" in response


def test_theorem_proof_generation_has_proof_environment():
    state = make_state()
    context = build_formal_math_context(state, "Show me that Theorem 1 holds.")
    response = generate_math_response(context, use_llm=False)
    assert "\\textbf{Theorem 1.}" in response
    assert "\\begin{proof}" in response
    assert "\\end{proof}" in response
    assert "Lagrangian" in response
    assert "node-product prices" in response
    assert "strong duality" in response
    assert "$z_P^* = z_D^*$" in response
    assert "validated_linear_problem_state" not in response
    assert "assumptions_verified" not in response
    assert "ProblemState" not in response
    assert "Benchmark interpretation" not in response


def test_theorem_proof_missing_assumptions_fails_cleanly():
    state = make_state()
    state.suppliers[0].capacity = None
    context = build_formal_math_context(state, "Show me that Theorem 1 holds.")
    response = generate_math_response(context, use_llm=False)
    assert "cannot certify" in response
    assert "\\begin{proof}" not in response
    assert "validated_linear_problem_state" not in response
    assert "assumptions_verified" not in response
    assert "ProblemState" not in response


def test_out_of_scope_theorem_request_is_explicit():
    state = make_state()
    context = build_formal_math_context(state, "Show me that Theorem 9 holds.")
    response = generate_math_response(context, use_llm=False)
    assert "out of scope" in response


def test_math_response_falls_back_when_gemini_is_misconfigured():
    registry = LLMProviderRegistry.get_instance()
    registry.reset()
    registry.set_provider(GeminiLLMProvider(client=None))

    try:
        with patch.dict("os.environ", {}, clear=True):
            state = make_state()
            context = build_formal_math_context(state, "Give me the dual problem in LaTeX.")
            response = generate_math_response(context, use_llm=True)
            assert "\\begin{aligned}" in response
    finally:
        registry.reset()
