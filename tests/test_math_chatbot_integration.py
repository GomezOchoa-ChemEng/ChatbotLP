import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.chatbot_engine import run_chatbot_session
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier


def make_state() -> ProblemState:
    state = ProblemState(problem_title="Integration Test")
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p1"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=10))
    state.add_consumer(Consumer(id="c1", node="n1", product="p1", capacity=10))
    state.add_bid(Bid(id="bs", owner_id="s1", owner_type="supplier", product_id="p1", price=1.0, quantity=10))
    state.add_bid(Bid(id="bc", owner_id="c1", owner_type="consumer", product_id="p1", price=5.0, quantity=10))
    return state


def test_chatbot_routes_dual_request():
    result = run_chatbot_session(make_state(), "Give me the dual problem in LaTeX.")
    assert result["intent"] == "formal_math"
    assert result["success"]
    assert "\\begin{aligned}" in result["response"]


def test_chatbot_routes_theorem_request():
    result = run_chatbot_session(make_state(), "Show me that Theorem 1 holds.")
    assert result["intent"] == "formal_math"
    assert result["success"]
    assert "Theorem" in result["response"]


def test_chatbot_out_of_scope_theorem_request_is_unsuccessful():
    result = run_chatbot_session(make_state(), "Show me that Theorem 9 holds.")
    assert result["intent"] == "formal_math"
    assert not result["success"]
    assert "out of scope" in result["response"]
