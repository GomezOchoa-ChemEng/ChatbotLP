import sys
import re
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

from src.chatbot_engine import run_chatbot_session
from src.math_response_generator import MathResponseGenerator
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
    first_block = re.findall(r"\$\$\s*(.*?)\s*\$\$", result["response"], flags=re.DOTALL)[0]
    assert result["intent"] == "formal_math"
    assert result["success"]
    assert result["render_mode"] == "markdown_latex"
    assert "The dual problem is formulated as follows:" in result["response"]
    assert result["response"].count("The dual problem is formulated as follows:") == 1
    assert "\\begin{aligned}" in result["response"]
    assert result["response"].count("\\begin{aligned}") == 2
    assert result["response"].count("$$") == 4
    assert "(D)\\qquad \\min \\quad" in result["response"]
    assert "\\text{s.t.} \\\\" in first_block
    assert "\\text{s.t.} \\quad &" not in first_block
    assert first_block.count("\\\\\n& ") == 2
    assert "\\text{sign restrictions}" not in result["response"]
    assert "(\\text{supplier bid})" not in result["response"]
    assert "balance_" not in result["response"]
    assert "\\documentclass" not in result["response"]


def test_chatbot_dual_request_hides_invalid_llm_output_and_validation_notes():
    rejected_response = r"""
The dual problem is formulated as follows:

Rejected raw response

$$
\begin{aligned}
\min \quad & \pi_{n1,p1}
\end{aligned}
$$

Validation notes:
- dual response is missing standard optimization LaTeX structure.
""".strip()

    with patch.object(
        MathResponseGenerator,
        "_generate_with_llm",
        return_value=rejected_response,
    ):
        result = run_chatbot_session(
            make_state(),
            "Give me the dual problem in LaTeX.",
            use_llm=True,
        )
    first_block = re.findall(r"\$\$\s*(.*?)\s*\$\$", result["response"], flags=re.DOTALL)[0]

    assert result["intent"] == "formal_math"
    assert result["success"]
    assert result["response"].count("The dual problem is formulated as follows:") == 1
    assert result["response"].count("\\begin{aligned}") == 2
    assert result["response"].count("$$") == 4
    assert "(D)\\qquad \\min \\quad" in result["response"]
    assert "\\text{s.t.} \\\\" in first_block
    assert "\\text{s.t.} \\quad &" not in first_block
    assert first_block.count("\\\\\n& ") == 2
    assert "Validation notes:" not in result["response"]
    assert "Rejected raw response" not in result["response"]
    assert "\\text{sign restrictions}" not in result["response"]
    assert "(\\text{supplier bid})" not in result["response"]
    assert "balance_" not in result["response"]
    assert "\\mu_{bs}" in result["response"]
    assert "\\nu_{bc}" in result["response"]


def test_chatbot_routes_theorem_request():
    result = run_chatbot_session(make_state(), "Show me that Theorem 1 holds.")
    assert result["intent"] == "formal_math"
    assert result["success"]
    assert "**Theorem 1**" in result["response"]
    assert result["render_mode"] == "markdown_latex"
    assert "**Primal Problem.**" in result["response"]
    assert "**Dual Problem.**" in result["response"]
    assert "**Proof.**" in result["response"]
    assert result["response"].count("\\begin{aligned}") >= 3
    assert result["response"].count("\n& ") >= 4
    assert "\\text{sign restrictions}" in result["response"]
    assert "\\mathcal{L}" in result["response"]
    assert "strong duality" in result["response"]
    assert "z_P^* = z_D^*" in result["response"]
    assert "dual objective induced by" not in result["response"].lower()
    assert "coordinated surplus over" not in result["response"].lower()
    assert "primal has an optimal solution" not in result["response"].lower()
    assert "\\begin{proof}" not in result["response"]
    assert "validated_linear_problem_state" not in result["response"]
    assert "assumptions_verified" not in result["response"]
    assert "ProblemState" not in result["response"]


def test_chatbot_out_of_scope_theorem_request_is_unsuccessful():
    result = run_chatbot_session(make_state(), "Show me that Theorem 9 holds.")
    assert result["intent"] == "formal_math"
    assert not result["success"]
    assert "out of scope" in result["response"]
