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


def test_validate_generated_math_response_accepts_two_block_dual_layout():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    response = r"""
The dual problem is formulated as follows:

$$
\begin{aligned}
(D)\qquad \min \quad & 6 \pi_{n1,p1} + 6 \mu_{bs} + 6 \nu_{bc} \\
\text{s.t.} \quad & \pi_{n1,p1} + \mu_{bs} \ge 1 \\
& \pi_{n1,p1} - \nu_{bc} \ge 2
\end{aligned}
$$
$$
\begin{aligned}
& \pi_{n1,p1} \in \mathbb{R} \\
& \mu_{bs} \ge 0 \\
& \nu_{bc} \ge 0
\end{aligned}
$$
""".strip()
    issues = validate_generated_math_response(context, response)
    assert issues == []


def test_validate_generated_math_response_rejects_extra_dual_prose():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    response = r"""
The dual problem is formulated as follows:

$$
\begin{aligned}
(D)\qquad \min \quad & 6 \pi_{n1,p1} + 6 \mu_{bs} + 6 \nu_{bc} \\
\text{s.t.} \quad & \pi_{n1,p1} + \mu_{bs} \ge 1 \\
& \pi_{n1,p1} - \nu_{bc} \ge 2
\end{aligned}
$$
$$
\begin{aligned}
& \pi_{n1,p1} \in \mathbb{R} \\
& \mu_{bs} \ge 0 \\
& \nu_{bc} \ge 0
\end{aligned}
$$

This note should not appear.
""".strip()
    issues = validate_generated_math_response(context, response)
    assert "Dual response must include the explanatory sentence exactly once." in issues


def test_validate_generated_math_response_rejects_non_sign_second_block():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    response = r"""
The dual problem is formulated as follows:

$$
\begin{aligned}
(D)\qquad \min \quad & 6 \pi_{n1,p1} + 6 \mu_{bs} + 6 \nu_{bc} \\
\text{s.t.} \quad & \pi_{n1,p1} + \mu_{bs} \ge 1 \\
& \pi_{n1,p1} - \nu_{bc} \ge 2
\end{aligned}
$$
$$
\begin{aligned}
& \pi_{n1,p1} \in \mathbb{R} \\
& \min \quad \mu_{bs}
\end{aligned}
$$
""".strip()
    issues = validate_generated_math_response(context, response)
    assert "Dual response second block must contain sign restrictions only." in issues


def test_validate_generated_math_response_flags_missing_proof_environment():
    context = build_formal_math_context(make_state(), "Show me that Theorem 1 holds.")
    issues = validate_generated_math_response(context, "Theorem 1 applies.")
    assert issues
