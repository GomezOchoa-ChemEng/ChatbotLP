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
    validation = validate_formal_math_context(context)
    assert validation == {"fatal": [], "warnings": []}


def test_validate_generated_math_response_flags_missing_dual_symbol_as_warning():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    validation = validate_generated_math_response(context, "dual text without symbols $$x$$")
    assert validation["fatal"] == []
    assert validation["warnings"]


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
    validation = validate_generated_math_response(context, response)
    assert validation == {"fatal": [], "warnings": []}


def test_validate_generated_math_response_allows_extra_dual_prose_under_relaxed_validation():
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
    validation = validate_generated_math_response(context, response)
    assert validation == {"fatal": [], "warnings": []}


def test_validate_generated_math_response_flags_missing_dual_structure():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    response = "Here is a vague dual description without any displayed math."
    validation = validate_generated_math_response(context, response)
    assert "Dual response should include a recognizable dual objective." in validation["fatal"]
    assert "Dual response should include at least one display-math block." in validation["fatal"]


def test_validate_generated_math_response_rejects_packed_dual_inequalities():
    context = build_formal_math_context(make_state(), "Give me the dual problem in LaTeX.")
    response = r"""
The dual problem is formulated as follows:

$$
\begin{aligned}
(D)\qquad \min \quad & 6 \pi_{n1,p1} + 6 \mu_{bs} + 6 \nu_{bc} \\
\text{s.t.} \\
& \pi_{n1,p1} + \mu_{bs} \ge 1,\quad \pi_{n1,p1} - \nu_{bc} \ge 2
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
    validation = validate_generated_math_response(context, response)
    assert "Dual response packs multiple inequalities into a single row." in validation["warnings"]


def test_validate_generated_math_response_flags_missing_proof_environment():
    context = build_formal_math_context(make_state(), "Show me that Theorem 1 holds.")
    validation = validate_generated_math_response(context, "Theorem 1 applies.")
    assert validation["warnings"]
