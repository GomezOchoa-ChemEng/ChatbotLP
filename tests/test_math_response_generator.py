import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

from src.formal_context_builder import build_formal_math_context
from src.math_response_generator import (
    MathResponseGenerator,
    generate_math_response,
    strip_full_latex_document,
)
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
    assert "The dual problem is formulated as follows:" in response
    assert response.count("The dual problem is formulated as follows:") == 1
    assert "$$" in response
    assert response.count("$$") == 4
    assert "\\begin{aligned}" in response
    assert response.count("\\begin{aligned}") == 2
    assert "(D)\\qquad \\min \\quad" in response
    assert "\\pi_{n1,p1}" in response
    assert "\\text{s.t.}" in response
    assert "\\mu_{bs}" in response
    assert "\\nu_{bc}" in response
    assert "\\text{sign restrictions}" not in response
    assert "(\\text{supplier bid})" not in response
    assert "(\\text{consumer bid})" not in response
    assert "(\\text{transport flow})" not in response
    assert "balance_" not in response
    assert "supplier_cap_" not in response
    assert "consumer_cap_" not in response
    assert "transport_cap_" not in response
    assert "\\begin{align*}" not in response
    assert "\\documentclass" not in response
    assert "\\usepackage" not in response
    assert "\\begin{document}" not in response
    assert "\\end{document}" not in response


def test_theorem_generation_without_llm():
    state = make_state()
    context = build_formal_math_context(state, "Explain why Theorem 1 applies in this case.")
    response = generate_math_response(context, use_llm=False)
    assert "Theorem 1" in response
    assert "Verified assumptions" in response


def test_theorem_proof_generation_is_notebook_friendly_fragment():
    state = make_state()
    context = build_formal_math_context(state, "Show me that Theorem 1 holds.")
    response = generate_math_response(context, use_llm=False)
    assert "**Theorem 1**" in response
    assert "**Primal Problem.**" in response
    assert "**Dual Problem.**" in response
    assert "**Proof.**" in response
    assert response.count("**Dual Problem.**") == 1
    assert response.count("(D)\\qquad \\min") == 1
    assert "$$" in response
    assert response.count("\\begin{aligned}") >= 3
    assert response.count("\n& ") >= 4
    assert "\\text{sign restrictions}" in response
    assert "\\text{Coefficient conditions:}" in response
    assert "\\mathcal{L}" in response
    assert "Lagrangian" in response
    assert "node-product prices" in response
    assert "strong duality" in response
    assert "z_P^* = z_D^*" in response
    assert "dual objective induced by" not in response.lower()
    assert "coordinated surplus over" not in response.lower()
    assert "concise or / mathematical programming proof" not in response.lower()
    assert "grounded in the verified structured context" not in response.lower()
    assert "primal has an optimal solution" not in response.lower()
    assert "primal optimum exists" not in response.lower()
    assert "q_{B1}=0q_{B1}=0" not in response
    assert "max⁡" not in response
    assert "\\begin{proof}" not in response
    assert "\\end{proof}" not in response
    assert "\\documentclass" not in response
    assert "\\usepackage" not in response
    assert "\\begin{document}" not in response
    assert "\\end{document}" not in response
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
    assert "\\documentclass" not in response
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
            assert "\\documentclass" not in response
    finally:
        registry.reset()


def test_dual_generation_discards_invalid_llm_output_and_shows_only_clean_fallback():
    state = make_state()
    context = build_formal_math_context(state, "Give me the dual problem in LaTeX.")
    rejected_response = r"""
The dual problem is formulated as follows:

$$
\begin{aligned}
\min \quad & \pi_{n1,p1}
\end{aligned}
$$

Validation notes:
- bad llm output

$$
\begin{aligned}
\min \quad & malformed duplicate
\end{aligned}
$$
""".strip()

    with patch.object(
        MathResponseGenerator,
        "_generate_with_llm",
        return_value=rejected_response,
    ):
        response = generate_math_response(context, use_llm=True)

    assert response.count("The dual problem is formulated as follows:") == 1
    assert response.count("\\begin{aligned}") == 2
    assert response.count("$$") == 4
    assert "(D)\\qquad \\min \\quad" in response
    assert "Validation notes:" not in response
    assert "bad llm output" not in response
    assert "malformed duplicate" not in response
    assert "\\text{sign restrictions}" not in response
    assert "(\\text{supplier bid})" not in response
    assert "balance_" not in response
    assert "\\mu_{bs}" in response
    assert "\\nu_{bc}" in response
    assert response.rstrip().endswith("$$")


def test_strip_full_latex_document_normalizes_notebook_fragment_artifacts():
    raw = r"""
```latex
\documentclass{article}
\usepackage{amsmath}
\begin{document}
\begin{proof}
\[
z_P^* = z_D^*.
\]
\end{proof}
\end{document}
```
""".strip()

    cleaned = strip_full_latex_document(raw)

    assert "```" not in cleaned
    assert "\\documentclass" not in cleaned
    assert "\\usepackage" not in cleaned
    assert "\\begin{document}" not in cleaned
    assert "\\end{document}" not in cleaned
    assert "\\begin{proof}" not in cleaned
    assert "\\end{proof}" not in cleaned
    assert "\\[" not in cleaned
    assert "\\]" not in cleaned
    assert "**Proof.**" in cleaned
    assert "$$" in cleaned
    assert "z_P^* = z_D^*" in cleaned


def test_theorem_1_response_requires_primal_and_dual_semantics():
    state = make_state()
    context = build_formal_math_context(state, "Show me that Theorem 1 holds.")
    response = generate_math_response(context, use_llm=False)

    assert "(P)" in response
    assert "(D)" in response
    assert response.count("**Dual Problem.**") == 1
    assert "strong duality theorem of linear programming" in response
    assert "z_P^* = z_D^*" in response
    assert "\\mathcal{L}" in response
    assert "\\text{sign restrictions}" in response
