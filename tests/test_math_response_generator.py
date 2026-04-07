import sys
import re
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

from src.formal_context_builder import build_formal_math_context
from src.math_response_generator import (
    MathResponseGenerator,
    generate_math_response,
    strip_full_latex_document,
)
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology
from src.llm_adapter import LLMProviderRegistry, GeminiLLMProvider
from src.solver import SolveResult


def make_state() -> ProblemState:
    state = ProblemState(problem_title="Math Response Test")
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p1"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=12))
    state.add_consumer(Consumer(id="c1", node="n1", product="p1", capacity=10))
    state.add_bid(Bid(id="bs", owner_id="s1", owner_type="supplier", product_id="p1", price=1.0, quantity=10))
    state.add_bid(Bid(id="bc", owner_id="c1", owner_type="consumer", product_id="p1", price=4.0, quantity=10))
    return state


def make_negative_bid_state() -> ProblemState:
    state = make_state()
    state.bids[0].price = -5.0
    return state


def make_case_c_state() -> ProblemState:
    state = ProblemState(problem_title="Case C Math Response Test")
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p1"))
    state.add_product(Product(id="p2"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=10))
    state.add_consumer(Consumer(id="c1", node="n1", product="p2", capacity=8))
    state.add_bid(Bid(id="B1", owner_id="s1", owner_type="supplier", product_id="p1", price=1.0, quantity=10))
    state.add_bid(Bid(id="B2", owner_id="c1", owner_type="consumer", product_id="p2", price=9.0, quantity=8))
    state.add_technology(
        Technology(
            id="K1",
            node="n1",
            capacity=6,
            yield_coefficients={"p1": -1.0, "p2": 0.8},
        )
    )
    return state


def test_dual_generation_without_llm():
    state = make_state()
    context = build_formal_math_context(state, "Give me the dual problem in LaTeX.")
    response, metadata = MathResponseGenerator(use_llm=False).generate_with_metadata(context)
    first_block = re.findall(r"\$\$\s*(.*?)\s*\$\$", response, flags=re.DOTALL)[0]
    assert "The dual problem is formulated as follows:" in response
    assert response.count("The dual problem is formulated as follows:") == 1
    assert "$$" in response
    assert response.count("$$") == 4
    assert "\\begin{aligned}" in response
    assert response.count("\\begin{aligned}") == 2
    assert "(D)\\qquad \\min \\quad" in response
    assert "\\pi_{n1,p1}" in response
    assert "\\text{s.t.}" in response
    assert "\\text{s.t.} \\\\" in first_block
    assert "\\text{s.t.} \\quad &" not in first_block
    assert first_block.count("\\ge") == 2
    assert first_block.count("\\\\\n& ") == 2
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
    assert metadata["response_source"] == "deterministic"
    assert metadata["fallback_triggered"] is False
    assert metadata["mode_used"] == "guided"
    assert metadata["grounding_mode"] == "model"


def test_primal_generation_without_llm():
    state = make_state()
    context = build_formal_math_context(
        state,
        "Formulate the current coordinated clearing problem as the primal linear program in LaTeX.",
    )
    response = generate_math_response(context, use_llm=False)

    assert "The primal problem is formulated as follows:" in response
    assert "**Primal Problem.**" in response
    assert "(P)\\qquad \\max" in response
    assert "\\text{s.t.}" in response
    assert "q_{" in response


def test_instance_specific_primal_includes_current_case_data_for_case_a():
    context = build_formal_math_context(
        make_state(),
        "Formulate the current coordinated clearing problem as the primal linear program in LaTeX.",
    )
    response = generate_math_response(context, use_llm=False)

    assert context.formulation_scope == "instantiated_current_formulation"
    assert "Current instance data:" in response
    assert "bs: price 1" in response
    assert "bc: price 4" in response
    assert "q_{bs}" in response
    assert "q_{bc}" in response
    assert "10" in response


def test_instance_specific_primal_preserves_negative_bid_value_for_case_b():
    context = build_formal_math_context(
        make_negative_bid_state(),
        "Formulate the current problem as the primal linear program in LaTeX.",
    )
    response = generate_math_response(context, use_llm=False)

    assert context.formulation_scope == "instantiated_current_formulation"
    assert "Current instance data:" in response
    assert "price -5" in response
    assert "q_{bs}" in response


def test_instance_specific_theorem_proof_mentions_case_c_technology_symbols_and_yields():
    context = build_formal_math_context(
        make_case_c_state(),
        "State Theorem 1 and prove it in LaTeX for the current coordinated clearing problem.",
    )
    response = generate_math_response(context, use_llm=False)

    assert context.formulation_scope == "instantiated_current_formulation"
    assert "Current instance data:" in response
    assert "K1" in response
    assert "p1 -> -1" in response
    assert "p2 -> 0.8" in response
    assert "x_{K1}" in response
    assert "\\pi_{n1,p1}" in response
    assert "\\pi_{n1,p2}" in response


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
""".strip()

    with patch.object(
        MathResponseGenerator,
        "_generate_with_llm",
        return_value=rejected_response,
    ):
        response, metadata = MathResponseGenerator(use_llm=True).generate_with_metadata(context)
    first_block = re.findall(r"\$\$\s*(.*?)\s*\$\$", response, flags=re.DOTALL)[0]

    assert response.count("The dual problem is formulated as follows:") == 1
    assert response.count("\\begin{aligned}") == 2
    assert response.count("$$") == 4
    assert "(D)\\qquad \\min \\quad" in response
    assert "\\text{s.t.} \\\\" in first_block
    assert "\\text{s.t.} \\quad &" not in first_block
    assert first_block.count("\\\\\n& ") == 2
    assert "Validation notes:" not in response
    assert "bad llm output" not in response
    assert "\\text{sign restrictions}" not in response
    assert "(\\text{supplier bid})" not in response
    assert "balance_" not in response
    assert "\\mu_{bs}" in response
    assert "\\nu_{bc}" in response
    assert response.rstrip().endswith("$$")
    assert metadata["response_source"] == "deterministic"
    assert metadata["fallback_triggered"] is True
    assert metadata["fallback_reason"] == "structurally_unusable_llm_output"
    assert metadata["raw_llm_output_present"] is True
    assert metadata["llm_output_length"] > 0


def test_instance_specific_dual_rejects_generic_family_level_llm_output():
    context = build_formal_math_context(
        make_state(),
        "Formulate the current coordinated clearing problem as the dual linear program in LaTeX.",
    )
    generic_dual = r"""
The dual problem is formulated as follows:

$$
\begin{aligned}
(D)\qquad \min \quad & \sum_{n,p} \pi_{np} + \sum_b \mu_b \\
\text{s.t.} \quad & \pi_{np} + \mu_b \ge c_b
\end{aligned}
$$
""".strip()

    with patch.object(MathResponseGenerator, "_generate_with_llm", return_value=generic_dual):
        response, metadata = MathResponseGenerator(use_llm=True).generate_with_metadata(context)

    assert "\\pi_{n1,p1}" in response
    assert "\\mu_{bs}" in response
    assert metadata["response_source"] == "deterministic"
    assert metadata["fallback_triggered"] is True
    assert metadata["validation_fatal"]
    assert metadata["fallback_reason"] == "instance_specific_validation_failed"


def test_dual_generation_keeps_llm_output_when_only_minor_grounding_warnings_exist():
    state = make_state()
    context = build_formal_math_context(state, "Give me the dual problem in LaTeX.")
    llm_response = r"""
The dual problem is formulated as follows:

$$
\begin{aligned}
(D)\qquad \min \quad & 6 \pi_{n1,p1} + 6 \mu_{bs} \\
\text{s.t.} \quad & \pi_{n1,p1} + \mu_{bs} \ge 1 \\
& \pi_{n1,p1} \ge 2
\end{aligned}
$$

This dual says the node-product price coordinates supplier and consumer acceptance.
""".strip()

    with patch.object(
        MathResponseGenerator,
        "_generate_with_llm",
        return_value=llm_response,
    ):
        response, metadata = MathResponseGenerator(use_llm=True).generate_with_metadata(context)

    assert "node-product price coordinates supplier and consumer acceptance" in response
    assert "This response may not be fully grounded in the deterministic model." in response
    assert response.count("The dual problem is formulated as follows:") == 1
    assert metadata["response_source"] == "llm"
    assert metadata["fallback_triggered"] is False
    assert metadata["grounding_warning_applied"] is True
    assert metadata["validation_warnings"]


def test_explanation_metadata_uses_solver_grounding_for_complementary_slackness():
    state = make_state()
    context = build_formal_math_context(
        state,
        "Verify the complementary slackness conditions for the current primal-dual pair and explain their economic interpretation.",
    )
    primal_result = SolveResult(object(), "optimal", "mock primal", 30.0, 0.01, {"q": {"bs": 10.0, "bc": 10.0}, "f": {}, "x": {}}, True)
    dual_result = SolveResult(object(), "optimal", "mock dual", 30.0, 0.01, {"y": {"\\pi_{n1,p1}": -1.0, "\\mu_{bs}": 0.0, "\\nu_{bc}": 3.0}}, True)

    with patch("src.math_response_generator.solve_model", side_effect=[primal_result, dual_result]):
        _, metadata = MathResponseGenerator(use_llm=False).generate_with_metadata(context)

    assert metadata["grounding_mode"] == "solver"


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


def test_general_math_explanation_for_dual_variable_meaning_avoids_full_dual_dump():
    state = make_state()
    context = build_formal_math_context(
        state,
        "Explain the economic meaning of the dual variables in the current model.",
    )
    response = generate_math_response(context, use_llm=False)

    assert "The dual problem is formulated as follows:" not in response
    assert "node-product prices" in response or "node-product price" in response
    assert "marginal" in response or "scarcity" in response
    assert "\\pi_{n1,p1}" in response


def test_general_math_explanation_for_strong_duality_answers_requested_concept():
    state = make_state()
    context = build_formal_math_context(
        state,
        "Explain strong duality for the current model and outline the proof structure.",
    )
    response = generate_math_response(context, use_llm=False)

    assert "strong duality" in response.lower()
    assert "z_P^* = z_D^*" in response
    assert "The dual problem is formulated as follows:" not in response


def test_mixed_dual_and_interpretation_request_keeps_both_parts():
    state = make_state()
    context = build_formal_math_context(
        state,
        "First write the dual problem in LaTeX, and then explain how that dual relates to prices and incentives in the model.",
    )
    response = generate_math_response(context, use_llm=False)

    assert "The dual problem is formulated as follows:" in response
    assert "prices" in response.lower() or "incentive" in response.lower()
    assert "\\pi_{n1,p1}" in response


def test_complementary_slackness_verification_uses_solver_backed_results():
    state = make_state()
    context = build_formal_math_context(
        state,
        "Verify the complementary slackness conditions for the current primal-dual pair and explain their economic interpretation.",
    )

    primal_result = SolveResult(
        model=object(),
        status="optimal",
        message="mock primal solve",
        objective_value=30.0,
        solver_time=0.01,
        solution={
            "q": {"bs": 10.0, "bc": 10.0},
            "f": {},
            "x": {},
        },
        success=True,
    )
    dual_result = SolveResult(
        model=object(),
        status="optimal",
        message="mock dual solve",
        objective_value=30.0,
        solver_time=0.01,
        solution={
            "y": {
                "\\pi_{n1,p1}": -1.0,
                "\\mu_{bs}": 0.0,
                "\\nu_{bc}": 3.0,
            }
        },
        success=True,
    )

    with patch("src.math_response_generator.solve_model", side_effect=[primal_result, dual_result]):
        response = generate_math_response(context, use_llm=False)

    assert "Complementary slackness was checked using solver-backed primal and dual solutions." in response
    assert "Primal status: optimal; dual status: optimal." in response
    assert "All checked complementary-slackness products are within tolerance." in response
    assert "Economic interpretation:" in response
