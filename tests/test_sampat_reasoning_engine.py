import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.reasoning_schema import SampatReasoningPackage
from src.sampat_reasoning_engine import SampatReasoningEngine
from src.schema import Bid, Consumer, Node, ProblemState, Product, Supplier, Technology


def make_state() -> ProblemState:
    state = ProblemState(problem_title="Sampat Reasoning Test")
    state.add_node(Node(id="n1"))
    state.add_product(Product(id="p1"))
    state.add_supplier(Supplier(id="s1", node="n1", product="p1", capacity=10))
    state.add_consumer(Consumer(id="c1", node="n1", product="p1", capacity=10))
    state.add_bid(Bid(id="bs", owner_id="s1", owner_type="supplier", product_id="p1", price=1.0, quantity=10))
    state.add_bid(Bid(id="bc", owner_id="c1", owner_type="consumer", product_id="p1", price=5.0, quantity=10))
    return state


def test_reasoning_engine_plans_price_question():
    engine = SampatReasoningEngine()

    package = engine.build_reasoning_package(
        "What do node-product prices represent?",
        make_state(),
    )

    assert isinstance(package, SampatReasoningPackage)
    assert package.plan.object == "prices"
    assert package.plan.operation == "interpret"
    assert package.response_mode == "paper_grounded_explanation"
    assert package.recommended_path == "sampat_reasoning"


def test_reasoning_engine_plans_case_comparison_as_paper_grounded():
    engine = SampatReasoningEngine()

    package = engine.build_reasoning_package(
        "Compare Case A and Case B.",
        make_state(),
    )

    assert package.plan.object == "benchmark_case"
    assert package.plan.operation == "compare"
    assert package.plan.grounding_mode == "paper"
    assert any("Case A" in line for line in package.answer_outline)


def test_reasoning_engine_records_missing_solver_artifact():
    engine = SampatReasoningEngine()
    state = ProblemState(problem_title="Incomplete")

    package = engine.build_reasoning_package(
        "Verify this with the solver.",
        state,
    )

    assert package.response_mode == "solver_grounded_verification"
    assert package.missing_artifacts
    assert any(item.name == "solver_result" for item in package.missing_artifacts)
    assert package.missing_information_text is not None


def test_reasoning_engine_collects_model_artifacts_for_technology_question():
    engine = SampatReasoningEngine()
    state = make_state()
    state.add_product(Product(id="p2"))
    state.add_technology(
        Technology(
            id="t1",
            node="n1",
            capacity=4,
            yield_coefficients={"p1": -1.0, "p2": 0.8},
        )
    )

    package = engine.build_reasoning_package(
        "How do technologies affect prices?",
        state,
    )

    artifact_names = {artifact.name for artifact in package.artifacts}
    assert package.plan.object == "technologies"
    assert "primal_scaffold" in artifact_names
    assert package.recommended_path == "sampat_reasoning"

