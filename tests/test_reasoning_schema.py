import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.reasoning_schema import GroundedArtifact, SampatReasoningPackage, SampatReasoningPlan


def test_reasoning_plan_accepts_supported_literals():
    plan = SampatReasoningPlan(
        user_query="What do node-product prices represent?",
        object="prices",
        operation="interpret",
        grounding_mode="paper",
        style="pedagogical",
        scope="section_22",
    )

    assert plan.object == "prices"
    assert plan.grounding_mode == "paper"


def test_reasoning_package_tracks_artifacts():
    plan = SampatReasoningPlan(user_query="Compare Case A and Case B.")
    package = SampatReasoningPackage(
        plan=plan,
        response_mode="paper_grounded_explanation",
        artifacts=[
            GroundedArtifact(
                name="benchmark_metadata",
                source="benchmark",
                summary="Current benchmark metadata.",
            )
        ],
    )

    assert package.response_mode == "paper_grounded_explanation"
    assert package.artifacts[0].name == "benchmark_metadata"

