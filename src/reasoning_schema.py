"""Typed schema for Sampat-specific reasoning requests and grounded artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ReasoningObject = Literal[
    "primal",
    "dual",
    "bids",
    "prices",
    "technologies",
    "theorem",
    "feasibility",
    "sensitivity",
    "section23",
    "benchmark_case",
    "unknown",
]

ReasoningOperation = Literal[
    "formulate",
    "explain",
    "compare",
    "verify",
    "interpret",
    "derive",
    "diagnose",
    "summarize",
]

GroundingMode = Literal["paper", "model", "solver", "theorem"]
ReasoningStyle = Literal["concise", "pedagogical", "latex", "proof"]
ReasoningScope = Literal[
    "section_21",
    "section_22",
    "section_23",
    "case_A",
    "case_B",
    "case_C",
    "unknown",
]

ResponseMode = Literal[
    "paper_grounded_explanation",
    "model_grounded_formulation",
    "solver_grounded_verification",
    "theorem_grounded_proof",
]

ReasoningPath = Literal[
    "sampat_reasoning",
    "math_response_generator",
    "validation",
    "solver",
    "theorem_checker",
]

ArtifactSource = Literal["paper", "model", "solver", "theorem", "benchmark", "state"]


class SampatReasoningPlan(BaseModel):
    """Compact structured interpretation of a Sampat-related request."""

    user_query: str
    object: ReasoningObject = "unknown"
    operation: ReasoningOperation = "explain"
    grounding_mode: GroundingMode = "paper"
    style: ReasoningStyle = "pedagogical"
    scope: ReasoningScope = "unknown"


class GroundedArtifact(BaseModel):
    """A reusable grounded artifact collected for answer generation."""

    name: str
    source: ArtifactSource
    available: bool = True
    summary: str
    data: Dict[str, Any] = Field(default_factory=dict)


class MissingArtifact(BaseModel):
    """A required artifact that could not be collected."""

    name: str
    required_for: List[str] = Field(default_factory=list)
    reason: str


class SampatReasoningPackage(BaseModel):
    """Structured package returned by the Sampat reasoning engine."""

    plan: SampatReasoningPlan
    response_mode: ResponseMode
    recommended_path: ReasoningPath = "sampat_reasoning"
    can_answer: bool = True
    grounded: bool = True
    artifacts: List[GroundedArtifact] = Field(default_factory=list)
    missing_artifacts: List[MissingArtifact] = Field(default_factory=list)
    source_notes: List[str] = Field(default_factory=list)
    answer_outline: List[str] = Field(default_factory=list)
    missing_information_text: Optional[str] = None

