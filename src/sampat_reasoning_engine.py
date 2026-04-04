"""Lightweight Sampat reasoning engine layered over deterministic artifacts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .dual_generator import build_dual_scaffold, build_primal_representation
from .formal_context_builder import build_formal_math_context, identify_formal_math_request
from .math_response_generator import MathResponseGenerator
from .model_builder import build_model_from_state
from .reasoning_schema import (
    GroundedArtifact,
    MissingArtifact,
    SampatReasoningPackage,
    SampatReasoningPlan,
)
from .schema import ProblemState
from .solver import solve_model
from .theorem_checker import check_theorems, get_theorem_check_map
from .validator import validate_state
from .domain.sampat2019 import (
    CANONICAL_NOTATION,
    DOMAIN_SOURCE,
    SECTION23_CONCEPTS,
    get_section_metadata,
    get_theorem_metadata,
    infer_benchmark_case,
)


class SampatReasoningEngine:
    """Infer a small reasoning plan and gather grounded Sampat artifacts."""

    PAPER_KEYWORDS = [
        "sampat",
        "section 2.1",
        "section 2.2",
        "section 2.3",
        "node-product price",
        "node product price",
        "nodal price",
        "negative bid",
        "negative price",
        "technology",
        "yield coefficient",
        "case a",
        "case b",
        "case c",
        "benchmark",
        "dual variable",
        "shadow price",
        "economic interpretation",
        "from the paper",
    ]

    def is_sampat_related(self, user_query: str) -> bool:
        """Return whether the request is in the supported Sampat scope."""

        text = user_query.lower()
        return any(token in text for token in self.PAPER_KEYWORDS)

    def should_handle(self, user_query: str, intent: str) -> bool:
        """Return whether this engine should preempt the generic route."""

        text = user_query.lower()
        explanation_like = any(
            token in text
            for token in ["explain", "what", "how", "why", "meaning", "represent", "compare", "interpret"]
        )

        if intent in {"validation", "solve", "scenario"}:
            return False
        if intent == "problem_formulation" and not explanation_like:
            return False
        if self.is_sampat_related(user_query):
            return True
        formal_request = identify_formal_math_request(user_query)
        return formal_request["request_type"] != "general_math_explanation"

    def build_reasoning_package(
        self,
        user_query: str,
        state: ProblemState,
        pedagogical_mode: str = "guided",
    ) -> SampatReasoningPackage:
        """Infer a plan, collect artifacts, and package a grounded response."""

        plan = self._infer_plan(user_query)
        artifacts: List[GroundedArtifact] = []
        missing_artifacts: List[MissingArtifact] = []
        source_notes = [DOMAIN_SOURCE]

        theorem_checks = state.theorem_checks or check_theorems(state)
        theorem_check_map = get_theorem_check_map(theorem_checks)

        benchmark_case = (
            state.benchmark.case_family
            if state.benchmark and state.benchmark.case_family
            else infer_benchmark_case(bool(state.technologies), any(bid.price < 0 for bid in state.bids))
        )
        artifacts.append(
            GroundedArtifact(
                name="benchmark_metadata",
                source="benchmark",
                summary=f"Current benchmark family inferred as {benchmark_case}.",
                data={
                    "case_family": benchmark_case,
                    "has_technology": bool(state.technologies),
                    "has_negative_bids": any(bid.price < 0 for bid in state.bids),
                },
            )
        )

        paper_artifact = self._collect_paper_artifact(plan)
        if paper_artifact is not None:
            artifacts.append(paper_artifact)

        model_artifacts, model_missing = self._collect_model_artifacts(plan, state)
        artifacts.extend(model_artifacts)
        missing_artifacts.extend(model_missing)

        theorem_artifact, theorem_missing = self._collect_theorem_artifact(plan, theorem_check_map)
        if theorem_artifact is not None:
            artifacts.append(theorem_artifact)
            theorem_metadata = theorem_artifact.data.get("theorem_metadata", {})
            source_notes.extend(theorem_metadata.get("source_notes", []))
        if theorem_missing is not None:
            missing_artifacts.append(theorem_missing)

        solver_artifact, solver_missing = self._collect_solver_artifact(plan, state)
        if solver_artifact is not None:
            artifacts.append(solver_artifact)
        if solver_missing is not None:
            missing_artifacts.append(solver_missing)

        response_mode = self._response_mode_for(plan)
        recommended_path = self._recommended_path_for(plan)
        answer_outline = self._build_answer_outline(plan, artifacts, missing_artifacts)
        grounded = len(missing_artifacts) == 0 or plan.grounding_mode == "paper"
        can_answer = bool(answer_outline)
        missing_information_text = (
            self._missing_information_text(missing_artifacts)
            if missing_artifacts
            else None
        )

        package = SampatReasoningPackage(
            plan=plan,
            response_mode=response_mode,
            recommended_path=recommended_path,
            can_answer=can_answer,
            grounded=grounded,
            artifacts=artifacts,
            missing_artifacts=missing_artifacts,
            source_notes=source_notes,
            answer_outline=answer_outline,
            missing_information_text=missing_information_text,
        )

        if recommended_path == "math_response_generator":
            formal_context = build_formal_math_context(
                state=state,
                user_message=user_query,
                pedagogical_mode=pedagogical_mode,
            )
            package.source_notes.extend(formal_context.source_notes)

        return package

    def render_response(
        self,
        package: SampatReasoningPackage,
        state: ProblemState,
        pedagogical_mode: str = "guided",
        use_llm: bool = False,
    ) -> Tuple[str, str]:
        """Render a user-facing response and preferred render mode."""

        if package.recommended_path == "math_response_generator":
            formal_context = build_formal_math_context(
                state=state,
                user_message=package.plan.user_query,
                pedagogical_mode=pedagogical_mode,
            )
            response = MathResponseGenerator(use_llm=use_llm).generate(formal_context)
            return response, MathResponseGenerator.infer_render_mode(formal_context)

        if use_llm:
            try:
                from .llm_adapter import LLMProviderRegistry

                provider = LLMProviderRegistry.get_instance()
                llm_gen = provider.get_explanation_generator()
                llm_response = llm_gen.generate(
                    "full" if pedagogical_mode == "full" else "guided",
                    {
                        "type": "sampat_reasoning",
                        "user_message": package.plan.user_query,
                        "response_mode": package.response_mode,
                        "reasoning_package": package.model_dump(),
                    },
                )
                if llm_response and llm_response.strip():
                    return llm_response.strip(), "markdown"
            except Exception:
                pass

        lines = [
            self._grounding_label(package.response_mode),
        ]
        lines.extend(package.answer_outline)
        if package.missing_information_text:
            lines.append("")
            lines.append(package.missing_information_text)
        return "\n\n".join(line for line in lines if line), "markdown"

    def _infer_plan(self, user_query: str) -> SampatReasoningPlan:
        text = user_query.lower()

        if "section 2.3" in text:
            obj = "section23"
        elif "theorem" in text or "proof" in text or "strong duality" in text:
            obj = "theorem"
        elif "technology" in text or "technologies" in text or "yield" in text or "transformation" in text:
            obj = "technologies"
        elif "price" in text or "shadow" in text:
            obj = "prices"
        elif "bid" in text:
            obj = "bids"
        elif "dual" in text:
            obj = "dual"
        elif "primal" in text or "formulation" in text:
            obj = "primal"
        elif "case a" in text or "case b" in text or "case c" in text or "benchmark" in text:
            obj = "benchmark_case"
        elif "feasible" in text or "ready" in text:
            obj = "feasibility"
        elif "sensitivity" in text or "marginal" in text:
            obj = "sensitivity"
        else:
            obj = "unknown"

        if "compare" in text:
            operation = "compare"
        elif any(token in text for token in ["verify", "check", "holds", "applies"]):
            operation = "verify"
        elif any(token in text for token in ["formulate", "write", "show me the dual", "show me the primal"]):
            operation = "formulate"
        elif any(token in text for token in ["interpret", "meaning", "represent", "role"]):
            operation = "interpret"
        elif "derive" in text:
            operation = "derive"
        elif any(token in text for token in ["diagnose", "why not", "missing"]):
            operation = "diagnose"
        elif any(token in text for token in ["summary", "summarize"]):
            operation = "summarize"
        else:
            operation = "explain"

        if "latex" in text:
            style = "latex"
        elif "proof" in text:
            style = "proof"
        elif any(token in text for token in ["brief", "concise", "short"]):
            style = "concise"
        else:
            style = "pedagogical"

        if "section 2.3" in text or "negative bid" in text or "negative price" in text:
            scope = "section_23"
        elif "section 2.2" in text or obj in {"dual", "prices", "theorem"}:
            scope = "section_22"
        elif "section 2.1" in text or obj == "primal":
            scope = "section_21"
        elif "case a" in text:
            scope = "case_A"
        elif "case b" in text:
            scope = "case_B"
        elif "case c" in text:
            scope = "case_C"
        else:
            scope = "unknown"

        if obj == "theorem" or style == "proof":
            grounding_mode = "theorem"
        elif operation == "verify" and any(token in text for token in ["solver", "solve", "verified"]):
            grounding_mode = "solver"
        elif any(token in text for token in ["current model", "current state", "formulate", "dual", "primal"]):
            grounding_mode = "model"
        elif any(token in text for token in ["from the paper", "from sampat", "section 2.1", "section 2.2", "section 2.3"]):
            grounding_mode = "paper"
        elif obj in {"benchmark_case", "bids", "prices", "technologies"} and operation in {"explain", "interpret", "compare", "summarize"}:
            grounding_mode = "paper"
        else:
            grounding_mode = "paper"

        return SampatReasoningPlan(
            user_query=user_query,
            object=obj,
            operation=operation,
            grounding_mode=grounding_mode,
            style=style,
            scope=scope,
        )

    def _response_mode_for(self, plan: SampatReasoningPlan) -> str:
        if plan.grounding_mode == "theorem":
            return "theorem_grounded_proof"
        if plan.grounding_mode == "solver":
            return "solver_grounded_verification"
        if plan.grounding_mode == "model":
            return "model_grounded_formulation"
        return "paper_grounded_explanation"

    def _recommended_path_for(self, plan: SampatReasoningPlan) -> str:
        if plan.object in {"dual", "primal", "theorem"}:
            return "math_response_generator"
        if plan.operation == "verify" and plan.grounding_mode in {"solver", "theorem"}:
            return "math_response_generator"
        return "sampat_reasoning"

    def _collect_paper_artifact(
        self,
        plan: SampatReasoningPlan,
    ) -> Optional[GroundedArtifact]:
        scope_map = {
            "section_21": "2.1",
            "section_22": "2.2",
            "section_23": "2.3",
        }
        section_id = scope_map.get(plan.scope)
        section_metadata = get_section_metadata(section_id) if section_id else None
        if plan.scope == "section_23":
            summary = "Section 2.3 concepts about negative bids, negative prices, and economic interpretation."
            data = {
                "section_id": "2.3",
                "section_metadata": section_metadata or {},
                "concepts": SECTION23_CONCEPTS,
            }
            return GroundedArtifact(name="paper_context", source="paper", summary=summary, data=data)
        if section_metadata is not None:
            return GroundedArtifact(
                name="paper_context",
                source="paper",
                summary=f"Structured notes for Section {section_id}: {section_metadata['title']}.",
                data={
                    "section_id": section_id,
                    "section_metadata": section_metadata,
                    "canonical_notation": CANONICAL_NOTATION,
                },
            )
        if plan.object in {"benchmark_case", "bids", "prices", "technologies"}:
            return GroundedArtifact(
                name="paper_context",
                source="paper",
                summary="Curated Sampat domain notes and canonical notation are available.",
                data={"canonical_notation": CANONICAL_NOTATION},
            )
        return None

    def _collect_model_artifacts(
        self,
        plan: SampatReasoningPlan,
        state: ProblemState,
    ) -> Tuple[List[GroundedArtifact], List[MissingArtifact]]:
        artifacts: List[GroundedArtifact] = []
        missing: List[MissingArtifact] = []

        if plan.grounding_mode not in {"model", "solver", "theorem"} and plan.object not in {"prices", "technologies"}:
            return artifacts, missing

        primal = build_primal_representation(state)
        artifacts.append(
            GroundedArtifact(
                name="primal_scaffold",
                source="model",
                summary=f"Primal scaffold with {len(primal.get('variables', []))} variables and {len(primal.get('constraints', []))} constraints.",
                data=primal,
            )
        )

        needs_dual = plan.object in {"dual", "prices", "theorem"} or plan.scope == "section_22"
        if needs_dual:
            dual = build_dual_scaffold(primal)
            artifacts.append(
                GroundedArtifact(
                    name="dual_scaffold",
                    source="model",
                    summary=f"Dual scaffold with {len(dual.get('dual_variables', []))} dual variables.",
                    data=dual,
                )
            )

        if not state.node_ids() or not state.product_ids():
            missing.append(
                MissingArtifact(
                    name="model_index_sets",
                    required_for=["model_grounding"],
                    reason="The current ProblemState does not yet contain both node and product sets.",
                )
            )

        return artifacts, missing

    def _collect_theorem_artifact(
        self,
        plan: SampatReasoningPlan,
        theorem_check_map: Dict[str, Any],
    ) -> Tuple[Optional[GroundedArtifact], Optional[MissingArtifact]]:
        if plan.grounding_mode != "theorem" and plan.object != "theorem":
            return None, None

        theorem_match = re.search(r"theorem\s+(\d+)", plan.user_query.lower())
        theorem_id = f"theorem_{theorem_match.group(1)}" if theorem_match else "theorem_1"
        theorem_metadata = get_theorem_metadata(theorem_id)
        theorem_check = theorem_check_map.get(theorem_id)

        if theorem_metadata is None:
            return None, MissingArtifact(
                name="theorem_metadata",
                required_for=["theorem_grounding"],
                reason=f"{theorem_id} is outside the supported Sampat Sections 2.1-2.3 theorem registry.",
            )

        summary = f"{theorem_metadata['title']} metadata is available."
        if theorem_check is not None:
            summary += " Applicability was checked deterministically."

        return GroundedArtifact(
            name="theorem_metadata",
            source="theorem",
            summary=summary,
            data={
                "theorem_id": theorem_id,
                "theorem_metadata": theorem_metadata,
                "theorem_check": (
                    theorem_check.model_dump() if theorem_check is not None else None
                ),
            },
        ), None

    def _collect_solver_artifact(
        self,
        plan: SampatReasoningPlan,
        state: ProblemState,
    ) -> Tuple[Optional[GroundedArtifact], Optional[MissingArtifact]]:
        if plan.grounding_mode != "solver":
            return None, None

        validation = validate_state(state)
        if not validation["solver_ready"]:
            return None, MissingArtifact(
                name="solver_result",
                required_for=["solver_grounding"],
                reason="The current ProblemState is not solver-ready, so solver-backed verification cannot be claimed.",
            )

        model = build_model_from_state(state)
        solve_result = solve_model(model)
        if not solve_result.success:
            return GroundedArtifact(
                name="solver_result",
                source="solver",
                available=False,
                summary=f"Solver attempt returned status {solve_result.status}.",
                data=solve_result.to_dict(),
            ), MissingArtifact(
                name="verified_optimal_solution",
                required_for=["solver_grounding"],
                reason=f"Solver-backed verification is unavailable because solve status was {solve_result.status}.",
            )

        return GroundedArtifact(
            name="solver_result",
            source="solver",
            summary=f"Solver-backed result available with status {solve_result.status}.",
            data=solve_result.to_dict(),
        ), None

    def _build_answer_outline(
        self,
        plan: SampatReasoningPlan,
        artifacts: List[GroundedArtifact],
        missing_artifacts: List[MissingArtifact],
    ) -> List[str]:
        artifact_map = {artifact.name: artifact for artifact in artifacts}
        lines: List[str] = []

        if plan.object == "prices":
            lines.append(
                "Node-product prices are the shadow values on the node-product balance equations, so they measure the marginal system value of one more unit of product at a node."
            )
            if plan.scope == "section_23":
                lines.append(SECTION23_CONCEPTS["negative_prices"])
            dual = artifact_map.get("dual_scaffold")
            if dual is not None:
                lines.append(
                    f"The current model-grounded dual scaffold is available, so this interpretation is tied to {len(dual.data.get('dual_variables', []))} explicit dual variable(s)."
                )

        elif plan.object == "bids":
            lines.append(SECTION23_CONCEPTS["negative_bids"])
            lines.append(
                "In Sampat Section 2.3, negative bids change the economic interpretation from simple supply cost to willingness to pay for disposal, remediation, storage, or related service."
            )

        elif plan.object == "technologies":
            lines.append(
                "Technologies affect prices by linking products through yield coefficients, so one technology activity can create scarcity relief for some products while consuming others."
            )
            primal = artifact_map.get("primal_scaffold")
            if primal is not None:
                tech_variables = [
                    variable
                    for variable in primal.data.get("variables", [])
                    if variable.get("variable_class") == "technology_activity"
                ]
                lines.append(
                    f"The current primal scaffold includes {len(tech_variables)} technology activity variable(s), which is the model-grounded channel through which yield coefficients affect balances and prices."
                )

        elif plan.object == "benchmark_case":
            benchmark = artifact_map.get("benchmark_metadata")
            current_case = benchmark.data.get("case_family") if benchmark is not None else "unknown"
            lines.append(
                "Case A is the no-transformation benchmark, Case B keeps the same clearing structure but allows negative bids, and Case C adds transformation technologies with explicit yield coefficients."
            )
            lines.append(
                "In economic terms, Case B changes how bids and prices should be interpreted because some accepted activities can represent disposal or remediation value rather than ordinary supply cost, while Case C changes prices by coupling products through technology yields."
            )
            lines.append(
                "So a Case A versus Case B comparison should focus on bid sign conventions and price interpretation, whereas a Case A versus Case C comparison should focus on how technology activity reshapes scarcity across products."
            )
            lines.append(f"The current state is closest to {current_case}.")

        elif plan.object == "section23":
            lines.append(
                "Section 2.3 changes the interpretation of bids and prices by allowing economically meaningful negative bids and negative prices in the coordinated clearing framework."
            )
            lines.append(SECTION23_CONCEPTS["negative_bids"])
            lines.append(
                "That means prices should not be read only as ordinary purchase prices; they can also encode disposal, remediation, storage, or value-of-service effects when the benchmark data supports those mechanisms."
            )

        elif plan.object == "theorem":
            theorem_artifact = artifact_map.get("theorem_metadata")
            if theorem_artifact is not None:
                theorem_check = theorem_artifact.data.get("theorem_check") or {}
                applies = theorem_check.get("applies")
                if applies is True:
                    lines.append("The requested theorem is grounded by deterministic applicability checks for the current ProblemState.")
                else:
                    lines.append("The theorem path is supported, but theorem applicability depends on the deterministic assumptions recorded in the current state.")

        else:
            lines.append(
                "This question is within the supported Sampat Sections 2.1-2.3 scope, so the answer is being composed from curated paper/domain knowledge plus any available deterministic model artifacts."
            )

        if missing_artifacts and plan.grounding_mode != "paper":
            lines.append(
                "The answer is partially grounded, but some requested verification or model-specific artifacts are still missing."
            )

        return lines

    def _missing_information_text(self, missing_artifacts: List[MissingArtifact]) -> str:
        parts = [
            f"- {artifact.name}: {artifact.reason}"
            for artifact in missing_artifacts
        ]
        return "Not fully grounded yet because:\n" + "\n".join(parts)

    def _grounding_label(self, response_mode: str) -> str:
        labels = {
            "paper_grounded_explanation": "Paper-grounded explanation",
            "model_grounded_formulation": "Model-grounded formulation",
            "solver_grounded_verification": "Solver-grounded verification",
            "theorem_grounded_proof": "Theorem-grounded proof path",
        }
        return labels.get(response_mode, "Sampat reasoning response")
