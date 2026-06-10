"""Run the Midterm 1 Manure Management manure benchmarks.

Local usage from the repository root:

    python3 examples/midterm_manure_q1_benchmark.py --no-llm --skip-reasoning
    python3 examples/midterm_manure_q1_benchmark.py --question q2 --no-llm --skip-reasoning
    python3 examples/midterm_manure_q1_benchmark.py --question q3 --no-llm --skip-reasoning
    python3 examples/midterm_manure_q1_benchmark.py --question q4 --no-llm --skip-reasoning

Set GEMINI_API_KEY, LLM_PROVIDER=gemini, and GEMINI_MODEL to run live Gemini
interpretation. Live LLM failures are reported as live_llm_failure metadata.
Use --no-llm for deterministic_fixture_mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.midterm_benchmark import (  # noqa: E402
    MidtermBenchmarkConfig,
    run_midterm_manure_q1_benchmark,
    run_midterm_manure_q2_benchmark,
    run_midterm_manure_q3_benchmark,
    run_midterm_manure_q4_benchmark,
    write_midterm_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Midterm Manure benchmark for ChatbotLP.")
    parser.add_argument(
        "--question",
        choices=["q1", "q2", "q3", "q4"],
        default="q1",
        help="Midterm manure question to run.",
    )
    parser.add_argument(
        "--strict-llm",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Live LLM failures are always reported "
            "without deterministic fixture substitution."
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable live LLM interpretation and use the deterministic benchmark fixture.",
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        help="Skip the reasoning prompt battery.",
    )
    parser.add_argument(
        "--prompt-id",
        choices=["canonical", "paraphrased", "incomplete", "ambiguous"],
        default=None,
        help="Run one prompt only. By default all prompts are evaluated.",
    )
    parser.add_argument(
        "--reasoning-prompt-id",
        default=None,
        help="Run one reasoning prompt only. By default all reasoning prompts are evaluated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("midterm_outputs"),
        help="Directory for CSV tables and run_summary.json.",
    )
    return parser.parse_args()


def print_table(name: str, table: pd.DataFrame) -> None:
    print(f"\n=== {name.replace('_', ' ').title()} ===")
    if table.empty:
        print("(empty)")
    else:
        print(table.to_string(index=False))


def main() -> None:
    args = parse_args()
    prompt_ids = (args.prompt_id,) if args.prompt_id else None
    reasoning_prompt_ids = (args.reasoning_prompt_id,) if args.reasoning_prompt_id else None
    use_llm = not args.no_llm

    config = MidtermBenchmarkConfig(
        prompt_ids=prompt_ids,
        reasoning_prompt_ids=reasoning_prompt_ids,
        use_llm=use_llm,
        use_llm_for_reasoning=use_llm,
        use_deterministic_fixture=not use_llm,
        run_reasoning=not args.skip_reasoning,
    )
    runners = {
        "q1": run_midterm_manure_q1_benchmark,
        "q2": run_midterm_manure_q2_benchmark,
        "q3": run_midterm_manure_q3_benchmark,
        "q4": run_midterm_manure_q4_benchmark,
    }
    runner = runners[args.question]
    report = runner(config=config)

    print(f"Midterm Manure {args.question.upper()} benchmark")
    print(json.dumps(report["metadata"], indent=2, default=str))

    print("\nReference solution:")
    reference = report["reference_solution"]
    print(
        json.dumps(
            {
                "objective_value": reference["objective_value"],
                "demand_revenue": reference["demand_revenue"],
                "transport_cost": reference["transport_cost"],
                "technology_cost": reference.get("technology_cost"),
                "transport_flows": reference["transport_flows"],
                "technology_activity": reference.get("technology_activity", {}),
            },
            indent=2,
        )
    )

    for name in [
        "case_summary",
        "solve_accuracy",
        "interpretation_errors",
        "semantic_count_metrics",
        "parameter_multiset_metrics",
        "topology_metrics",
        "technology_yield_metrics",
        "route_economics_metrics",
        "route_association_metrics",
        "transport_link_attribute_metrics",
        "solver_aggregate_metrics",
        "balance_residual_metrics",
        "formulation_completeness_metrics",
        "solve_correctness_metrics",
        "reasoning_readiness_metrics",
        "removal_incentive_diagnostics",
        "alias_resolution_diagnostics",
        "reasoning_prompt_success",
        "interpretation_metadata",
    ]:
        print_table(name, report["tables"][name])

    write_midterm_outputs(report, args.output_dir)
    print(f"\nWrote benchmark tables to {args.output_dir}")


if __name__ == "__main__":
    main()
