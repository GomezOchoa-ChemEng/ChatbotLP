"""Run the Midterm 1 Manure Management Q1 benchmark.

Local usage from the repository root:

    python examples/midterm_manure_q1_benchmark.py --no-llm --skip-reasoning

Set GEMINI_API_KEY, LLM_PROVIDER=gemini, and GEMINI_MODEL to run live Gemini
interpretation. Use --strict-llm to record live failures without fixture
fallback.
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
    write_midterm_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Midterm Manure Q1 benchmark for ChatbotLP.")
    parser.add_argument(
        "--strict-llm",
        action="store_true",
        help="Do not fall back to the deterministic fixture if live LLM interpretation fails.",
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
    use_llm = not args.no_llm

    config = MidtermBenchmarkConfig(
        prompt_ids=prompt_ids,
        use_llm=use_llm,
        use_llm_for_reasoning=use_llm,
        fallback_to_reference_fixture=not args.strict_llm,
        run_reasoning=not args.skip_reasoning,
    )
    report = run_midterm_manure_q1_benchmark(config=config)

    print("Midterm Manure Q1 benchmark")
    print(json.dumps(report["metadata"], indent=2, default=str))

    print("\nReference solution:")
    reference = report["reference_solution"]
    print(
        json.dumps(
            {
                "objective_value": reference["objective_value"],
                "demand_revenue": reference["demand_revenue"],
                "transport_cost": reference["transport_cost"],
                "transport_flows": reference["transport_flows"],
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
        "route_economics_metrics",
        "solver_aggregate_metrics",
        "balance_residual_metrics",
        "alias_resolution_diagnostics",
        "reasoning_prompt_success",
        "interpretation_metadata",
    ]:
        print_table(name, report["tables"][name])

    write_midterm_outputs(report, args.output_dir)
    print(f"\nWrote benchmark tables to {args.output_dir}")


if __name__ == "__main__":
    main()
