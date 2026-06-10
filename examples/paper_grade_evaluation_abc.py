"""Run paper-grade ABC evaluation for the ChatbotLP prose-to-solver pipeline.

Local usage from the repository root:

    python examples/paper_grade_evaluation_abc.py

Set GEMINI_API_KEY in the script environment to run live Gemini interpretation.
Without a key, live mode reports live_llm_failure metadata. Use --no-llm for
deterministic_fixture_mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paper_grade_evaluation import (  # noqa: E402
    EVALUATION_PRESETS,
    EvaluationConfig,
    build_evaluation_config,
    plot_error_categories,
    plot_passed_checks_by_case,
    run_paper_grade_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade ABC evaluation for ChatbotLP.")
    parser.add_argument("--mode", default="guided", choices=["hint", "guided", "full", "exploration"])
    parser.add_argument(
        "--preset",
        default="full_evaluation",
        choices=sorted(EVALUATION_PRESETS),
        help="Named staged evaluation preset.",
    )
    parser.add_argument(
        "--quota-safe-demo",
        action="store_true",
        help="Run canonical Case A, negative-bid Case B, and transformation Case C interpretation only.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case names, for example canonical_case_a,negative_bid_case_b.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable live LLM interpretation and use expected benchmark fixtures explicitly.",
    )
    parser.add_argument(
        "--strict-llm",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Live LLM failures are always reported "
            "without expected fixture substitution."
        ),
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        help="Skip the reasoning prompt battery.",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Alias for --skip-reasoning.",
    )
    parser.add_argument(
        "--reasoning-only",
        action="store_true",
        help="Skip live interpretation and run reasoning prompts against fixture state.",
    )
    parser.add_argument(
        "--reasoning-prompts",
        default=None,
        help="Comma-separated reasoning prompt ids, for example primal_lp,dual_lp.",
    )
    parser.add_argument(
        "--no-solve",
        action="store_true",
        help="Skip solver attempts even for solver-ready cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV tables and PNG figures.",
    )
    return parser.parse_args()


def parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def print_table(name: str, table: pd.DataFrame) -> None:
    print(f"\n=== {name.replace('_', ' ').title()} ===")
    if table.empty:
        print("(empty)")
    else:
        print(table.to_string(index=False))


def write_outputs(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    plot_passed_checks_by_case(tables["case_level_summary"], ax=ax1)
    fig1.tight_layout()
    fig1.savefig(output_dir / "passed_checks_by_case.png", dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_error_categories(tables["error_category_counts"], ax=ax2)
    fig2.tight_layout()
    fig2.savefig(output_dir / "error_categories_across_cases.png", dpi=160)
    plt.close(fig2)


def main() -> None:
    args = parse_args()
    selected_cases = parse_csv(args.cases)
    reasoning_prompt_subset = parse_csv(args.reasoning_prompts)
    preset = "quota_safe_demo" if args.quota_safe_demo else args.preset
    skip_reasoning = args.skip_reasoning or args.no_reasoning

    overrides = {
        "mode": args.mode,
        "selected_cases": selected_cases,
        "use_llm": not args.no_llm,
        "use_llm_for_reasoning": not args.no_llm,
        "use_deterministic_fixture": args.no_llm,
        "attempt_solve": False if args.no_solve else None,
        "run_reasoning": False if skip_reasoning else None,
        "reasoning_prompt_subset": reasoning_prompt_subset,
    }

    if args.reasoning_only:
        overrides["run_interpretation"] = False
        overrides["run_reasoning"] = True
        if selected_cases is None:
            overrides["selected_cases"] = ("canonical_case_a",)

    config = build_evaluation_config(
        preset=preset,
        **overrides,
    )
    report = run_paper_grade_evaluation(config=config)

    print("Paper-grade ABC evaluation")
    print(f"LLM_PROVIDER={os.getenv('LLM_PROVIDER')!r}")
    print(f"GEMINI_MODEL={os.getenv('GEMINI_MODEL')!r}")
    print(f"Gemini configured: {report['metadata']['gemini_configured']}")
    print(f"Config: {report['metadata']['config']}")

    metadata_rows = [
        {
            "case": case["name"],
            "evaluation_mode": case["interpretation_metadata"].get("evaluation_mode"),
            "interpretation_source": case["interpretation_metadata"].get("interpretation_source"),
            "deterministic_fixture_used": case["interpretation_metadata"].get("deterministic_fixture_used"),
            "failure_type": case["interpretation_metadata"].get("failure_type"),
            "llm_failure": case["interpretation_metadata"].get("llm_failure"),
        }
        for case in report["cases"]
    ]
    tables = dict(report["tables"])
    tables["interpretation_metadata"] = pd.DataFrame(metadata_rows)

    for name in [
        "case_level_summary",
        "interpretation_accuracy",
        "solver_readiness_accuracy",
        "solve_accuracy",
        "reasoning_prompt_success",
        "error_category_counts",
        "interpretation_metadata",
    ]:
        print_table(name, tables[name])

    if args.output_dir is not None:
        write_outputs(tables, args.output_dir)
        print(f"\nWrote CSV tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
