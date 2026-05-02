"""Run a lightweight stress-test for the prose interpreter milestone."""

from __future__ import annotations

import json

from src.prose_interpreter_evaluation import evaluate_benchmark_cases


def main() -> None:
    report = evaluate_benchmark_cases(mode="guided")

    print("\nProse Interpreter Stress Test\n")
    print(report["summary"])

    if not report.get("ran"):
        return

    for case in report["cases"]:
        print(f"\n=== {case['label']} ===\n")
        print("Prose input:\n")
        print(case["prose_input"])
        print("\nSemantic plan:\n")
        print(json.dumps(case["semantic_plan"], indent=2, default=str))
        print("\nProblemState summary:\n")
        print(json.dumps(case["problem_state_summary"], indent=2, default=str))
        print("\nValidation result:\n")
        print(json.dumps(case["validation_result"], indent=2, default=str))
        print("\nSolve succeeded:\n")
        print(case["solve_succeeded"])
        print("\nExplanation output:\n")
        print(case["explanation_output"])
        print("\nComparison summary:\n")
        print(json.dumps(case["comparison"], indent=2, default=str))


if __name__ == "__main__":
    main()
