#!/usr/bin/env python3
"""Small benchmark-sized demo for imbalance-style metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_problem_interpreter import build_state_from_semantic_plan
from src.model_builder import build_market_instance, build_model_from_market_instance
from src.solver import solve_model
from src.solver_results import SolverResults


def build_case_a_state(transport_capacity: float):
    return build_state_from_semantic_plan(
        {
            "problem_title": f"Case A Imbalance Demo - Transport {transport_capacity}",
            "nodes": [
                {"id": "N1", "name": "Supply Node"},
                {"id": "N2", "name": "Demand Node"},
            ],
            "products": [{"id": "P1", "name": "Product 1"}],
            "suppliers": [{"id": "S1", "node": "N1", "product": "P1", "capacity": 10.0}],
            "consumers": [{"id": "C1", "node": "N2", "product": "P1", "capacity": 8.0}],
            "transport_links": [{"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": transport_capacity}],
            "bids": [
                {"id": "B_SUP", "owner_id": "S1", "owner_type": "supplier", "product_id": "P1", "price": 2.0, "quantity": 10.0},
                {"id": "B_CON", "owner_id": "C1", "owner_type": "consumer", "product_id": "P1", "price": 9.0, "quantity": 8.0},
            ],
            "technologies": [],
        }
    )


def solve_case(transport_capacity: float, scenario_id: str, baseline_results: SolverResults | None = None) -> SolverResults:
    state = build_case_a_state(transport_capacity)
    market_instance = build_market_instance(state)
    model = build_model_from_market_instance(market_instance)
    raw_solve_result = solve_model(model, solver_name="glpk", fallback_solver="glpk")
    raw_solve_result.scenario_metadata = {
        "run_id": f"run_{scenario_id}",
        "scenario_id": scenario_id,
        "parameter_name": "transport_capacity",
        "parameter_value": transport_capacity,
        "baseline_id": "baseline_transport_10",
        "region_map": {
            "N1": "upstream_region",
            "N2": "downstream_region",
        },
        "baseline_results": baseline_results,
    }
    return SolverResults.from_solve_result(raw_solve_result, state)


def run_demo():
    baseline = solve_case(10.0, "baseline_transport_10")
    constrained = solve_case(4.0, "constrained_transport_4", baseline_results=baseline)
    return baseline, constrained


def main():
    _, constrained = run_demo()
    table = pd.DataFrame(constrained.plotting_data["imbalance_metrics_table"])
    print("Imbalance metrics table:")
    print(table.to_string(index=False))
    print("\nLong-form records:")
    print(json.dumps(table.to_dict(orient="records"), indent=2, default=str))


if __name__ == "__main__":
    main()
