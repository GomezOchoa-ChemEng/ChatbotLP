#!/usr/bin/env python3
"""Prototype scenario-sweep figure workflow using SolverResults.figure-data tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_problem_interpreter import build_state_from_semantic_plan
from src.model_builder import build_market_instance, build_model_from_market_instance
from src.solver import solve_model
from src.solver_results import SolverResults


def build_case_a_state(transport_capacity: float):
    semantic_plan = {
        "problem_title": f"Case A Sweep - Transport {transport_capacity}",
        "nodes": [
            {"id": "N1", "name": "Supply Node"},
            {"id": "N2", "name": "Demand Node"},
        ],
        "products": [
            {"id": "P1", "name": "Product 1"},
        ],
        "suppliers": [
            {"id": "S1", "node": "N1", "product": "P1", "capacity": 10.0},
        ],
        "consumers": [
            {"id": "C1", "node": "N2", "product": "P1", "capacity": 8.0},
        ],
        "transport_links": [
            {"id": "T1", "origin": "N1", "destination": "N2", "product": "P1", "capacity": transport_capacity, "cost": 0.0},
        ],
        "bids": [
            {
                "id": "B_SUP",
                "owner_id": "S1",
                "owner_type": "supplier",
                "product_id": "P1",
                "price": 2.0,
                "quantity": 10.0,
            },
            {
                "id": "B_CON",
                "owner_id": "C1",
                "owner_type": "consumer",
                "product_id": "P1",
                "price": 9.0,
                "quantity": 8.0,
            },
        ],
        "technologies": [],
    }
    return build_state_from_semantic_plan(semantic_plan)


def solve_case_a_scenario(transport_capacity: float, run_index: int) -> SolverResults:
    state = build_case_a_state(transport_capacity)
    market_instance = build_market_instance(state)
    model = build_model_from_market_instance(market_instance)
    solve_result = solve_model(model, solver_name="glpk", fallback_solver="glpk")
    solve_result.scenario_metadata = {
        "run_id": f"run_case_a_{run_index}",
        "scenario_id": f"transport_cap_{transport_capacity:g}",
        "parameter_name": "transport_capacity",
        "parameter_value": transport_capacity,
        "baseline_id": "transport_cap_10",
        "comparison_metrics": {},
    }
    return SolverResults.from_solve_result(solve_result, state)


def build_scenario_sweep_dataframe(results_list: Iterable[SolverResults]) -> pd.DataFrame:
    rows: List[dict] = []
    for results in results_list:
        rows.extend(results.plotting_data["scenario_sweep_table"])
    return pd.DataFrame(rows)


def build_total_welfare_plot(table: pd.DataFrame):
    plot_table = table.loc[table["metric_name"] == "total_welfare"].copy()
    plot_table = plot_table.sort_values(["parameter_value", "scenario_id"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        plot_table["parameter_value"],
        plot_table["metric_value"],
        marker="o",
        linewidth=2,
        color="#1f5aa6",
    )
    ax.set_xlabel("Transport Capacity")
    ax.set_ylabel("Total Welfare")
    ax.set_title("Case A Scenario Sweep: Welfare vs Transport Capacity")
    ax.grid(True, alpha=0.3)
    return fig, ax, plot_table


def build_utilization_dataframe(results_list: Iterable[SolverResults]) -> pd.DataFrame:
    rows: List[dict] = []
    for results in results_list:
        scenario_id = results.scenario_metadata.get("scenario_id")
        parameter_name = results.scenario_metadata.get("parameter_name")
        parameter_value = results.scenario_metadata.get("parameter_value")
        for row in results.plotting_data["utilization_table"]:
            enriched_row = dict(row)
            enriched_row["scenario_id"] = scenario_id
            enriched_row["parameter_name"] = parameter_name
            enriched_row["parameter_value"] = parameter_value
            rows.append(enriched_row)
    return pd.DataFrame(rows)


def build_supplier_utilization_plot(table: pd.DataFrame):
    plot_table = table.loc[table["asset_type"] == "supplier"].copy()
    plot_table = plot_table.sort_values(["parameter_value", "asset_id"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        plot_table["parameter_value"].astype(str),
        plot_table["utilization_fraction"],
        color="#2c7c31",
        width=0.6,
    )
    ax.set_xlabel("Transport Capacity")
    ax.set_ylabel("Supplier Utilization Fraction")
    ax.set_title("Case A Scenario Sweep: Supplier Utilization")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax, plot_table


def run_demo():
    capacities = [4.0, 8.0, 10.0]
    results_list = [solve_case_a_scenario(capacity, idx + 1) for idx, capacity in enumerate(capacities)]
    scenario_sweep_df = build_scenario_sweep_dataframe(results_list)
    utilization_df = build_utilization_dataframe(results_list)
    fig, ax, plot_table = build_total_welfare_plot(scenario_sweep_df)
    utilization_fig, utilization_ax, utilization_plot_table = build_supplier_utilization_plot(utilization_df)
    return {
        "solver_results": results_list,
        "scenario_sweep_df": scenario_sweep_df,
        "utilization_df": utilization_df,
        "plot_table": plot_table,
        "utilization_plot_table": utilization_plot_table,
        "figure": fig,
        "axes": ax,
        "utilization_figure": utilization_fig,
        "utilization_axes": utilization_ax,
    }


def main():
    demo = run_demo()
    print("Scenario sweep table used for plotting:")
    print(demo["plot_table"].to_string(index=False))
    print("\nExact plotting call:")
    print(
        'ax.plot(plot_table["parameter_value"], plot_table["metric_value"], '
        'marker="o", linewidth=2, color="#1f5aa6")'
    )

    print("\nUtilization table used for plotting:")
    print(demo["utilization_plot_table"].to_string(index=False))
    print("\nExact utilization plotting call:")
    print(
        'ax.bar(plot_table["parameter_value"].astype(str), '
        'plot_table["utilization_fraction"], color="#2c7c31", width=0.6)'
    )

    output_path = Path(__file__).with_name("scenario_sweep_case_a.png")
    demo["figure"].tight_layout()
    demo["figure"].savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")

    utilization_output_path = Path(__file__).with_name("scenario_sweep_case_a_utilization.png")
    demo["utilization_figure"].tight_layout()
    demo["utilization_figure"].savefig(utilization_output_path, dpi=150)
    print(f"Saved utilization plot to: {utilization_output_path}")

    serializable = demo["plot_table"].to_dict(orient="records")
    print("\nPlot-table records:")
    print(json.dumps(serializable, indent=2))

    utilization_serializable = demo["utilization_plot_table"].to_dict(orient="records")
    print("\nUtilization plot-table records:")
    print(json.dumps(utilization_serializable, indent=2))


if __name__ == "__main__":
    main()
