"""Plotting utilities for reproducing Sampat et al. (2019) style figures.

This module provides functions to generate plotting-ready data from SolverResults
and create visualizations similar to those in the coordinated supply chain paper.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .solver_results import SolverResults


def prepare_imbalance_ratio_data(
    solver_results_list: List[SolverResults],
    parameter_values: List[float],
    parameter_name: str = "Parameter"
) -> pd.DataFrame:
    """Prepare data for imbalance ratio vs parameter sweep plots.

    Args:
        solver_results_list: List of SolverResults for different parameter values
        parameter_values: Corresponding parameter values
        parameter_name: Name of the parameter being swept

    Returns:
        DataFrame with columns: parameter, imbalance_ratio, objective_value
    """
    data = []
    for results, param in zip(solver_results_list, parameter_values):
        total_flow = sum(abs(flow) for flow in results.transport_flows.values()) + sum(
            abs(qty) for qty in results.bid_allocations.values()
        )
        slacks = list(results.constraint_slacks.values())
        imbalance_ratio = max(abs(slack) for slack in slacks) / total_flow if total_flow > 0 and slacks else 0

        data.append({
            parameter_name: param,
            "imbalance_ratio": imbalance_ratio,
            "objective_value": results.objective_value or 0,
        })

    return pd.DataFrame(data)


def prepare_nodal_price_map_data(solver_results: SolverResults) -> pd.DataFrame:
    """Prepare data for nodal price maps.

    Args:
        solver_results: SolverResults object

    Returns:
        DataFrame with columns: node, product, price
    """
    data = []
    for key, price in solver_results.node_product_duals.items():
        if price is not None:
            try:
                node, product = key.split(":")
                data.append({
                    "node": node,
                    "product": product,
                    "price": price,
                })
            except ValueError:
                continue

    return pd.DataFrame(data)


def prepare_transport_flow_data(solver_results: SolverResults) -> pd.DataFrame:
    """Prepare data for transport flow summaries.

    Args:
        solver_results: SolverResults object

    Returns:
        DataFrame with columns: origin, destination, product, flow, capacity, utilization
    """
    data = []
    # This requires linking back to the problem state - simplified version
    for row in solver_results.plotting_data.get("transport_flows", []):
        data.append({
            "origin": row.get("origin"),
            "destination": row.get("destination"),
            "product": row.get("product"),
            "flow": row.get("flow"),
            "utilization": row.get("utilization", 0.0),
        })

    return pd.DataFrame(data)


def prepare_scenario_comparison_data(
    base_results: SolverResults,
    scenario_results: List[SolverResults],
    scenario_names: List[str]
) -> pd.DataFrame:
    """Prepare data for scenario comparison plots.

    Args:
        base_results: Baseline SolverResults
        scenario_results: List of scenario SolverResults
        scenario_names: Names for each scenario

    Returns:
        DataFrame with comparison metrics
    """
    data = [{"scenario": "Base", "objective": base_results.objective_value or 0}]

    for results, name in zip(scenario_results, scenario_names):
        data.append({
            "scenario": name,
            "objective": results.objective_value or 0,
        })

    return pd.DataFrame(data)


def generate_sample_plot_data(solver_results: SolverResults) -> Dict[str, Any]:
    """Generate sample plotting data for demonstration.

    Args:
        solver_results: SolverResults object

    Returns:
        Dictionary with plotting-ready data
    """
    return dict(solver_results.plotting_data)


# Example plotting functions (for demonstration)

def plot_nodal_prices(solver_results: SolverResults, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Create a simple nodal price plot.

    Args:
        solver_results: SolverResults object
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    price_data = prepare_nodal_price_map_data(solver_results)
    if not price_data.empty:
        # Simple bar plot of prices by node-product
        labels = [f"{row['node']}-{row['product']}" for _, row in price_data.iterrows()]
        prices = price_data['price'].values

        ax.bar(labels, prices)
        ax.set_xlabel('Node-Product')
        ax.set_ylabel('Price')
        ax.set_title('Nodal Prices')
        plt.xticks(rotation=45)
    else:
        ax.text(0.5, 0.5, 'No price data available', ha='center', va='center', transform=ax.transAxes)

    return fig


def plot_transport_flows(solver_results: SolverResults, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Create a simple transport flow plot.

    Args:
        solver_results: SolverResults object
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    flow_data = prepare_transport_flow_data(solver_results)
    if not flow_data.empty:
        # Simple bar plot of flows
        labels = [f"{row['origin']}→{row['destination']}" for _, row in flow_data.iterrows()]
        flows = flow_data['flow'].values

        ax.bar(labels, flows)
        ax.set_xlabel('Transport Link')
        ax.set_ylabel('Flow')
        ax.set_title('Transport Flows')
        plt.xticks(rotation=45)
    else:
        ax.text(0.5, 0.5, 'No flow data available', ha='center', va='center', transform=ax.transAxes)

    return fig


__all__ = [
    "prepare_imbalance_ratio_data",
    "prepare_nodal_price_map_data",
    "prepare_transport_flow_data",
    "prepare_scenario_comparison_data",
    "generate_sample_plot_data",
    "plot_nodal_prices",
    "plot_transport_flows",
]
