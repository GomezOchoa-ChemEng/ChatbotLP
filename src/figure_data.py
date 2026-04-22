"""Thin figure-data layer for plotting-oriented outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .imbalance_metrics import prepare_imbalance_metric_rows

if TYPE_CHECKING:
    from .solver_results import SolverResults


PRICE_SIGN_CONVENTION_NOTE = (
    "Raw Pyomo duals are stored as imported from node_balance constraints; "
    "normalized_price = - raw_shadow_price for an economic price interpretation."
)


def build_figure_data(results: SolverResults) -> Dict[str, Any]:
    """Build plotting-ready long-form tables from SolverResults."""

    return {
        "scenario_sweep_table": prepare_scenario_sweep_table(results),
        "utilization_table": prepare_utilization_table(results),
        "nodal_price_table": prepare_nodal_price_table(results),
        "bid_acceptance_table": prepare_bid_acceptance_table(results),
        "imbalance_metrics_table": prepare_imbalance_table(results),
        "derived_metrics_table": prepare_derived_metrics_table(results),
    }


def prepare_scenario_sweep_table(results: SolverResults) -> List[Dict[str, Any]]:
    """Prepare a long-form scenario sweep table."""

    run_id = _scenario_value(results, "run_id", "run_1")
    scenario_id = _scenario_value(results, "scenario_id", run_id)
    parameter_name = _scenario_value(results, "parameter_name")
    parameter_value = _scenario_value(results, "parameter_value")
    baseline_id = _scenario_value(results, "baseline_id")

    rows: List[Dict[str, Any]] = []

    for product, totals in results.totals_by_product.items():
        for metric_name, metric_value in totals.items():
            rows.append(
                _scenario_row(
                    run_id=run_id,
                    scenario_id=scenario_id,
                    parameter_name=parameter_name,
                    parameter_value=parameter_value,
                    baseline_id=baseline_id,
                    product=product,
                    node=None,
                    metric_name=f"product_{metric_name}",
                    metric_value=metric_value,
                )
            )

    for key, raw_shadow_price in results.node_product_duals.items():
        node, product = key.split(":", 1)
        rows.append(
            _scenario_row(
                run_id=run_id,
                scenario_id=scenario_id,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                baseline_id=baseline_id,
                product=product,
                node=node,
                metric_name="normalized_nodal_price",
                metric_value=-raw_shadow_price,
            )
        )

    for metric in prepare_derived_metrics_table(results):
        rows.append(
            _scenario_row(
                run_id=run_id,
                scenario_id=scenario_id,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                baseline_id=baseline_id,
                product=metric.get("product"),
                node=metric.get("node"),
                metric_name=metric["metric_name"],
                metric_value=metric["metric_value"],
            )
        )

    return rows


def prepare_utilization_table(results: SolverResults) -> List[Dict[str, Any]]:
    """Prepare a utilization table for suppliers, transport links, and technologies."""

    market_instance = results.market_instance
    if market_instance is None:
        return []

    rows: List[Dict[str, Any]] = []

    for bid in market_instance.bids:
        if bid.owner_type != "supplier":
            continue
        accepted_quantity = results.bid_allocations.get(bid.id, 0.0)
        capacity = bid.quantity
        rows.append(
            {
                "asset_type": "supplier",
                "asset_id": bid.owner_id,
                "bid_id": bid.id,
                "node": bid.node,
                "product": bid.product_id,
                "capacity": capacity,
                "cleared_quantity": accepted_quantity,
                "utilization_fraction": _fraction(accepted_quantity, capacity),
            }
        )

    for link in market_instance.transport_links:
        flow_key = str((link.origin, link.destination))
        flow = results.transport_flows.get(flow_key, 0.0)
        rows.append(
            {
                "asset_type": "transport",
                "asset_id": link.id,
                "bid_id": None,
                "node": None,
                "product": link.product_id,
                "capacity": link.capacity,
                "cleared_quantity": flow,
                "utilization_fraction": _fraction(flow, link.capacity),
                "origin": link.origin,
                "destination": link.destination,
            }
        )

    for technology in market_instance.technologies:
        extent = results.technology_extents.get(technology.id, 0.0)
        rows.append(
            {
                "asset_type": "technology",
                "asset_id": technology.id,
                "bid_id": None,
                "node": technology.node,
                "product": None,
                "capacity": technology.capacity,
                "cleared_quantity": extent,
                "utilization_fraction": _fraction(extent, technology.capacity),
            }
        )

    return rows


def prepare_nodal_price_table(results: SolverResults) -> List[Dict[str, Any]]:
    """Prepare an explicit nodal price table with normalized sign convention."""

    rows = []
    for key, raw_shadow_price in results.node_product_duals.items():
        node, product = key.split(":", 1)
        rows.append(
            {
                "node": node,
                "product": product,
                "raw_shadow_price": raw_shadow_price,
                "normalized_price": -raw_shadow_price,
                "sign_convention_note": PRICE_SIGN_CONVENTION_NOTE,
            }
        )
    return rows


def prepare_bid_acceptance_table(results: SolverResults) -> List[Dict[str, Any]]:
    """Prepare accepted-bid rows for plotting."""

    market_instance = results.market_instance
    if market_instance is None:
        return []

    rows = []
    for bid in market_instance.bids:
        accepted_quantity = results.bid_allocations.get(bid.id, 0.0)
        capacity = bid.quantity
        rows.append(
            {
                "bid_id": bid.id,
                "owner": bid.owner_id,
                "node": bid.node,
                "product": bid.product_id,
                "bid_type": bid.owner_type,
                "bid_price": bid.price,
                "capacity": capacity,
                "accepted_quantity": accepted_quantity,
                "acceptance_fraction": _fraction(accepted_quantity, capacity),
            }
        )
    return rows


def prepare_imbalance_table(results: SolverResults) -> List[Dict[str, Any]]:
    """Prepare a dedicated long-form imbalance metrics table."""

    baseline_results = results.scenario_metadata.get("baseline_results")
    return prepare_imbalance_metric_rows(results, baseline_results=baseline_results)


def prepare_derived_metrics_table(results: SolverResults) -> List[Dict[str, Any]]:
    """Prepare a small long-form table of figure-oriented derived metrics."""

    run_id = _scenario_value(results, "run_id", "run_1")
    scenario_id = _scenario_value(results, "scenario_id", run_id)
    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "metric_group": "welfare",
            "metric_name": "total_welfare",
            "metric_value": results.objective_value,
            "product": None,
            "node": None,
        }
    )

    total_cleared_quantity = sum(results.bid_allocations.values())
    total_transport_flow = sum(results.transport_flows.values())
    max_constraint_slack = max((abs(value) for value in results.constraint_slacks.values()), default=0.0)
    rows.extend(
        [
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "metric_group": "system",
                "metric_name": "total_cleared_quantity",
                "metric_value": total_cleared_quantity,
                "product": None,
                "node": None,
            },
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "metric_group": "system",
                "metric_name": "total_transport_flow",
                "metric_value": total_transport_flow,
                "product": None,
                "node": None,
            },
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "metric_group": "imbalance",
                "metric_name": "max_constraint_slack",
                "metric_value": max_constraint_slack,
                "product": None,
                "node": None,
            },
        ]
    )

    for product, totals in results.totals_by_product.items():
        for metric_name, metric_value in totals.items():
            rows.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "metric_group": "product_totals",
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "product": product,
                    "node": None,
                }
            )

    rows.extend(prepare_imbalance_table(results))

    comparison_metrics = results.scenario_metadata.get("comparison_metrics", {})
    for metric_name, metric_value in comparison_metrics.items():
        rows.append(
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "metric_group": "scenario_comparison",
                "metric_name": metric_name,
                "metric_value": metric_value,
                "product": None,
                "node": None,
            }
        )

    return rows


def _scenario_row(
    *,
    run_id: str,
    scenario_id: str,
    parameter_name: Optional[str],
    parameter_value: Optional[Any],
    baseline_id: Optional[str],
    product: Optional[str],
    node: Optional[str],
    metric_name: str,
    metric_value: Any,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "baseline_id": baseline_id,
        "product": product,
        "node": node,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


def _fraction(numerator: float, denominator: Optional[float]) -> Optional[float]:
    if denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _scenario_value(results: SolverResults, key: str, default: Optional[Any] = None) -> Optional[Any]:
    return results.scenario_metadata.get(key, default)


__all__ = [
    "PRICE_SIGN_CONVENTION_NOTE",
    "build_figure_data",
    "prepare_bid_acceptance_table",
    "prepare_derived_metrics_table",
    "prepare_imbalance_table",
    "prepare_nodal_price_table",
    "prepare_scenario_sweep_table",
    "prepare_utilization_table",
]
