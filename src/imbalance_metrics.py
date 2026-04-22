"""Thin imbalance-metrics layer for figure-oriented derived metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .solver_results import SolverResults


IMBALANCE_SIGN_NOTE = (
    "Positive imbalance means net surplus: supply + inflow + technology output "
    "exceeds demand + outflow + technology input."
)


def prepare_imbalance_metric_rows(
    results: SolverResults,
    baseline_results: Optional[SolverResults] = None,
) -> List[Dict[str, Any]]:
    """Build long-form imbalance-style metrics from a solved instance."""

    run_id = _scenario_value(results, "run_id", "run_1")
    scenario_id = _scenario_value(results, "scenario_id", run_id)
    rows: List[Dict[str, Any]] = []

    node_product = _compute_node_product_imbalance(results)
    product_totals = _compute_product_imbalance(node_product)
    regional_totals = _compute_regional_imbalance(node_product, results)

    for key, payload in node_product.items():
        node, product = key.split(":", 1)
        rows.extend(
            [
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_node_product",
                    "node_product_net_surplus",
                    payload["net_surplus"],
                    product=product,
                    node=node,
                    region=payload["region"],
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_node_product",
                    "node_product_local_surplus",
                    payload["local_surplus"],
                    product=product,
                    node=node,
                    region=payload["region"],
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_node_product",
                    "node_product_local_absolute_imbalance",
                    payload["local_absolute_imbalance"],
                    product=product,
                    node=node,
                    region=payload["region"],
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_node_product",
                    "node_product_absolute_imbalance",
                    payload["absolute_imbalance"],
                    product=product,
                    node=node,
                    region=payload["region"],
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_node_product",
                    "node_product_handled_volume",
                    payload["handled_volume"],
                    product=product,
                    node=node,
                    region=payload["region"],
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_node_product",
                    "node_product_local_normalized_imbalance_ratio",
                    payload["local_normalized_imbalance_ratio"],
                    product=product,
                    node=node,
                    region=payload["region"],
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_node_product",
                    "node_product_normalized_imbalance_ratio",
                    payload["normalized_imbalance_ratio"],
                    product=product,
                    node=node,
                    region=payload["region"],
                ),
            ]
        )

    for product, payload in product_totals.items():
        rows.extend(
            [
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_product",
                    "product_net_surplus",
                    payload["net_surplus"],
                    product=product,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_product",
                    "product_absolute_imbalance",
                    payload["absolute_imbalance"],
                    product=product,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_product",
                    "product_normalized_imbalance_ratio",
                    payload["normalized_imbalance_ratio"],
                    product=product,
                ),
            ]
        )

    for region, payload in regional_totals.items():
        rows.extend(
            [
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_region",
                    "regional_local_surplus",
                    payload["local_surplus"],
                    region=region,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_region",
                    "regional_local_absolute_imbalance",
                    payload["local_absolute_imbalance"],
                    region=region,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_region",
                    "regional_local_normalized_imbalance_ratio",
                    payload["local_normalized_imbalance_ratio"],
                    region=region,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_region",
                    "regional_net_surplus",
                    payload["net_surplus"],
                    region=region,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_region",
                    "regional_absolute_imbalance",
                    payload["absolute_imbalance"],
                    region=region,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_region",
                    "regional_normalized_imbalance_ratio",
                    payload["normalized_imbalance_ratio"],
                    region=region,
                ),
            ]
        )

    max_abs_node_product = max(
        (payload["absolute_imbalance"] for payload in node_product.values()),
        default=0.0,
    )
    rows.append(
        _metric_row(
            run_id,
            scenario_id,
            "imbalance_summary",
            "max_absolute_node_product_imbalance",
            max_abs_node_product,
        )
    )

    if baseline_results is not None:
        rows.extend(compare_imbalance_metric_rows(baseline_results, results))

    return rows


def compare_imbalance_metric_rows(
    baseline_results: SolverResults,
    scenario_results: SolverResults,
) -> List[Dict[str, Any]]:
    """Build scenario-to-baseline comparison rows for imbalance metrics."""

    run_id = _scenario_value(scenario_results, "run_id", "run_1")
    scenario_id = _scenario_value(scenario_results, "scenario_id", run_id)
    baseline_id = _scenario_value(
        scenario_results,
        "baseline_id",
        _scenario_value(baseline_results, "scenario_id", "baseline"),
    )

    baseline_node = _compute_node_product_imbalance(baseline_results)
    scenario_node = _compute_node_product_imbalance(scenario_results)
    all_keys = sorted(set(baseline_node) | set(scenario_node))

    rows: List[Dict[str, Any]] = []
    for key in all_keys:
        node, product = key.split(":", 1)
        baseline_payload = baseline_node.get(
            key,
            {
                "net_surplus": 0.0,
                "absolute_imbalance": 0.0,
                "normalized_imbalance_ratio": None,
                "local_surplus": 0.0,
                "local_absolute_imbalance": 0.0,
                "local_normalized_imbalance_ratio": None,
                "region": node,
            },
        )
        scenario_payload = scenario_node.get(
            key,
            {
                "net_surplus": 0.0,
                "absolute_imbalance": 0.0,
                "normalized_imbalance_ratio": None,
                "local_surplus": 0.0,
                "local_absolute_imbalance": 0.0,
                "local_normalized_imbalance_ratio": None,
                "region": node,
            },
        )
        rows.extend(
            [
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_comparison",
                    "node_product_local_surplus_delta",
                    scenario_payload["local_surplus"] - baseline_payload["local_surplus"],
                    product=product,
                    node=node,
                    region=scenario_payload["region"],
                    baseline_id=baseline_id,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_comparison",
                    "node_product_local_absolute_imbalance_delta",
                    scenario_payload["local_absolute_imbalance"] - baseline_payload["local_absolute_imbalance"],
                    product=product,
                    node=node,
                    region=scenario_payload["region"],
                    baseline_id=baseline_id,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_comparison",
                    "node_product_local_normalized_imbalance_ratio_delta",
                    _difference(
                        scenario_payload["local_normalized_imbalance_ratio"],
                        baseline_payload["local_normalized_imbalance_ratio"],
                    ),
                    product=product,
                    node=node,
                    region=scenario_payload["region"],
                    baseline_id=baseline_id,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_comparison",
                    "node_product_net_surplus_delta",
                    scenario_payload["net_surplus"] - baseline_payload["net_surplus"],
                    product=product,
                    node=node,
                    region=scenario_payload["region"],
                    baseline_id=baseline_id,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_comparison",
                    "node_product_absolute_imbalance_delta",
                    scenario_payload["absolute_imbalance"] - baseline_payload["absolute_imbalance"],
                    product=product,
                    node=node,
                    region=scenario_payload["region"],
                    baseline_id=baseline_id,
                ),
                _metric_row(
                    run_id,
                    scenario_id,
                    "imbalance_comparison",
                    "node_product_normalized_imbalance_ratio_delta",
                    _difference(
                        scenario_payload["normalized_imbalance_ratio"],
                        baseline_payload["normalized_imbalance_ratio"],
                    ),
                    product=product,
                    node=node,
                    region=scenario_payload["region"],
                    baseline_id=baseline_id,
                ),
            ]
        )

    return rows


def _compute_node_product_imbalance(results: SolverResults) -> Dict[str, Dict[str, Any]]:
    market_instance = results.market_instance
    if market_instance is None:
        return {}

    transport_by_arc = {
        (link.origin, link.destination, link.product_id): results.transport_flows.get(str((link.origin, link.destination)), 0.0)
        for link in market_instance.transport_links
    }
    bid_by_product_node = {}
    for bid in market_instance.bids:
        key = (bid.node, bid.product_id, bid.owner_type)
        bid_by_product_node[key] = bid_by_product_node.get(key, 0.0) + results.bid_allocations.get(bid.id, 0.0)

    region_map = _scenario_value(results, "region_map", {}) or {}
    rows: Dict[str, Dict[str, Any]] = {}
    for node in market_instance.nodes:
        for product in market_instance.products:
            supply = bid_by_product_node.get((node, product, "supplier"), 0.0)
            demand = bid_by_product_node.get((node, product, "consumer"), 0.0)
            inflow = sum(
                flow
                for (origin, destination, flow_product), flow in transport_by_arc.items()
                if destination == node and flow_product == product
            )
            outflow = sum(
                flow
                for (origin, destination, flow_product), flow in transport_by_arc.items()
                if origin == node and flow_product == product
            )
            technology_net = 0.0
            for technology in market_instance.technologies:
                if technology.node != node:
                    continue
                extent = results.technology_extents.get(technology.id, 0.0)
                technology_net += technology.yield_coefficients.get(product, 0.0) * extent

            local_surplus = supply + technology_net - demand
            net_surplus = supply + inflow + technology_net - demand - outflow
            handled_volume = supply + demand + inflow + outflow + abs(technology_net)
            local_handled_volume = supply + demand + abs(technology_net)
            normalized_ratio = abs(net_surplus) / handled_volume if handled_volume > 0 else None
            local_normalized_ratio = abs(local_surplus) / local_handled_volume if local_handled_volume > 0 else None
            key = f"{node}:{product}"
            rows[key] = {
                "net_surplus": net_surplus,
                "absolute_imbalance": abs(net_surplus),
                "handled_volume": handled_volume,
                "normalized_imbalance_ratio": normalized_ratio,
                "local_surplus": local_surplus,
                "local_absolute_imbalance": abs(local_surplus),
                "local_handled_volume": local_handled_volume,
                "local_normalized_imbalance_ratio": local_normalized_ratio,
                "region": region_map.get(node, node),
                "sign_convention_note": IMBALANCE_SIGN_NOTE,
            }

    return rows


def _compute_product_imbalance(node_product: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    products: Dict[str, Dict[str, Any]] = {}
    for key, payload in node_product.items():
        _, product = key.split(":", 1)
        target = products.setdefault(
            product,
            {
                "net_surplus": 0.0,
                "absolute_imbalance": 0.0,
                "handled_volume": 0.0,
                "normalized_imbalance_ratio": None,
                "local_surplus": 0.0,
                "local_absolute_imbalance": 0.0,
                "local_normalized_imbalance_ratio": None,
                "local_handled_volume": 0.0,
            },
        )
        target["net_surplus"] += payload["net_surplus"]
        target["absolute_imbalance"] += payload["absolute_imbalance"]
        target["handled_volume"] += payload["handled_volume"]
        target["local_surplus"] += payload["local_surplus"]
        target["local_absolute_imbalance"] += payload["local_absolute_imbalance"]
        target["local_handled_volume"] += payload["local_handled_volume"]

    for payload in products.values():
        payload["normalized_imbalance_ratio"] = (
            payload["absolute_imbalance"] / payload["handled_volume"]
            if payload["handled_volume"] > 0
            else None
        )
        payload["local_normalized_imbalance_ratio"] = (
            payload["local_absolute_imbalance"] / payload["local_handled_volume"]
            if payload["local_handled_volume"] > 0
            else None
        )
    return products


def _compute_regional_imbalance(
    node_product: Dict[str, Dict[str, Any]],
    results: SolverResults,
) -> Dict[str, Dict[str, Any]]:
    # `results` is kept in the signature so this helper can evolve alongside
    # scenario metadata or region-level aggregation options without changing
    # the public call shape used by the figure-data layer.
    regions: Dict[str, Dict[str, Any]] = {}
    for payload in node_product.values():
        region = payload["region"]
        target = regions.setdefault(
            region,
            {
                "net_surplus": 0.0,
                "absolute_imbalance": 0.0,
                "handled_volume": 0.0,
                "normalized_imbalance_ratio": None,
                "local_surplus": 0.0,
                "local_absolute_imbalance": 0.0,
                "local_handled_volume": 0.0,
                "local_normalized_imbalance_ratio": None,
            },
        )
        target["net_surplus"] += payload["net_surplus"]
        target["absolute_imbalance"] += payload["absolute_imbalance"]
        target["handled_volume"] += payload["handled_volume"]
        target["local_surplus"] += payload["local_surplus"]
        target["local_absolute_imbalance"] += payload["local_absolute_imbalance"]
        target["local_handled_volume"] += payload["local_handled_volume"]

    for payload in regions.values():
        payload["normalized_imbalance_ratio"] = (
            payload["absolute_imbalance"] / payload["handled_volume"]
            if payload["handled_volume"] > 0
            else None
        )
        payload["local_normalized_imbalance_ratio"] = (
            payload["local_absolute_imbalance"] / payload["local_handled_volume"]
            if payload["local_handled_volume"] > 0
            else None
        )
    return regions


def _metric_row(
    run_id: str,
    scenario_id: str,
    metric_group: str,
    metric_name: str,
    metric_value: Any,
    *,
    product: Optional[str] = None,
    node: Optional[str] = None,
    region: Optional[str] = None,
    baseline_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "baseline_id": baseline_id,
        "metric_group": metric_group,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "product": product,
        "node": node,
        "region": region,
        "sign_convention_note": IMBALANCE_SIGN_NOTE,
    }


def _difference(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return value - baseline


def _scenario_value(results: SolverResults, key: str, default: Optional[Any] = None) -> Optional[Any]:
    return results.scenario_metadata.get(key, default)


__all__ = [
    "IMBALANCE_SIGN_NOTE",
    "compare_imbalance_metric_rows",
    "prepare_imbalance_metric_rows",
]
