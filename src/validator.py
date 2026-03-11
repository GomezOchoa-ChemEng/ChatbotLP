"""Validator for ProblemState objects.

Detects missing parameters, invalid references, incomplete technologies,
solver readiness, and benchmark compatibility for Cases A, B, and C.
"""

from typing import Dict, List, Tuple

from .schema import ProblemState


def validate_state(state: ProblemState) -> Dict:
    """Run validation on a ProblemState and return a diagnostics dict.

    The diagnostics dict contains:
    - issues: general problems (strings)
    - missing_parameters: list of missing parameter messages
    - invalid_references: list of invalid reference messages
    - incomplete_technologies: list of technology ids with incomplete yields
    - solver_ready: bool
    - benchmark_compatibility: dict case->{compatible: bool, explanation: str}
    """
    issues: List[str] = []
    missing: List[str] = []
    invalid_refs: List[str] = []
    incomplete_techs: List[str] = []

    node_ids = set(state.node_ids())
    product_ids = set(state.product_ids())

    # Basic missing sets
    if not node_ids:
        missing.append("no nodes defined")
    if not product_ids:
        missing.append("no products defined")

    # Duplicate id checks across collections
    def find_duplicates(ids: List[str]) -> List[str]:
        seen = set()
        dups = []
        for i in ids:
            if i in seen:
                dups.append(i)
            else:
                seen.add(i)
        return dups

    all_node_ids = [n.id for n in state.nodes]
    dup_nodes = find_duplicates(all_node_ids)
    if dup_nodes:
        invalid_refs.append(f"duplicate node ids: {dup_nodes}")

    # Check suppliers
    for s in state.suppliers:
        if s.node not in node_ids:
            invalid_refs.append(f"supplier:{s.id} references unknown node {s.node}")
        if s.product not in product_ids:
            invalid_refs.append(f"supplier:{s.id} references unknown product {s.product}")
        # capacity should be provided and non-negative
        if s.capacity is None:
            missing.append(f"supplier:{s.id} missing capacity")
        else:
            try:
                if s.capacity < 0:
                    missing.append(f"supplier:{s.id} has negative capacity {s.capacity}")
                elif s.capacity == 0:
                    issues.append(f"supplier:{s.id} has zero capacity")
            except TypeError:
                missing.append(f"supplier:{s.id} capacity not numeric: {s.capacity}")

    # Check consumers
    for c in state.consumers:
        if c.node not in node_ids:
            invalid_refs.append(f"consumer:{c.id} references unknown node {c.node}")
        if c.product not in product_ids:
            invalid_refs.append(f"consumer:{c.id} references unknown product {c.product}")
        if c.capacity is None:
            missing.append(f"consumer:{c.id} missing capacity")
        else:
            try:
                if c.capacity < 0:
                    missing.append(f"consumer:{c.id} has negative capacity {c.capacity}")
                elif c.capacity == 0:
                    issues.append(f"consumer:{c.id} has zero capacity")
            except TypeError:
                missing.append(f"consumer:{c.id} capacity not numeric: {c.capacity}")

    # Check transport links
    for t in state.transport_links:
        if t.origin not in node_ids:
            invalid_refs.append(f"transport:{t.id} unknown origin {t.origin}")
        if t.destination not in node_ids:
            invalid_refs.append(f"transport:{t.id} unknown destination {t.destination}")
        if t.product not in product_ids:
            invalid_refs.append(f"transport:{t.id} unknown product {t.product}")
        if t.capacity is None:
            missing.append(f"transport:{t.id} missing capacity")
        else:
            try:
                if t.capacity < 0:
                    missing.append(f"transport:{t.id} has negative capacity {t.capacity}")
                elif t.capacity == 0:
                    issues.append(f"transport:{t.id} has zero capacity")
            except TypeError:
                missing.append(f"transport:{t.id} capacity not numeric: {t.capacity}")

    # Check technologies (transformation)
    for tech in state.technologies:
        if tech.node not in node_ids:
            invalid_refs.append(f"technology:{tech.id} references unknown node {tech.node}")
        yields = tech.yield_coefficients
        if not yields:
            incomplete_techs.append(tech.id)
            missing.append(f"technology:{tech.id} has empty yield_coefficients")
            continue
        # verify product references
        pos = neg = 0
        for pid, coef in yields.items():
            if pid not in product_ids:
                invalid_refs.append(f"technology:{tech.id} yield references unknown product {pid}")
            if coef is None:
                missing.append(f"technology:{tech.id} yield for {pid} is None")
            else:
                if coef > 0:
                    pos += 1
                elif coef < 0:
                    neg += 1
        # incomplete if yields do not show transformation (need both input and output)
        if not (pos >= 1 and neg >= 1):
            incomplete_techs.append(tech.id)
        # technology capacity should be provided for transforming techs
        if tech.yield_coefficients and (tech.capacity is None or tech.capacity <= 0):
            incomplete_techs.append(tech.id)
            missing.append(f"technology:{tech.id} missing positive capacity for transformation")

    # Bids and references
    owner_sets = {
        "supplier": {s.id for s in state.suppliers},
        "consumer": {c.id for c in state.consumers},
        "transport": {t.id for t in state.transport_links},
        "technology": {x.id for x in state.technologies},
    }

    for b in state.bids:
        if b.owner_type not in owner_sets:
            invalid_refs.append(f"bid:{b.id} has unknown owner_type {b.owner_type}")
            continue
        if b.owner_id not in owner_sets[b.owner_type]:
            invalid_refs.append(f"bid:{b.id} owner {b.owner_id} not found in {b.owner_type}")
        if b.product_id not in product_ids:
            invalid_refs.append(f"bid:{b.id} references unknown product {b.product_id}")
        # quantity if provided must be non-negative
        if b.quantity is not None:
            try:
                if b.quantity < 0:
                    missing.append(f"bid:{b.id} has negative quantity {b.quantity}")
                elif b.quantity == 0:
                    issues.append(f"bid:{b.id} has zero quantity")
            except TypeError:
                missing.append(f"bid:{b.id} quantity not numeric: {b.quantity}")

    # Solver readiness heuristics
    # solver ready only if no missing, no invalid refs, and no incomplete techs
    solver_ready = (len(missing) == 0 and len(invalid_refs) == 0 and len(incomplete_techs) == 0)
    # additional check: there must be at least one supply and one sink for products
    any_supply = bool(state.suppliers or state.technologies)
    any_demand = bool(state.consumers or state.technologies)
    if not any_supply:
        missing.append("no supply-side entities (suppliers or technologies) defined")
        solver_ready = False
    if not any_demand:
        missing.append("no demand-side entities (consumers or technologies) defined")
        solver_ready = False

    # Benchmark compatibility checks
    benchmark_compat: Dict[str, Dict[str, object]] = {}

    # Case A: no transformation => there should be no technologies
    def case_a(state: ProblemState) -> Tuple[bool, str]:
        if state.technologies:
            return False, "technologies present (transformation detected)"
        return True, "no technologies present"

    # Case B: negative bidding costs => at least one bid.price < 0
    def case_b(state: ProblemState) -> Tuple[bool, str]:
        neg_bids = [b for b in state.bids if b.price < 0]
        if neg_bids:
            return True, f"found {len(neg_bids)} negative bids"
        return False, "no negative bids found"

    # Case C: transformation => at least one technology with both negative and positive yields
    def case_c(state: ProblemState) -> Tuple[bool, str]:
        for tech in state.technologies:
            pos = any(v > 0 for v in tech.yield_coefficients.values())
            neg = any(v < 0 for v in tech.yield_coefficients.values())
            cap_ok = tech.capacity is not None and tech.capacity > 0
            if pos and neg and cap_ok:
                return True, f"technology {tech.id} has transformation yields and positive capacity"
            if pos and neg and not cap_ok:
                return False, f"technology {tech.id} has yields but missing positive capacity"
        return False, "no technology with both input (neg) and output (pos) yields found"

    benchmark_compat["Case A"] = dict(compatible=case_a(state)[0], explanation=case_a(state)[1])
    benchmark_compat["Case B"] = dict(compatible=case_b(state)[0], explanation=case_b(state)[1])
    benchmark_compat["Case C"] = dict(compatible=case_c(state)[0], explanation=case_c(state)[1])

    diagnostics = {
        "issues": issues,
        "missing_parameters": missing,
        "invalid_references": invalid_refs,
        "incomplete_technologies": incomplete_techs,
        "solver_ready": solver_ready,
        "benchmark_compatibility": benchmark_compat,
    }

    return diagnostics


__all__ = ["validate_state"]
