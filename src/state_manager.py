"""State manager for ProblemState objects.

Provides a small helper class to manage a `ProblemState` instance,
serialize/deserialize it, apply simple scenario modifications, and run
lightweight validation that fills `missing_parameters` for downstream
modules to check before building/solving models.

Designed to be Colab-friendly (pure Python + pydantic).
"""

from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from pydantic import ValidationError

from .schema import ProblemState, ScenarioRecord


class StateManager:
    """Manage a ProblemState instance.

    Typical usage:
    ```py
    mgr = StateManager()
    mgr.load_from_dict(data)
    mgr.validate()
    mgr.save_to_file("state.json")
    ```
    """

    def __init__(self, state: Optional[ProblemState] = None) -> None:
        self.state = state or ProblemState()

    def get_state(self) -> ProblemState:
        return self.state

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        try:
            self.state = ProblemState.from_dict(data)
        except ValidationError as e:
            raise

    def to_dict(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def save_to_file(self, path: str) -> None:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), default=str, indent=2), encoding="utf-8")

    def load_from_file(self, path: str) -> None:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        self.load_from_dict(data)

    def add_scenario_record(self, name: str, description: Optional[str] = None) -> None:
        rec = ScenarioRecord(name=name, description=description)
        self.state.add_scenario(rec)

    def apply_scenario_changes(self, changes: Dict[str, Any]) -> None:
        """Apply lightweight scenario changes described by a dict.

        Supported keys (partial): `add_nodes`, `add_products`, `add_suppliers`,
        `add_consumers`, `add_transport_links`, `add_technologies`, `add_bids`.

        Each value should be a list of dictionaries convertible to the
        corresponding Pydantic model. This method performs minimal parsing
        then calls `validate()` to refresh missing-parameter diagnostics.
        """
        if not isinstance(changes, dict):
            raise ValueError("changes must be a dict")

        # Defensive mapping from change-key -> state add method
        from .schema import (
            Node,
            Product,
            Supplier,
            Consumer,
            TransportLink,
            Technology,
            Bid,
        )

        mapping = {
            "add_nodes": (self.state.add_node, Node),
            "add_products": (self.state.add_product, Product),
            "add_suppliers": (self.state.add_supplier, Supplier),
            "add_consumers": (self.state.add_consumer, Consumer),
            "add_transport_links": (self.state.add_transport, TransportLink),
            "add_technologies": (self.state.add_technology, Technology),
            "add_bids": (self.state.add_bid, Bid),
        }

        for key, payload in changes.items():
            if key not in mapping:
                continue
            add_fn, _ = mapping[key]
            if not isinstance(payload, list):
                continue
            for item in payload:
                Model = mapping[key][1]
                try:
                    obj = Model.parse_obj(item) if isinstance(item, dict) else Model.parse_obj(item)
                except Exception:
                    # fallback: try to pass through (may raise later)
                    obj = item
                add_fn(obj)

        # refresh validation diagnostics
        self.validate()

    def validate(self) -> List[str]:
        """Run light validation and update `state.missing_parameters`.

        Checks performed:
        - referenced nodes/products exist for suppliers/consumers/links/tech
        - bids reference existing owners and products
        - technology yield coefficients reference known products

        Returns the list of missing parameter messages.
        """
        missing: List[str] = []

        node_ids = set(self.state.node_ids())
        product_ids = set(self.state.product_ids())
        # report completely empty sets
        if not node_ids:
            missing.append("no nodes defined")
        if not product_ids:
            missing.append("no products defined")

        # check suppliers
        for s in self.state.suppliers:
            if s.node not in node_ids:
                missing.append(f"supplier:{s.id} missing node {s.node}")
            if s.product not in product_ids:
                missing.append(f"supplier:{s.id} missing product {s.product}")

        # consumers
        for c in self.state.consumers:
            if c.node not in node_ids:
                missing.append(f"consumer:{c.id} missing node {c.node}")
            if c.product not in product_ids:
                missing.append(f"consumer:{c.id} missing product {c.product}")

        # transport
        for t in self.state.transport_links:
            if t.origin not in node_ids:
                missing.append(f"transport:{t.id} missing origin {t.origin}")
            if t.destination not in node_ids:
                missing.append(f"transport:{t.id} missing destination {t.destination}")
            if t.product not in product_ids:
                missing.append(f"transport:{t.id} missing product {t.product}")

        # technologies
        for tech in self.state.technologies:
            if tech.node not in node_ids:
                missing.append(f"technology:{tech.id} missing node {tech.node}")
            for pid in tech.yield_coefficients.keys():
                if pid not in product_ids:
                    missing.append(f"technology:{tech.id} unknown product in yields: {pid}")

        # bids
        owner_sets = {
            "supplier": {s.id for s in self.state.suppliers},
            "consumer": {c.id for c in self.state.consumers},
            "transport": {t.id for t in self.state.transport_links},
            "technology": {x.id for x in self.state.technologies},
        }

        for b in self.state.bids:
            if b.owner_type not in owner_sets:
                missing.append(f"bid:{b.id} unknown owner_type {b.owner_type}")
                continue
            if b.owner_id not in owner_sets[b.owner_type]:
                missing.append(f"bid:{b.id} owner {b.owner_id} not found in {b.owner_type}")
            if b.product_id not in product_ids:
                missing.append(f"bid:{b.id} unknown product {b.product_id}")

        # store diagnostics
        self.state.missing_parameters = missing
        return missing
