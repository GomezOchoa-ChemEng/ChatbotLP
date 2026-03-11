"""Rule-based parser for extracting supply chain entities from natural language.

This module provides a simple rule-based parser that uses regular expressions
and keyword matching to extract structured entities (nodes, products,
suppliers, consumers, etc.) from plain text descriptions of supply chain
problems. The design is modular to allow easy extension with additional rules
or replacement with an LLM-based parser in the future.

The main entry point is ``parse_supply_chain_text(text)``, which returns a
dictionary of entity lists that can be used to populate a ProblemState.
"""

import re
from typing import Dict, List, Any


class RuleBasedParser:
    """A simple rule-based parser for supply chain entity extraction.

    This class encapsulates the parsing logic using predefined regular
    expression patterns. It can be extended by adding more patterns or
    overridden for more sophisticated parsing.
    """

    def __init__(self):
        # Define patterns for each entity type
        # These are simple examples; in practice, more robust patterns would be needed
        self.patterns = {
            "nodes": re.compile(r"node\s+(\w+)", re.IGNORECASE),
            "products": re.compile(r"product\s+(\w+)", re.IGNORECASE),
            "suppliers": re.compile(
                r"supplier\s+(\w+)\s+in\s+(\w+)\s+supplies\s+(\w+)\s+with\s+capacity\s+(\d+)",
                re.IGNORECASE
            ),
            "consumers": re.compile(
                r"consumer\s+(\w+)\s+in\s+(\w+)\s+demands\s+(\w+)\s+with\s+capacity\s+(\d+)",
                re.IGNORECASE
            ),
            "transport_links": re.compile(
                r"transport\s+(\w+)\s+from\s+(\w+)\s+to\s+(\w+)\s+for\s+(\w+)\s+with\s+capacity\s+(\d+)",
                re.IGNORECASE
            ),
            "technologies": re.compile(
                r"technology\s+(\w+)\s+in\s+(\w+)\s+with\s+capacity\s+(\d+)",
                re.IGNORECASE
            ),
            "bids": re.compile(
                r"bid\s+(\w+)\s+by\s+(\w+)\s+\((\w+)\)\s+for\s+(\w+)\s+price\s+([\d.-]+)\s+quantity\s+(\d+)",
                re.IGNORECASE
            ),
        }

    def parse_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse the given text and extract entities.

        Returns a dictionary where keys are entity types and values are lists
        of entity dictionaries. Each entity dict contains the extracted fields.
        """
        entities: Dict[str, List[Dict[str, Any]]] = {
            "nodes": [],
            "products": [],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }

        # Parse nodes
        for match in self.patterns["nodes"].finditer(text):
            name = match.group(1)
            entities["nodes"].append({"id": name, "name": name})

        # Parse products
        for match in self.patterns["products"].finditer(text):
            name = match.group(1)
            entities["products"].append({"id": name, "name": name})

        # Parse suppliers
        for match in self.patterns["suppliers"].finditer(text):
            sid, node, product, cap = match.groups()
            entities["suppliers"].append({
                "id": sid,
                "node": node,
                "product": product,
                "capacity": int(cap),
            })

        # Parse consumers
        for match in self.patterns["consumers"].finditer(text):
            cid, node, product, cap = match.groups()
            entities["consumers"].append({
                "id": cid,
                "node": node,
                "product": product,
                "capacity": int(cap),
            })

        # Parse transport links
        for match in self.patterns["transport_links"].finditer(text):
            tid, origin, dest, product, cap = match.groups()
            entities["transport_links"].append({
                "id": tid,
                "origin": origin,
                "destination": dest,
                "product": product,
                "capacity": int(cap),
            })

        # Parse technologies (simplified, no yields yet)
        for match in self.patterns["technologies"].finditer(text):
            tid, node, cap = match.groups()
            entities["technologies"].append({
                "id": tid,
                "node": node,
                "capacity": int(cap),
                "yield_coefficients": {},  # placeholder
            })

        # Parse bids
        for match in self.patterns["bids"].finditer(text):
            bid_id, owner_id, owner_type, product, price, qty = match.groups()
            entities["bids"].append({
                "id": bid_id,
                "owner_id": owner_id,
                "owner_type": owner_type,
                "product_id": product,
                "price": float(price),
                "quantity": int(qty),
            })

        return entities


def parse_supply_chain_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to parse supply chain text using the rule-based parser.

    This is the main entry point for parsing natural language descriptions.
    """
    parser = RuleBasedParser()
    return parser.parse_entities(text)


__all__ = ["RuleBasedParser", "parse_supply_chain_text"]
