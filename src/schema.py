"""src.schema

Pydantic-based canonical schema for the coordinated supply chain problem.

This module defines the domain entities used by `ProblemState` and
provides light utility methods for building and validating the problem
description. The design targets the benchmark cases from the supporting
information (Case A: no transformation, Case B: negative bids,
Case C: transformation).
"""

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator


# ------------------------------------------------
# Core entities
# ------------------------------------------------


class Node(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    name: Optional[str] = Field(None, description="Human-friendly name")


class Product(BaseModel):
    id: str = Field(..., description="Unique product identifier")
    name: Optional[str] = Field(None, description="Human-friendly name")


class Bid(BaseModel):
    """Generic bid/offering record.

    Price can be negative to support Case B (negative bidding costs).
    Quantity is optional; omitting quantity can represent a price-only bid.
    """

    id: str
    owner_id: str
    owner_type: Literal["supplier", "consumer", "transport", "technology"]
    product_id: str
    price: float
    quantity: Optional[float] = None

    @validator("price")
    def price_is_finite(cls, v):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("price must be a finite number")
        return v


class Supplier(BaseModel):
    id: str
    node: str
    product: str
    capacity: Optional[float] = None


class Consumer(BaseModel):
    id: str
    node: str
    product: str
    capacity: Optional[float] = None


class TransportLink(BaseModel):
    id: str
    origin: str
    destination: str
    product: str
    capacity: Optional[float] = None


class Technology(BaseModel):
    id: str
    node: str
    capacity: Optional[float] = None
    # yield_coefficients maps product_id -> coefficient (output per unit input)
    # Negative coefficients are allowed when a technology *consumes* a resource
    yield_coefficients: Dict[str, float] = Field(default_factory=dict)


# ------------------------------------------------
# Supporting / analytical objects
# ------------------------------------------------


class TheoremCheck(BaseModel):
    theorem_name: str
    theorem_id: Optional[str] = None
    target_section: Optional[str] = None
    applies: Optional[bool] = None
    explanation: Optional[str] = None
    assumptions_verified: List[str] = Field(default_factory=list)
    assumptions_missing: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScenarioRecord(BaseModel):
    name: str
    description: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkMetadata(BaseModel):
    source: Optional[str] = None
    case_family: Optional[str] = None
    case_id: Optional[str] = None
    reference_section: Optional[str] = None
    notes: Optional[str] = None


class FormalMathContext(BaseModel):
    """Structured context for theorem-, proof-, and dual-style responses."""

    request_type: str
    domain_source: str
    target_section: Optional[str] = None
    theorem_id: Optional[str] = None
    applicable: Optional[bool] = None
    assumptions_verified: List[str] = Field(default_factory=list)
    assumptions_missing: List[str] = Field(default_factory=list)
    notation_profile: Dict[str, Any] = Field(default_factory=dict)
    primal_formulation: Optional[Dict[str, Any]] = None
    dual_formulation: Optional[Dict[str, Any]] = None
    objective: Optional[Dict[str, Any]] = None
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    variables: List[Dict[str, Any]] = Field(default_factory=list)
    dual_variables: List[Dict[str, Any]] = Field(default_factory=list)
    profit_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    lagrangian_components: List[Dict[str, Any]] = Field(default_factory=list)
    benchmark_case: Optional[str] = None
    supporting_equations: List[str] = Field(default_factory=list)
    source_notes: List[str] = Field(default_factory=list)
    latex_mode: str = "align"
    pedagogical_mode: str = "guided"
    user_request: Optional[str] = None


# ------------------------------------------------
# Central ProblemState
# ------------------------------------------------


class ProblemState(BaseModel):
    """Central object representing the supply-chain problem.

    All modules must accept and return this object when modifying the
    problem (see AGENTS.md). The structure intentionally keeps domain
    lists (nodes, products, entities) simple and flat to ease
    deterministic model construction.
    """

    problem_title: str = Field("Untitled Coordinated Supply Chain Problem")

    # core sets
    nodes: List[Node] = Field(default_factory=list)
    products: List[Product] = Field(default_factory=list)

    # entities
    suppliers: List[Supplier] = Field(default_factory=list)
    consumers: List[Consumer] = Field(default_factory=list)
    transport_links: List[TransportLink] = Field(default_factory=list)
    technologies: List[Technology] = Field(default_factory=list)

    # bids (supports negative prices explicitly)
    bids: List[Bid] = Field(default_factory=list)

    # bookkeeping / metadata
    assumptions: List[str] = Field(default_factory=list)
    missing_parameters: List[str] = Field(default_factory=list)
    theorem_checks: List[TheoremCheck] = Field(default_factory=list)
    scenario_history: List[ScenarioRecord] = Field(default_factory=list)
    benchmark: Optional[BenchmarkMetadata] = None

    # --------------------------------------------
    # Convenience utilities
    # --------------------------------------------

    def solver_ready(self) -> bool:
        return len(self.missing_parameters) == 0

    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def product_ids(self) -> List[str]:
        return [p.id for p in self.products]

    def add_node(self, node: Node) -> None:
        if node.id in self.node_ids():
            return
        self.nodes.append(node)

    def add_product(self, product: Product) -> None:
        if product.id in self.product_ids():
            return
        self.products.append(product)

    def add_supplier(self, supplier: Supplier) -> None:
        if supplier.id in [s.id for s in self.suppliers]:
            return
        self.suppliers.append(supplier)

    def add_consumer(self, consumer: Consumer) -> None:
        if consumer.id in [c.id for c in self.consumers]:
            return
        self.consumers.append(consumer)

    def add_transport(self, link: TransportLink) -> None:
        if link.id in [t.id for t in self.transport_links]:
            return
        self.transport_links.append(link)

    def add_technology(self, tech: Technology) -> None:
        if tech.id in [t.id for t in self.technologies]:
            return
        self.technologies.append(tech)

    def add_bid(self, bid: Bid) -> None:
        if bid.id in [b.id for b in self.bids]:
            return
        # negative prices are allowed by design (Case B)
        self.bids.append(bid)

    def add_scenario(self, record: ScenarioRecord) -> None:
        self.scenario_history.append(record)

    def to_dict(self) -> Dict:
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict) -> "ProblemState":
        return cls.parse_obj(data)
