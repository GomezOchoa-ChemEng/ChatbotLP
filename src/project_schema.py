"""
project_schema.py

Starter schema and state-management utilities for a classroom-oriented
coordinated supply chain chatbot.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Supplier(BaseModel):
    name: str
    node: str
    product: str
    capacity: Optional[float] = None
    bid: Optional[float] = None


class Consumer(BaseModel):
    name: str
    node: str
    product: str
    capacity: Optional[float] = None
    bid: Optional[float] = None


class TransportLink(BaseModel):
    name: str
    origin: str
    destination: str
    product: str
    capacity: Optional[float] = None
    bid: Optional[float] = None


class Technology(BaseModel):
    name: str
    node: str
    capacity: Optional[float] = None
    bid: Optional[float] = None
    yield_coefficients: Dict[str, float] = Field(default_factory=dict)


class ProblemState(BaseModel):

    problem_title: str = "Untitled Coordinated Supply Chain Problem"

    nodes: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)

    suppliers: List[Supplier] = Field(default_factory=list)
    consumers: List[Consumer] = Field(default_factory=list)
    transport_links: List[TransportLink] = Field(default_factory=list)
    technologies: List[Technology] = Field(default_factory=list)

    assumptions: List[str] = Field(default_factory=list)
    missing_parameters: List[str] = Field(default_factory=list)

    scenario_history: List[dict] = Field(default_factory=list)

    def solver_ready(self) -> bool:
        """
        Check if the model has enough information to run a solver.
        """
        return len(self.missing_parameters) == 0


def create_example_problem() -> ProblemState:

    state = ProblemState(
        problem_title="Starter manure supply chain example",
        nodes=["CountyA", "CountyB"],
        products=["manure", "pellet"],
        suppliers=[
            Supplier(name="Farm1", node="CountyA", product="manure", capacity=100, bid=0),
            Supplier(name="Farm2", node="CountyB", product="manure", capacity=80, bid=0),
        ],
        consumers=[
            Consumer(name="Field1", node="CountyA", product="pellet", capacity=60, bid=15),
            Consumer(name="Field2", node="CountyB", product="pellet", capacity=70, bid=14),
        ],
        transport_links=[
            TransportLink(
                name="Truck_A_B",
                origin="CountyA",
                destination="CountyB",
                product="manure",
                capacity=100,
                bid=2,
            )
        ],
        technologies=[
            Technology(
                name="Pelletizer",
                node="CountyB",
                capacity=120,
                bid=3,
                yield_coefficients={"manure": -1, "pellet": 0.8},
            )
        ],
    )

    return state


def get_missing_parameter_suggestions(state: ProblemState):

    missing = []

    for s in state.suppliers:
        if s.capacity is None:
            missing.append(f"supplier_capacity:{s.name}")

        if s.bid is None:
            missing.append(f"supplier_bid:{s.name}")

    for d in state.consumers:
        if d.capacity is None:
            missing.append(f"consumer_capacity:{d.name}")

        if d.bid is None:
            missing.append(f"consumer_bid:{d.name}")

    for t in state.technologies:
        if t.capacity is None:
            missing.append(f"technology_capacity:{t.name}")

        if t.bid is None:
            missing.append(f"technology_bid:{t.name}")

    return missing


if __name__ == "__main__":

    state = create_example_problem()

    print("Problem summary")
    print("----------------")

    print("Nodes:", state.nodes)
    print("Products:", state.products)

    print("Suppliers:", len(state.suppliers))
    print("Consumers:", len(state.consumers))

    print("Solver ready:", state.solver_ready())

    print("Missing parameters:", get_missing_parameter_suggestions(state))