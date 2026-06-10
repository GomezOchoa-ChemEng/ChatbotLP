"""Deterministic fixture for HW#2: plastic waste processing supply chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .design_model import extract_design_solution
from .llm_problem_interpreter import build_state_from_semantic_plan
from .schema import ProblemState


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "Benchmarks" / "hw2_plastic_waste"
DEFAULT_HOMEWORK_PDF = REPO_ROOT / "docs" / "hw2_supply_chains_spring2022.pdf"

PLASTIC_WASTE = "PlasticWaste"
PYROLYSIS_OIL = "PyrolysisOil"
ETHYLENE = "Ethylene"


HW2_STRUCTURED_PROSE = """# HW#2

Due: February 17th, 2022

CBE450: Process Design - Spring 2022

Department of Chemical and Biological Engineering, University of Wisconsin-Madison

## Supply Chain for Plastic Waste Processing (100 pts)

Plastic waste affects the environment in many ways; it is estimated that 3.5
million tons of plastic waste are generated every year in the Upper Midwest
region of the United States. Fortunately, there is a new promising approach for
dealing with plastic waste in a scalable manner. This pathway uses
thermochemical technologies (pyrolysis and steam cracking) to break plastic and
recover value-added chemicals. As shown in Figure 1, pyrolysis converts the
plastic waste into a pyrolysis oil, while steam cracking produces ethylene from
the pyrolysis oil.

Figure 1: Schematic of plastic waste processing pathway

We would like to design and operate a supply chain that collects and processes
waste and obtains value-added products (ethylene). You are given the following
data:

- Supply Information: The goal is to recycle all the plastic waste generated in
  Madison (MAD) and Milwaukee (MKE). The total plastic waste generated in MAD
  and MKE is 33100, 77300 ton/yr, respectively. The supply cost for the plastic
  waste from these cities is 0 $/ton (plastic waste is offered for free).

- Demand Information: Each city has a consumer of ethylene and we wish to
  satisfy the demands of such consumers. The demand capacities for ethylene are
  both 100,000 tons/yr and the consumers are both willing to pay 1050 $/ton.

- Transportation Information: The transportation cost for the three products
  (plastic, pyrolysis oil, and ethylene) are the same. The cost to move one ton
  of product from MAD to MKE (and from MKE to MAD) is 10$/ton. The
  transportation capacity for all products is 100,000 ton/yr.

- Technology Information:
  - You have the possibility of installing a pyrolysis process in MAD and
    another one in MKE; you also have the possibility of installing a steam
    cracking process in MAD and another one in MKE.
  - For the pyrolysis process, 0.7 ton of pyrolysis oil can be produced from
    breaking one ton of plastic waste. The operational cost of pyrolyzing one
    ton of plastic waste is 14 $. The maximum capacity of a single pyrolysis
    facility is 100,000 tons of plastic waste per year. The investment cost for
    this technology is 2,000,000 $/yr (annualized investment).
  - The steam cracking process converts one ton of pyrolysis oil into 0.25 ton
    of ethylene. The operational cost of steam cracking is 71$ per ton of
    pyrolysis oil. A single steam cracking facility process 100,000 ton of
    pyrolysis oil each year. The investment cost for this technology is
    2,800,000 $/yr (annualized investment).
"""


def build_hw2_plastic_waste_semantic_plan(
    ethylene_wtp: float = 1050.0,
    missing_investment_cost: bool = False,
) -> Dict[str, Any]:
    """Return a deterministic semantic plan for the plastic-waste homework."""

    pyrolysis_fixed_cost = None if missing_investment_cost else 2_000_000.0
    steam_cracking_fixed_cost = None if missing_investment_cost else 2_800_000.0
    transport_links = []
    for product in [PLASTIC_WASTE, PYROLYSIS_OIL, ETHYLENE]:
        for origin, destination in [("MAD", "MKE"), ("MKE", "MAD")]:
            transport_links.append(
                {
                    "id": f"T_{origin}_to_{destination}_{product}",
                    "origin": origin,
                    "destination": destination,
                    "product": product,
                    "capacity": 100000.0,
                    "cost": 10.0,
                }
            )

    return {
        "problem_title": "HW#2: Supply Chain for Plastic Waste Processing",
        "problem_type": "case_c",
        "nodes": [
            {"id": "MAD", "name": "Madison"},
            {"id": "MKE", "name": "Milwaukee"},
        ],
        "products": [
            {"id": PLASTIC_WASTE, "name": "plastic waste"},
            {"id": PYROLYSIS_OIL, "name": "pyrolysis oil"},
            {"id": ETHYLENE, "name": "ethylene"},
        ],
        "suppliers": [
            {"id": "S_MAD_PlasticWaste", "node": "MAD", "product": PLASTIC_WASTE, "capacity": 33100.0},
            {"id": "S_MKE_PlasticWaste", "node": "MKE", "product": PLASTIC_WASTE, "capacity": 77300.0},
        ],
        "consumers": [
            {"id": "C_MAD_Ethylene", "node": "MAD", "product": ETHYLENE, "capacity": 100000.0},
            {"id": "C_MKE_Ethylene", "node": "MKE", "product": ETHYLENE, "capacity": 100000.0},
        ],
        "transport_links": transport_links,
        "technologies": [
            {
                "id": "Pyrolysis_MAD",
                "node": "MAD",
                "capacity": 100000.0,
                "cost": 14.0,
                "fixed_cost": pyrolysis_fixed_cost,
                "yield_coefficients": {PLASTIC_WASTE: -1.0, PYROLYSIS_OIL: 0.7},
            },
            {
                "id": "Pyrolysis_MKE",
                "node": "MKE",
                "capacity": 100000.0,
                "cost": 14.0,
                "fixed_cost": pyrolysis_fixed_cost,
                "yield_coefficients": {PLASTIC_WASTE: -1.0, PYROLYSIS_OIL: 0.7},
            },
            {
                "id": "SteamCracking_MAD",
                "node": "MAD",
                "capacity": 100000.0,
                "cost": 71.0,
                "fixed_cost": steam_cracking_fixed_cost,
                "yield_coefficients": {PYROLYSIS_OIL: -1.0, ETHYLENE: 0.25},
            },
            {
                "id": "SteamCracking_MKE",
                "node": "MKE",
                "capacity": 100000.0,
                "cost": 71.0,
                "fixed_cost": steam_cracking_fixed_cost,
                "yield_coefficients": {PYROLYSIS_OIL: -1.0, ETHYLENE: 0.25},
            },
        ],
        "bids": [
            {
                "id": "B_S_MAD_PlasticWaste",
                "owner_id": "S_MAD_PlasticWaste",
                "owner_type": "supplier",
                "product_id": PLASTIC_WASTE,
                "price": 0.0,
                "quantity": 33100.0,
            },
            {
                "id": "B_S_MKE_PlasticWaste",
                "owner_id": "S_MKE_PlasticWaste",
                "owner_type": "supplier",
                "product_id": PLASTIC_WASTE,
                "price": 0.0,
                "quantity": 77300.0,
            },
            {
                "id": "B_C_MAD_Ethylene",
                "owner_id": "C_MAD_Ethylene",
                "owner_type": "consumer",
                "product_id": ETHYLENE,
                "price": float(ethylene_wtp),
                "quantity": 100000.0,
            },
            {
                "id": "B_C_MKE_Ethylene",
                "owner_id": "C_MKE_Ethylene",
                "owner_type": "consumer",
                "product_id": ETHYLENE,
                "price": float(ethylene_wtp),
                "quantity": 100000.0,
            },
        ],
        "missing_information": (
            ["technology investment costs"]
            if missing_investment_cost
            else []
        ),
        "ambiguities": [],
    }


def build_hw2_plastic_waste_state(
    ethylene_wtp: float = 1050.0,
    missing_investment_cost: bool = False,
) -> ProblemState:
    """Build the HW2 fixture as a ``ProblemState``."""

    return build_state_from_semantic_plan(
        build_hw2_plastic_waste_semantic_plan(
            ethylene_wtp=ethylene_wtp,
            missing_investment_cost=missing_investment_cost,
        )
    )


def total_available_plastic_waste(state: ProblemState) -> float:
    """Return total plastic waste supply capacity in the state."""

    total = 0.0
    for supplier in state.suppliers:
        if supplier.product != PLASTIC_WASTE:
            continue
        if supplier.capacity is None:
            raise ValueError(f"supplier:{supplier.id} missing capacity")
        total += supplier.capacity
    return total


def total_recycled_plastic_waste(
    state: ProblemState,
    solve_result: Any,
) -> float:
    """Return accepted plastic-waste supply in a solved design model."""

    summary = extract_design_solution(state, solve_result)
    supplier_ids = {
        supplier.id
        for supplier in state.suppliers
        if supplier.product == PLASTIC_WASTE
    }
    return sum(
        quantity
        for supplier_id, quantity in summary["accepted_supply"].items()
        if supplier_id in supplier_ids
    )


def total_accepted_ethylene(
    state: ProblemState,
    solve_result: Any,
) -> float:
    """Return accepted ethylene demand in a solved design model."""

    summary = extract_design_solution(state, solve_result)
    consumer_ids = {
        consumer.id
        for consumer in state.consumers
        if consumer.product == ETHYLENE
    }
    return sum(
        quantity
        for consumer_id, quantity in summary["accepted_demand"].items()
        if consumer_id in consumer_ids
    )


__all__ = [
    "DEFAULT_BENCHMARK_DIR",
    "DEFAULT_HOMEWORK_PDF",
    "ETHYLENE",
    "HW2_STRUCTURED_PROSE",
    "PLASTIC_WASTE",
    "PYROLYSIS_OIL",
    "build_hw2_plastic_waste_semantic_plan",
    "build_hw2_plastic_waste_state",
    "total_accepted_ethylene",
    "total_available_plastic_waste",
    "total_recycled_plastic_waste",
]
