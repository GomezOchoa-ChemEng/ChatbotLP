# Coordinated Supply Chain Optimization Chatbot

## Project Overview

This project builds a domain-specific chatbot for coordinated supply chain optimization.

The system allows users to progressively formulate, analyze, and solve supply chain models through natural language interaction.

The chatbot is designed for classroom use in advanced undergraduate courses.

The architecture combines:

- natural language interaction
- structured schemas
- optimization modeling
- validation logic
- theorem and assumption checking
- scenario analysis

---

## Theoretical References

The main references for this project are:

- `docs/Sampat_2019_Coordinated_Supply_Chain.pdf`
- `docs/Supporting_Information_CoorditatedManagement.pdf`

The first project milestone is to support the illustrative case-study families in Section 3 of the supporting information:

- Case A: no transformation
- Case B: negative bidding costs
- Case C: transformation

These benchmark cases will guide the first implementation and early tests.

---

## Key Features

The target system should be able to:

- interpret supply chain problem descriptions
- extract model entities
- maintain a progressive problem state
- generate mathematical formulations in LaTeX
- build optimization models in Pyomo
- solve optimization models
- detect missing parameters
- check theorem or assumption applicability
- run what-if scenarios

---

## Repository Structure

```text
CHATBOTLP/
│
├── docs/
├── notebooks/
├── prompts/
├── src/
├── tests/
│
├── AGENTS.md
├── README.md
└── requirements.txt
```

## Getting Started

The following example demonstrates creating a simple problem state,
validating it, building a Pyomo model, and solving it. It can be run in a Python
interactive session or script:

```python
from src.schema import ProblemState, Node, Product, Supplier, Consumer, Bid
from src.state_manager import StateManager
from src.validator import validate_state
from src.model_builder import build_model_from_state
from src.solver import solve_model

# initialize state and add one supplier and one consumer
state = ProblemState(problem_title="Example")
state.add_node(Node(id="n1"))
state.add_product(Product(id="p1"))
state.add_supplier(Supplier(id="sup1", node="n1", product="p1", capacity=10))
state.add_consumer(Consumer(id="con1", node="n1", product="p1", capacity=5))
state.add_bid(Bid(id="b1", owner_id="sup1", owner_type="supplier", product_id="p1", price=1.0, quantity=5))
state.add_bid(Bid(id="b2", owner_id="con1", owner_type="consumer", product_id="p1", price=2.0, quantity=5))

# validate
mgr = StateManager(state)
issues = mgr.validate()
print("missing/invalid parameters", issues)
print("benchmark flags", validate_state(state)["benchmark_compatibility"])

# build model
model = build_model_from_state(state)
print("model built with", len(list(model.BIDS)), "bids")

# attempt solve; may return solver_unavailable if no external solver is installed
result = solve_model(model)
if result.status == "solver_unavailable":
    print("No solver installed on this system.  Results not computed.")
else:
    print("Solve finished, status=", result.status, "objective=", result.objective_value)
```

You can run the unit tests with `pytest`.
