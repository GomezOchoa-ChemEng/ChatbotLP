# AGENTS.md

Guidelines for Codex when assisting with this repository.

---

## Project Mission

Build a domain-specific chatbot for coordinated supply chain optimization designed for classroom use by advanced undergraduate students.

The chatbot should support progressive inquiry similar to ChatGPT but specialized for the coordinated supply chain framework.

The theoretical references are:

- `docs/Sampat_2019_Coordinated_Supply_Chain.pdf`
- `docs/Supporting_Information_CoorditatedManagement.pdf`

Use the main paper as the theoretical foundation of the coordinated market-clearing model.

Use Section 3 of the supporting information as the benchmark reference for the first working implementation.

The first project milestone is that the system must be able to represent and later solve the illustrative benchmark case-study families:

- Case A: no transformation
- Case B: negative bidding costs
- Case C: transformation

The system should not rely on fine-tuning.

Instead it should use:

- structured problem schemas
- rule-based validation
- retrieval of local project documents
- deterministic model generation
- optimization solvers
- progressive conversational state tracking

---

## Development Philosophy

This system must follow a hybrid architecture:

language interface → structured problem state → validator → model builder → solver → explanation

The language model is not the mathematical authority.

All final mathematical claims must come from:

- explicit schema validation
- deterministic model construction
- solver-backed computation
- theorem or assumption checks

---

## Key Initial Requirement

The first version must support the benchmark structures described in the supporting information:

1. no-transformation systems
2. systems with negative supplier or consumer bids
3. systems with transformation technologies and yield coefficients

The schema, validator, model builder, and tests must all be designed with these benchmark families in mind.

---

## Architecture Overview

The project should eventually contain these logical modules in `src/`:

- `schema.py`
- `state_manager.py`
- `parser.py`
- `retrieval.py`
- `validator.py`
- `model_builder.py`
- `solver.py`
- `theorem_checker.py`
- `scenario_engine.py`
- `response_generator.py`
- `chatbot_engine.py`

Do not implement all of them at once.

---

## Implementation Order

Codex should follow this implementation sequence:

1. `schema.py`
2. `state_manager.py`
3. `validator.py`
4. `model_builder.py`
5. `solver.py`
6. benchmark tests for Case A, B, and C
7. `theorem_checker.py`
8. `parser.py`
9. `response_generator.py`
10. `scenario_engine.py`
11. `chatbot_engine.py`

Do not skip steps.

---

## File Editing Policy

Codex may modify:

- `src/`
- `tests/`
- `notebooks/`

Codex should not rewrite:

- `AGENTS.md`
- `README.md`
- `docs/system_architecture.md`

unless explicitly instructed.

---

## Mathematical Safety Rules

Codex must follow these rules:

1. Do not fabricate parameter values.
2. Do not claim a model was solved unless the solver ran.
3. Do not claim theorem applicability without explicit checks.
4. Always distinguish between:
   - user-provided data
   - article-derived data
   - assumed values
   - externally sourced values
5. If essential parameters are missing, request them or mark the instance as not solver-ready.
6. Support negative bids explicitly where allowed by the benchmark cases.
7. Support transformation yield coefficients explicitly.

---

## Code Style Guidelines

Python version: 3.10+

Use:

- type hints when appropriate
- modular design
- clear docstrings
- deterministic transformations
- Pydantic models for the schema

Avoid:

- hard-coding undocumented assumptions
- mixing symbolic and numerical logic
- large monolithic functions

---

## Testing Policy

Each core module should include unit tests.

Add tests for:

- schema validation
- solver readiness checks
- benchmark case structure
- model generation
- solver execution for small cases

The first benchmark tests should target:

- Case A
- Case B
- Case C

from the supporting information.

---

## Preferred Libraries

Use:

- `pydantic`
- `pyomo`
- `pandas`
- `numpy`
- `networkx`
- `matplotlib`

Notebook development should remain Colab compatible.

---

## Classroom Mode

The chatbot should support:

- Hint Mode
- Guided Mode
- Full Solution Mode

Do not always give the final answer immediately.

---

## Instructions for Codex

When implementing code:

- propose modular architecture
- preserve clear interfaces
- keep the benchmark cases in mind
- prefer small testable components
- maintain readability for teaching purposes

Favor clarity and correctness over clever shortcuts.

---

# Module Interface Contracts

To ensure that all modules integrate correctly, the following interface rules must be respected.

## ProblemState as the Core Object

All modules must use `ProblemState` as the central object.

Modules must **receive and return `ProblemState` whenever the problem structure is modified**.

Example pattern:

```python
def update_state(state: ProblemState, data: dict) -> ProblemState:
    ...

## End of AGENTS.md