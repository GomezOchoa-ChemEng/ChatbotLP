# Math Exposition Layer

## Purpose

The math exposition layer adds specialized theorem, proof, dual, and Section 2.3 explanation responses on top of the deterministic optimization pipeline.

It is designed for requests such as:

- "Give me the dual problem in LaTeX."
- "Show me that Theorem 1 holds."
- "Explain why Theorem 1 applies in this case."
- "Explain the role of negative bids in Section 2.3."

## Authority Boundary

The LLM is not the mathematical authority.

Authoritative inputs come from:

- `ProblemState`
- `validator`
- `theorem_checker`
- deterministic primal and dual scaffolds

The LLM may polish exposition, but it must not invent:

- theorem applicability
- missing assumptions
- unsupported symbols
- claims beyond the structured context

## Components

- `src/domain/sampat2019.py`
  Narrow registry for Sections 2.1-2.3: notation, theorem metadata, and Section 2.3 concepts.

- `src/formal_context_builder.py`
  Builds `FormalMathContext` from `ProblemState`, validation output, and theorem-check metadata.

- `src/dual_generator.py`
  Produces a deterministic primal scaffold and a minimal dual scaffold.

- `src/proof_validator.py`
  Performs lightweight pre-generation and post-generation checks.

- `src/math_response_generator.py`
  Produces bounded mathematical exposition and uses the existing `llm_adapter` abstraction when available.

## Supported Sampat Scope

### Section 2.1

- primal coordinated clearing
- bid acceptance variables
- node-product balance constraints

### Section 2.2

- dual interpretation
- Lagrangian-style explanation
- scoped Theorem 1 applicability and proof requests

### Section 2.3

- negative bids
- disposal
- remediation
- storage
- VOS-style interpretation
- negative prices

## Current Limits

- The registry is hand curated, not a general PDF QA system.
- The dual helper is a scaffold for the currently supported linear model, not a symbolic algebra engine.
- The theorem-proof path is limited to theorem IDs explicitly present in the registry.
- Applicability remains structural and deterministic, not formally verified in a proof-assistant sense.
