# System Architecture

## Project Title

Classroom-Oriented Chatbot for Coordinated Supply Chain Optimization

---

## 1. Purpose

This project aims to build a domain-specific chatbot that helps users progressively formulate, analyze, and solve coordinated supply chain optimization problems.

The chatbot is intended for advanced undergraduate classroom use.

The system should behave like a conversational assistant, but its mathematical reasoning and outputs must be grounded in:

- structured problem representations
- deterministic optimization workflows
- benchmark-driven validation
- solver-backed computations
- theorem and assumption checks
- pedagogical explanations

The theoretical references are:

- `docs/Sampat_2019_Coordinated_Supply_Chain.pdf`
- `docs/Supporting_Information_CoorditatedManagement.pdf`

The main paper provides the coordinated supply chain formulation.

The supporting information, especially Section 3, provides the first benchmark case-study families that the initial implementation must support:

- Case A: no transformation
- Case B: negative bidding costs
- Case C: transformation

The chatbot should not rely on fine-tuning.

Instead, it should use:

- modular Python code
- structured schemas
- retrieval of local reference material
- rule-based validation
- mathematical programming with Pyomo
- progressive conversational state tracking

---

## 2. Design Philosophy

The system should not be implemented as a monolithic chatbot that tries to do everything in one step.

Instead, it should use a tool-oriented scientific assistant architecture in which:

1. the chatbot interprets the user request
2. routes the request to the appropriate modules
3. updates a persistent problem state
4. validates whether enough information is available
5. generates symbolic or numerical outputs as appropriate
6. explains results in a classroom-friendly manner

The language model should mainly be used for:

- interpreting user input
- extracting structure from text
- generating explanations
- polishing bounded theorem, proof, and dual exposition from structured context
- supporting conversational flow

Mathematical correctness must come from:

- validated schemas
- explicit model generation
- optimization solving
- rule-based theorem checks
- formal context assembly from deterministic module outputs

---

## 3. High-Level Architecture

```text
User
  ->
Chatbot Engine (Orchestrator)
  ->
Intent / Task Router
  |- Parser
  |- State Manager
  |- Retrieval Module
  |- Validator
  |- Sampat Reasoning Engine
  |- Model Builder
  |- Solver
  |- Theorem Checker
  |- Formal Context Builder
  |- Dual Generator
  |- Proof Validator
  |- Math Response Generator
  |- Scenario Engine
  `- Response Generator
  ->
Chatbot Response
```

---

## 4. Math Exposition Extension

The repository includes a bounded mathematical exposition layer specialized to Sampat et al. (2019) Sections 2.1, 2.2, and 2.3.

This layer supports:

- dual-problem exposition in LaTeX form
- theorem applicability explanations
- theorem-proof style outputs scoped to supported theorem IDs
- Section 2.3 explanations for negative bids and negative prices

The extension does not change the mathematical authority of the system.

Instead it adds the bridge:

`ProblemState + validator + theorem_checker + primal scaffold -> FormalMathContext -> math_response_generator`

When an LLM is enabled, it is only allowed to polish exposition from the supplied structured context.

---

## 5. Sampat Reasoning Extension

The repository now also includes a lightweight Sampat reasoning layer for broader question handling within the same architectural backbone.

This layer does not replace the deterministic math pipeline.

Instead it adds a reusable bridge from natural-language Sampat questions to grounded artifacts:

`user question -> reasoning plan -> grounded artifacts -> response composition`

The reasoning plan captures:

- object
- operation
- grounding mode
- style
- scope

The engine can collect artifacts from:

- curated local Sampat domain knowledge
- the current primal or dual scaffold
- theorem applicability metadata
- solver results when solver-backed verification is actually available
- benchmark-case metadata

The supported grounding distinctions are:

- paper-grounded explanation
- model-grounded formulation
- solver-grounded verification
- theorem-grounded proof

If a requested artifact is unavailable, the engine records what is missing instead of silently improvising.

For narrow theorem, proof, primal, and dual requests, the reasoning engine routes into the existing formal-math path so current behavior is preserved.
