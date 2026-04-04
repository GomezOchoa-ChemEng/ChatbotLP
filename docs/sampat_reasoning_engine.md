# Sampat Reasoning Engine

## Purpose

The Sampat reasoning engine is a lightweight extension layer inside ChatbotLP for broader question handling around Sampat et al. (2019), especially Sections 2.1-2.3.

It exists to improve generalization without replacing the deterministic backbone:

- `ProblemState`
- `validator`
- `model_builder`
- `solver`
- `theorem_checker`
- `math_response_generator`

The engine does not turn ChatbotLP into a pure retrieval chatbot.

Instead, it maps a user question into a small reasoning plan:

- `object`
- `operation`
- `grounding_mode`
- `style`
- `scope`

Then it gathers grounded artifacts from:

- curated Sampat domain knowledge in `src/domain/sampat2019.py`
- current model scaffolds
- theorem applicability metadata
- solver results when explicitly required and available
- benchmark-case metadata

## Why This Layer Exists

The existing theorem/dual support is strong for narrowly scoped formal math requests such as:

- dual formulation
- theorem proof
- Section 2.3 explanation

But broader Sampat questions often mix paper interpretation and deterministic model context, for example:

- "What do node-product prices represent?"
- "Compare Case A and Case B."
- "How do technologies affect prices?"
- "Explain this from the paper even if it is not solver-verified."

Those questions benefit from a reusable reasoning layer instead of adding one-off keyword branches.

## Grounding Modes

The engine distinguishes among four response modes:

1. `paper_grounded_explanation`
2. `model_grounded_formulation`
3. `solver_grounded_verification`
4. `theorem_grounded_proof`

This distinction is stored in the structured reasoning package before user-facing text is produced.

If a requested grounding artifact is missing, the package records that explicitly instead of bluffing.

## Relation to Existing Math Support

The engine is an extension layer, not a replacement.

- For narrow deterministic math requests, it routes into the existing `math_response_generator`.
- For broader Sampat questions, it returns a grounded reasoning package and a lightweight explanation path.

This keeps deterministic authority where it belongs:

- model claims come from model scaffolds
- solver claims come from solver execution
- theorem claims come from theorem metadata and theorem checks
- paper interpretation comes from curated local Sampat knowledge

## Current Scope

The first version is intentionally limited to:

- Sampat Sections 2.1-2.3
- benchmark families Case A, Case B, and Case C
- existing deterministic structures already implemented in ChatbotLP

For Case C specifically, the deterministic pipeline now carries `ProblemState.technologies`
through `model_builder` into explicit Pyomo technology activity variables, node-product
yield terms, and optional technology capacity constraints.

It is not a generic paper-QA engine and does not claim arbitrary symbolic generalization beyond the current supported scope.
