# Math Exposition Layer

## Purpose

The math exposition layer adds specialized theorem, proof, dual, and Section 2.3 explanation responses on top of the deterministic optimization pipeline.

Formal math responses are now emitted as notebook-friendly Markdown + LaTeX fragments rather than standalone LaTeX documents. This keeps them directly usable in Google Colab and Jupyter with `IPython.display`.

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
  Produces bounded mathematical exposition and uses the existing `llm_adapter` abstraction when available. When Gemini is configured through the provider registry, polishing can be Gemini-backed while deterministic checks remain authoritative.

- `src/notebook_rendering.py`
  Small optional helper for rendering chatbot responses directly in Colab/Jupyter.

## Colab / Jupyter Rendering

Formal math output is designed for direct rendering as a fragment:

- no `\documentclass`
- no `\usepackage`
- no `\begin{document}` / `\end{document}`
- theorem/proof responses prefer bold headings, prose, and display-math blocks over standalone theorem environments

Typical usage:

```python
from IPython.display import Markdown, display
from src.chatbot_engine import run_chatbot_session

result = run_chatbot_session(state, "Show me that Theorem 1 holds.")
display(Markdown(result["response"]))
```

Optional helper:

```python
from src.notebook_rendering import render_chatbot_result

result = run_chatbot_session(state, "Give me the dual problem in LaTeX.")
render_chatbot_result(result)
```

When the request is routed through the formal math layer, the chatbot result may also include a small rendering hint such as:

- `render_mode = "markdown_latex"`

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
- Deterministic authority remains unchanged: theorem applicability, primal/dual structure, and fallback behavior still come from the repository's validated structured pipeline, while Gemini only polishes exposition when enabled.
