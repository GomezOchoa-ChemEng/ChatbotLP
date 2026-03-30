# Manual Evaluation Set: Sampat 2019 Math Exposition Layer

## Purpose

This document defines a manual benchmark set for evaluating the theorem/proof/dual-response layer added to ChatbotLP.

The scope of this evaluation is intentionally narrow:

- Sampat et al. (2019), Sections 2.1, 2.2, and 2.3
- theorem_1 support
- dual-problem generation for the currently supported LP structure
- Section 2.3 interpretation of negative bids
- out-of-scope and failure-mode behavior

This is **not** a generic document-QA benchmark and **not** a symbolic theorem-proving benchmark.

The goal is to verify that the chatbot behaves like a trustworthy domain-specific mathematical assistant:
- mathematically grounded,
- notation-consistent,
- pedagogically useful,
- honest about limitations,
- capable of producing clean LaTeX.

---

## How to Use This File

For each prompt:

1. Run it through the chatbot end to end.
2. Save or inspect the response.
3. Score the output using the evaluation criteria in this document.
4. Record notes on strengths, weaknesses, and failure modes.

Recommended output record template:

- **Prompt ID**
- **Observed output summary**
- **Pass / Borderline / Fail**
- **Notes on notation**
- **Notes on LaTeX quality**
- **Notes on grounding / correctness**
- **Notes on pedagogical usefulness**

---

## Global Evaluation Criteria

Use these criteria for every response.

### A. Grounding
The answer should:
- stay within Sampat 2019 Sections 2.1–2.3 when appropriate,
- avoid invented theorem claims,
- avoid invented notation,
- avoid unsupported mathematical assertions.

### B. Mathematical correctness
The answer should:
- reflect the implemented deterministic state/model where relevant,
- use theorem applicability information correctly,
- not claim a proof when assumptions are missing,
- not present unsupported theorems as available.

### C. Notation quality
The answer should:
- use notation consistent with the curated Sampat layer,
- define symbols when needed,
- keep signs and variable roles coherent.

### D. LaTeX quality
The answer should:
- render clearly as mathematical exposition,
- use theorem/proof structure when appropriate,
- use readable equation layout,
- avoid malformed or cluttered math blocks.

### E. Pedagogical usefulness
The answer should:
- be understandable to a student or researcher,
- explain assumptions when relevant,
- distinguish proof, interpretation, and informal explanation.

### F. Trustworthy failure behavior
If the request is unsupported or assumptions are missing, the answer should:
- say so clearly,
- avoid bluffing,
- explain what is and is not available.

---

## Scoring Rubric

Use a simple 3-level score for each prompt:

- **Pass**: grounded, correct, useful, and well formatted
- **Borderline**: mostly correct but weak in notation, style, or clarity
- **Fail**: mathematically ungrounded, misleading, unsupported, or poorly formatted

Optional detailed scoring (0–2 each):
- Grounding
- Correctness
- Notation
- LaTeX quality
- Pedagogical usefulness
- Failure behavior

Total possible: 12

---

# Benchmark Prompts

## Category 1 — Dual Formulation

### Prompt D1
**Prompt:**  
Write the dual problem in LaTeX for the current coordinated supply chain formulation.

**What a good answer should contain:**
- a clearly formatted LaTeX dual formulation,
- notation aligned with the supported primal structure,
- dual variables associated with the relevant constraints,
- no placeholder notation,
- no unexplained symbol drift.

**Failure conditions:**
- generic prose with little math,
- placeholder coefficients,
- inconsistent signs,
- unsupported claim of full symbolic generality.

---

### Prompt D2
**Prompt:**  
Show me the dual of the current model and explain what each dual variable means economically.

**What a good answer should contain:**
- dual formulation in LaTeX,
- interpretation of dual variables in economic terms,
- linkage between constraints and corresponding dual quantities,
- clear distinction between formulation and interpretation.

**Failure conditions:**
- dual presented without interpretation,
- interpretation detached from the actual modeled constraints,
- invented economics beyond the supported scope.

---

### Prompt D3
**Prompt:**  
Give me only the LaTeX version of the dual problem, with no prose.

**What a good answer should contain:**
- concise, valid LaTeX,
- formulation only,
- consistent notation.

**Failure conditions:**
- extra explanatory prose despite the request,
- malformed LaTeX,
- incomplete formulation.

---

### Prompt D4
**Prompt:**  
Compare the primal and dual problems for the current state, and explain the relationship in the style of mathematical programming.

**What a good answer should contain:**
- concise comparison of primal and dual roles,
- mathematically literate explanation,
- no overclaiming beyond supported structure,
- preferably clear LaTeX snippets for both.

**Failure conditions:**
- vague high-level explanation only,
- no reference to actual modeled structure,
- incorrect duality statements.

---

## Category 2 — Theorem 1 Proof Generation

### Prompt T1
**Prompt:**  
Show that Theorem 1 holds and provide the answer as a polished LaTeX proof.

**What a good answer should contain:**
- theorem statement or clearly identified theorem target,
- explicit assumptions or applicability status,
- proof environment or theorem/proof style formatting,
- logic aligned with the curated theorem_1 interpretation,
- no bluffing if assumptions are missing.

**Failure conditions:**
- proof-like language without actual proof structure,
- unsupported claims,
- no mention of applicability conditions when needed.

---

### Prompt T2
**Prompt:**  
Write a rigorous mathematical proof in LaTeX that Theorem 1 applies in this case.

**What a good answer should contain:**
- either a proof or a clean explanation that applicability is not fully verified,
- explicit connection to verified assumptions,
- concise operations-research style writing.

**Failure conditions:**
- claiming applicability without verified assumptions,
- purely informal explanation instead of proof-style response.

---

### Prompt T3
**Prompt:**  
State Theorem 1 and then prove it using the notation of the current system.

**What a good answer should contain:**
- statement + proof structure,
- notation consistent with current curated context,
- no invented theorem numbering or statement changes outside supported scope.

**Failure conditions:**
- theorem statement that does not match supported interpretation,
- symbol drift,
- proof disconnected from the theorem statement.

---

### Prompt T4
**Prompt:**  
Give me a shorter version of the proof of Theorem 1, but still in proper mathematical LaTeX.

**What a good answer should contain:**
- shortened but rigorous proof,
- preserved mathematical integrity,
- concise LaTeX formatting.

**Failure conditions:**
- oversimplified prose replacing proof,
- omission of essential logical steps,
- degraded notation consistency.

---

## Category 3 — Theorem Applicability and Explanation

### Prompt A1
**Prompt:**  
Explain why Theorem 1 applies here.

**What a good answer should contain:**
- applicability discussion,
- reference to verified assumptions,
- distinction between “applies” and “is proved,”
- mathematically grounded explanation.

**Failure conditions:**
- vague statement like “it applies because conditions are satisfied” without identifying conditions,
- unsupported certainty.

---

### Prompt A2
**Prompt:**  
Which assumptions are needed for Theorem 1, and which of them are currently verified?

**What a good answer should contain:**
- list of required assumptions,
- indication of verified vs missing assumptions,
- honest status reporting.

**Failure conditions:**
- merged or ambiguous assumptions,
- no distinction between verified and unverified conditions.

---

### Prompt A3
**Prompt:**  
If Theorem 1 cannot be established from the current state, explain exactly why.

**What a good answer should contain:**
- precise missing assumptions or unsupported elements,
- no fake proof,
- useful next-step guidance.

**Failure conditions:**
- generic failure message,
- vague “insufficient information” without specifics,
- hidden assumption gaps.

---

## Category 4 — Section 2.3 Negative Bids

### Prompt S1
**Prompt:**  
Explain the role of negative bids in Section 2.3 of Sampat et al. (2019).

**What a good answer should contain:**
- clear explanation of negative bids,
- grounding in Section 2.3 concepts,
- distinction between interpretation and proof,
- pedagogically useful description.

**Failure conditions:**
- generic market commentary not tied to Sampat,
- lack of Section 2.3 specialization.

---

### Prompt S2
**Prompt:**  
Write a mathematically clean explanation of how disposal or remediation can be represented through negative bids.

**What a good answer should contain:**
- Section 2.3 style modeling explanation,
- clear connection between economics and formulation,
- preferably some math notation or symbolic description.

**Failure conditions:**
- purely verbal explanation without mathematical content,
- incorrect sign interpretation.

---

### Prompt S3
**Prompt:**  
Can negative nodal prices arise in this framework? Explain using the logic of Section 2.3.

**What a good answer should contain:**
- clear yes/no with conditions,
- Section 2.3 grounded interpretation,
- no unsupported extension beyond the curated scope.

**Failure conditions:**
- generic electricity-market discussion detached from the current domain layer,
- unsupported claims about all cases.

---

### Prompt S4
**Prompt:**  
Explain Section 2.3 to a graduate student, but keep the answer mathematically disciplined.

**What a good answer should contain:**
- pedagogically useful explanation,
- still grounded in formal concepts,
- balanced prose and math.

**Failure conditions:**
- too informal,
- too generic,
- loss of domain specialization.

---

## Category 5 — Mixed Math Exposition

### Prompt M1
**Prompt:**  
First write the dual problem in LaTeX, and then explain how that dual relates to prices and incentives in the model.

**What a good answer should contain:**
- dual first,
- explanation second,
- good separation of formulation and interpretation,
- no mixing of unsupported claims.

**Failure conditions:**
- missing dual,
- poor structure,
- confused economic interpretation.

---

### Prompt M2
**Prompt:**  
Show me a theorem-style response: state Theorem 1, prove it, and then briefly interpret its meaning for the coordinated supply chain model.

**What a good answer should contain:**
- theorem statement,
- proof,
- short interpretation,
- clean structure.

**Failure conditions:**
- interpretation overwhelms proof,
- lack of clear theorem/proof separation.

---

### Prompt M3
**Prompt:**  
Write the answer in beautiful LaTeX, suitable for inclusion in a classroom handout.

**What a good answer should contain:**
- especially clean formatting,
- polished theorem/proof or derivation style,
- readable equations.

**Failure conditions:**
- ugly or cluttered formatting,
- weak structure,
- plain text instead of mathematical presentation.

---

## Category 6 — Out-of-Scope / Failure Behavior

### Prompt O1
**Prompt:**  
Prove Theorem 3 in LaTeX.

**What a good answer should contain:**
- clear statement that this is unsupported if theorem_3 is not implemented,
- honest scope limitation,
- possibly a note that the current layer is focused on theorem_1.

**Failure conditions:**
- fabricated theorem_3 proof,
- pretending support exists.

---

### Prompt O2
**Prompt:**  
Use the supporting information to derive a completely general symbolic dual for any future extension of the model.

**What a good answer should contain:**
- clear limitation statement,
- explanation that the dual support is scaffolded for the currently supported LP structure,
- no false claim of general symbolic derivation.

**Failure conditions:**
- bluffing generality,
- unsupported symbolic engine claims.

---

### Prompt O3
**Prompt:**  
Answer any theorem question from the paper, even if it is not in Sections 2.1–2.3.

**What a good answer should contain:**
- explicit scope boundary,
- refusal to overclaim,
- clear supported range.

**Failure conditions:**
- pretending full-paper support exists.

---

### Prompt O4
**Prompt:**  
Prove Theorem 1 even though the necessary assumptions are not satisfied.

**What a good answer should contain:**
- refusal to provide a grounded proof,
- explanation of which assumptions are missing,
- possibly a conditional explanation instead of a false proof.

**Failure conditions:**
- proving it anyway,
- hiding failed assumptions.

---

## Category 7 — LLM / Fallback Behavior

### Prompt F1
**Prompt:**  
Show that Theorem 1 holds.

**Evaluation note:**  
Run once with LLM enabled and once with LLM disabled if possible.

**What a good answer should contain:**
- with LLM enabled: polished theorem/proof style output,
- with LLM disabled: graceful fallback, still honest and coherent.

**Failure conditions:**
- system crash,
- unreadable fallback,
- inconsistent theorem status.

---

### Prompt F2
**Prompt:**  
Write the dual problem in LaTeX.

**Evaluation note:**  
Run once with LLM enabled and once with LLM disabled if possible.

**What a good answer should contain:**
- stable behavior in both modes,
- clearer rendering with LLM,
- no loss of mathematical honesty in fallback mode.

**Failure conditions:**
- output drift between modes that changes the mathematics,
- broken fallback.

---

# Evaluation Record Template

Copy and fill this for each prompt.

## Prompt ID: [ID]
**Prompt:**  
[Paste prompt here]

**Observed outcome summary:**  
[Short summary]

**Score:**  
- Grounding:
- Correctness:
- Notation:
- LaTeX quality:
- Pedagogical usefulness:
- Failure behavior:
- Overall: Pass / Borderline / Fail

**Strengths:**  
- 
- 

**Weaknesses:**  
- 
- 

**Action items:**  
- 
- 

---

# Recommended First-Pass Priority Prompts

If you do not want to run the full set immediately, start with these:

1. D1
2. T1
3. A1
4. S1
5. O1
6. O4
7. F1
8. F2

These will tell you quickly whether the current system is:
- mathematically grounded,
- well formatted,
- trustworthy under failure,
- actually using the LLM effectively.

---

# Exit Criteria for This Phase

The current theorem/dual layer is ready for broader use only if:

- theorem_1 responses are consistently correct and well structured,
- dual responses are visually clean and notation-consistent,
- Section 2.3 explanations sound domain-specific rather than generic,
- unsupported requests fail honestly and clearly,
- fallback behavior is graceful when LLM support is unavailable.

If these conditions are not met, refine prompt design, formal context completeness, or domain registry quality before expanding scope.