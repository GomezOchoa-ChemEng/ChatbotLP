# Midterm 1 Benchmark: Manure Management Q1

This benchmark is a focused evaluation layer for Midterm Problem 1, Question 1:
Manure Management.

Source/reference files:

- `Benchmarks/midterm1/midtermI_cbe450_spring2026.pdf`
- `Benchmarks/midterm1/SupplyChain_Manure_Q1.ipynb`

The goal is to test whether the current ChatbotLP pipeline can interpret a
realistic prose problem as a coordinated supply-chain instance, validate it,
solve it when enough information is present, and generate grounded reasoning.

This benchmark is intentionally separate from the Case A/B/C paper-grade
evaluation. It does not change the existing benchmark behavior.

Expected Question 1 solution:

- Eau Claire to Menomonie flow: 500 tons
- Eau Claire to Black River Falls flow: 500 tons
- Dairy supply accepted: 1000 tons
- Menomonie demand accepted: 500 tons
- Black River Falls demand accepted: 500 tons
- Demand revenue: 1000
- Transport cost: 150
- Supply cost: 0
- Total profit: 850

Run locally from the repository root:

```bash
python examples/midterm_manure_q1_benchmark.py --no-llm --skip-reasoning
```

For live Gemini interpretation, set `GEMINI_API_KEY`, `LLM_PROVIDER=gemini`,
and optionally `GEMINI_MODEL`, then run:

```bash
python examples/midterm_manure_q1_benchmark.py --strict-llm
```
