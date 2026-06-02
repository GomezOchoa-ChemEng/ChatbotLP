# Midterm 1 Benchmark: Manure Management Q3

This benchmark evaluates Midterm Problem 1, Question 3: the DNR policy setting
from Question 2 with an added dairy-farmer removal incentive.

Question 3 keeps Menomonie's effective bid at -0.5 dollars per ton and Black
River Falls' bid at 1.5 dollars per ton. The new feature is that the dairy
farmer is willing to pay 0.7 dollars per ton to anyone who takes the manure.
In the coordinated clearing model this is represented as a supplier bid of
-0.7 dollars per ton.

Both transport routes have capacity 1000 tons. The negative supplier bid
contributes 700 dollars to the objective because the objective subtracts
supplier cost. The route net values become:

```text
Menomonie:         -0.5 - (-0.7) - 0.1 = 0.1
Black River Falls:  1.5 - (-0.7) - 0.2 = 2.0
```

The benchmark uses the same ID-independent scoring layer as Q1 and Q2. Exact
entity IDs are secondary diagnostics; primary scoring checks semantic
structure, negative-bid detection, route-specific attribute binding, topology,
route economics, solver aggregates, balance residuals, and reasoning
readiness.

Expected Question 3 solution:

- Eau Claire to Menomonie flow: 500 tons
- Eau Claire to Black River Falls flow: 500 tons
- Dairy supply accepted: 1000 tons
- Menomonie demand accepted: 500 tons
- Black River Falls demand accepted: 500 tons
- Demand revenue: 500
- Transport cost: 150
- Supply cost: -700
- Supply contribution: 700
- Total profit: 1050

Run locally from the repository root:

```bash
python3 examples/midterm_manure_q1_benchmark.py --question q3 --no-llm --skip-reasoning
```
