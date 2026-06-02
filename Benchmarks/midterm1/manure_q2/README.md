# Midterm 1 Benchmark: Manure Management Q2

This benchmark evaluates Midterm Problem 1, Question 2: the DNR policy/remediation
charge variant of the manure management problem.

Question 2 changes the Menomonie receiving farm economics. Menomonie originally
pays 0.5 dollars per ton, but the DNR remediation charge is 1.0 dollars per ton,
so its effective bid is:

```text
0.5 - 1.0 = -0.5 dollars per ton
```

Both transport routes have capacity 1000 tons. Their coordinated route net
values are:

```text
Menomonie:         -0.5 - 0.0 - 0.1 = -0.6
Black River Falls:  1.5 - 0.0 - 0.2 =  1.3
```

The benchmark uses the same ID-independent scoring layer as Q1. Exact entity
IDs are secondary diagnostics; primary scoring checks semantic structure,
negative-bid detection, route-specific attribute binding, topology, route
economics, solver aggregates, balance residuals, and reasoning readiness.

Expected Question 2 solution:

- Eau Claire to Menomonie flow: 0 tons
- Eau Claire to Black River Falls flow: 500 tons
- Dairy supply accepted: 500 tons
- Menomonie demand accepted: 0 tons
- Black River Falls demand accepted: 500 tons
- Demand revenue: 750
- Transport cost: 100
- Supply cost: 0
- Total profit: 650

Run locally from the repository root:

```bash
python3 examples/midterm_manure_q1_benchmark.py --question q2 --no-llm --skip-reasoning
```
