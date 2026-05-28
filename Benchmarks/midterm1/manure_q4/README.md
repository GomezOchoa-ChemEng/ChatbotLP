# Midterm 1 Benchmark: Manure Management Q4

This benchmark evaluates Midterm Problem 1, Question 4: the Q3 manure removal
setting with an added compost pathway.

Question 4 keeps the Eau Claire dairy supplier bid at -0.7 dollars per ton,
Menomonie's effective bid at -0.5 dollars per ton, and Black River Falls' bid
at 1.5 dollars per ton. It adds a Madison compost consumer with willingness to
pay 100 dollars per ton of compost and a composter that consumes dairy manure
and produces compost.

The composter has capacity 500 tons of manure processed, operating cost 1
dollar per ton of manure processed, and yield coefficients:

- DM/manure: -1.0
- Compost: 0.1

The benchmark uses ID-independent scoring. Exact IDs and aliases are secondary
diagnostics; primary scoring checks semantic structure, parameter multisets,
topology, technology/yield structure, route/pathway economics, solver
aggregates, and balance residuals.

Expected Question 4 solution:

- Eau Claire/DF supply accepted: 1000 tons DM
- Menomonie/CF demand accepted: 0 tons DM
- Black River Falls/SF demand accepted: 500 tons DM
- Madison/DC demand accepted: 50 tons Compost
- Eau Claire/DF to Menomonie/CF flow: 0 tons DM
- Eau Claire/DF to Black River Falls/SF flow: 500 tons DM
- Eau Claire/DF to Composter flow: 500 tons DM
- Composter to Madison/DC flow: 50 tons Compost
- Composter activity: 500 tons manure processed
- Compost produced: 50 tons
- Demand revenue: 5750
- Supply contribution: 700
- Transport cost: 150
- Technology operating cost: 500
- Total profit: 5800

Run locally from the repository root:

```bash
python3 examples/midterm_manure_q1_benchmark.py --question q4 --no-llm --skip-reasoning
```
