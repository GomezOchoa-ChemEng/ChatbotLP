# HW2 Plastic Waste Benchmark

This benchmark supports `notebooks/HW2_PlasticWaste_Showcase.ipynb`.

The deterministic fixture lives in `src.hw2_plastic_waste` and builds a
`ProblemState` with explicit numerical values.  Missing values remain `None`.
In particular, the fixed-charge design model requires technology investment
costs before solving.

The homework PDF is already present at:

```text
docs/hw2_supply_chains_spring2022.pdf
```

If needed later, place a copy at:

```text
Benchmarks/hw2_plastic_waste/hw2_supply_chains_spring2022.pdf
```

Expected design solution at ethylene WTP 1050 $/ton:

- Select `Pyrolysis_MKE`
- Select `SteamCracking_MKE`
- Process 100000 ton/year plastic waste
- Leave 10400 ton/year plastic waste unprocessed
- Objective value: 6978000 $/year

Expected full-recycling WTP threshold:

- Approximately 1465.485 $/ton ethylene
- Select `Pyrolysis_MAD`, `Pyrolysis_MKE`, and `SteamCracking_MKE`
