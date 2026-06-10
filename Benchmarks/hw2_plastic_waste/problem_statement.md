# HW#2: Supply Chain for Plastic Waste Processing

This benchmark encodes the plastic waste processing homework as a coordinated
supply-chain design problem.

The homework PDF is available in this repository at:

```text
docs/hw2_supply_chains_spring2022.pdf
```

If that file is ever absent, place the homework PDF at:

```text
Benchmarks/hw2_plastic_waste/hw2_supply_chains_spring2022.pdf
```

## Structured Problem Data

Cities/nodes:

- Madison (`MAD`)
- Milwaukee (`MKE`)

Products:

- Plastic waste (`PlasticWaste`)
- Pyrolysis oil (`PyrolysisOil`)
- Ethylene (`Ethylene`)

Plastic waste supply:

- `MAD`: 33100 ton/year at explicit supply cost 0.0 $/ton
- `MKE`: 77300 ton/year at explicit supply cost 0.0 $/ton

Ethylene demand:

- `MAD`: capacity 100000 ton/year, willingness to pay 1050 $/ton
- `MKE`: capacity 100000 ton/year, willingness to pay 1050 $/ton

Transportation:

- Products can move `MAD -> MKE` and `MKE -> MAD`
- Products are plastic waste, pyrolysis oil, and ethylene
- Every product-direction transport link has cost 10 $/ton
- Every product-direction transport link has capacity 100000 ton/year

Technology candidates:

- Pyrolysis can be installed in `MAD` and `MKE`
- Pyrolysis converts 1 ton plastic waste into 0.7 ton pyrolysis oil
- Pyrolysis operating cost is 14 $/ton plastic processed
- Pyrolysis capacity is 100000 ton plastic/year per facility
- Pyrolysis investment cost is 2000000 $/year per facility
- Steam cracking can be installed in `MAD` and `MKE`
- Steam cracking converts 1 ton pyrolysis oil into 0.25 ton ethylene
- Steam cracking operating cost is 71 $/ton pyrolysis oil processed
- Steam cracking capacity is 100000 ton pyrolysis oil/year per facility
- Steam cracking investment cost is 2800000 $/year per facility

The design objective maximizes demand value minus supplier costs, transport
costs, processing operating costs, and technology investment costs.
