# Problem 1, Question 4: Manure Management With Composting

Continue from the Question 3 setting.

A dairy farm in Eau Claire produces 1000 tons of dairy manure and is willing to
pay 0.7 dollars per ton to anyone who takes its manure.

Menomonie is under the DNR policy and effectively requests manure at -0.5
dollars per ton. Menomonie can accept at most 500 tons of manure.

Black River Falls values manure at 1.5 dollars per ton and can accept at most
500 tons of manure.

Question 4 adds a compost pathway. A Madison compost consumer is willing to pay
100 dollars per ton of compost and can accept up to 100 tons of compost.
Transporting compost to Madison costs 1 dollar per ton of compost.

The dairy farmer has access to a composting technology represented as a
Composter. The Composter consumes dairy manure and produces compost. Its
capacity is 500 tons of manure processed, and its operating cost is 1 dollar
per ton of manure processed. The yield coefficients are:

- DM/manure: -1.0
- Compost: 0.1

The transport links are:

- DF to CF transports DM, capacity 1000, cost 0.1
- DF to SF transports DM, capacity 1000, cost 0.2
- DF to Composter transports DM, capacity 1000, cost 0.0
- Composter to DC transports Compost, capacity 1000, cost 1.0

The cooperative maximizes total profit. Determine the optimal manure and
compost allocation, the composter activity, total profit, whether all 1000 tons
of manure are removed, and verify the balances.
