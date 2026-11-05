# Post-hoc selector timing

These are CPU timings on the saved Fashion-MNIST objectives, excluding retraining and surrogate fitting.
QAOA uses recorded original statevector-simulator wall time (three starts, up to 360 exact expectation
evaluations, 32 final shots). Classical selectors are measured now: median of five batches of 1000
solves per objective, including candidate construction, scoring and selection. Imports/data loading
are excluded. Runs were not simultaneous or load-controlled; this is an indicative toy-instance
comparison, not a quantum-hardware or asymptotic speedup benchmark.

| Pruning | Selector | Mean milliseconds across five initializations |
|---|---|---:|
| 50% | Best of 32 random by QUBO | 0.0340 |
| 50% | Exact QUBO enumeration | 0.0456 |
| 50% | QAOA statevector simulation | 2019.2453 |
| 75% | Best of 32 random by QUBO | 0.0339 |
| 75% | Exact QUBO enumeration | 0.0376 |
| 75% | QAOA statevector simulation | 1985.8340 |

There are only 70 feasible masks at 50% pruning and 28 at 75%. Exhaustive classical selection
is cheap at this scale. QAOA was not faster in these runs. Its simulator runtime cannot establish
how actual quantum hardware would compare. Full experiment cost also includes 885 mask retrainings;
all surrogate-based methods share this expensive label-generation stage. No end-to-end speedup has
been demonstrated. QAOA timings include the solver's encoding/diagnostic checks as well as optimization.
