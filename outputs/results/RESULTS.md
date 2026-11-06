# Executed Qismet experiment

Completed all five milestones on a 2→8→2 PyTorch model using a local Qiskit simulator.
Across 3 fixed initializations on one fixed split, after 50% hidden-neuron pruning:

| Method | Mean test accuracy |
|---|---:|
| Dense | 97.44% |
| QAOA selected + rewind | 97.67% |
| Single random mask + rewind | 97.67% |
| Best of 32 random feasible + rewind | 97.44% |
| Exact surrogate optimum + rewind | 97.44% |
| Fresh initialization, same width | 93.11% |

The selected sparse model has 22 parameters versus
42 in the dense model.
Mean held-out within-budget Spearman correlation: **0.427**.
QAOA's mean probability of a top-decile surrogate mask, conditional on feasibility:
**10.06%**, versus **10.00%** for uniform feasible sampling.

This demonstrates an executed pruning/rewinding/QUBO/QAOA pipeline. It does not establish
quantum advantage or superiority over random/classical pruning. Interpret the accuracy table
together with the surrogate holdout metrics and sampling diagnostics. Test data did not guide
surrogate fitting, optimizer settings, or mask selection.

Defaults: 200 full-batch Adam steps, learning rate 0.03; seeds [11, 22, 33];
two QAOA layers; three COBYLA starts; at most 120 objectives/start; 32 final shots.
Each initialization uses 128 fit-mask retrainings plus 48 audit-mask retrainings at the default
50% pruning budget. This is expensive label collection, not a speedup demonstration.

### What this executed run supports

The QAOA-selected four-neuron networks retained the dense model's accuracy on this split,
but **random pruning achieved the same mean test accuracy (97.67%)**. Magnitude pruning
averaged 98.00%, and classical swap search averaged 97.67%. These small differences are
descriptive; no statistical superiority is established.

The quadratic approximation has limited predictive quality at the requested pruning budget:
held-out-mask R² values are **−0.262, 0.300, and −0.122**. Two are negative. The mean
rank correlation is 0.427. QAOA's conditional top-decile probability is **10.06%**, against
10.00% for uniform feasible masks, so it learns cardinality much more strongly than good-mask ranking.

All nine COBYLA restarts reached the declared evaluation limit; the result is the best observed
within that budget, not a claim of optimizer convergence. The next research steps are to improve
the surrogate and evaluate a cardinality-preserving mixer under a new, predeclared protocol.

