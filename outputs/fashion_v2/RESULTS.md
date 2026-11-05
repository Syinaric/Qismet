# Fashion-MNIST executed results

784 pixel inputs → 16 ReLU hidden neurons → 10 clothing classes. Fixed groups of two neurons; 8 QAOA qubits.

5,000 training / 1,000 validation / 10,000 official test examples; five initialization seeds; 20 epochs per model.

## Observed comparison

At 50% neuron pruning, mean test accuracy was 80.41% for QAOA, 80.28% for the ten-mask random mean, 80.98% for exact QUBO, 80.73% for exact linear, and 81.11% for best-of-32 random selection by QUBO cost.

At 75% neuron pruning, mean test accuracy was 76.43% for QAOA, 71.34% for the ten-mask random mean, 76.28% for exact QUBO, 76.25% for exact linear, and 76.53% for best-of-32 random selection by QUBO cost.

The comparison with random pruning tests useful mask selection; comparison with exact linear and best-of-random is needed before attributing any gain to pairwise terms or QAOA.

## Final test accuracy

| pruning | method | test accuracy (%) | SD (pp) | parameters |
| --- | --- | --- | --- | --- |
| 50.0 | Dense | 82.18 | 0.47 | 12730.0 |
| 50.0 | Random (10-mask mean) | 80.28 | 0.73 | 6370.0 |
| 50.0 | Magnitude | 80.65 | 0.63 | 6370.0 |
| 50.0 | Exact linear | 80.73 | 0.47 | 6370.0 |
| 50.0 | Exact QUBO | 80.98 | 0.66 | 6370.0 |
| 50.0 | QAOA | 80.41 | 0.55 | 6370.0 |
| 50.0 | Best of 32 random | 81.11 | 0.49 | 6370.0 |
| 75.0 | Dense | 82.18 | 0.47 | 12730.0 |
| 75.0 | Random (10-mask mean) | 71.34 | 1.62 | 3190.0 |
| 75.0 | Magnitude | 75.59 | 1.55 | 3190.0 |
| 75.0 | Exact linear | 76.25 | 0.83 | 3190.0 |
| 75.0 | Exact QUBO | 76.28 | 0.5 | 3190.0 |
| 75.0 | QAOA | 76.43 | 0.9 | 3190.0 |
| 75.0 | Best of 32 random | 76.53 | 0.46 | 3190.0 |

## Paired differences: seed-level descriptive 95% t intervals

| pruning | difference | n_pairs | mean_pp | lower_95_pp | upper_95_pp |
| --- | --- | --- | --- | --- | --- |
| 50.0 | QAOA minus Random (10-mask mean) | 5 | 0.128 | -0.362 | 0.618 |
| 50.0 | QAOA minus Exact QUBO | 5 | -0.572 | -1.782 | 0.638 |
| 50.0 | QAOA minus Magnitude | 5 | -0.242 | -0.472 | -0.012 |
| 50.0 | Exact QUBO minus Exact linear | 5 | 0.246 | -0.675 | 1.167 |
| 50.0 | Exact QUBO minus Random (10-mask mean) | 5 | 0.7 | -0.599 | 2.0 |
| 50.0 | QAOA minus Dense | 5 | -1.768 | -2.566 | -0.97 |
| 75.0 | QAOA minus Random (10-mask mean) | 5 | 5.09 | 3.346 | 6.834 |
| 75.0 | QAOA minus Exact QUBO | 5 | 0.152 | -1.278 | 1.582 |
| 75.0 | QAOA minus Magnitude | 5 | 0.836 | -1.03 | 2.702 |
| 75.0 | Exact QUBO minus Exact linear | 5 | 0.026 | -1.447 | 1.499 |
| 75.0 | Exact QUBO minus Random (10-mask mean) | 5 | 4.938 | 2.339 | 7.537 |
| 75.0 | QAOA minus Dense | 5 | -5.748 | -7.059 | -4.437 |

## Surrogate holdout quality and audit-only selection

| pruning | model | mae_mean | r2_mean | spearman_mean | selection_regret_mean | selected_val_loss_mean | audit_masks_per_seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50.0 | Linear | 0.0512 | -7.0262 | 0.5745 | 0.0121 | 0.5538 | 35 |
| 50.0 | Quadratic | 0.0357 | -3.4653 | 0.5718 | 0.0189 | 0.5606 | 35 |
| 75.0 | Linear | 0.0882 | 0.1663 | 0.7574 | 0.0221 | 0.6663 | 14 |
| 75.0 | Quadratic | 0.1144 | -0.7416 | 0.7363 | 0.0116 | 0.6558 | 14 |

## QAOA optimization and sampling

| pruning | exact_hits | trials | mean_surrogate_gap | mean_feasible_probability | mean_top_decile_conditional | uniform_top_decile | mean_feasible_shots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50.0 | 1 | 5 | 0.01175 | 0.76757 | 0.10008 | 0.1 | 24.8 |
| 75.0 | 2 | 5 | 0.00939 | 0.52988 | 0.10789 | 0.10714 | 17.8 |

The random baseline is a ten-mask average within each seed. The five initialization seeds, not individual masks or epochs, are the replication units. Intervals describe this fixed split and are not a claim of superiority or generalization across datasets.

Both surrogate models use the same fitting labels and CV splits. The test set does not guide masks, regression, or optimization. Exact QUBO is an oracle for predicted loss, not actual retraining performance. QAOA uses noiseless classical statevector simulation and 32 final shots. These results do not establish quantum advantage.

References: [Fashion-MNIST official repository](https://github.com/zalandoresearch/fashion-mnist); [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635).
