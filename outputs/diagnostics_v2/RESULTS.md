# QUBO diagnostic: original Two Moons experiment

This is an **exploratory follow-up** on the original three initializations and mask audit split.
It reuses existing audit labels; it is not a fresh confirmatory test or a new dataset trial.
Test data was not evaluated or used for any audit decisions.

## Does QAOA solve the surrogate?

| seed | qaoa_mask | random_mask | exact_qubo_mask | qaoa_random_hamming | qaoa_equals_exact | qaoa_surrogate_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 10111000 | 01010101 | 11101000 | 6 | False | 0.004643 |
| 22 | 10011001 | 01010101 | 10001011 | 4 | False | 0.009324 |
| 33 | 01010101 | 00101011 | 11010001 | 6 | False | 0.020972 |

QAOA's selected mask differs from the single random mask in every initialization, and does not
equal the exact surrogate optimum in any of the three. The gap is measured in unpenalized
predicted validation cross-entropy; all compared masks satisfy the same cardinality.

## Do pairwise terms improve prediction?

Both models use exactly the same original 128 fitting labels and five-fold CV with the same
alpha grid. The 35 held-out feasible masks per initialization are excluded from fitting and CV.
The original quadratic coefficients are reused unchanged. Positive cost/loss correlation is
desirable; negative cost/accuracy correlation is desirable.

| seed | model | n_audit | MAE | R2 | Spearman_cost_vs_loss | ridge_alpha | Spearman_cost_vs_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | Linear only | 35 | 0.0432 | -0.3978 | 0.2658 | 1.0 | -0.3385 |
| 11 | Quadratic | 35 | 0.032 | -0.2622 | 0.2874 | 0.1 | -0.3135 |
| 22 | Linear only | 35 | 0.0392 | 0.1991 | 0.5588 | 1.0 | -0.1768 |
| 22 | Quadratic | 35 | 0.0362 | 0.2997 | 0.5246 | 1.0 | -0.1642 |
| 33 | Linear only | 35 | 0.031 | -0.013 | 0.7059 | 1.0 | -0.5166 |
| 33 | Quadratic | 35 | 0.0287 | -0.1224 | 0.4678 | 0.0001 | -0.4565 |

Mean audit metrics:

| model | MAE | R2 | Spearman_cost_vs_loss | Spearman_cost_vs_accuracy |
| --- | --- | --- | --- | --- |
| Linear only | 0.0378 | -0.0706 | 0.5102 | -0.344 |
| Quadratic | 0.0323 | -0.0283 | 0.4266 | -0.3114 |

## Does selecting by the surrogate help?

All 70 feasible masks were retrained from saved initializations, with exactly the original
200 full-batch Adam steps and fresh optimizer. Final validation losses matched the saved
labels within 1e-7. Validation accuracy was measured at every epoch. The oracle below is a
retrospective validation-loss reference, not a deployable selector or test-set optimum.
Audit-only selection compares both surrogates on masks whose labels neither saw during fitting.

| candidate_pool | method | val_loss | val_accuracy | val_loss_regret |
| --- | --- | --- | --- | --- |
| all70 | Exact linear | 0.1547 | 0.93333 | 0.00344 |
| all70 | Exact quadratic | 0.15489 | 0.93556 | 0.00362 |
| all70 | QAOA | 0.15777 | 0.93556 | 0.0065 |
| all70 | Random | 0.1556 | 0.93333 | 0.00434 |
| all70 | Validation oracle (diagnostic) | 0.15126 | 0.93556 | 0.0 |
| audit35 | Audit-only linear | 0.15551 | 0.93111 | 0.00385 |
| audit35 | Audit-only oracle (diagnostic) | 0.15166 | 0.93667 | 0.0 |
| audit35 | Audit-only quadratic | 0.15394 | 0.93556 | 0.00228 |

## How much performance variation is available?

| seed | min_val_accuracy | max_val_accuracy | median_val_accuracy | fraction_within_1pp_best | min_val_loss | max_val_loss |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 0.86333 | 0.94667 | 0.93333 | 0.11429 | 0.15129 | 0.30902 |
| 22 | 0.86667 | 0.94667 | 0.93333 | 0.21429 | 0.15246 | 0.31192 |
| 33 | 0.87667 | 0.94667 | 0.935 | 0.5 | 0.15004 | 0.30398 |

The plot shows held-out cost/loss scatter, the pairwise ablation, the complete feasible-mask
performance spread, and selected-mask retraining curves. These are descriptions of three
initializations, not statistically established advantages. Pairwise features can improve fit
without reliably improving selection; read both tables together. A stronger dataset will not
automatically repair either surrogate error or QAOA optimization error.

Reproducibility: run `python run_audit.py` from this folder (or any working directory), with
the original sibling `results/` and `experiment_utils.py` available. It writes six CSVs and
`qubo_audit.png`. No original files are modified. Audit elapsed time: 42.8 seconds.
