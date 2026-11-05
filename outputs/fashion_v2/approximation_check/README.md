# Does the quadratic approximation match actual retraining loss?

These plots reuse the frozen original Fashion-MNIST surrogates. No fitting, coefficient tuning,
or neural-network retraining was performed to create them. All plotted masks were excluded
from surrogate fitting and CV. There are 35 audit masks at 50% pruning and 14 at 75%, for each
of five initializations. Each label is validation cross-entropy after 20 epochs from surviving
original weights, using the original fixed training protocol. The final test set is not used.

In qubo_loss_fidelity.png, each dot is one mask/initialization pair. The x coordinate is actual
retraining loss and the y coordinate is predicted loss. Closeness to the dashed identity line
indicates accurate loss prediction. Metrics are calculated per initialization, then averaged;
they are not pooled estimates or uncertainty intervals.

| Neuron pruning | Mean absolute error (CE units) | Mean R² | Mean rank correlation |
|---|---:|---:|---:|
| 50% | 0.036 | -3.47 | 0.572 |
| 75% | 0.114 | -0.74 | 0.736 |

The current QUBO is not a reliably close approximation of absolute retraining loss.
Negative per-seed R² means larger squared error than a constant equal to that audit set's
mean measured loss; this is a diagnostic baseline, not a proposed deployable predictor.
MAE is in cross-entropy units, not accuracy percentage points.

In qubo_ranking_fidelity.png, ranks are calculated within each initialization before plotting.
The model retains moderate ranking information at 50% pruning and stronger ranking information
at 75%. A score can rank candidates usefully while systematically misestimating their loss.
That motivates surrogate-guided search, but does not establish that QAOA is a superior solver.
The original classical best-of-random selection matched or exceeded QAOA's mean test accuracy.
Linear-only ranking was also slightly better on average in the existing ablation.

Suggested presentation statement: "Our quadratic surrogate captures some retrainability ranking,
especially at 75% pruning (mean held-out Spearman 0.736), but absolute loss prediction remains
poor. This motivates improving the surrogate and comparing quantum and classical searches;
it does not demonstrate quantum advantage."
