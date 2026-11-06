# Qismet: QUBO audit and Fashion-MNIST follow-up

The revised strategy was executed in order: audit the original objective, compare linear and pairwise models, then run Fashion-MNIST with five initializations and both pruning budgets. Original results were preserved.

## What the Two Moons audit established

- QAOA missed the exact surrogate minimum in all three original runs. Its selected masks differed from the random masks in all three.
- Retraining every feasible four-neuron mask revealed validation accuracy from approximately 86–88% at the low end to 94.67% at the high end. There is meaningful variation, although many masks perform similarly.
- Pairwise terms reduced mean held-out loss-prediction MAE from 0.0378 to 0.0323, but mean rank correlation decreased from 0.510 to 0.427. Selection results were mixed. This does not establish that pairwise information improves pruning.
- This audit reused the original validation examples and mask audit splits. It is exploratory, not a fresh confirmatory evaluation.

See [the detailed audit](diagnostics_v2/QUBO_Diagnostic.html).

## Fashion-MNIST results

The model is 784 → 16 ReLU → 10 logits. It takes a grayscale clothing image and predicts its class. Eight fixed pairs of neurons provide eight pruning variables. The run used 5,000 training images, 1,000 separate validation images, all 10,000 official test images, and 20 epochs per model. Five initialization seeds share one fixed data split.

Mean test accuracy across five initializations:

| Selection method | 50% neuron pruning | 75% neuron pruning |
|---|---:|---:|
| Random masks, ten-mask mean per seed | 80.28% | 71.34% |
| Magnitude | 80.65% | 75.59% |
| Exact linear surrogate | 80.73% | 76.25% |
| Exact quadratic surrogate | 80.98% | 76.28% |
| QAOA sampled selection | 80.41% | 76.43% |
| Best of 32 random masks by QUBO score | 81.11% | 76.53% |

The unpruned model averaged **82.18%**. Parameters decrease from **12,730** to **6,370** or **3,190**. All pruned methods remain below dense-model accuracy on average.

At 75% pruning, QAOA's paired improvement over the ten-mask random baseline averaged **5.09 percentage points**, with a descriptive five-seed 95% t interval of **3.35 to 6.83 points**. However, best-of-32 random selection using the same surrogate was numerically better. The gain cannot be attributed to quantum search. These intervals describe initialization variability on this fixed split, not generalization across datasets or protocols.

QAOA reached the exact quadratic optimum in **1/5** runs at 50% pruning and **2/5** at 75%. Conditional low-cost sampling probabilities remained close to uniform feasible sampling. Simulation used exact statevector expectations for optimization and 32 finite final shots; this is not quantum hardware evidence.

The quadratic model's held-out rank correlation averaged **0.572** and **0.736** at the two pruning levels, versus **0.574** and **0.757** for linear-only. Pairwise terms have not shown a clear advantage. Quadratic held-out R² averaged **−3.47** and **−0.74**, so absolute loss calibration remains weak even where ranking is useful.

The supported conclusion is that **retraining-aware scoring can help select useful heavily pruned subnetworks in this experiment**. The results do not establish a pairwise-model advantage, a QAOA advantage, or sparse subnetworks matching the dense network's performance.

## Files and checks

- [Executed Fashion-MNIST analysis notebook](fashion_v2/Fashion_Qismet.ipynb)
- [Readable Fashion-MNIST report](fashion_v2/Fashion_Qismet.html)
- [Implementation](fashion_v2/fashion_experiment.py) and [run instructions](fashion_v2/README.md)
- [Actual input image examples](fashion_v2/fashion_examples.png)
- [Dataset subset](fashion_v2/dataset_subset.npz)

The notebook transparently reads the saved experiment by default; its runner can resume or reproduce training. Saved files include original initializations, selected-model checkpoints, every fit/audit label, epoch-wise validation curves, exact coefficients, QAOA samples and optimizer traces, uncertainty tables, and source/runtime details.

Validation passed for 885 Fashion-MNIST fit/audit mask labels, 70 benchmark rows, 1,400 method-level learning-curve rows, all ten QAOA sampled selections, Ising energy mappings, and test accuracies reconstructed from the saved QAOA checkpoints. The original audit additionally evaluated all 210 feasible seed/mask combinations without evaluating test data.
