# Qismet: revised performance experiment

**Real hardware result:** the revised sampler achieved **81.651% expected best-of-32 test accuracy**, versus **81.478%** for uniform random search, a **+0.172 percentage-point** improvement at the same 32-raw-proposal budget. This is one model initialization and one finite measurement batch per depth; it is not a speedup claim.

The revised ideal QAOA sampler improved expected retrained test accuracy over uniform random pruning on **all five fresh initializations**. Mean gain per accepted proposal: **+0.771 percentage points**. Mean expected best-of-32 gain: **+0.233 percentage points**. These are ideal-simulation results; hardware results are stated separately below.

## What changed and why

The dataset remains Fashion-MNIST and the model remains 784→16→10. We fit the QUBO on **35 masks at the desired size**, rather than mixing multiple network sizes. Regularization is selected by cross-validation on those masks. Another 35 masks audit prediction fidelity. Only half the hidden neurons survive, reducing parameters from **12,730 to 6,370**.

The new circuit starts with four Bell pairs and uses paired RXX/RYY rotations on a ring. These XY mixers preserve the number of retained groups in an ideal circuit. Removing the large size penalty lets the cost phases represent quality differences directly. This is the constrained alternating-operator approach described by [Hadfield et al.](https://arxiv.org/abs/1709.03489).

Three QAOA layers use six angles, optimized on a CPU simulator with three fixed starts. Hardware executes frozen angles; it does not perform the earlier online SPSA loop. Both the earlier and revised methods are hybrids with classical preparation costs.

## Independent ideal-simulation follow-up

Fresh initializations: 101, 202, 303, 404, 505. Each network, surrogate, circuit and selected mask is frozen before test evaluation. The same Fashion-MNIST split is reused. This is an exploratory follow-up; fresh seeds check initialization dependence, not a new dataset.

| Metric | Uniform feasible masks | Revised ideal QAOA |
|---|---:|---:|
| Expected test accuracy per proposal | 80.130% | 80.901% |
| Expected test accuracy, best of 32 | 81.155% | 81.389% |
| Chance of top-10% QUBO mask | 10.00% | 53.52% |

Dense mean accuracy: **82.50%**. The revised ideal best-of-32 subnetworks retain half the parameters with an average accuracy drop of **1.11 percentage points**.

The paired 95% t interval across five initializations is **[0.569, 0.974] pp** for the one-proposal gain and **[0.012, 0.455] pp** for best-of-32. These small-sample intervals condition on this dataset and ideal probabilities. They do not include dataset uncertainty, hardware noise or all researcher choices.

Expected metrics integrate the frozen sampling distribution against the actual retrained performance of all 70 valid masks. Best-of-32 selects by the QUBO, not by true test performance. This avoids presenting one lucky quantum draw against one unlucky random draw. It does not remove the simulator's optimization cost from the experiment.

![Fresh initializations](fresh_seed_performance.png)

## Is the QUBO more accurate?

The revised surrogate's mean held-out R² is **0.450** and mean absolute validation-loss error is **0.0132** on the five fresh initializations. On the original five development seeds, fixed-size fitting improved mean R² from **−3.465 to 0.333**, and mean absolute error from **0.0357 to 0.0139**, using the same original audit masks. Prediction is useful but still imperfect.

![Surrogate and sampling](surrogate_and_sampling.png)

## Actual hardware follow-up

Four real follow-up jobs ran on **ibm_fez**, consuming **12 QPU seconds**. All completed jobs across the project report **32 seconds** of QPU usage. The earlier queued Kingston attempt was cancelled before any QAOA result because of queue latency; its metrics currently report zero charge with pending usage status. Its 60-second reservation remains conservatively accounted. Both the partial attempt and all four Fez comparisons are preserved.

The chosen depth is **2**, selected using mean QUBO score before test evaluation. Its feasible fraction is **38.6%**. Among feasible outputs it samples top-10% masks **27.6%** of the time; counting every raw shot this is **10.6%**. Uniform feasible sampling is 10%.

Expected test accuracy per accepted hardware proposal is **80.59%**, versus **80.16%** for uniform feasible masks. Expected best-of-32 accuracy, including the raw-shot acceptance process, is **81.65%**, versus **81.48%** for random search. Probability of no feasible mask in 32 shots: **1.6e-07**.

The single frozen 32-shot selection achieved **81.81%** test accuracy. The corresponding seeded random selection achieved **81.29%**. Classical exact QUBO search found the same selected mask (state 60) and therefore matches its 81.81% test accuracy. Single draws are not the primary evidence of distribution quality.

The one-layer circuit also showed a small positive best-of-32 gain. The three-layer circuit did not improve that metric, illustrating the tradeoff between ideal expressiveness and device noise. The two-layer circuit accepts only **38.6%** of raw shots. Conditional sampling gains must be reported alongside this rejection overhead.

All four depths, job IDs, shot counts, validation loss and test accuracy are in `hardware/depth_comparison.csv`. Hardware uncertainty and drift are not captured by the initialization intervals below.

A multinomial bootstrap of these finite-shot batches, repeating QUBO-based depth selection, gives a sampling-only 95% interval of **[0.140, 0.202] percentage points** for the expected best-of-32 gain. It excludes device drift, dataset uncertainty and earlier experiment-design choices. It is not evidence of universal advantage.

## What the success does and does not establish

The ideal follow-up demonstrates better proposal quality and a small positive best-of-32 accuracy gain against the specified uniform-random baseline. It also improves held-out surrogate fidelity. All five fresh initialization results are reported.

Classical exact QUBO search remains cheap for 70 masks and is a stronger optimization reference than random sampling. This experiment does **not** demonstrate quantum speedup, superiority over all classical methods, or quantum advantage. A cost-aware comparison must include label generation, regression, simulator angle optimization, circuit preparation, QPU usage, queue latency and final retraining. Repeatedly evaluating candidates through a learned surrogate is a benefit available to classical optimizers too.

## Files

- `Revised_Walkthrough.ipynb` and `Results.html`: executed explanatory notebook and readable export.
- `fresh_seed_summary.csv`: all five independent ideal results.
- `fresh_seeds/seed_*/selection_frozen.json`: choices frozen before test evaluation.
- `hardware/depth_comparison.csv`: all real circuits and their quality metrics, when complete.
- `constrained_experiment.py`: actual revised circuit and IBM execution code.
