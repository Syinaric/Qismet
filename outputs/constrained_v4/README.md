# Revised pruning experiment

This follow-up keeps Fashion-MNIST and the 784→16→10 network so the formulation changes can be assessed directly. It aims to improve useful mask sampling and pruning quality. It does not claim quantum speedup.

**Complete:** the four Fez hardware comparisons, final CPU evaluations, five fresh initialization experiments, report, figures and executed notebook are ready. Start with `Results.html` or `Revised_Walkthrough.ipynb`. The two-layer hardware sampler improved expected best-of-32 test accuracy by 0.172 percentage points against uniform random selection at the same raw-proposal budget. Classical exact QUBO remains faster and matches the selected network's accuracy.

## Changes

1. Fit the loss surrogate using only the target size: 35 masks retaining four groups. Select ridge regularization by five-fold cross-validation on those fit masks. Audit on 35 disjoint masks.
2. Remove the cardinality penalty. Prepare four Bell pairs, which contain exactly four retained groups, then apply ring XY mixing. This mixing preserves the number retained in the ideal circuit.
3. Scale the cost by its variation among valid masks, so the circuit responds to quality differences. Test one, two and three QAOA layers. Each uses simulator-optimized angles; the follow-up hardware does not optimize its angles online.
4. Repeat the three-layer ideal experiment on five fresh weight initializations, 101–505. Freeze each model, circuit and mask selection before evaluating test images.
5. Run an initial-state control and all three depths on IBM hardware, at 4,096 shots per circuit. Select the final depth using QUBO score, never test accuracy. Report all depths, acceptance rates and failures rather than only the winning circuit.

## Read the results

`Results.md`, `Results.html` and `Revised_Walkthrough.ipynb` are generated after the hardware comparisons and CPU evaluations finish. `fresh_seed_summary.csv` contains the independent ideal-simulation results; `hardware/depth_comparison.csv` contains actual hardware evidence.

The key comparisons are the chance of sampling the best-scoring 10% of masks, expected accuracy per accepted proposal, and expected best-of-32 accuracy. Expected metrics integrate over distributions using retrained per-mask performance; they are not a lucky selection of a few sampled masks. Random baselines use the same QUBO and feasible masks. Classical exact QUBO remains a reference; it is cheap at this problem size.

The original local credentials file is used automatically. No credentials are copied into this folder. The follow-up moved from Kingston to Fez because of the queue, before any new QAOA result was available. The initial Kingston control and cancelled queued job remain in `hardware_kingston_partial/`. All four comparison circuits are rerun on Fez. The ledger subtracts both earlier usage and the cancelled job's uncertain 60-second reservation from the original 600-second allowance, retaining a 120-second buffer. It permits only four Fez jobs.

## Source files

- `constrained_experiment.py`: transparent Qiskit circuit, numerical circuit checks and real IBM jobs.
- `confirm_fresh_seeds.py`: fixed training protocol and independent ideal-simulation follow-up.
- `evaluate_hardware.py`: freeze selection by QUBO, retrain from original weights, then evaluate test results.

This remains a small exploratory experiment with shared Fashion-MNIST splits. Results across seeds measure variation in initialization; they do not establish generalization to all datasets or quantum advantage over strong classical optimization.
