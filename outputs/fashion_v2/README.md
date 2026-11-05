# Qismet: Fashion-MNIST experiment

Open `Fashion_Qismet.html` for the executed report, or `Fashion_Qismet.ipynb` for the analysis notebook. `fashion_examples.png` shows actual images used to train the classifier. `fashion_results.png` summarizes the results.

The neural network receives a 28×28 grayscale clothing image as **784 pixel values** and predicts one of **10 clothing classes**. Its 16 hidden neurons are grouped into eight adjacent pairs. The experiment keeps four groups (50% neuron pruning) or two groups (75%), with five initialization seeds and 20 training epochs each.

## Run

Keep the original package layout:

```text
outputs/
  requirements.txt
  qaoa_utils.py
  fashion_v2/
    fashion_experiment.py
    build_report.py
    protocol.json
    dataset_subset.npz
    results/
```

Use the same Python environment as the original Qismet experiment, or install `../requirements.txt`. From this folder:

```powershell
python fashion_experiment.py
python build_report.py
```

The first command runs training and QAOA simulation; it resumes seeds with complete output records. The second executes the cached-results analysis notebook and creates its HTML export, figures, summary CSVs and `RESULTS.md`. The notebook defaults to `RUN_EXPERIMENT = False`; set it to `True` to invoke the runner before analysis. This does not force already completed seeds to train again. Use a new output folder for a changed protocol.

Dataset downloads are automatic and verified against the official IDX checksums. `dataset_subset.npz` stores the exact 5,000 training, 1,000 validation and 10,000 test images and labels. Training/validation indices refer to the original official training set; all 10,000 official test images are used. Input pixels are divided by 255.

The tested runtime and package versions are recorded in [environment.json](environment.json).

## Evidence and interpretation

Each seed uses 128 fitting masks and 49 separate audit masks. Linear-only and pairwise quadratic surrogates share fitting labels and cross-validation splits. Neither sees the audit labels during fitting. Masks at both pruning budgets are fixed before final test evaluation. All subnetworks rewind to surviving original weights and use fresh Adam optimizers with identical epoch shuffles within a seed.

`results/` contains aggregate comparisons, learning curves, surrogate audits and QAOA diagnostics. Each `results/seed_*/` subfolder preserves original initialization, selected subnetworks, fitting/audit labels, per-epoch label curves, fitted coefficients, selection records and QAOA samples/optimizer traces.

Error bars show standard deviation over five initializations. Paired 95% t intervals use five seed-level differences; the random baseline averages ten masks within each seed, rather than treating those masks as independent initialization trials. These are descriptive results on one fixed split. They do not establish statistical superiority across tasks, the Lottery Ticket Hypothesis, or quantum advantage. QAOA runs as a noiseless classical statevector simulation with exact optimization expectations and 32 final measurement shots.

The exact QUBO optimum minimizes the approximation, not actual retraining loss. Surrogate ranking, audit-only selection regret, QAOA optimality gaps, and conditional low-cost sampling probabilities must be considered alongside final test accuracy.

Sources: [official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist), [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635), [IBM QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm).
