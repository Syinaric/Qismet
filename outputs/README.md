# Qismet

A reproducible experiment that trains a small PyTorch classifier, prunes hidden neurons, restores their original initialization, fits a quadratic approximation to retraining loss, and uses Qiskit QAOA to propose pruning masks.

**Start with `Qismet.html` to read the executed experiment in a browser, or `Qismet.ipynb` to inspect and rerun it.** The notebook contains its own implementation; the Python helpers are convenience copies.

## Files

| File | Purpose |
|---|---|
| `Qismet.ipynb` | Self-contained notebook covering all five milestones, with saved outputs |
| `Qismet.html` | Executed notebook as a browser-readable report |
| `requirements.txt` | Tested Python package versions |
| `experiment_utils.py`, `qaoa_utils.py` | Importable copies of the experiment and solver |
| `results/RESULTS.md` | Summary generated from the actual run |
| `results/` | Comparison tables, mask labels, diagnostics, QAOA traces and samples, QUBO coefficients, plots, configuration, and model checkpoints |

## Run locally

Python **3.13.7** was used for the supplied execution. The experiment runs on a CPU and needs no quantum account, API key, GPU, or quantum hardware. Install the requirements into a new environment; the development environment under `work/venv` is not part of this portable deliverable.

Run the following from the directory containing this README.

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name python3
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Qismet.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to html Qismet.ipynb
```

**macOS/Linux, with Python 3.13 available:**

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name python3
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Qismet.ipynb
.venv/bin/python -m jupyter nbconvert --to html Qismet.ipynb
```

Alternatively, open the notebook in a notebook editor such as VS Code, select the new environment as its Python kernel, and run all cells in order. Rerunning replaces the notebook outputs and matching files in `results/`. Retraining hundreds of masks is the main computational cost.

## Supported model and controls

The supported architecture is exactly **2 inputs → 8 tanh hidden neurons → 2 class logits**, on the included two-moons classification dataset. Arbitrary neural networks are unsupported. The first configuration cell controls pruning percentage, training steps, initialization seeds, and final QAOA shots. Defaults are 50% hidden-neuron pruning, 200 full-batch Adam steps, seeds `[11, 22, 33]`, and 32 shots.

The helper API accepts an **untrained** instance. Its current weights become the saved initialization. Seed PyTorch before constructing it; the function's `seed` argument controls the experiment's sampling and fitting choices.

```python
import torch
from experiment_utils import TinyMLP, prepare_data, run_experiment

torch.manual_seed(11)
data = prepare_data()
result = run_experiment(
    TinyMLP(8), data, pruning_percentage=50, seed=11
)
print(result["table"])
```

The helper returns results in memory; the notebook also saves them. Retained neuron counts are rounded to the nearest integer, with halves rounded up, and must be between 1 and 7. At the default 50% neuron pruning, parameter count falls from 42 to 22. Original weights are saved as `results/original_initialization_seed_*.pt`; trained QAOA selections and their masks are saved as `results/qaoa_subnetwork_seed_*.pt`.

## Interpret the evidence

Validation loss after retraining supplies the surrogate labels. At the default pruning budget, each initialization uses 128 fitting masks and 48 separate audit masks; audit labels do not guide fitting or selection. Test data is used only for final comparisons. Surrogate diagnostics measure prediction quality on held-out masks, including those within the requested pruning budget.

QAOA uses a noiseless **classical statevector simulation** with exact expectations during optimization and finite shots for final mask selection. Comparisons include random pruning, best-of-random selection, greedy swap search, an exact surrogate optimum, magnitude pruning, and fresh initialization. The exact QUBO optimum need not minimize actual retraining loss.

Read `results/RESULTS.md` and the notebook's diagnostics together. Three initializations on one fixed dataset split demonstrate the pipeline; they do not prove the lottery-ticket hypothesis, quantum advantage, or superiority over classical search. The expensive surrogate-label collection is included explicitly, so this is not a speedup claim.
