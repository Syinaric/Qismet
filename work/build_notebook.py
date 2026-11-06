from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
cells = []
def md(text): cells.append(nbformat.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbformat.v4.new_code_cell(text.strip()))

md(r'''
# Qismet: can QAOA propose useful neural subnetworks?

**An executed, reproducible feasibility experiment — not a claim of quantum advantage.**

We train a small PyTorch classifier, structurally remove hidden neurons, rewind surviving
weights to their **original initialization**, and retrain. We then learn a quadratic
approximation to validation loss after retraining and use **Qiskit QAOA on a noiseless
CPU statevector simulator** to propose masks. No quantum account or hardware is required.

The supported input is one architecture: **2 → 8 tanh → 2 logits**, on a fixed two-moons
classification dataset. Change the pruning percentage below; the retained count is rounded
to the nearest integer (halves round up). Arbitrary neural networks are deliberately unsupported.

The default experiment has three fixed initializations, 50% hidden-neuron pruning, 200 full-batch
Adam steps per training run, and 32 final QAOA shots. The code is self-contained in this notebook.
Run all cells from top to bottom. It writes reproducibility artifacts to `results/` beside it.

**What counts as evidence?** A selected mask must retrain well on untouched test data.
We separately measure whether the surrogate predicts unseen-mask performance and whether
QAOA favors good masks *within the correct pruning budget*. None is assumed to succeed.
''')
md('''## Setup and fixed protocol

Install the accompanying requirements in a Python environment, then select that environment
as the notebook kernel. The saved notebook already includes outputs from an actual execution.
The configuration is declared before any test results are inspected. Three initialization seeds
share the same data split; they do not measure uncertainty across datasets or establish significance.
''')
code('''import sys, json, time, importlib.metadata
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

HIDDEN = 8
PRUNING_PERCENTAGE = 50
EPOCHS = 200
INITIALIZATION_SEEDS = [11, 22, 33]
SHOTS = 32
RESULT_DIR = Path('results')
RESULT_DIR.mkdir(exist_ok=True)
VERSIONS = {name: importlib.metadata.version(name) for name in
            ['torch', 'numpy', 'scipy', 'scikit-learn', 'qiskit', 'pandas', 'matplotlib']}
print('Python:', sys.version.split()[0])
print(json.dumps(VERSIONS, indent=2))
''')

source = (OUT / 'experiment_utils.py').read_text(encoding='utf-8')
chunks = source.split('# %% ')
code(chunks[0])
md(r'''## 1. Train, prune, rewind, retrain

There are 900 training, 300 validation, and 300 test examples. Standardization is fit on
training examples only. The immutable checkpoint is saved **before dense training**.
Pruning physically slices hidden-layer rows, hidden biases, and output-layer columns.
The output bias is also reset. Every retraining creates a fresh Adam optimizer.

Assertions check exact restoration and equality between the sliced network and the original
network with hidden activations masked. The first milestone reports validation accuracy;
test examples are reserved for the final comparison. Magnitude pruning here ranks the trained
outgoing-weight norms; it is a simple structured-pruning baseline, not iterative magnitude pruning.
''')
code(chunks[1].split('\n',1)[1])
code('''data = prepare_data()
keep = int(np.floor(HIDDEN*(1-PRUNING_PERCENTAGE/100)+0.5))
milestone_table, original_init, first_mask, dense_history, sparse_history = milestone_one(
    data, seed=INITIALIZATION_SEEDS[0], hidden=HIDDEN, keep=keep, epochs=EPOCHS)
torch.save(original_init, RESULT_DIR / 'milestone1_original_initialization.pt')
display(milestone_table)
print('PASS: exact weight rewind; structural equivalence; immutable initialization.')
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(dense_history, label='Dense')
ax.plot(sparse_history, label='Magnitude-pruned, rewound')
ax.set(xlabel='Training step', ylabel='Training cross-entropy', title='Classical foundation')
ax.legend(); fig.tight_layout(); plt.show()
''')
md(r'''## 2. Define a binary structured-pruning problem

Each of the eight hidden neurons has a binary variable $m_i$: 1 retains it, 0 removes it.
At 50% pruning, $\sum_i m_i=4$: there are 70 feasible masks out of 256 bitstrings.
The dense model has 42 parameters; a four-unit model has 22 (47.6% fewer parameters).

## 3. Fit a QUBO approximation to actual retraining loss

For each initialization we retrain **128 fitting masks** from the saved weights and label
them with validation cross-entropy. Another **48 masks** audit surrogate quality and are
excluded from fitting and hyperparameter selection. At the default budget each set includes
35 distinct four-unit masks. Audit labels are diagnostics; masks remain eligible for search,
but their observed losses never guide QAOA or the classical selectors.

$$\hat L(m)=c+\sum_i a_i m_i+\sum_{i<j} b_{ij}m_i m_j.$$

A ridge regression fits the individual and pairwise effects; 5-fold CV on fitting masks
selects the regularization strength. Terms are **empirical approximations**, not causal
neuron contributions or an exact representation of retraining. We report holdout MAE, $R^2$,
and Spearman rank correlation, especially within the requested pruning budget.
The coefficient convention stores each pair once, in the strict upper triangle.

Collecting retraining labels is the expensive part of this toy demonstration. It may cost
more than directly searching masks; we make no end-to-end acceleration claim.
''')
code(chunks[2].split('\n',1)[1])
md(r'''## 4. Convert to Ising and run QAOA in Qiskit

Add a cardinality penalty:

$$C(m)=\hat L(m)+\lambda(\sum_i m_i-k)^2,\quad
\lambda=1+\sum_i|a_i|+\sum_{i<j}|b_{ij}|.$$

This sufficient penalty exceeds the maximum possible range of the unpenalized surrogate.
It guarantees that all feasible masks have lower cost than all infeasible masks, but it can
dominate the useful retraining signal. The diagnostics expose this tradeoff.

Substitute $m_i=(I-Z_i)/2$. If $a'_i=a_i+\lambda(1-2k)$ and
$b'_{ij}=b_{ij}+2\lambda$, then
$J_{ij}=b'_{ij}/4$ and
$h_i=-a'_i/2-\sum_{j\ne i}b'_{\min(i,j),\max(i,j)}/4$.
The constant offset is retained for energy checks and omitted only as a circuit global phase.

The circuit starts in $|+\rangle^{\otimes8}$ and alternates two Ising cost layers with
X mixers. RZ/RZZ/RX angles implement the exponential with the correct factor of two.
Energy is rescaled by the RMS nonconstant Ising energy for angle optimization.
Three seeded COBYLA starts each receive up to 120 objective evaluations. Their exact
statevector expectations are used during optimization; **32 finite shots** are used only
for the final candidate selection. This is an ideal simulation, not a finite-shot hardware benchmark.

The returned mask is the lowest-surrogate-cost feasible **observed** sample. There is no
fallback to an enumerated optimum. All 256 basis energies are checked against the Qiskit
Hamiltonian. `mask[i]` corresponds to qubit i (least-significant bit); printed mask strings
use neuron order, the reverse of ordinary Qiskit bitstrings.
''')
qcode = (OUT / 'qaoa_utils.py').read_text(encoding='utf-8').replace('from __future__ import annotations\n', '')
code(qcode)
md(r'''## 5. Retrain selections and compare on the untouched test set

All selectors use the surrogate or trained dense weights, never test loss. Compare:

- Dense model and trained-magnitude pruning with rewind.
- One uniform feasible random mask.
- Best of 32 uniform feasible masks by surrogate cost (a stronger sparsity-aware baseline).
- Best feasible mask among 32 uniformly random raw bitstrings (same raw candidate budget as QAOA).
- Classical one-out/one-in greedy swap search and exhaustive feasible QUBO optimization.
- QAOA's best feasible measured mask and a fresh randomly initialized four-unit model.

The exact QUBO solution is an oracle for the **surrogate**, not actual retraining performance.
The 32-shot comparisons match final candidate counts, not total compute: QAOA additionally
uses up to 360 exact-statevector objectives; swap search and exhaustive search have their own
evaluation budgets. Repeated retraining is deterministic full-batch CPU training. Fresh weights
control whether retaining the original initialization matters for this architecture.
''')
code(chunks[3].split('\n',1)[1].replace('    from qaoa_utils import solve_qaoa\n', ''))
code('''runs = []
for seed in INITIALIZATION_SEEDS:
    print(f'Running initialization seed {seed}', flush=True)
    torch.manual_seed(seed)
    input_model = TinyMLP(HIDDEN)  # Pass your untrained instance of the supported architecture.
    run = run_experiment(input_model, data, pruning_percentage=PRUNING_PERCENTAGE,
                         seed=seed, epochs=EPOCHS, shots=SHOTS)
    runs.append(run)
    run['table'].to_csv(RESULT_DIR / f'comparison_seed_{seed}.csv', index=False)
    run['labels'].to_csv(RESULT_DIR / f'surrogate_labels_seed_{seed}.csv', index=False)
    (RESULT_DIR / f'qaoa_seed_{seed}.json').write_text(json.dumps(run['qaoa'], indent=2))
    (RESULT_DIR / f'quality_seed_{seed}.json').write_text(json.dumps(run['quality'], indent=2))
    np.savez(RESULT_DIR / f'qubo_seed_{seed}.npz', linear=run['linear'],
             pairwise=run['pairwise'], constant=run['constant'])
    torch.save(run['initial'], RESULT_DIR / f'original_initialization_seed_{seed}.pt')
    torch.save({'mask': run['qaoa']['selected_mask'], 'state_dict': run['qaoa_model'].state_dict()},
               RESULT_DIR / f'qaoa_subnetwork_seed_{seed}.pt')
    print(f"Finished seed {seed} in {run['quality']['total_seconds']:.1f}s", flush=True)

comparison = pd.concat([r['table'] for r in runs], ignore_index=True)
comparison.to_csv(RESULT_DIR / 'comparison_all_seeds.csv', index=False)
summary = comparison.groupby('method', sort=False).agg(
    test_accuracy_mean=('test_accuracy', 'mean'), test_accuracy_sd=('test_accuracy', 'std'),
    test_loss_mean=('test_loss', 'mean'), val_loss_mean=('val_loss', 'mean'),
    parameters=('parameters', 'first'))
summary.to_csv(RESULT_DIR / 'summary.csv')
display(summary.style.format({'test_accuracy_mean': '{:.2%}', 'test_accuracy_sd': '{:.2%}',
                              'test_loss_mean': '{:.4f}', 'val_loss_mean': '{:.4f}'}))
display(comparison[['seed','method','mask','test_accuracy','test_loss']])
''')
md('''### Surrogate validity and what QAOA actually learned

Read the within-budget holdout correlations before interpreting the selected masks.
Feasibility alone is not evidence of finding better subnetworks: compare QAOA's probability
of a top-decile surrogate mask *conditional on feasibility* with uniform feasible sampling.
Values near 10% indicate little ranking improvement even if nearly every shot is feasible.
''')
code('''quality_rows = []
for run in runs:
    q, v = run['qaoa'], run['quality']
    quality_rows.append({'seed': v['seed'], 'audit_R2': v['all_audit']['r2'],
        'feasible_audit_R2': v['feasible_audit']['r2'], 'feasible_audit_MAE': v['feasible_audit']['mae'],
        'feasible_audit_Spearman': v['feasible_audit']['spearman'],
        'QAOA_feasible_probability': q['feasible_probability'],
        'QAOA_top10_given_feasible': q['conditional_top_decile_probability'],
        'uniform_top10_given_feasible': q['uniform_conditional_top_decile_probability'],
        'feasible_shots': q['feasible_shots'], 'unique_feasible_samples': q['unique_feasible_samples'],
        'surrogate_gap_to_exact': q['selected_surrogate']-q['exact_feasible_minimum'],
        'objective_evaluations': q['total_evaluations'],
        'fit_retrains': v['fit_retrains'], 'audit_retrains': v['audit_retrains'],
        'QAOA_seconds': v['qaoa_seconds'], 'total_seconds': v['total_seconds']})
diagnostics = pd.DataFrame(quality_rows)
diagnostics.to_csv(RESULT_DIR / 'diagnostics.csv', index=False)
display(diagnostics.round(4))
print('Encoding/selection checks:', [r['qaoa']['checks'] for r in runs])
print('Optimizer statuses:', [[o['message'] for o in r['qaoa']['optimizer_records']] for r in runs])
''')
code('''plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
means = summary.test_accuracy_mean*100
axes[0,0].barh(means.index, means.values, xerr=summary.test_accuracy_sd.fillna(0)*100,
              color=['#2563eb' if x=='QAOA' else '#94a3b8' for x in means.index])
axes[0,0].set(xlabel='Test accuracy (%) — mean ± SD across initializations',
              title='Retrained models', xlim=(0,100))
axes[0,0].invert_yaxis()
for r in runs:
    audit = r['labels'].query("split == 'audit' and kept == @keep")
    axes[0,1].scatter(audit.val_loss, audit.predicted_val_loss, s=18, alpha=.65,
                      label=f"Seed {r['quality']['seed']}")
axes[0,1].set(xlabel='Measured validation CE after retraining', ylabel='QUBO prediction',
              title='Unseen masks at the requested pruning budget')
axes[0,1].legend()
xx = np.arange(len(runs)); width=.35
axes[1,0].bar(xx-width/2, diagnostics.QAOA_top10_given_feasible*100, width, label='QAOA')
axes[1,0].bar(xx+width/2, diagnostics.uniform_top10_given_feasible*100, width, label='Uniform feasible')
axes[1,0].set(xticks=xx, xticklabels=INITIALIZATION_SEEDS, xlabel='Initialization seed',
              ylabel='Conditional probability (%)', title='Top-decile surrogate masks, given feasibility')
axes[1,0].legend()
for r in runs:
    best = min(r['qaoa']['optimizer_records'], key=lambda o:o['best_scaled_expectation'])
    axes[1,1].plot(np.minimum.accumulate(best['scaled_trace']), label=f"Seed {r['quality']['seed']}")
axes[1,1].set(xlabel='Objective evaluation (best restart)', ylabel='Best scaled penalized energy',
              title='QAOA optimization traces')
axes[1,1].legend()
fig.suptitle(f'Qismet · {HIDDEN} hidden neurons → {keep} · ideal Qiskit simulation', fontsize=16)
fig.tight_layout()
fig.savefig(RESULT_DIR / 'experiment_summary.png', dpi=160, bbox_inches='tight')
plt.show()
''')
md('''### Interpretation and limits

The report below is generated from the executed results. Accuracy differences here are descriptive,
not significance tests. A successful pipeline does not establish quantum advantage, prove the
lottery-ticket hypothesis, or guarantee the same outcome for other networks or data.
If the quadratic surrogate ranks masks poorly, improving the optimizer cannot repair that mismatch.
If QAOA's conditional low-cost probability is near uniform, selecting a good mask from many shots
is largely best-of-random search. A cardinality-preserving mixer and a surrogate fitted specifically
within the pruning budget are reasonable future experiments, not changes made after seeing test scores.
''')
code('''qacc = summary.loc['QAOA','test_accuracy_mean']
dacc = summary.loc['Dense','test_accuracy_mean']
racc = summary.loc['Random mask','test_accuracy_mean']
bacc = summary.loc[f'Best of {SHOTS} random feasible','test_accuracy_mean']
eacc = summary.loc['Exact QUBO','test_accuracy_mean']
freshacc = summary.loc['Fresh initialization','test_accuracy_mean']
spearman = diagnostics.feasible_audit_Spearman.mean()
topmass = diagnostics.QAOA_top10_given_feasible.mean()
uniformmass = diagnostics.uniform_top10_given_feasible.mean()
report = f"""# Executed Qismet experiment

Completed all five milestones on a 2→8→2 PyTorch model using a local Qiskit simulator.
Across {len(runs)} fixed initializations on one fixed split, after {PRUNING_PERCENTAGE}% hidden-neuron pruning:

| Method | Mean test accuracy |
|---|---:|
| Dense | {dacc:.2%} |
| QAOA selected + rewind | {qacc:.2%} |
| Single random mask + rewind | {racc:.2%} |
| Best of {SHOTS} random feasible + rewind | {bacc:.2%} |
| Exact surrogate optimum + rewind | {eacc:.2%} |
| Fresh initialization, same width | {freshacc:.2%} |

The selected sparse model has {int(summary.loc['QAOA','parameters'])} parameters versus
{int(summary.loc['Dense','parameters'])} in the dense model.
Mean held-out within-budget Spearman correlation: **{spearman:.3f}**.
QAOA's mean probability of a top-decile surrogate mask, conditional on feasibility:
**{topmass:.2%}**, versus **{uniformmass:.2%}** for uniform feasible sampling.

This demonstrates an executed pruning/rewinding/QUBO/QAOA pipeline. It does not establish
quantum advantage or superiority over random/classical pruning. Interpret the accuracy table
together with the surrogate holdout metrics and sampling diagnostics. Test data did not guide
surrogate fitting, optimizer settings, or mask selection.

Defaults: {EPOCHS} full-batch Adam steps, learning rate 0.03; seeds {INITIALIZATION_SEEDS};
two QAOA layers; three COBYLA starts; at most 120 objectives/start; {SHOTS} final shots.
Each initialization uses 128 fit-mask retrainings plus 48 audit-mask retrainings at the default
50% pruning budget. This is expensive label collection, not a speedup demonstration.
"""
display(Markdown(report))
(RESULT_DIR / 'RESULTS.md').write_text(report, encoding='utf-8')
config = {'hidden': HIDDEN, 'pruning_percentage': PRUNING_PERCENTAGE, 'epochs': EPOCHS,
          'seeds': INITIALIZATION_SEEDS, 'shots': SHOTS, 'versions': VERSIONS,
          'python': sys.version, 'data_seed': 2026, 'data_samples': 1500}
(RESULT_DIR / 'run_config.json').write_text(json.dumps(config, indent=2))
''')
md('''## References and reproducibility

- [Frankle & Carbin: The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635).
- [IBM Qiskit: Quantum approximate optimization algorithm](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm).
- [IBM Qiskit: SparsePauliOp](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparsePauliOp).
- [PyTorch: Module and state_dict](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html).

See `results/` for every fitting/audit mask label, all QAOA sample counts and optimizer traces,
QUBO coefficients, original initialization checkpoints, selected subnetworks, metrics and configuration.
The `.py` helpers are convenience copies; the notebook itself includes the implementations.
''')

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {'display_name': 'Python 3 (Qismet)', 'language': 'python', 'name': 'python3'}
nb.metadata.language_info = {'name': 'python', 'version': '3.13.7'}
nbformat.write(nb, OUT / 'Qismet.ipynb')
print(f'Created {len(cells)} cells')
