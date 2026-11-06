from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root=Path(__file__).resolve().parents[1]
out=root/'outputs/diagnostics_v2'
md=nbformat.v4.new_markdown_cell
code=nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# Qismet: audit the QUBO before scaling

This executed notebook reads the diagnostic artifacts produced by `run_audit.py`.
It asks whether QAOA finds the surrogate optimum, whether pairwise interactions improve
prediction over a linear-only model, and whether either model selects useful subnetworks.

**Scope:** original three Two Moons initializations; 8 hidden neurons; retain 4;
128 fit masks and 35 held-out feasible masks per seed. All 70 feasible masks per seed
were retrained for 200 full-batch Adam steps, with validation metrics at every epoch.
Test data was not evaluated. This is exploratory reuse of the original audit split,
not a new confirmatory experiment.

To regenerate underlying measurements, run `python run_audit.py` in this folder using
the original Qismet environment. The script needs the sibling `results/` checkpoints
and `experiment_utils.py`; it does not modify them. Then rerun this notebook.
'''),
code('''from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown, Image
BASE=Path('.')
metrics=pd.read_csv(BASE/'surrogate_metrics.csv')
solver=pd.read_csv(BASE/'solver_comparison.csv',dtype={'qaoa_mask':str,'random_mask':str,'exact_qubo_mask':str})
selection=pd.read_csv(BASE/'selection_comparison.csv',dtype={'mask':str})
all_masks=pd.read_csv(BASE/'all_feasible_masks.csv',dtype={'mask':str})
curves=pd.read_csv(BASE/'validation_curves.csv')
assert len(all_masks)==210 and len(curves)==42000
assert all_masks.groupby('seed').size().eq(70).all()
assert metrics.n_audit.eq(35).all()
print('Verified: 210 masks, 42,000 validation checkpoints, 35 audit masks per initialization.')
'''),
md('''## QAOA versus exact search and random masks

Mask strings are in neuron-index order: a 1 retains that neuron. Hamming distance counts
positions that differ. The surrogate gap is measured against the exact minimum over all
70 four-neuron masks. A gap above zero means the finite-shot QAOA selection missed the optimum.
'''),
code('display(solver.round(6))'),
md('''## Linear-only versus pairwise surrogate

The linear model uses the same original fitting labels, ridge-alpha candidates, and CV splits
as the original quadratic model. The quadratic coefficients are reused unchanged. Higher
Spearman cost-versus-loss correlation is better; cost-versus-accuracy should be negative.
R² below zero means predictions are worse in squared error than the held-out mean.
'''),
code("display(metrics.round(4))\ndisplay(metrics.groupby('model')[['MAE','R2','Spearman_cost_vs_loss','Spearman_cost_vs_accuracy']].mean().round(4))"),
md('''## Selecting masks and retraining

`all70` selection searches the entire feasible set. `audit35` compares surrogate selection
on masks whose retraining labels were excluded from fitting. Validation-loss regret is the
excess over the best actual validation loss in that same candidate set. The oracle is a
retrospective reference, not an independently validated winning network.
'''),
code("display(selection.round(5))\ndisplay(selection.groupby(['candidate_pool','method'])[['val_loss','val_accuracy','val_loss_regret']].mean().round(5))"),
code("display(Image(filename=str(BASE/'qubo_audit.png')))"),
code("display(Markdown((BASE/'RESULTS.md').read_text(encoding='utf-8')))"),
])
nb.metadata.kernelspec={'name':'python3','display_name':'Python 3 (Qismet)','language':'python'}
NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
nbformat.validate(nb)
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
nbformat.write(nb,out/'QUBO_Diagnostic.ipynb')
html,_=HTMLExporter().from_notebook_node(nb)
(out/'QUBO_Diagnostic.html').write_text(html,encoding='utf-8')
print('Executed diagnostic notebook and HTML generated.')
