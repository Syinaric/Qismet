from pathlib import Path
import json
import nbformat
import numpy as np
import pandas as pd
import torch

root = Path(__file__).resolve().parents[1] / 'outputs'
nb = nbformat.read(root / 'Qismet.ipynb', as_version=4)
nbformat.validate(nb)
codes = [c for c in nb.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in codes)
assert not [o for c in codes for o in c.outputs if o.output_type == 'error']
table = pd.read_csv(root / 'results/comparison_all_seeds.csv')
assert len(table) == 27
assert table.test_accuracy.between(0, 1).all()
assert (table.loc[table.method != 'Dense', 'kept'] == 4).all()
assert (table.loc[table.method != 'Dense', 'parameters'] == 22).all()
for seed in [11, 22, 33]:
    q = json.loads((root / f'results/qaoa_seed_{seed}.json').read_text())
    selected = q['selected_index']
    assert selected in q['sampled_indices']
    assert sum(q['selected_mask']) == 4
    assert q['checks']['maximum_basis_encoding_error'] < 1e-9
    masks = np.array(q['basis_masks'])
    observed = np.array(q['sampled_indices'])
    feasible_observed = observed[masks[observed].sum(1) == 4]
    assert q['selected_surrogate'] == min(q['surrogate_energies'][i] for i in feasible_observed)
    labels = pd.read_csv(root / f'results/surrogate_labels_seed_{seed}.csv')
    assert len(labels.query("split == 'fit'")) == 128
    assert len(labels.query("split == 'audit'")) == 48
    assert not labels.state.duplicated().any()
    saved = torch.load(root / f'results/qaoa_subnetwork_seed_{seed}.pt', weights_only=True)
    assert saved['mask'] == q['selected_mask']
    assert saved['state_dict']['fc1.weight'].shape == (4, 2)
assert (root / 'Qismet.html').stat().st_size > 10000
print(f'PASS: {len(codes)} executed code cells, 27 benchmark rows, 528 retraining labels, three sampled masks/checkpoints and Ising energy checks.')
print(table.groupby('method').test_accuracy.mean().to_string())
