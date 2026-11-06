from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
root=Path(__file__).resolve().parents[1]/'outputs'
sys.path.insert(0,str(root/'fashion_v2'))
import fashion_experiment as f
out=root/'fashion_v2'; results=out/'results'
comparison=pd.read_csv(results/'comparison.csv')
assert len(comparison)==70
assert comparison.test_accuracy.between(0,1).all()
assert comparison.groupby(['seed','pruning']).size().eq(7).all()
assert len(pd.read_csv(results/'learning_curves.csv'))==1400
arrays=np.load(out/'dataset_subset.npz')
assert not set(arrays['train_ids'])&set(arrays['val_ids'])
assert len(arrays['train_labels'])==5000 and len(arrays['val_labels'])==1000
test=(torch.tensor(arrays['test_images'].reshape(-1,784)/255,dtype=torch.float32),torch.tensor(arrays['test_labels'],dtype=torch.long))
for seed in f.CONFIG['seeds']:
    folder=results/f'seed_{seed}'
    labels=pd.read_csv(folder/'labels.csv',dtype={'mask':str})
    assert len(labels)==177 and not labels.state.duplicated().any()
    assert (labels.split=='fit').sum()==128 and (labels.split=='audit').sum()==49
    for keep in [4,2]:
        q=json.loads((folder/f'qaoa_keep_{keep}.json').read_text())
        assert sum(q['selected_mask'])==keep
        assert q['selected_index'] in q['sampled_indices']
        feasible=[i for i in q['sampled_indices'] if sum(q['basis_masks'][i])==keep]
        assert q['selected_surrogate']==min(q['surrogate_energies'][i] for i in feasible)
        assert q['checks']['maximum_basis_encoding_error']<1e-9
        checkpoint=torch.load(folder/f"selected_state_{q['selected_index']}.pt",weights_only=True)
        model=f.FashionMLP(keep*2); model.load_state_dict(checkpoint['state_dict'])
        metrics=f.evaluate(model,test)
        row=comparison.query("seed == @seed and keep_groups == @keep and method == 'QAOA'").iloc[0]
        assert abs(row.test_accuracy-metrics['accuracy'])<1e-7
        assert int(row.parameters)==(6370 if keep==4 else 3190)
        actual_budget=all(r['evaluations']>=f.CONFIG['qaoa_maxiter'] for r in q['optimizer_records'])
        diag=pd.read_csv(folder/'qaoa_diagnostics.csv')
        recorded=diag.loc[diag.keep_groups==keep,'all_optimizer_starts_hit_budget'].iloc[0]
        assert recorded==actual_budget
print('PASS: 5 initializations x 2 sparsities; 885 disjoint fit/audit labels; 70 comparison rows; 1400 curve rows; all10 sampled QAOA masks and saved test accuracies verified.')
