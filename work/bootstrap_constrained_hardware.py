"""Finite-shot uncertainty, including QUBO-based selection among three depths.

This does not capture hardware drift, dataset uncertainty or all experiment choices.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

out=Path(__file__).resolve().parents[1]/'outputs'/'constrained_v4'
a=np.load(out/'pilot_p3_seed_11.npz'); ids=a['ids']; predicted=a['predictions']; order=np.argsort(predicted)
acc=pd.read_csv(out/'hardware'/'all_mask_test_metrics.csv').test_accuracy.to_numpy()
s=json.loads((out/'hardware'/'summary.json').read_text())
rng=np.random.default_rng(20261107); B=2000
means=[]; single=[]; best=[]; raw_top=[]; per_depth={}
for layer in [1,2,3]:
    counts=json.loads((out/'hardware'/f'constrained_p{layer}'/'result.json').read_text())['counts']
    p=np.array([counts.get(format(int(i),'08b'),0)/4096 for i in ids])
    probabilities=np.r_[p,max(0,1-p.sum())]
    samples=rng.multinomial(4096,probabilities,size=B)[:,:70]/4096
    feasible=samples.sum(1); conditional=samples/feasible[:,None]
    means.append(conditional@predicted); single.append(conditional@acc)
    remaining=np.maximum(0,np.c_[np.ones(B),1-np.cumsum(samples[:,order],axis=1)])
    win=remaining[:,:-1]**32-remaining[:,1:]**32
    values=(win@acc[order])/win.sum(1)
    best.append(values); raw_top.append(samples[:,order[:7]].sum(1))
    per_depth[str(layer)]={'best32_accuracy_gain_pp_interval_95':np.quantile(100*(values-s['uniform_best32_expected_test_accuracy']),[.025,.975]).tolist()}
selection=np.argmin(np.array(means),axis=0); index=np.arange(B)
selected_single=np.array(single)[selection,index]; selected_best=np.array(best)[selection,index]
summary={'bootstrap_samples':B,'depth_selection_frequencies':{str(i+1):float(np.mean(selection==i)) for i in range(3)},
         'selected_single_accuracy_gain_pp_interval_95':np.quantile(100*(selected_single-s['uniform_expected_test_accuracy']),[.025,.975]).tolist(),
         'selected_best32_accuracy_gain_pp_interval_95':np.quantile(100*(selected_best-s['uniform_best32_expected_test_accuracy']),[.025,.975]).tolist(),
         'per_depth':per_depth,'scope':'Multinomial resampling of this one batch per circuit, including QUBO-based depth reselection. No hardware drift or dataset uncertainty.'}
(out/'hardware'/'finite_shot_uncertainty.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
