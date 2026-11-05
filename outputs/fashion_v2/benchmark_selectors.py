"""Post-hoc CPU timing of classical selection; QAOA timings are from the original run.

No retraining or surrogate fitting included. QAOA is a statevector simulation.
Classical timings include candidate construction, objective evaluation and selection.
Median of five batches of 1000 solves; tiny-instance microbenchmarks, not quantum hardware predictions.
"""
from pathlib import Path
import json,timeit
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
rows=[]
for seed in [11,22,33,44,55]:
    folder=HERE/'results'/f'seed_{seed}'
    coef=np.load(folder/'surrogates.npz')
    a=coef['quadratic_linear']; b=coef['quadratic_pairs']; c=float(coef['quadratic_constant'])
    for keep in [4,2]:
        q=json.loads((folder/f'qaoa_keep_{keep}.json').read_text())
        rng=np.random.default_rng(seed+keep)
        def exact():
            masks=((np.arange(256)[:,None]>>np.arange(8))&1).astype(int)
            feasible=masks[masks.sum(1)==keep]
            costs=c+feasible@a+np.einsum('bi,ij,bj->b',feasible,b,feasible)
            return feasible[np.argmin(costs)]
        def best32():
            # Independent uniform fixed-cardinality masks, with replacement.
            masks=np.zeros((32,8),dtype=int)
            selected=np.argsort(rng.random((32,8)),axis=1)[:,:keep]
            np.put_along_axis(masks,selected,1,axis=1)
            costs=c+masks@a+np.einsum('bi,ij,bj->b',masks,b,masks)
            return masks[np.argmin(costs)]
        for name,fn in [('Exact QUBO enumeration',exact),('Best of 32 random by QUBO',best32)]:
            fn()
            seconds=float(np.median(timeit.repeat(fn,repeat=5,number=1000))/1000)
            rows.append(dict(seed=seed,pruning=100*(1-keep/8),method=name,seconds=seconds,
                             timing_source='new CPU microbenchmark, median 5x1000 solves'))
        rows.append(dict(seed=seed,pruning=100*(1-keep/8),method='QAOA statevector simulation',
                         seconds=q['elapsed_seconds'],timing_source='recorded original solver wall time'))
table=pd.DataFrame(rows)
table.to_csv(HERE/'runtime_benchmark.csv',index=False)
summary=table.groupby(['pruning','method']).seconds.agg(['mean','min','max'])
print(summary.to_string())
training=[json.loads((HERE/'results'/f'seed_{s}'/'complete.json').read_text())['seconds'] for s in [11,22,33,44,55]]
print('Total recorded per-seed experiment durations:',sum(training),'seconds (sum, not parallel wall time).')
report='''# Post-hoc selector timing

These are CPU timings on the saved Fashion-MNIST objectives, excluding retraining and surrogate fitting.
QAOA uses recorded original statevector-simulator wall time (three starts, up to 360 exact expectation
evaluations, 32 final shots). Classical selectors are measured now: median of five batches of 1000
solves per objective, including candidate construction, scoring and selection. Imports/data loading
are excluded. Runs were not simultaneous or load-controlled; this is an indicative toy-instance
comparison, not a quantum-hardware or asymptotic speedup benchmark.

| Pruning | Selector | Mean milliseconds across five initializations |
|---|---|---:|
'''
for (pruning,method),values in summary.iterrows():
    report+=f'| {pruning:.0f}% | {method} | {values["mean"]*1000:.4f} |\n'
report+='''
There are only 70 feasible masks at 50% pruning and 28 at 75%. Exhaustive classical selection
is cheap at this scale. QAOA was not faster in these runs. Its simulator runtime cannot establish
how actual quantum hardware would compare. Full experiment cost also includes 885 mask retrainings;
all surrogate-based methods share this expensive label-generation stage. No end-to-end speedup has
been demonstrated. QAOA timings include the solver's encoding/diagnostic checks as well as optimization.
'''
(HERE/'RUNTIME.md').write_text(report,encoding='utf-8')
