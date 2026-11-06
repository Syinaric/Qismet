"""Direct QUBO baselines: no exhaustive cost table is used by greedy or annealing."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import experiment as e

HERE=Path(__file__).resolve().parent

def scores(masks,linear,b,constant):
    return constant+masks@linear+np.einsum('bi,ij,bj->b',masks,b,masks)

def greedy(keep,linear,b,constant,rng,starts=8):
    best_mask=None; best_value=np.inf; evaluations=0
    for _ in range(starts):
        current=np.zeros(16); current[rng.choice(16,keep,replace=False)]=1
        value=float(scores(current[None],linear,b,constant)[0]); evaluations+=1
        while True:
            on,off=np.flatnonzero(current),np.flatnonzero(1-current)
            removed=np.repeat(on,len(off)); added=np.tile(off,len(on))
            candidates=np.tile(current,(len(removed),1)); rows=np.arange(len(removed))
            candidates[rows,removed]=0; candidates[rows,added]=1
            values=scores(candidates,linear,b,constant); evaluations+=len(values)
            idx=int(values.argmin())
            if values[idx]>=value-1e-12: break
            current=candidates[idx]; value=float(values[idx])
        if value<best_value: best_mask=current.copy(); best_value=value
    return best_mask,best_value,evaluations

def anneal(keep,linear,b,constant,rng,scale,chains=32,steps=300):
    current=np.zeros((chains,16)); choices=np.argsort(rng.random((chains,16)),axis=1)[:,:keep]
    current[np.arange(chains)[:,None],choices]=1
    values=scores(current,linear,b,constant); best=int(values.argmin())
    best_mask=current[best].copy(); best_value=float(values[best]); rows=np.arange(chains)
    for temperature in np.geomspace(max(scale,1e-5),max(scale*.005,1e-7),steps):
        on=np.argmax(np.where(current==1,rng.random((chains,16)),-1),axis=1)
        off=np.argmax(np.where(current==0,rng.random((chains,16)),-1),axis=1)
        proposals=current.copy(); proposals[rows,on]=0; proposals[rows,off]=1
        candidate_values=scores(proposals,linear,b,constant); delta=candidate_values-values
        accept=(delta<0)|(rng.random(chains)<np.exp(-np.maximum(delta,0)/temperature))
        current[accept]=proposals[accept]; values[accept]=candidate_values[accept]
        idx=int(values.argmin())
        if values[idx]<best_value: best_mask=current[idx].copy(); best_value=float(values[idx])
    return best_mask,best_value,chains*(steps+1)

def run(seed,keep):
    folder=HERE/f'seed_{seed}'/f'keep_{keep}'
    if (folder/'direct_classical.json').exists(): return
    a=np.load(folder/'qaoa_p1.npz'); linear=a['linear']; b=np.zeros((16,16)); b[np.triu_indices(16,1)]=a['pairs']; constant=float(a['constant'])
    minimum=float(a['predictions'].min()); rng=np.random.default_rng(seed+keep*999)
    output=[]
    for name,fn in [('Greedy swaps',lambda:greedy(keep,linear,b,constant,rng)),
                    ('Simulated annealing',lambda:anneal(keep,linear,b,constant,rng,float(a['scale'])))]:
        t=time.perf_counter(); mask,value,evaluations=fn(); elapsed=time.perf_counter()-t
        output.append({'method':name,'state':int(mask@(2**np.arange(16))),
                       'qubo':value,'qubo_gap':value-minimum,'evaluations':evaluations,'seconds':elapsed,
                       'cached_cost_table':False})
    timings=[]
    for _ in range(5):
        t=time.perf_counter(); masks,ids,_=e.design(keep); values=scores(masks,linear,b,constant); best=int(values.argmin()); timings.append(time.perf_counter()-t)
    exact={'method':'Exact QUBO, enumerate and score','state':int(ids[best]),'qubo':float(values[best]),
           'qubo_gap':float(values[best]-minimum),'evaluations':len(ids),'seconds':float(np.median(timings)),
           'cached_cost_table':False,'scope':'Includes mask enumeration and feature construction as in the main experiment, direct score evaluation and selecting the minimum.'}
    output.append(exact)
    # Save selection before looking up any neural-network performance.
    e.save(folder/'direct_classical_frozen.json',{'choices':output,'uses_test_for_selection':False})
    previous=pd.read_csv(folder/'comparison.csv').drop_duplicates('state').set_index('state')
    arrays=None; data=None; initial=None
    for result in output:
        state=result['state']
        if state in previous.index:
            result['test_accuracy']=float(previous.loc[state,'test_accuracy']); result['val_accuracy']=float(previous.loc[state,'val_accuracy'])
        else:
            if data is None:
                import torch
                arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
                data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val','test']}
                initial=torch.load(folder.parent/'original_initialization.pt',weights_only=True)
            mask=((state>>np.arange(16))&1); model=e.rewind(initial,mask); e.f.train(model,data,seed)
            result['test_accuracy']=e.f.evaluate(model,data['test'])['accuracy']; result['val_accuracy']=e.f.evaluate(model,data['val'])['accuracy']
    e.save(folder/'direct_classical.json',output)
    print('Direct classical baselines complete:',seed,keep,flush=True)

if __name__=='__main__':
    while True:
        complete=True
        for seed in e.PROTOCOL['seeds']:
            for keep in [8,4]:
                folder=HERE/f'seed_{seed}'/f'keep_{keep}'
                if (folder/'summary.json').exists(): run(seed,keep)
                else: complete=False
        if complete: break
        time.sleep(20)
