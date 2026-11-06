"""Fixed 16-group scaling experiment. CPU training, explicit ideal QAOA simulation.

No IBM job is submitted by this script. Dataset and training protocol are retained.
All selectors use a frozen learned QUBO; test metrics never choose masks or depth.
"""
from pathlib import Path
from itertools import combinations
import json,sys,time,math
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fashion_v2'))
import fashion_experiment as f
threadpool_limits(limits=1)
N=16; WIDTH=32; PAIRS=list(combinations(range(N),2))
EDGES=[(i,i+1) for i in range(0,N,2)]+[(i,(i+1)%N) for i in range(1,N,2)]
PROTOCOL={'dataset':'existing Fashion-MNIST split, 5000 train/1000 validation/10000 test',
          'hidden':WIDTH,'groups':N,'group_size':2,'keep_groups':[8,4],
          'seeds':[1001,2002,3003],'epochs':20,'batch_size':256,'learning_rate':.003,
          'fit_masks_per_budget':192,'audit_masks_per_budget':64,
          'layers':[1,2],'optimizer':'BFGS','restarts':3,'max_iterations':100,
          'final_candidate_budget':32,'greedy_restarts':8,'annealing_chains':32,'annealing_steps':300,
          'max_validation_drop_pp':4,'selection':'QUBO score only, masks and depth frozen before test evaluation',
          'hardware_gate':'positive audit R2, lower predicted best32 cost than uniform; inspect transpilation before submitting',
          'scope':'Exploratory scaling study. No guaranteed quantum advantage.'}

def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(obj,indent=2),encoding='utf-8'); temp.replace(path)

def design(keep):
    masks=np.zeros((math.comb(N,keep),N),dtype=np.int8)
    for r,indices in enumerate(combinations(range(N),keep)): masks[r,list(indices)]=1
    ids=masks.astype(np.int64)@(2**np.arange(N))
    x=np.column_stack([masks]+[masks[:,i]*masks[:,j] for i,j in PAIRS]).astype(float)
    return masks,ids,x

def rewind(initial,mask):
    ix=torch.tensor(np.flatnonzero(np.repeat(mask,2)))
    net=f.FashionMLP(len(ix)); net.load_state_dict({'fc1.weight':initial['fc1.weight'][ix],
       'fc1.bias':initial['fc1.bias'][ix],'fc2.weight':initial['fc2.weight'][:,ix],'fc2.bias':initial['fc2.bias']})
    return net

class Subspace:
    def __init__(self,masks,ids,keep):
        self.masks=masks; self.ids=ids
        lookup={int(idx):i for i,idx in enumerate(ids)}
        self.swaps=[]
        for a,b in EDGES:
            left=np.flatnonzero((masks[:,a]==0)&(masks[:,b]==1))
            right=np.array([lookup[int(ids[i])^(1<<a)^(1<<b)] for i in left])
            self.swaps.append((left,right))
        block=N//keep
        valid=np.all(masks.reshape(-1,keep,block).sum(2)==1,axis=1)
        self.start=valid.astype(complex)/np.sqrt(valid.sum())
    def probabilities(self,theta,values,layers):
        state=self.start.copy()
        for l in range(layers):
            state*=np.exp(-1j*theta[l]*values)
            c,s=np.cos(theta[layers+l]),-1j*np.sin(theta[layers+l])
            for left,right in self.swaps:
                a,b=state[left].copy(),state[right].copy()
                state[left]=c*a+s*b; state[right]=s*a+c*b
        return np.abs(state)**2

def best_distribution(pr,values,k=32):
    order=np.argsort(values); remaining=np.maximum(0,1-np.r_[0,np.cumsum(pr[order])])
    weights=remaining[:-1]**k-remaining[1:]**k
    result=np.zeros(len(pr)); result[order]=weights
    return result/result.sum()

def greedy(masks,pred,starts,rng):
    ids=masks.astype(np.int64)@(2**np.arange(N)); lookup={int(s):i for i,s in enumerate(ids)}
    best=None; evaluations=0
    for current in rng.integers(0,len(masks),starts):
        while True:
            on=np.flatnonzero(masks[current]); off=np.flatnonzero(1-masks[current])
            neighbors=np.array([lookup[int(ids[current])^(1<<i)^(1<<j)] for i in on for j in off])
            evaluations+=len(neighbors)
            candidate=int(neighbors[np.argmin(pred[neighbors])])
            if pred[candidate]>=pred[current]-1e-12: break
            current=candidate
        if best is None or pred[current]<pred[best]: best=int(current)
    return best,evaluations

def anneal(masks,pred,rng,chains=32,steps=300):
    ids=masks.astype(np.int64)@(2**np.arange(N)); lookup=np.full(2**N,-1,dtype=np.int32); lookup[ids]=np.arange(len(ids))
    current=rng.integers(0,len(masks),chains); best=int(current[pred[current].argmin()])
    temperatures=np.geomspace(max(float(pred.std()),1e-5),max(float(pred.std())*.005,1e-7),steps)
    for temperature in temperatures:
        # Random retained/removed positions give valid cardinality-preserving moves.
        on=np.array([rng.choice(np.flatnonzero(masks[i])) for i in current])
        off=np.array([rng.choice(np.flatnonzero(1-masks[i])) for i in current])
        proposal=lookup[ids[current]^(1<<on)^(1<<off)]
        delta=pred[proposal]-pred[current]
        accept=(delta<0)|(rng.random(chains)<np.exp(-np.maximum(delta,0)/temperature))
        current=np.where(accept,proposal,current)
        candidate=int(current[pred[current].argmin()])
        if pred[candidate]<pred[best]: best=candidate
    return best,chains*steps

def run_budget(seed,keep,data,initial,dense_metrics):
    out=HERE/f'seed_{seed}'/f'keep_{keep}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): return json.loads((out/'summary.json').read_text())
    masks,ids,x=design(keep); space=Subspace(masks,ids,keep)
    rng=np.random.default_rng(seed+keep*10000)
    permutation=rng.permutation(len(ids)); fit=permutation[:192]; audit=permutation[192:256]
    assert not set(fit)&set(audit)
    label_path=out/'labels.csv'
    records=pd.read_csv(label_path).to_dict('records') if label_path.exists() else []
    seen={int(r['row']) for r in records}
    for j,row in enumerate(np.r_[fit,audit]):
        if int(row) in seen: continue
        net=rewind(initial,masks[row]); start=time.perf_counter(); f.train(net,data,seed)
        metrics=f.evaluate(net,data['val'])
        records.append({'row':int(row),'state':int(ids[row]),'split':'fit' if row in fit else 'audit',
                        'val_loss':metrics['loss'],'val_accuracy':metrics['accuracy'],
                        'training_seconds':time.perf_counter()-start})
        if len(records)%16==0 or j==255:
            pd.DataFrame(records).to_csv(label_path,index=False)
            print(f'seed {seed}, keep {keep}: {len(records)}/256 labels',flush=True)
    labels=pd.DataFrame(records).set_index('row'); y=labels.loc[fit].val_loss.to_numpy()
    start=time.perf_counter(); estimators={}; predictions={}; audits=[]
    ring_indices=list(range(N))+[N+PAIRS.index(tuple(sorted(edge))) for edge in EDGES]
    for name,features in [('Linear',list(range(N))),('Quadratic',list(range(x.shape[1]))),('Ring quadratic',ring_indices)]:
        ridge=RidgeCV(alphas=np.logspace(-4,3,15),cv=KFold(5,shuffle=True,random_state=seed),scoring='neg_mean_squared_error')
        ridge.fit(x[fit][:,features],y); predicted=ridge.predict(x[:,features])
        estimators[name]=ridge; predictions[name]=predicted
        truth=labels.loc[audit].val_loss.to_numpy()
        audits.append({'model':name,'alpha':float(ridge.alpha_),'r2':r2_score(truth,predicted[audit]),
                       'mae':mean_absolute_error(truth,predicted[audit]),
                       'spearman':float(spearmanr(truth,predicted[audit]).statistic)})
    regression_seconds=time.perf_counter()-start
    pd.DataFrame(audits).to_csv(out/'surrogate_audit.csv',index=False)
    # The full quadratic is the predeclared primary objective. Alternatives are ablations.
    model=estimators['Quadratic']; pred=predictions['Quadratic']; scale=float(pred.std()); normalized=(pred-pred.mean())/scale
    distributions={}; traces=[]
    for layers in [1,2]:
        start=time.perf_counter(); runs=[]
        def objective(theta): return space.probabilities(theta,normalized,layers)@normalized
        for restart in range(3):
            start_angles=np.random.default_rng(seed+keep*100+layers*10+restart).uniform(-1,1,2*layers)
            result=minimize(objective,start_angles,method='BFGS',options={'maxiter':100,'gtol':1e-5})
            runs.append(result)
        optimum=min(runs,key=lambda r:r.fun); pr=space.probabilities(optimum.x,normalized,layers)
        assert abs(pr.sum()-1)<1e-9
        distributions[layers]=pr
        np.savez(out/f'qaoa_p{layers}.npz',linear=model.coef_[:N],pairs=model.coef_[N:],constant=model.intercept_,
                 theta=optimum.x,probabilities=pr,ids=ids,predictions=pred,scale=scale,keep=keep)
        traces.append({'layers':layers,'seconds':time.perf_counter()-start,'evaluations':sum(r.nfev for r in runs),
                       'mean_qubo':float(pr@pred),'top10_probability':float(pr[np.argsort(pred)[:math.ceil(len(ids)/10)]].sum())})
        print(f'seed {seed}, keep {keep}, QAOA p={layers}: {traces[-1]}',flush=True)
    pd.DataFrame(traces).to_csv(out/'qaoa_optimization.csv',index=False)
    chosen=min(traces,key=lambda r:r['mean_qubo'])['layers']; pr=distributions[chosen]
    uniform=np.ones(len(ids))/len(ids)
    selection_rng=np.random.default_rng(seed+keep*1000+777)
    qdraw=selection_rng.choice(len(ids),32,p=pr); rdraw=selection_rng.choice(len(ids),32)
    choices={'QAOA best32':int(qdraw[np.argmin(pred[qdraw])]),'Random best32':int(rdraw[np.argmin(pred[rdraw])]),
             'Exact linear':int(predictions['Linear'].argmin())}
    timings={}
    t=time.perf_counter(); choices['Exact QUBO']=int(pred.argmin()); timings['exact_lookup_seconds']=time.perf_counter()-t
    t=time.perf_counter(); choices['Greedy swaps'],ge=greedy(masks,pred,8,selection_rng); timings['greedy_seconds']=time.perf_counter()-t
    t=time.perf_counter(); choices['Simulated annealing'],se=anneal(masks,pred,selection_rng); timings['annealing_seconds']=time.perf_counter()-t
    # Search timings above use a cached table, including its construction below separately.
    times=[]
    for _ in range(30):
        t=time.perf_counter(); scores=model.intercept_+x@model.coef_; int(scores.argmin()); times.append(time.perf_counter()-t)
    timings['exact_score_all_median_seconds']=float(np.median(times))
    timings['cost_table_scope']='Enumeration and feature construction were earlier; greedy/annealing use cached QUBO scores. Classical exact score-all time includes matrix scoring and argmin.'
    # Distribution summaries use independent retraining labels, but never test data.
    audit_truth=labels.loc[audit].val_loss.to_numpy()
    summaries={}
    for name,p in [('QAOA',pr),('Random',uniform)]:
        conditional=p[audit]/p[audit].sum()
        summaries[name+'_audit_expected_validation_loss']=float(conditional@audit_truth)
        summaries[name+'_audit_best32_validation_loss']=float(best_distribution(conditional,pred[audit])@audit_truth)
    frozen={'seed':seed,'keep':keep,'chosen_layers':chosen,'selected_states':{name:int(ids[row]) for name,row in choices.items()},
            'fit_states':ids[fit].tolist(),'audit_states':ids[audit].tolist(),'frozen_before_test':True}
    save(out/'selection_frozen.json',frozen)
    rows=[]
    for method,row in choices.items():
        net=rewind(initial,masks[row]); t=time.perf_counter(); f.train(net,data,seed); seconds=time.perf_counter()-t
        val=f.evaluate(net,data['val']); test=f.evaluate(net,data['test'])
        torch.save({'state_dict':f.clone_state(net),'mask':masks[row].tolist()},out/(method.replace(' ','_')+'.pt'))
        rows.append({'method':method,'state':int(ids[row]),'parameters':sum(p.numel() for p in net.parameters()),
                     'val_accuracy':val['accuracy'],'val_loss':val['loss'],'test_accuracy':test['accuracy'],'test_loss':test['loss'],
                     'training_seconds':seconds,'validation_drop_pp':100*(dense_metrics['val_accuracy']-val['accuracy']),
                     'qubo_gap':float(pred[row]-pred.min())})
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    summary={'seed':seed,'keep':keep,'candidate_masks':len(ids),'chosen_layers':chosen,
             'surrogate_audit':audits,'qaoa_traces':traces,**summaries,'classical_timings':timings,
             'label_training_seconds':float(labels.training_seconds.sum()),'regression_seconds':regression_seconds,
             'dense':dense_metrics,'greedy_evaluations':ge,'annealing_evaluations':se}
    save(out/'summary.json',summary); print('COMPLETE',seed,keep,flush=True)
    return summary

def run():
    HERE.mkdir(exist_ok=True); protocol_path=HERE/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=PROTOCOL: raise ValueError('Protocol changed; preserve the original run')
    save(protocol_path,PROTOCOL)
    arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
    data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),
             torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val','test']}
    for seed in PROTOCOL['seeds']:
        folder=HERE/f'seed_{seed}'; folder.mkdir(exist_ok=True)
        dense_path=folder/'dense.json'
        if dense_path.exists():
            initial=torch.load(folder/'original_initialization.pt',weights_only=True); dense_metrics=json.loads(dense_path.read_text())
        else:
            torch.manual_seed(seed); dense=f.FashionMLP(WIDTH); initial=f.clone_state(dense)
            torch.save(initial,folder/'original_initialization.pt'); t=time.perf_counter(); f.train(dense,data,seed); seconds=time.perf_counter()-t
            # Dense test is reported but never passed to any selection objective.
            val=f.evaluate(dense,data['val']); test=f.evaluate(dense,data['test'])
            dense_metrics={'parameters':sum(p.numel() for p in dense.parameters()),'training_seconds':seconds,
                           'val_accuracy':val['accuracy'],'val_loss':val['loss'],'test_accuracy':test['accuracy']}
            save(dense_path,dense_metrics)
        for keep in PROTOCOL['keep_groups']: run_budget(seed,keep,data,initial,dense_metrics)
    print('ALL SIX LOCAL EXPERIMENTS COMPLETE. No hardware submitted.',flush=True)

if __name__=='__main__': run()
