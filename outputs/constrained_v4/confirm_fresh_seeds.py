"""Fixed follow-up on five fresh initializations. Never tune using test accuracy.

This is an exploratory follow-up to the earlier experiment, not a speedup claim.
The frozen model, splits, fit rule and optimizer match the documented local pilot.
"""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fashion_v2'))
import fashion_experiment as f

SEEDS=[101,202,303,404,505]; P=3
IDS=np.flatnonzero(f.MASKS.sum(1)==4); M=f.MASKS[IDS]
EDGES=[(i,i+1) for i in range(0,8,2)]+[(i,(i+1)%8) for i in range(1,8,2)]
SWAPS=[]; lookup={int(idx):i for i,idx in enumerate(IDS)}
for a,b in EDGES:
    left=np.flatnonzero((M[:,a]==0)&(M[:,b]==1))
    right=np.array([lookup[int(IDS[i])^(1<<a)^(1<<b)] for i in left])
    SWAPS.append((left,right))
START=np.all(M[:,::2]+M[:,1::2]==1,axis=1).astype(complex)/4

def probabilities(theta,values):
    state=START.copy()
    for l in range(P):
        state*=np.exp(-1j*theta[l]*values)
        c,s=np.cos(theta[P+l]),-1j*np.sin(theta[P+l])
        for left,right in SWAPS:
            a,b=state[left].copy(),state[right].copy()
            state[left]=c*a+s*b; state[right]=s*a+c*b
    return np.abs(state)**2

def best_distribution(pr,values,k):
    order=np.argsort(values)
    remaining=np.maximum(0,1-np.r_[0,np.cumsum(pr[order])])
    win=remaining[:-1]**k-remaining[1:]**k
    result=np.zeros(70); result[order]=win
    return result/result.sum()

def run_seed(seed,data,arrays):
    out=HERE/'fresh_seeds'/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
    complete=out/'summary.json'
    if complete.exists(): return json.loads(complete.read_text())
    torch.manual_seed(seed); dense=f.FashionMLP(); initial=f.clone_state(dense)
    torch.save(initial,out/'original_initialization.pt')
    t=time.perf_counter(); f.train(dense,data,seed); dense_train_seconds=time.perf_counter()-t
    fit_ids=np.random.default_rng(seed+4000).permutation(IDS)[:35]
    fit=np.isin(IDS,fit_ids); audit=~fit
    models=[]; labels=[]; t=time.perf_counter()
    for index,idx in enumerate(IDS):
        model=f.rewind(initial,int(idx)); f.train(model,data,seed)
        labels.append(f.evaluate(model,data['val'])); models.append(model)
        if (index+1)%20==0: print(seed,index+1,'/70 trained',flush=True)
    labeling_seconds=time.perf_counter()-t
    truth=np.array([v['loss'] for v in labels])
    t=time.perf_counter(); ridge=RidgeCV(alphas=np.logspace(-5,2,15),cv=KFold(5,shuffle=True,random_state=seed),scoring='neg_mean_squared_error')
    ridge.fit(f.FEATURES[IDS[fit]],truth[fit]); predicted=ridge.predict(f.FEATURES[IDS])
    regression_seconds=time.perf_counter()-t
    normalized=(predicted-predicted.mean())/predicted.std()
    def objective(theta): return probabilities(theta,normalized)@normalized
    t=time.perf_counter(); rng=np.random.default_rng(seed+9000); results=[]
    for _ in range(3):
        results.append(minimize(objective,rng.uniform(-1,1,6),method='BFGS',options={'maxiter':150,'gtol':1e-6}))
    optimum=min(results,key=lambda r:r.fun); pr=probabilities(optimum.x,normalized)
    optimization_seconds=time.perf_counter()-t
    uniform=np.ones(70)/70
    q32=best_distribution(pr,predicted,32); r32=best_distribution(uniform,predicted,32)
    chooser=np.random.default_rng(seed+7000)
    qdraw=chooser.choice(70,32,p=pr); rdraw=chooser.choice(70,32,p=uniform)
    choices={'QAOA best32':int(qdraw[np.argmin(predicted[qdraw])]),
             'Random best32':int(rdraw[np.argmin(predicted[rdraw])]),'Exact QUBO':int(predicted.argmin())}
    frozen={'seed':seed,'theta':optimum.x.tolist(),'alpha':float(ridge.alpha_),
            'selected_states':{k:int(IDS[v]) for k,v in choices.items()},
            'fit_states':fit_ids.tolist(),'audit_states':IDS[audit].tolist(),
            'selection_frozen_before_test':True}
    (out/'selection_frozen.json').write_text(json.dumps(frozen,indent=2))
    # Test data first becomes available after every selection is frozen.
    test=(torch.tensor(arrays['test_images'].reshape(-1,784)/255.,dtype=torch.float32),
          torch.tensor(arrays['test_labels'],dtype=torch.long))
    test_metrics=[f.evaluate(model,test) for model in models]
    accuracies=np.array([v['accuracy'] for v in test_metrics])
    losses=np.array([v['loss'] for v in test_metrics]); dense_metrics=f.evaluate(dense,test)
    dense_val=f.evaluate(dense,data['val'])
    table=pd.DataFrame({'state':IDS,'fit':fit,'val_loss':truth,'predicted_loss':predicted,
                        'val_accuracy':[v['accuracy'] for v in labels],
                        'test_accuracy':accuracies,'test_loss':losses,'ideal_qaoa_probability':pr})
    table.to_csv(out/'all_mask_metrics.csv',index=False)
    summary={'seed':seed,'execution':'ideal simulation, not hardware','alpha':float(ridge.alpha_),
             'audit_r2':r2_score(truth[audit],predicted[audit]),
             'audit_mae':mean_absolute_error(truth[audit],predicted[audit]),
             'audit_spearman':float(spearmanr(truth[audit],predicted[audit]).statistic),
             'top10_probability':float(pr[np.argsort(predicted)[:7]].sum()),
             'dense_test_accuracy':dense_metrics['accuracy'],'dense_val_accuracy':dense_val['accuracy'],
             'dense_parameters':12730,'pruned_parameters':6370,
             'qaoa_expected_test_accuracy':float(pr@accuracies),'random_expected_test_accuracy':float(uniform@accuracies),
             'qaoa_best32_expected_test_accuracy':float(q32@accuracies),'random_best32_expected_test_accuracy':float(r32@accuracies),
             'qaoa_expected_validation_loss':float(pr@truth),'random_expected_validation_loss':float(uniform@truth),
             'qaoa_best32_expected_validation_loss':float(q32@truth),'random_best32_expected_validation_loss':float(r32@truth),
             'dense_training_seconds':dense_train_seconds,'all70_labeling_seconds':labeling_seconds,
             'regression_seconds':regression_seconds,'qaoa_simulator_optimization_seconds':optimization_seconds}
    for name,idx in choices.items():
        summary[name+'_state']=int(IDS[idx]); summary[name+'_test_accuracy']=float(accuracies[idx])
    np.savez(out/'frozen_model.npz',linear=ridge.coef_[:8],pairs=ridge.coef_[8:],constant=ridge.intercept_,
             theta=optimum.x,probabilities=pr,ids=IDS,predictions=predicted,validation_truth=truth,scale=predicted.std())
    complete.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    protocol={'seeds':SEEDS,'hidden':16,'keep_groups':4,'fit_masks':35,'audit_masks':35,
              'layers':P,'restarts':3,'bfgs_maxiter':150,'baseline':'uniform over the same 70 masks',
              'endpoints':['expected single-proposal test accuracy','expected best-of-32 test accuracy'],
              'note':'Exploratory follow-up. No quantum speed advantage claim. Final test metrics do not tune any choice.'}
    path=HERE/'fresh_seed_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2))
    arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
    data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val']}
    summaries=[run_seed(seed,data,arrays) for seed in SEEDS]
    pd.DataFrame(summaries).to_csv(HERE/'fresh_seed_summary.csv',index=False)
    print('All five fresh seeds complete. No hardware jobs in this script.',flush=True)
