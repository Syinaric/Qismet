"""Exploratory validation-only pilot. No test images, credentials or QPU calls."""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'constrained_v4'; OUT.mkdir(exist_ok=True)
N=8; KEEP=4; P=int(sys.argv[1]) if len(sys.argv)>1 else 3
MASKS=((np.arange(256)[:,None]>>np.arange(8))&1)
IDS=np.flatnonzero(MASKS.sum(1)==KEEP); M=MASKS[IDS]
PAIRS=[(i,j) for i in range(8) for j in range(i+1,8)]
X=np.column_stack([MASKS]+[MASKS[:,i]*MASKS[:,j] for i,j in PAIRS])
EDGES=[(i,i+1) for i in range(0,8,2)]+[(i,(i+1)%8) for i in range(1,8,2)]
SWAPS=[]
for a,b in EDGES:
    left=np.flatnonzero((M[:,a]==0)&(M[:,b]==1))
    lookup={int(idx):i for i,idx in enumerate(IDS)}
    right=np.array([lookup[int(IDS[i])^(1<<a)^(1<<b)] for i in left])
    SWAPS.append((left,right))
START=np.all(M[:,::2]+M[:,1::2]==1,axis=1).astype(complex)/4

def probabilities(theta,costs):
    state=START.copy()
    for l in range(P):
        state*=np.exp(-1j*theta[l]*costs)
        c,s=np.cos(theta[P+l]),-1j*np.sin(theta[P+l])
        for left,right in SWAPS:
            a,b=state[left].copy(),state[right].copy()
            state[left]=c*a+s*b; state[right]=s*a+c*b
    return np.abs(state)**2

rows=[]
for seed in [11,22,33,44,55]:
    start=time.perf_counter()
    labels=pd.read_csv(ROOT/'outputs'/'fashion_v2'/'results'/f'seed_{seed}'/'labels.csv')
    fit=labels.query('keep_groups==4 and split=="fit"')
    audit=labels.query('keep_groups==4 and split=="audit"')
    model=RidgeCV(alphas=np.logspace(-5,2,15),cv=KFold(5,shuffle=True,random_state=seed),scoring='neg_mean_squared_error')
    model.fit(X[fit.state],fit.val_loss)
    predictions=model.predict(X)
    truth=labels.set_index('state').loc[IDS].val_loss.to_numpy()
    costs=predictions[IDS]; normalized=(costs-costs.mean())/costs.std()
    def objective(t): return probabilities(t,normalized)@normalized
    rng=np.random.default_rng(seed+9000); results=[]
    for _ in range(3):
        theta=np.r_[rng.uniform(-1,1,P),rng.uniform(-1,1,P)]
        results.append(minimize(objective,theta,method='BFGS',options={'maxiter':150,'gtol':1e-6}))
    best=min(results,key=lambda r:r.fun); pr=probabilities(best.x,normalized)
    assert abs(pr.sum()-1)<1e-10
    top=np.argsort(costs)[:7]
    # Expected best-of-k, computed from the distributions, not lucky selected shots.
    rank=np.argsort(costs)
    def best_k_values(distribution,k):
        remaining=1-np.r_[0,np.cumsum(distribution[rank])]
        win=np.maximum(0,remaining[:-1])**k-np.maximum(0,remaining[1:])**k
        return win@costs[rank],win@truth[rank]
    row=dict(seed=seed,layers=P,alpha=float(model.alpha_),audit_r2=r2_score(audit.val_loss,predictions[audit.state]),
             audit_mae=mean_absolute_error(audit.val_loss,predictions[audit.state]),
             audit_spearman=spearmanr(audit.val_loss,predictions[audit.state]).statistic,
             top10_probability=float(pr[top].sum()),uniform_top10_probability=.1,
             uniform_expected_val_loss=float(truth.mean()),initial_expected_val_loss=float(np.abs(START)**2@truth),qaoa_expected_val_loss=float(pr@truth),
             initial_top10_probability=float((np.abs(START)**2)[top].sum()),
             mode_mask=int(IDS[pr.argmax()]),mode_val_loss=float(truth[pr.argmax()]),seconds=time.perf_counter()-start)
    for k in [1,4,8,32]:
        for name,distribution in [('qaoa',pr),('random',np.ones(70)/70)]:
            predicted,actual=best_k_values(distribution,k)
            row[f'{name}_best{k}_predicted_loss']=predicted
            row[f'{name}_best{k}_validation_loss']=actual
    rows.append(row)
    np.savez(OUT/f'pilot_p{P}_seed_{seed}.npz',linear=model.coef_[:8],pairs=model.coef_[8:],constant=model.intercept_,
             theta=best.x,probabilities=pr,ids=IDS,predictions=costs,validation_truth=truth,scale=costs.std())
    print(json.dumps(row),flush=True)
pd.DataFrame(rows).to_csv(OUT/f'pilot_p{P}_validation.csv',index=False)
print('VALIDATION EXPLORATION ONLY. No hardware or test evaluation.',flush=True)
