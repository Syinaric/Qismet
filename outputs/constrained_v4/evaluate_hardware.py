"""Analyze all predeclared hardware depths, freeze selection, then measure test accuracy."""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from confirm_fresh_seeds import best_distribution
sys.path.insert(0,str(HERE.parent/'fashion_v2'))
import fashion_experiment as f

def run():
    ledger=json.loads((HERE/'hardware'/'ledger.json').read_text())
    if len(ledger['jobs'])!=4 or any(j['state']!='complete' for j in ledger['jobs']): return False
    assets=np.load(HERE/'pilot_p3_seed_11.npz')
    ids=assets['ids']; predictions=assets['predictions']; order=np.argsort(predictions); top=order[:7]
    truth=assets['validation_truth']; uniform=np.ones(70)/70
    rows=[]; distributions={}; raw_counts={}
    for layers in [0,1,2,3]:
        path=HERE/'hardware'/f'constrained_p{layers}'/'result.json'
        result=json.loads(path.read_text()); counts=result['counts']; raw_counts[layers]=counts
        full=np.zeros(256)
        for bits,n in counts.items(): full[int(bits,2)]=n/sum(counts.values())
        raw=full[ids]; acceptance=float(raw.sum()); pr=raw/acceptance; distributions[layers]=pr
        best32=best_distribution(raw,predictions,32)
        rows.append({'layers':layers,'shots':sum(counts.values()),'feasible_fraction':acceptance,
                     'top10_probability_raw':float(raw[top].sum()),'top10_probability_conditional':float(pr[top].sum()),
                     'predicted_loss_conditional':float(pr@predictions),'validation_loss_conditional':float(pr@truth),
                     'best32_validation_loss':float(best32@truth),'no_feasible_in32_probability':float((1-acceptance)**32),
                     'qpu_seconds':result['reported_usage_seconds'],'job_id':result['job_id']})
    chosen=min(rows[1:],key=lambda r:r['predicted_loss_conditional'])['layers']
    rng=np.random.default_rng(20261106)
    bag=[int(b,2) for b,n in raw_counts[chosen].items() for _ in range(n)]
    draws=rng.choice(bag,32,replace=False); candidates=np.array([idx for idx in draws if idx in ids])
    lookup={int(idx):i for i,idx in enumerate(ids)}
    selected=int(min(candidates,key=lambda idx:predictions[lookup[int(idx)]]))
    random_draws=rng.choice(ids,32,replace=True)
    random_selected=int(min(random_draws,key=lambda idx:predictions[lookup[int(idx)]]))
    frozen={'chosen_layers':chosen,'depth_selection_criterion':'lowest conditional mean QUBO score among all three hardware depths',
            'selected_state':selected,'random_best32_state':random_selected,'exact_qubo_state':int(ids[predictions.argmin()]),
            'selection_frozen_before_test':True,'raw_candidate_budget':32,'candidate_seed':20261106,
            'note':'Exploratory hardware depth selection by QUBO only. All depths and control reported.'}
    (HERE/'hardware'/'selection_frozen.json').write_text(json.dumps(frozen,indent=2))
    arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
    data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),
             torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val','test']}
    initial=torch.load(HERE.parent/'fashion_v2'/'results'/'seed_11'/'original_initialization.pt',weights_only=True)
    metrics=[]
    for index,state in enumerate(ids):
        model=f.rewind(initial,int(state)); f.train(model,data,11)
        val=f.evaluate(model,data['val']); test=f.evaluate(model,data['test'])
        np.testing.assert_allclose(val['loss'],truth[index],atol=1e-6)
        metrics.append({'state':int(state),'val_loss':val['loss'],'test_loss':test['loss'],
                        'val_accuracy':val['accuracy'],'test_accuracy':test['accuracy']})
        if (index+1)%20==0: print('hardware evaluation:',index+1,'/70 retrained',flush=True)
    table=pd.DataFrame(metrics); table.to_csv(HERE/'hardware'/'all_mask_test_metrics.csv',index=False)
    acc=table.test_accuracy.to_numpy()
    for row in rows:
        p=distributions[row['layers']]; raw=p*row['feasible_fraction']
        row['expected_test_accuracy']=float(p@acc)
        row['best32_expected_test_accuracy']=float(best_distribution(raw,predictions,32)@acc)
        row['expected_validation_loss_se']=float(np.sqrt((p@(truth-p@truth)**2)/(row['shots']*row['feasible_fraction'])))
    pd.DataFrame(rows).to_csv(HERE/'hardware'/'depth_comparison.csv',index=False)
    protocol=json.loads((HERE/'hardware'/'protocol.json').read_text())
    summary={**frozen,'uniform_expected_test_accuracy':float(uniform@acc),
             'uniform_best32_expected_test_accuracy':float(best_distribution(uniform,predictions,32)@acc),
             'uniform_expected_validation_loss':float(uniform@truth),
             'uniform_best32_expected_validation_loss':float(best_distribution(uniform,predictions,32)@truth),
             'selected_test_accuracy':float(table.set_index('state').loc[selected,'test_accuracy']),
             'random_selected_test_accuracy':float(table.set_index('state').loc[random_selected,'test_accuracy']),
             'additional_qpu_seconds':sum(j['reported_usage_seconds'] for j in ledger['jobs']),
             'total_project_qpu_seconds':protocol['prior_qpu_seconds']+sum(j['reported_usage_seconds'] for j in ledger['jobs']),
             'earlier_cancelled_job_reservation_seconds':protocol['prior_budget_accounted_seconds']-protocol['prior_qpu_seconds'],
             'backend':ledger['backend'],'quantum_speedup_demonstrated':False}
    (HERE/'hardware'/'summary.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True); print(json.dumps(summary),flush=True)
    return True

if __name__=='__main__':
    while True:
        path=HERE/'hardware'/'ledger.json'
        if path.exists() and run(): break
        time.sleep(20)
