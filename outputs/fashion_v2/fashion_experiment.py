"""Fixed-protocol Fashion-MNIST grouped pruning experiment. CPU, no quantum hardware."""
from pathlib import Path
import sys, json, time, gzip, hashlib, urllib.request, copy
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from qaoa_utils import solve_qaoa

CONFIG = dict(dataset_seed=2026, train_size=5000, validation_size=1000, test_size=10000,
              seeds=[11,22,33,44,55], hidden=16, groups=8, group_size=2,
              keep_groups=[4,2], epochs=20, batch_size=256, lr=0.003,
              fit_masks=128, audit_masks=49, shots=32, qaoa_reps=2,
              qaoa_restarts=3, qaoa_maxiter=120, random_masks=10,
              ridge_alphas=[0.0001,0.01,0.1,1.0,10.0])
CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
HASHES = {'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
          'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
          't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
          't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def prepare_data(cache=None):
    """Verified official IDX files; 5000 train/1000 val, official10000 test; scale /255."""
    cache = Path(cache) if cache else HERE / 'data_cache'
    cache.mkdir(parents=True, exist_ok=True)
    raw = {}
    for name, expected in HASHES.items():
        path = cache / name
        if not path.exists():
            print('Downloading', name, flush=True)
            urllib.request.urlretrieve('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+name, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Dataset checksum failed: '+name)
        unpacked = gzip.decompress(path.read_bytes())
        is_image = 'images' in name
        magic = int.from_bytes(unpacked[:4], 'big')
        assert magic == (2051 if is_image else 2049)
        count = int.from_bytes(unpacked[4:8], 'big')
        array = np.frombuffer(unpacked, np.uint8, offset=16 if is_image else 8).copy()
        raw[name] = array.reshape(count,28,28) if is_image else array
    images, labels = raw['train-images-idx3-ubyte.gz'], raw['train-labels-idx1-ubyte.gz']
    ids = np.arange(len(labels))
    train_ids, rest = train_test_split(ids, train_size=CONFIG['train_size'], stratify=labels, random_state=CONFIG['dataset_seed'])
    val_ids, _ = train_test_split(rest, train_size=CONFIG['validation_size'], stratify=labels[rest], random_state=CONFIG['dataset_seed']+1)
    assert not set(train_ids)&set(val_ids)
    arrays = {'train_images': images[train_ids], 'train_labels': labels[train_ids], 'train_ids':train_ids,
              'val_images': images[val_ids], 'val_labels':labels[val_ids], 'val_ids':val_ids,
              'test_images':raw['t10k-images-idx3-ubyte.gz'], 'test_labels':raw['t10k-labels-idx1-ubyte.gz']}
    np.savez_compressed(HERE / 'dataset_subset.npz', **arrays)
    data = {s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),
               torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val','test']}
    return data, arrays


class FashionMLP(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
    def forward(self,x):
        return self.fc2(torch.relu(self.fc1(x)))


def clone_state(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()}


MASKS = ((np.arange(256)[:,None] >> np.arange(8))&1).astype(int)
PAIRS = list(combinations(range(8),2))
FEATURES = np.column_stack([MASKS]+[MASKS[:,i]*MASKS[:,j] for i,j in PAIRS])


def rewind(initial, state):
    groups = MASKS[int(state)]
    neurons = np.flatnonzero(np.repeat(groups,2))
    if not len(neurons): raise ValueError('Cannot retain zero neurons')
    ix = torch.tensor(neurons)
    net = FashionMLP(len(ix))
    expected = {'fc1.weight':initial['fc1.weight'][ix], 'fc1.bias':initial['fc1.bias'][ix],
                'fc2.weight':initial['fc2.weight'][:,ix], 'fc2.bias':initial['fc2.bias']}
    net.load_state_dict(expected)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in expected.items())
    return net


def evaluate(net, split):
    net.eval()
    with torch.no_grad():
        logits = net(split[0])
        return dict(loss=float(nn.functional.cross_entropy(logits,split[1])),
                    accuracy=float((logits.argmax(1)==split[1]).float().mean()))


def train(net, data, seed):
    """Fresh Adam and same seeded epoch permutations for every mask at a seed."""
    optimizer=torch.optim.Adam(net.parameters(),lr=CONFIG['lr'])
    generator=torch.Generator().manual_seed(seed+100)
    x,y=data['train']; history=[]
    for epoch in range(CONFIG['epochs']):
        net.train()
        order=torch.randperm(len(y), generator=generator)
        for idx in order.split(CONFIG['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(net(x[idx]), y[idx])
            loss.backward(); optimizer.step()
        val=evaluate(net,data['val'])
        history.append(dict(epoch=epoch+1,val_loss=val['loss'],val_accuracy=val['accuracy']))
    return history


def mask_design(seed):
    rng=np.random.default_rng(seed+4000)
    fit=[]; audit=[]
    for keep in [4,2]:
        ids=rng.permutation(np.flatnonzero(MASKS.sum(1)==keep))
        half=len(ids)//2; fit.extend(ids[:half]); audit.extend(ids[half:])
    others=np.flatnonzero((MASKS.sum(1)>0)&~np.isin(MASKS.sum(1),[2,4]))
    fit.extend(rng.choice(others,128-len(fit),replace=False))
    assert len(fit)==128 and len(audit)==49 and not set(fit)&set(audit)
    return np.array(fit),np.array(audit)


def fit_models(labels,fit,seed):
    models={}; predictions={}
    for name,features in [('Linear',MASKS),('Quadratic',FEATURES)]:
        grid=GridSearchCV(Ridge(),{'alpha':CONFIG['ridge_alphas']},
                          cv=KFold(5,shuffle=True,random_state=seed),scoring='neg_mean_squared_error')
        grid.fit(features[fit],[labels[int(i)]['val_loss'] for i in fit])
        models[name]=grid.best_estimator_; predictions[name]=grid.predict(features)
    return models,predictions


def run_seed(seed,data):
    folder=HERE/'results'/f'seed_{seed}'; folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); torch.manual_seed(seed)
    dense=FashionMLP(); initial=clone_state(dense)
    torch.save(initial,folder/'original_initialization.pt')
    dense_history=train(dense,data,seed)
    dense_state=clone_state(dense)
    # Verify slicing equals activation masking before learning.
    original=FashionMLP(); original.load_state_dict(initial)
    check=rewind(initial,15)
    with torch.no_grad():
        x=data['train'][0][:20]
        expected=original.fc2(torch.relu(original.fc1(x))*torch.tensor(np.repeat(MASKS[15],2)))
        torch.testing.assert_close(check(x),expected)
    fit,audit=mask_design(seed); labels={}; histories={}; trained={}
    for index,state in enumerate(np.r_[fit,audit]):
        state=int(state); net=rewind(initial,state)
        histories[state]=train(net,data,seed)
        labels[state]=histories[state][-1].copy()
        trained[state]=clone_state(net)
        if (index+1)%20==0:
            print(f'seed {seed}: {index+1}/177 retraining labels ({time.perf_counter()-start:.1f}s)',flush=True)
    label_table=pd.DataFrame([dict(seed=seed,state=i,mask=''.join(map(str,MASKS[i])),keep_groups=int(MASKS[i].sum()),
                                   split='fit' if i in fit else 'audit',**v) for i,v in labels.items()])
    label_table.to_csv(folder/'labels.csv',index=False)
    pd.DataFrame([dict(state=i,**row) for i,h in histories.items() for row in h]).to_csv(folder/'label_learning_curves.csv',index=False)
    models,predictions=fit_models(labels,fit,seed)
    quadratic=models['Quadratic']; linear=quadratic.coef_[:8]; pair=np.zeros((8,8))
    for coeff,(i,j) in zip(quadratic.coef_[8:],PAIRS): pair[i,j]=coeff
    np.testing.assert_allclose(predictions['Quadratic'],quadratic.intercept_+MASKS@linear+np.einsum('bi,ij,bj->b',MASKS,pair,MASKS))
    np.savez(folder/'surrogates.npz',quadratic_linear=linear,quadratic_pairs=pair,quadratic_constant=quadratic.intercept_,
             linear_coefficients=models['Linear'].coef_,linear_constant=models['Linear'].intercept_)
    rows=[]; curves=[]; audits=[]; diagnostics=[]; selections={}
    # Freeze every selection at both pruning budgets BEFORE any test evaluation.
    scores=dense.fc2.weight.detach().square().sum(0).reshape(8,2).sum(1).numpy()
    for keep in CONFIG['keep_groups']:
        feasible=np.flatnonzero(MASKS.sum(1)==keep)
        rng=np.random.default_rng(seed+5000+keep)
        random_states=rng.choice(feasible,CONFIG['random_masks'],replace=True).tolist()
        candidates=rng.choice(feasible,CONFIG['shots'],replace=True)
        mag=np.zeros(8,int); mag[np.argsort(scores,kind='stable')[-keep:]]=1
        q=solve_qaoa(linear,pair,float(quadratic.intercept_),keep,seed=seed+1000+keep,
                     shots=CONFIG['shots'],reps=CONFIG['qaoa_reps'],restarts=CONFIG['qaoa_restarts'],maxiter=CONFIG['qaoa_maxiter'])
        (folder/f'qaoa_keep_{keep}.json').write_text(json.dumps(q,indent=2))
        choices={'Dense':[255],'Random (10-mask mean)':random_states,
                 'Magnitude':[int(mag@(2**np.arange(8)))],
                 'Exact linear':[int(feasible[np.argmin(predictions['Linear'][feasible])])],
                 'Exact QUBO':[int(feasible[np.argmin(predictions['Quadratic'][feasible])])],
                 'QAOA':[q['selected_index']],
                 'Best of 32 random':[int(candidates[np.argmin(predictions['Quadratic'][candidates])])]}
        selections[keep]=choices
        for name in ['Linear','Quadratic']:
            ids=audit[MASKS[audit].sum(1)==keep]
            truth=np.array([labels[int(i)]['val_loss'] for i in ids]); pred=predictions[name][ids]
            # Audit-only choice evaluates selection on masks excluded from surrogate fitting.
            chosen=int(ids[np.argmin(pred)])
            audits.append(dict(seed=seed,keep_groups=keep,pruning=100*(1-keep/8),model=name,n=len(ids),
                               mae=mean_absolute_error(truth,pred),r2=r2_score(truth,pred),
                               spearman=float(spearmanr(truth,pred).statistic),alpha=float(models[name].alpha),
                               audit_selected_state=chosen,audit_selected_val_loss=labels[chosen]['val_loss'],
                               audit_selection_regret=labels[chosen]['val_loss']-truth.min()))
        exact=choices['Exact QUBO'][0]
        diagnostics.append(dict(seed=seed,keep_groups=keep,pruning=100*(1-keep/8),
            qaoa_mask=''.join(map(str,MASKS[q['selected_index']])), exact_mask=''.join(map(str,MASKS[exact])),
            hit_exact=q['selected_index']==exact, surrogate_gap=q['selected_surrogate']-q['exact_feasible_minimum'],
            qaoa_feasible_probability=q['feasible_probability'],feasible_shots=q['feasible_shots'],
            top_decile_conditional=q['conditional_top_decile_probability'],uniform_top_decile=q['uniform_conditional_top_decile_probability'],
            evaluations=q['total_evaluations'],ising_max_error=q['checks']['maximum_basis_encoding_error'],
            all_optimizer_starts_hit_budget=all(r['evaluations'] >= CONFIG['qaoa_maxiter'] for r in q['optimizer_records'])))
    (folder/'selections.json').write_text(json.dumps(selections,indent=2))
    # Final, untouched-test evaluation. No test metric is available to selectors above.
    metric_cache={}
    def get_metrics(state):
        if state in metric_cache: return metric_cache[state]
        if state==255:
            net=FashionMLP(); net.load_state_dict(dense_state); hist=dense_history
        else:
            net=rewind(initial,state)
            if state in trained: net.load_state_dict(trained[state]); hist=histories[state]
            else: hist=train(net,data,seed)
        metric={f'{s}_{k}':v for s in ['val','test'] for k,v in evaluate(net,data[s]).items()}
        metric['parameters']=sum(p.numel() for p in net.parameters())
        torch.save({'group_mask':MASKS[state].tolist(),'state_dict':clone_state(net)},folder/f'selected_state_{state}.pt')
        metric_cache[state]=(metric,hist)
        return metric,hist
    for keep,choices in selections.items():
        for method,states in choices.items():
            evaluated=[get_metrics(s) for s in states]
            metric={key:float(np.mean([e[0][key] for e in evaluated])) for key in evaluated[0][0]}
            rows.append(dict(seed=seed,pruning=100*(1-keep/8),keep_groups=keep,method=method,
                             states=json.dumps(states),neurons=16 if method=='Dense' else keep*2,**metric))
            for epoch in range(CONFIG['epochs']):
                curves.append(dict(seed=seed,pruning=100*(1-keep/8),method=method,epoch=epoch+1,
                    val_loss=float(np.mean([e[1][epoch]['val_loss'] for e in evaluated])),
                    val_accuracy=float(np.mean([e[1][epoch]['val_accuracy'] for e in evaluated]))))
    # Label prediction table facilitates reproducible scatter plots; audit status is explicit.
    for name in predictions: label_table[name+'_prediction']=[predictions[name][s] for s in label_table.state]
    label_table.to_csv(folder/'labels.csv',index=False)
    tables={'comparison':pd.DataFrame(rows),'learning_curves':pd.DataFrame(curves),
            'surrogate_audit':pd.DataFrame(audits),'qaoa_diagnostics':pd.DataFrame(diagnostics)}
    for name,table in tables.items(): table.to_csv(folder/(name+'.csv'),index=False)
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in initial.items())
    manifest={'seed':seed,'seconds':time.perf_counter()-start,'fit_retrains':len(fit),'audit_retrains':len(audit),
              'distinct_final_masks':len(metric_cache),'initialization_unchanged':True,'structural_check':True}
    (folder/'complete.json').write_text(json.dumps(manifest,indent=2))
    print('Finished',manifest,flush=True)
    return tables


def run_all():
    HERE.mkdir(exist_ok=True); (HERE/'results').mkdir(exist_ok=True)
    protocol=HERE/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=CONFIG:
        raise ValueError('Existing run uses a different protocol; use a new output folder')
    protocol.write_text(json.dumps(CONFIG,indent=2))
    data,_=prepare_data()
    tables=[]
    for seed in CONFIG['seeds']:
        folder=HERE/'results'/f'seed_{seed}'
        if (folder/'complete.json').exists():
            print('Reusing completed seed',seed,flush=True)
            tables.append({n:pd.read_csv(folder/(n+'.csv')) for n in ['comparison','learning_curves','surrogate_audit','qaoa_diagnostics']})
        else: tables.append(run_seed(seed,data))
    for name in tables[0]:
        pd.concat([t[name] for t in tables],ignore_index=True).to_csv(HERE/'results'/(name+'.csv'),index=False)
    return tables


if __name__=='__main__': run_all()
