"""Small, resumable hardware QAOA experiment. Dry-run is explicitly simulation.

One replay + three SPSA updates (two costs each) + independent final sampling =8 jobs.
Saved simulation angles warm-start the hardware optimizer; prior classical work is disclosed.
"""
from pathlib import Path
import argparse, json, sys, time, hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from circuit_core import load_problem,build_qaoa,bind_qaoa,ideal_sample,counts_stats

PROTOCOL={'seed':11,'keep_groups':4,'reps':2,'optimizer':'SPSA','steps':3,
          'optimization_shots':1024,'final_shots':2048,'selection_shots':32,
          'max_accuracy_drop_pp':4.0,'preferred_accuracy_drop_pp':3.0,
          'qpu_allowance_seconds':600,'reserve_seconds':120,'max_job_seconds':60,
          'max_jobs':8,'spsa_a':0.12,'spsa_c':0.10,'random_seed':20261105,
          'description':'One preselected initialization and 50% pruning; not a significance or speedup demonstration.'}


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8'); temp.replace(path)


def problem_and_warmstart():
    base=HERE.parent/'fashion_v2'/'results'/'seed_11'
    problem=load_problem(base/'surrogates.npz',keep=4)
    previous=json.loads((base/'qaoa_keep_4.json').read_text())
    return problem,np.array(previous['parameters']),previous


def fresh_state(mode):
    _,theta,previous=problem_and_warmstart()
    return {'mode':mode,'protocol':PROTOCOL,'phase':'replay','iteration':0,
            'theta':theta.tolist(),'history':[],'complete':False,
            'prior_simulator_optimization_seconds':previous['elapsed_seconds'],
            'prior_simulator_evaluations':previous['total_evaluations']}


def pending_parameters(state):
    theta=np.array(state['theta'])
    phase=state['phase']; iteration=state['iteration']
    if phase=='final':
        best=min(state['history'],key=lambda x:x['scaled_mean_cost'])
        return np.array(best['parameters']),PROTOCOL['final_shots'],'final'
    if phase=='replay': return theta,PROTOCOL['optimization_shots'],'replay'
    rng=np.random.default_rng(PROTOCOL['random_seed']+iteration)
    direction=rng.choice([-1.,1.],len(theta)); c=PROTOCOL['spsa_c']/(iteration+1)**0.101
    sign=1 if phase=='plus' else -1
    return theta+sign*c*direction,PROTOCOL['optimization_shots'],f'step_{iteration+1}_{phase}'


def accept_result(state,problem,parameters,result):
    stats=counts_stats(problem,result['counts'])
    # Recompute cost expectation directly over observed strings, no exhaustive optimum.
    from circuit_core import cost,bitstring_to_mask
    total=sum(result['counts'].values())
    expectation=sum(int(v)*cost(problem,bitstring_to_mask(k,problem.n),penalized=True)
                    for k,v in result['counts'].items())/total
    record={'phase':state['phase'],'iteration':state['iteration'],'parameters':parameters.tolist(),
            'scaled_mean_cost':float((expectation-problem.offset)/problem.scale),
            'stats':stats,'job_id':result.get('job_id'),'shots':total}
    state['history'].append(record)
    if state['phase']=='replay': state['phase']='plus'
    elif state['phase']=='plus':
        state['plus_cost']=record['scaled_mean_cost']; state['phase']='minus'
    elif state['phase']=='minus':
        iteration=state['iteration']; rng=np.random.default_rng(PROTOCOL['random_seed']+iteration)
        direction=rng.choice([-1.,1.],len(parameters))
        c=PROTOCOL['spsa_c']/(iteration+1)**0.101
        a=PROTOCOL['spsa_a']/(iteration+1)**0.602
        gradient=(state['plus_cost']-record['scaled_mean_cost'])/(2*c)*direction
        state['theta']=(np.array(state['theta'])-a*gradient).tolist()
        state['iteration']+=1
        state['phase']='final' if state['iteration']==PROTOCOL['steps'] else 'plus'
    elif state['phase']=='final':
        # Random subset of actual observed shots; primary mask budget matches best-of32.
        bag=[bits for bits,count in result['counts'].items() for _ in range(int(count))]
        rng=np.random.default_rng(PROTOCOL['random_seed']+999)
        chosen=rng.choice(len(bag),size=PROTOCOL['selection_shots'],replace=False)
        selected_counts={}
        for index in chosen: selected_counts[bag[index]]=selected_counts.get(bag[index],0)+1
        from circuit_core import bitstring_to_mask,cost
        feasible=[bits for bits in selected_counts if sum(bitstring_to_mask(bits,problem.n))==problem.keep]
        state['selection_counts']=selected_counts
        state['selected_mask']=None if not feasible else bitstring_to_mask(
            min(feasible,key=lambda bits:cost(problem,bitstring_to_mask(bits,problem.n),penalized=False)),problem.n).tolist()
        state['final_counts']=result['counts']; state['complete']=True
    return record


def advance(mode='dry-run',account_name='qismet-open',backend_name=None):
    folder=HERE/'runs'/mode; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'state.json'
    state=json.loads(path.read_text()) if path.exists() else fresh_state(mode)
    if state['protocol']!=PROTOCOL: raise ValueError('Protocol changed; use a new experiment folder')
    if state['complete']:
        print('Already complete. Mode:',mode,'Selected mask:',state['selected_mask']); return state
    problem,_,_=problem_and_warmstart()
    params,shots,tag=pending_parameters(state)
    # The IBM adapter adds the explicit mask register before transpilation.
    circuit=bind_qaoa(build_qaoa(problem,reps=PROTOCOL['reps'],measure=False),params)
    save(path,state)
    if mode=='dry-run':
        raw=ideal_sample(problem,params,shots=shots,seed=PROTOCOL['random_seed']+len(state['history']),reps=2)
        counts=raw['counts'] if 'counts' in raw else raw
        result={'counts':counts,'job_id':None,'status':'completed','execution':'ideal simulator'}
        save(folder/(tag+'.json'),result)
    else:
        from ibm_backend import HardwareRunner,connect
        ledger_path=folder/'ledger.json'
        if ledger_path.exists():
            recorded_backend=json.loads(ledger_path.read_text())['backend']
            if backend_name is not None and backend_name!=recorded_backend:
                raise ValueError('Resume on the recorded backend to keep one hardware experiment consistent')
            backend_name=recorded_backend
        service,backend=connect(account_name=account_name,backend_name=backend_name)
        runner=HardwareRunner(service=service,backend=backend,ledger_path=folder/'ledger.json',
                              allowance_seconds=600,reserve_seconds=120,max_job_seconds=60,max_jobs=8)
        existing=runner.find_job(tag)
        if existing is not None:
            if not existing.get('job_id'):
                raise RuntimeError('An ambiguous submission exists. Reconcile it in IBM Workloads before continuing; do not resubmit.')
            result=runner.collect(existing['job_id'],wait=False)
        else:
            result=runner.sample(circuit,shots=shots,output_dir=folder,tag=tag,wait=False)
        if 'counts' not in result:
            if result.get('state')=='failed':
                raise RuntimeError(f"Hardware job {result.get('job_id')} failed; inspect its saved status before any further submission.")
            print('Hardware job:',result.get('job_id'),result.get('state'),result.get('remote_status'))
            print('Run the same hardware command again to collect it; it will not resubmit the same job.')
            return state
    record=accept_result(state,problem,params,result); save(path,state)
    print(mode,tag,'scaled mean cost:',round(record['scaled_mean_cost'],5))
    if state['complete']: print('Selection:',state['selected_mask'],'(32 observed-shot candidate budget)')
    return state


def evaluate_selected(mode):
    """Final retraining and inference latency; these do not demonstrate quantum search speed."""
    import torch,pandas as pd
    sys.path.insert(0,str(HERE.parent/'fashion_v2'))
    import fashion_experiment as f
    folder=HERE/'runs'/mode; state=json.loads((folder/'state.json').read_text())
    if not state['complete'] or state['selected_mask'] is None:
        raise ValueError('A completed run with a feasible observed mask is required')
    arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
    data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),
             torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val','test']}
    initial=torch.load(HERE.parent/'fashion_v2'/'results'/'seed_11'/'original_initialization.pt',weights_only=True)
    problem,_,_=problem_and_warmstart()
    feasible=np.flatnonzero(f.MASKS.sum(1)==4)
    rng=np.random.default_rng(PROTOCOL['random_seed']+888)
    from circuit_core import cost
    start=time.perf_counter(); exact=min(feasible,key=lambda i:cost(problem,f.MASKS[i],penalized=False)); exact_seconds=time.perf_counter()-start
    start=time.perf_counter(); candidates=rng.choice(feasible,32,replace=True)
    random_best=min(candidates,key=lambda i:cost(problem,f.MASKS[i],penalized=False)); random_seconds=time.perf_counter()-start
    chosen=int(np.array(state['selected_mask'])@(2**np.arange(8)))
    methods={'Dense':255,'QAOA selected':chosen,'Exact QUBO':int(exact),'Best of 32 random':int(random_best)}
    rows=[]
    for name,idx in methods.items():
        net=f.rewind(initial,idx); start=time.perf_counter(); history=f.train(net,data,11); train_seconds=time.perf_counter()-start
        metrics={f'{s}_{k}':v for s in ['val','test'] for k,v in f.evaluate(net,data[s]).items()}
        net.eval(); timing={}
        with torch.no_grad():
            for batch in [1,32]:
                x=data['test'][0][:batch]
                for _ in range(30): net(x)
                repeats=[]
                for _ in range(5):
                    t=time.perf_counter()
                    for _ in range(500): net(x)
                    repeats.append((time.perf_counter()-t)/500*1000)
                timing[f'inference_batch_{batch}_ms']=float(np.median(repeats))
        rows.append({'mode':mode,'method':name,'mask_state':idx,'parameters':sum(p.numel() for p in net.parameters()),
                     'retraining_seconds':train_seconds,**metrics,**timing})
    frame=pd.DataFrame(rows); dense_val=float(frame.loc[frame.method=='Dense','val_accuracy'].iloc[0])
    frame['validation_drop_pp']=(dense_val-frame.val_accuracy)*100
    frame['meets_4pp_target']=frame.validation_drop_pp<=4.0
    frame.to_csv(folder/'final_comparison.csv',index=False)
    prior=json.loads((HERE.parent/'fashion_v2'/'results'/'seed_11'/'complete.json').read_text())
    save(folder/'timing_scope.json',{'classical_exact_seconds':exact_seconds,'classical_best32_seconds':random_seconds,
         'prior_full_seed_experiment_seconds':prior['seconds'],
         'prior_experiment_includes':'128fit+49audit mask retrains, regression CV, two QAOA solves, final comparisons',
         'prior_warmstart_solver_seconds':state['prior_simulator_optimization_seconds'],
         'end_to_end_speedup_established':False,
         'limitation':'Historic label-generation stages were not separately timed; do not claim exact end-to-end comparison. Hardware job timings reside in its ledger. Inference speed is distinct from pruning-search speed.'})
    print(frame.to_string(index=False))
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['dry-run','hardware','evaluate-dry-run','evaluate-hardware'])
    parser.add_argument('--account',default='qismet-open'); parser.add_argument('--backend')
    args=parser.parse_args()
    if args.action=='dry-run':
        for _ in range(8):
            state=advance('dry-run')
            if state['complete']: break
    elif args.action=='hardware': advance('hardware',args.account,args.backend)
    else: evaluate_selected(args.action.replace('evaluate-',''))
