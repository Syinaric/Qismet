"""Constraint-preserving QAOA. Local validation pilot precedes real IBM sampling.

Circuit angles are optimized on a CPU simulator; hardware evaluates these frozen
circuits. No claim of hardware angle optimization or quantum speedup is made.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hardware_v3'))
from ibm_backend import HardwareRunner,connect

EDGES=[(i,i+1) for i in range(0,8,2)]+[(i,(i+1)%8) for i in range(1,8,2)]

def build(seed=11,layers=3):
    a=np.load(HERE/f'pilot_p{max(1,layers)}_seed_{seed}.npz')
    qc=QuantumCircuit(8,name=f'constrained_p{layers}')
    # Four Bell pairs: one excitation per pair, exactly four overall.
    for i in range(0,8,2): qc.h(i); qc.x(i+1); qc.cx(i,i+1)
    if layers==0: return qc
    linear=a['linear']; b=np.zeros((8,8)); b[np.triu_indices(8,1)]=a['pairs']
    h=-linear/2-(b+b.T).sum(1)/4; J=b/4
    theta=a['theta']; scale=float(a['scale'])
    for l in range(layers):
        for i in range(8): qc.rz(2*theta[l]*h[i]/scale,i)
        for i in range(8):
            for j in range(i+1,8): qc.rzz(2*theta[l]*J[i,j]/scale,i,j)
        # Each adjacent XY rotation swaps |01> and |10> without changing count.
        for i,j in EDGES:
            qc.rxx(theta[layers+l],i,j); qc.ryy(theta[layers+l],i,j)
    return qc

def verify():
    worst=0.
    for seed in [11,22,33,44,55]:
        for layers in [1,2,3]:
            a=np.load(HERE/f'pilot_p{layers}_seed_{seed}.npz')
            pr=Statevector.from_instruction(build(seed,layers)).probabilities()
            error=float(np.max(np.abs(pr[a['ids']]-a['probabilities'])))
            assert error<1e-10 and abs(pr[a['ids']].sum()-1)<1e-10
            worst=max(worst,error)
    print('PASS: all 15 actual Qiskit circuits match the independent subspace simulator;',worst)

def run():
    # Freeze the protocol before hardware results and test evaluations.
    previous=HERE.parent/'hardware_v3'/'runs'/'hardware'/'ledger.json'
    old=json.loads(previous.read_text())
    if not all(j['state']=='complete' and j.get('reported_usage_seconds') for j in old['jobs']):
        raise RuntimeError('Resolve prior usage before reserving additional QPU time.')
    spent=sum(j['reported_usage_seconds'] for j in old['jobs'])
    accounted=spent
    partial_path=HERE/'hardware_kingston_partial'/'ledger.json'
    if partial_path.exists():
        partial=json.loads(partial_path.read_text())
        if any(j['state'] not in ('complete','failed') for j in partial['jobs']):
            raise RuntimeError('Resolve the previous backend attempt before continuing.')
        spent+=sum(j.get('reported_usage_seconds',0) for j in partial['jobs'])
        accounted+=sum(j['accounted_seconds'] for j in partial['jobs'])
    protocol={'seed':11,'layers':[0,1,2,3],'shots_per_circuit':4096,'prior_qpu_seconds':spent,
              'prior_budget_accounted_seconds':accounted,'backend':'ibm_fez',
              'backend_change':'The kingston control completed; the next job was cancelled while queued, before any QAOA result. Preserve its 60-second uncertain reservation and rerun all four circuits on fez.',
              'additional_allowance_seconds':600-accounted,'reserve_seconds':120,'max_jobs':4,
              'max_job_seconds':60,'selection_shots':32,'selection_seed':20261106,
              'description':'Exploratory comparison; CPU-optimized angles evaluated on real hardware. Initial-state control plus three depths. No test-based selection.'}
    folder=HERE/'hardware'; folder.mkdir(exist_ok=True)
    path=folder/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Protocol mismatch')
    path.write_text(json.dumps(protocol,indent=2))
    service,backend=connect(backend_name=protocol['backend'])
    runner=HardwareRunner(service,backend,folder/'ledger.json',allowance_seconds=600-accounted,
                          reserve_seconds=120,max_job_seconds=60,max_jobs=4)
    for layers in protocol['layers']:
        tag=f'constrained_p{layers}'
        existing=runner.find_job(tag)
        if existing:
            if not existing.get('job_id'): raise RuntimeError('Ambiguous submission; reconcile before continuing.')
            result=runner.collect(existing['job_id'],wait=False)
        else:
            result=runner.sample(build(layers=layers),shots=4096,output_dir=folder,tag=tag,wait=False)
        if result['state']=='failed': raise RuntimeError('Hardware job failed; inspect ledger')
        if result['state']!='complete':
            print(tag,result['job_id'],result['state'],result.get('remote_status'),flush=True)
            return False
        print(tag,'complete',result.get('reported_usage_seconds'),'QPU seconds',flush=True)
    return True

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','hardware'])
    args=parser.parse_args()
    if args.action=='verify': verify()
    else:
        while not run(): time.sleep(20)
        print('All four comparison jobs complete.',flush=True)
