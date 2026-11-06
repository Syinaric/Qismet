"""16-qubit hardware scaling check, with usage carried forward from prior runs."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); import experiment as e
sys.path.insert(0,str(HERE.parent/'hardware_v3'))
from ibm_backend import HardwareRunner,connect

def build(keep,layers=1):
    a=np.load(HERE/'seed_1001'/f'keep_{keep}'/f'qaoa_p{max(layers,1)}.npz')
    qc=QuantumCircuit(16,name=f'scale16_keep{keep}_p{layers}')
    block=16//keep
    for start in range(0,16,block):
        if block==2:
            qc.h(start); qc.x(start+1); qc.cx(start,start+1)
        else:
            # Uniform single excitation in each four-neuron-group block.
            qc.x(start)
            for i in range(block-1):
                angle=2*np.arccos(1/np.sqrt(block-i))
                qc.cry(angle,start+i,start+i+1); qc.cx(start+i+1,start+i)
    if layers==0: return qc
    linear=a['linear']; b=np.zeros((16,16)); b[np.triu_indices(16,1)]=a['pairs']
    h=-linear/2-(b+b.T).sum(1)/4; J=b/4; theta=a['theta']; scale=float(a['scale'])
    for l in range(layers):
        for i in range(16): qc.rz(2*theta[l]*h[i]/scale,i)
        for i in range(16):
            for j in range(i+1,16): qc.rzz(2*theta[l]*J[i,j]/scale,i,j)
        for i,j in e.EDGES:
            qc.rxx(theta[layers+l],i,j); qc.ryy(theta[layers+l],i,j)
    return qc

def verify(keep):
    for layers in [1,2]:
        a=np.load(HERE/'seed_1001'/f'keep_{keep}'/f'qaoa_p{layers}.npz')
        pr=Statevector.from_instruction(build(keep,layers)).probabilities()
        error=float(np.max(np.abs(pr[a['ids']]-a['probabilities'])))
        assert error<1e-10 and abs(pr[a['ids']].sum()-1)<1e-10
        print('Verified 16-qubit circuit',keep,layers,'probability error',error,flush=True)

def prior_usage():
    paths=[HERE.parent/'hardware_v3'/'runs'/'hardware'/'ledger.json',
           HERE.parent/'constrained_v4'/'hardware_kingston_partial'/'ledger.json',
           HERE.parent/'constrained_v4'/'hardware'/'ledger.json']
    accounted=0.; reported=0.
    for path in paths:
        ledger=json.loads(path.read_text())
        if any(j['state'] not in ('complete','failed') for j in ledger['jobs']):
            raise RuntimeError('Resolve prior pending jobs before reserving additional usage')
        accounted+=sum(j['accounted_seconds'] for j in ledger['jobs'])
        reported+=sum(j.get('reported_usage_seconds',0) for j in ledger['jobs'])
    return accounted,reported

def run():
    folder=HERE/'hardware'; folder.mkdir(exist_ok=True)
    accounted,reported=prior_usage()
    protocol={'seed':1001,'budgets':[8,4],'layers':[0,1],'shots':4096,'backend':'ibm_fez',
              'previous_accounted_seconds':accounted,'previous_reported_seconds':reported,
              'allowance_seconds':600-accounted,'reserve_seconds':120,'max_job_seconds':60,'max_jobs':4,
              'description':'First preselected initialization, both sparsities; initial-state controls and one QAOA layer. Deeper circuits are local ablations, not additional QPU jobs. All results are reported.'}
    path=folder/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Protocol mismatch')
    e.save(path,protocol)
    for keep in [8,4]:
        base=HERE/'seed_1001'/f'keep_{keep}'
        if not (base/'summary.json').exists():
            print('Waiting for local model',keep,flush=True); return False
        summary=json.loads((base/'summary.json').read_text())
        audit=next(r for r in summary['surrogate_audit'] if r['model']=='Quadratic')
        if audit['r2']<=0:
            e.save(folder/f'skip_keep_{keep}.json',{'reason':'nonpositive held-out QUBO R2','audit':audit})
            print('Skipping unpromising hardware budget:',keep,flush=True); continue
        verified=folder/f'verified_keep_{keep}.json'
        if not verified.exists(): verify(keep); e.save(verified,{'circuit_equivalence':True})
        service,backend=connect(backend_name='ibm_fez')
        runner=HardwareRunner(service,backend,folder/'ledger.json',allowance_seconds=600-accounted,
                              reserve_seconds=120,max_job_seconds=60,max_jobs=4,logical_qubits=16)
        for layers in [0,1]:
            tag=f'keep_{keep}_p{layers}'; existing=runner.find_job(tag)
            if existing:
                if not existing.get('job_id'): raise RuntimeError('Ambiguous submission; reconcile before continuing')
                result=runner.collect(existing['job_id'],wait=False)
            else:
                result=runner.sample(build(keep,layers),shots=4096,output_dir=folder,tag=tag,wait=False)
            if result['state']=='failed': raise RuntimeError('Hardware job failed; inspect saved ledger')
            if result['state']!='complete':
                print(tag,result['job_id'],result['state'],result.get('remote_status'),flush=True); return False
            print(tag,'complete',result.get('reported_usage_seconds'),'QPU seconds',flush=True)
    e.save(folder/'complete.json',{'complete':True,'note':'Inspect all results and skips; no speedup is implied.'})
    return True

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=['verify','run']); args=parser.parse_args()
    if args.action=='verify':
        for keep in [8,4]: verify(keep)
    else:
        while not run(): time.sleep(20)
        print('16-group hardware follow-up complete.',flush=True)
