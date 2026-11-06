"""Offline boundary test: no IBM account, network, or QPU calls."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs'/'hardware_v3'))
import workflow as w

problem,theta,prior=w.problem_and_warmstart()
w.problem_and_warmstart=lambda:(problem,theta,prior)
jobs={}
connections=[]

def connect(account_name,backend_name):
    connections.append(backend_name)
    return object(),SimpleNamespace(name='offline-boundary-test')

class Runner:
    def __init__(self,service,backend,ledger_path,**kwargs):
        assert kwargs==dict(allowance_seconds=600,reserve_seconds=120,max_job_seconds=60,max_jobs=8)
        self.path=ledger_path
    def find_job(self,tag):
        return jobs.get(tag)
    def sample(self,circuit,shots,output_dir,tag,wait):
        assert not wait and not circuit.num_parameters and circuit.num_clbits==0
        assert circuit.num_qubits==8 and tag not in jobs
        assert output_dir==self.path.parent
        job=dict(job_id=tag,state='pending',shots=shots,remote_status='QUEUED')
        jobs[tag]=job
        self.path.write_text(json.dumps({'backend':'offline-boundary-test'}))
        return job
    def collect(self,job_id,wait):
        assert not wait
        job=jobs[job_id]
        return dict(job,state='complete',counts={'00111100':job['shots']})

sys.modules['ibm_backend']=SimpleNamespace(HardwareRunner=Runner,connect=connect)
with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
    w.HERE=Path(temp)
    for i in range(8):
        pending=w.advance('hardware')
        assert len(pending['history'])==i and len(jobs)==i+1
        complete=w.advance('hardware')
        assert len(complete['history'])==i+1 and len(jobs)==i+1
    assert complete['complete'] and complete['selected_mask']==[0,0,1,1,1,1,0,0]
    assert len(complete['history'])==8
    assert connections[0] is None and all(x=='offline-boundary-test' for x in connections[1:])
    w.advance('hardware')
    assert len(jobs)==8
print('PASS: eight stages, pending resume, no duplicate submission, backend pinning, unitary circuit contract. OFFLINE ONLY.')
