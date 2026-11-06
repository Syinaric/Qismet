"""Offline 8/16-bit adapter checks, never contacts IBM."""
from pathlib import Path
import sys,tempfile
from types import SimpleNamespace as NS
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'outputs'/'hardware_v3'))
from ibm_backend import HardwareRunner

class Service:
    def active_instance(self): return 'offline'

for width in [8,16]:
    class Job:
        def job_id(self): return f'offline-{width}'
        def status(self): return 'DONE'
        def metrics(self): return {}
        def usage(self): return 2.
        def result(self): return [NS(data=NS(mask=NS(get_counts=lambda:{'10'*(width//2):32})))]
    job=Job()
    class Sampler:
        def __init__(self,backend):
            self.options=NS(execution=NS(),twirling=NS(),dynamical_decoupling=NS(),environment=NS())
        def run(self,pubs,shots):
            assert pubs[0].num_clbits==width and shots==32
            return job
    with tempfile.TemporaryDirectory(dir=root/'work') as temp:
        runner=HardwareRunner(Service(),GenericBackendV2(width),Path(temp)/'ledger.json',
                              sampler_factory=Sampler,verified_open=False,logical_qubits=width)
        qc=QuantumCircuit(width); qc.h(0); qc.cx(0,1)
        result=runner.sample(qc,shots=32,output_dir=temp,tag='check')
        assert result['counts']=={'10'*(width//2):32} and result['logical_mask_bits']==width
        assert result['reported_usage_seconds']==2
        assert runner.find_job('check')['state']=='complete'
print('PASS: real transpilation, explicit measurement mapping and count recovery for both 8 and 16 bits. Offline only.')
