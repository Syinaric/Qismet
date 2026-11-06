"""IBM Open Plan sampling with explicit credentials, recoverable jobs and usage guards.

Importing this module performs NO network call and submits NO job. Credentials load
from the gitignored ibm_credentials.local.json beside this module, or a named IBM
account if that file is absent. Requires qiskit-ibm-runtime >= 0.50.

This adapter measures eight logical mask bits on a physical IBM backend. It does
not simulate statevectors and does not itself optimize QAOA parameters. Usage
guards reserve each pending job's server execution-time limit, retain uncertain
charges, and stop at allowance minus reserve. This is a conservative client guard,
not an IBM billing guarantee; unrelated jobs/account users are outside this ledger.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import time
from typing import Any

LOCAL_CREDENTIALS = Path(__file__).resolve().with_name('ibm_credentials.local.json')


def load_local_credentials(path=None):
    """Read local JSON without evaluating code or including secrets in errors.

    An existing but incomplete file fails explicitly; it never silently selects
    another account. The path is relative to this module, independent of cwd.
    """
    path = Path(path) if path is not None else LOCAL_CREDENTIALS
    if not path.exists():
        return None
    try:
        values = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        raise ValueError('Cannot read local IBM credentials. Use valid JSON with api_key and crn strings.') from None
    if not isinstance(values, dict):
        raise ValueError('Local IBM credentials must be a JSON object with api_key and crn strings.')
    token, instance = values.get('api_key'), values.get('crn')
    if not isinstance(token, str) or not isinstance(instance, str):
        raise ValueError('Fill in both api_key and crn in ibm_credentials.local.json.')
    token, instance = token.strip(), instance.strip()
    if not token or not instance or token.startswith('PASTE_') or instance.startswith('PASTE_'):
        raise ValueError('Replace both placeholders in ibm_credentials.local.json with your API key and Open Plan instance CRN.')
    if not instance.startswith('crn:'):
        raise ValueError('The crn field must contain the full IBM instance CRN beginning with crn:.')
    return {'token': token, 'instance': instance}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(temp, path)


def _verify_open(service, instance):
    matches = [i for i in service.instances() if i.get('crn') == instance]
    if len(matches) != 1 or str(matches[0].get('plan', '')).lower() != 'open':
        raise RuntimeError('The explicitly selected IBM instance is not verified as Open Plan. No jobs submitted.')
    if service.active_instance() != instance:
        raise RuntimeError('The active IBM instance differs from the selected Open instance.')


def configure_account(instance=None, account_name='qismet-open', overwrite=False):
    """Load local JSON or prompt, verify the explicit Open instance, then save.

    Never paste the key into chat, source code, command arguments, or notebook cells.
    The named account is saved outside this project in IBM's standard local store.
    No circuits or paid workloads are submitted by this function.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    local = load_local_credentials()
    if local is not None and instance is not None and instance != local['instance']:
        raise ValueError('The requested instance differs from the local credentials CRN.')
    if local is not None:
        instance = local['instance']
    elif instance is None:
        instance = input('IBM Open Plan instance CRN (Instances page): ').strip()
    if not instance or instance == 'auto':
        raise ValueError('Provide the exact Open Plan instance CRN.')
    token = local['token'] if local is not None else getpass.getpass('IBM Quantum API key (hidden input): ').strip()
    if not token:
        raise ValueError('No API key entered.')
    try:
        service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token, instance=instance)
        _verify_open(service, instance)
        QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=token,
            instance=instance, name=account_name, set_as_default=False,
            plans_preference=['open'], overwrite=overwrite)
    finally:
        token = None
    return {'account_name': account_name, 'plan': 'open', 'saved': True, 'jobs_submitted': 0}


def connect(account_name='qismet-open', backend_name=None):
    """Read the ignored JSON (preferred) or named account, then verify Open Plan.

    Does not auto-select an instance or submit any circuit. Missing credentials fail
    with instructions; account dictionaries containing tokens are never returned.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    local = load_local_credentials()
    if local is not None:
        instance = local['instance']
        service = QiskitRuntimeService(channel='ibm_quantum_platform',
                                      token=local['token'], instance=instance)
    else:
        account = QiskitRuntimeService.saved_accounts().get(account_name)
        if not account:
            raise RuntimeError('Fill in ibm_credentials.local.json or configure a named IBM account locally.')
        instance = account.get('instance')
        if not instance or instance == 'auto':
            raise RuntimeError('The named account must specify an explicit Open Plan instance.')
        service = QiskitRuntimeService(name=account_name, instance=instance)
    _verify_open(service, instance)
    backend = (service.backend(backend_name, instance=instance) if backend_name else
               service.least_busy(operational=True, simulator=False, min_num_qubits=8))
    if backend.num_qubits < 8 or not backend.status().operational:
        raise RuntimeError('Choose an operational backend with at least eight programmable qubits.')
    if backend.configuration().simulator or getattr(backend, 'is_mock', False):
        raise RuntimeError('This connection must use a real IBM QPU, not a simulator or mock.')
    _verify_open(service, instance)
    return service, backend


class HardwareRunner:
    """Eight-bit hardware sampler with a persistent, single-writer usage ledger.

    ``sample(..., wait=False)`` prepares artifacts and submits one job. ``collect``
    resumes it by ID; nonblocking collection returns its status without resubmitting.
    No more than one unresolved job may exist; ambiguous submission failures block
    further work until manually reconciled with the IBM Workloads page. Tags are
    unique across the ledger, even for completed jobs, to prevent accidental reruns.

    Defaults: 600s total allowance, 120s reserve, 60s maximum server execution/job,
    8 submitted jobs, 4096 shots/job. Waiting is bounded to at most 60s per call.
    Results retain counts in Qiskit's c7...c0 order: int(bitstring,2) gives the mask
    integer; mask[i]=(integer>>i)&1. The measurement map is attached before routing.
    """
    def __init__(self, service, backend, ledger_path, *, allowance_seconds=600,
                 reserve_seconds=120, max_job_seconds=60, max_jobs=8,
                 max_shots=4096, seed_transpiler=2026, sampler_factory=None,
                 verified_open=True):
        if allowance_seconds > 600 or allowance_seconds <= 0:
            raise ValueError('The authorized hardware allowance is at most 600 seconds.')
        if reserve_seconds < 120 or reserve_seconds >= allowance_seconds:
            raise ValueError('Retain at least 120 seconds of the allowance as reserve.')
        if not 1 <= max_job_seconds <= 120:
            raise ValueError('Server execution limit must be between 1 and 120 seconds.')
        if not 1 <= max_jobs <= 100 or not 1 <= max_shots <= 65536:
            raise ValueError('Invalid job/shot guard.')
        self.service, self.backend = service, backend
        self.ledger_path = Path(ledger_path).resolve()
        self.settings = dict(allowance_seconds=float(allowance_seconds), reserve_seconds=float(reserve_seconds),
            max_job_seconds=int(max_job_seconds), max_jobs=int(max_jobs), max_shots=int(max_shots))
        self.seed_transpiler = int(seed_transpiler)
        self.sampler_factory = sampler_factory
        self._jobs = {}
        self.instance = service.active_instance()
        # Test injection must be explicit; production always rechecks the plan.
        if verified_open:
            _verify_open(service, self.instance)
        elif sampler_factory is None:
            raise ValueError('Bypassing plan verification is permitted only with an injected test sampler.')
        self.test_mode = not verified_open
        with self._lock():
            if self.ledger_path.exists():
                ledger = self._read()
                if ledger['settings'] != self.settings or ledger['backend'] != self.backend.name:
                    raise ValueError('Existing ledger uses different budget/backend settings; preserve its budget accounting.')
                if ledger.get('instance') != self.instance or ledger.get('test_mode') != self.test_mode:
                    raise ValueError('Existing ledger belongs to a different instance or execution mode.')
            else:
                _write_json(self.ledger_path, {'schema':1,'created':_now(),'backend':self.backend.name,
                    'instance':self.instance,'test_mode':self.test_mode,'settings':self.settings,'jobs':[]})

    @contextlib.contextmanager
    def _lock(self):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.ledger_path.with_suffix('.lock')
        try:
            handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError('Ledger is locked. Do not submit concurrently; inspect any interrupted owner before removing the lock.')
        try:
            os.write(handle, str(os.getpid()).encode())
            yield
        finally:
            os.close(handle)
            lock.unlink()

    def _read(self):
        return json.loads(self.ledger_path.read_text(encoding='utf-8'))

    def budget_status(self):
        ledger = self._read()
        accounted = sum(float(j.get('accounted_seconds', self.settings['max_job_seconds'])) for j in ledger['jobs'])
        available = self.settings['allowance_seconds']-self.settings['reserve_seconds']-accounted
        return dict(accounted_seconds=accounted, available_for_new_reservations_seconds=available,
                    jobs=len(ledger['jobs']), allowance_seconds=self.settings['allowance_seconds'],
                    reserve_seconds=self.settings['reserve_seconds'])

    def find_job(self, tag):
        """Return saved metadata for a tag, or None, without submitting/contacting IBM.

        Recover after interruption using collect(record['job_id']). If job_id is
        None, submission was interrupted before its ID was safely persisted; do
        not retry. Reconcile the job tag on IBM Workloads before manual recovery.
        """
        matches = [j for j in self._read()['jobs'] if j['tag'] == tag]
        if len(matches)>1: raise RuntimeError('Duplicate tags in ledger; manual reconciliation required.')
        return dict(matches[0]) if matches else None

    def _prepare(self, circuit, shots, folder):
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, qpy
        from qiskit.transpiler import generate_preset_pass_manager
        if circuit.num_qubits != 8 or circuit.num_parameters:
            raise ValueError('Pass a fully bound eight-logical-qubit circuit.')
        if circuit.num_clbits or any(x.operation.name in ('measure','reset') for x in circuit.data):
            raise ValueError('Pass a unitary circuit without classical bits, measurements or reset; adapter adds the mask register.')
        qreg, creg = QuantumRegister(8,'q'), ClassicalRegister(8,'mask')
        measured = QuantumCircuit(qreg,creg)
        measured.compose(circuit,qubits=range(8),inplace=True)
        measured.measure(range(8),range(8))
        pm = generate_preset_pass_manager(backend=self.backend,optimization_level=2,
                                          seed_transpiler=self.seed_transpiler)
        isa = pm.run(measured)
        artifacts = {}
        for name, qc in [('logical',measured),('transpiled',isa)]:
            with (folder/(name+'.qpy')).open('wb') as handle: qpy.dump(qc,handle)
            (folder/(name+'.txt')).write_text(str(qc.draw(output='text',idle_wires=False,fold=120)),encoding='utf-8')
            artifacts[name] = {'qubits_in_register':qc.num_qubits,'classical_bits':qc.num_clbits,
                'depth':qc.depth(),'two_qubit_gates':sum(len(x.qubits)==2 for x in qc.data),
                'gate_counts':dict(qc.count_ops())}
            try:
                fig = qc.draw(output='mpl',idle_wires=False,fold=28)
                fig.savefig(folder/(name+'.png'),dpi=140,bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as exc:
                # Visualization is optional; exact text and QPY remain available.
                artifacts[name]['diagram_unavailable'] = type(exc).__name__
        layout = isa.layout.final_index_layout() if isa.layout is not None else list(range(8))
        measurement_map = [{'physical_qubit':isa.find_bit(x.qubits[0]).index,
                            'mask_classical_bit':isa.find_bit(x.clbits[0]).index}
                           for x in isa.data if x.operation.name=='measure']
        if sorted(m['mask_classical_bit'] for m in measurement_map) != list(range(8)):
            raise AssertionError('Transpiled mask measurement map is not a bijection.')
        # A scheduling estimate is informative; the server max_execution_time is
        # the actual per-job cancellation guard. No custom repetition delay used.
        try:
            circuit_seconds=float(isa.estimate_duration(self.backend.target,unit='s'))
        except Exception:
            circuit_seconds=None
        rep_delay=getattr(self.backend,'default_rep_delay',None)
        rep_delay=float(rep_delay) if rep_delay is not None else .00025
        estimate = 2+shots*(rep_delay+(circuit_seconds if circuit_seconds is not None else .0001))
        metadata={'logical_mask_bits':8,'logical_to_physical_final':layout,'measurement_map':measurement_map,
            'circuits':artifacts,'shots':shots,'transpiler_seed':self.seed_transpiler,
            'circuit_seconds_estimate':circuit_seconds,'rep_delay_seconds':rep_delay,
            'baseline_usage_estimate_seconds':estimate,
            'estimate_warning':'Approximation only; server overhead and runtime behavior can differ. No billing guarantee.'}
        _write_json(folder/'circuit_metadata.json',metadata)
        return isa,metadata

    def sample(self, bound_logical_circuit, shots, output_dir, tag, *, wait=False, timeout=60):
        """Submit once, persist ID before waiting, return status or completed counts.

        A timeout or interrupted submission must be reconciled, never blindly retried.
        A completed duplicate tag must be retrieved with collect instead of resubmitted.
        """
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',tag):
            raise ValueError('Tag must be 1..80 letters, digits, underscores or hyphens.')
        if int(shots)!=shots or not 1 <= shots <= self.settings['max_shots']:
            raise ValueError('Shots exceed the configured guard.')
        shots=int(shots)
        folder=Path(output_dir).resolve()/tag
        folder.mkdir(parents=True,exist_ok=True)
        with self._lock():
            ledger=self._read()
            if any(j['tag']==tag for j in ledger['jobs']):
                raise RuntimeError('This tag already exists. Use collect(job_id); automatic replay is disabled.')
            if any(j['state'] in ('submitting','submission_uncertain','pending') for j in ledger['jobs']):
                raise RuntimeError('Resolve the existing pending/uncertain job before submitting another.')
            if len(ledger['jobs'])>=self.settings['max_jobs']:
                raise RuntimeError('Maximum job count reached.')
            if self.budget_status()['available_for_new_reservations_seconds']<self.settings['max_job_seconds']:
                raise RuntimeError('Insufficient authorized allowance after usage reservations and reserve.')
            if not self.test_mode: _verify_open(self.service,self.instance)
            isa,metadata=self._prepare(bound_logical_circuit,shots,folder)
            if metadata['baseline_usage_estimate_seconds'] > self.settings['max_job_seconds']:
                raise RuntimeError('Estimated usage exceeds the configured per-job execution limit.')
            if self.sampler_factory is None:
                from qiskit_ibm_runtime.executor_sampler import Sampler
                sampler=Sampler(mode=self.backend)
            else:
                sampler=self.sampler_factory(self.backend)
            sampler.options.max_execution_time=self.settings['max_job_seconds']
            sampler.options.default_shots=shots
            sampler.options.execution.init_qubits=True
            sampler.options.twirling.enable_gates=False
            sampler.options.twirling.enable_measure=False
            sampler.options.dynamical_decoupling.enable=False
            sampler.options.environment.job_tags=['qismet',tag]
            record={'tag':tag,'job_id':None,'state':'submitting','submitted_at':_now(),
                'submitted_epoch':time.time(),'output_dir':str(folder),'shots':shots,
                'accounted_seconds':self.settings['max_job_seconds'],
                'usage_estimate_seconds':metadata['baseline_usage_estimate_seconds']}
            ledger['jobs'].append(record)
            _write_json(self.ledger_path,ledger)  # reservation precedes remote request
            try:
                job=sampler.run([isa],shots=shots)
                record.update(job_id=job.job_id(),state='pending')
                self._jobs[record['job_id']]=job
                _write_json(self.ledger_path,ledger)
                _write_json(folder/'submission.json',record)
            except BaseException:
                record['state']='submission_uncertain'
                _write_json(self.ledger_path,ledger)
                raise
        return self.collect(record['job_id'],wait=wait,timeout=timeout)

    def collect(self, job_id, *, wait=False, timeout=60):
        """Recover a job without submitting; wait is bounded to <=60 seconds.

        Counts are read from the explicit `mask` register. Unknown/zero finalized
        usage retains the full reservation. Recollect later to reconcile delayed
        usage. Cancelled/failed jobs can consume usage and are never assumed free.
        """
        from qiskit.providers.exceptions import JobTimeoutError
        if not 0 < timeout <= 60: raise ValueError('Use a bounded timeout up to 60 seconds.')
        with self._lock():
            ledger=self._read()
            matching=[j for j in ledger['jobs'] if j['job_id']==job_id]
            if len(matching)!=1: raise ValueError('Job ID is not uniquely recorded in this ledger.')
            record=matching[0]
            job=self._jobs.get(job_id) or self.service.job(job_id)
            state=str(job.status()).upper().split('.')[-1]
            if wait and state not in ('DONE','COMPLETED','ERROR','FAILED','CANCELLED','CANCELED'):
                try:
                    job.wait_for_final_state(timeout=timeout)
                except (TimeoutError, JobTimeoutError):
                    pass
                state=str(job.status()).upper().split('.')[-1]
            finished=state in ('DONE','COMPLETED','ERROR','FAILED','CANCELLED','CANCELED')
            record['remote_status']=state
            record['state']='complete' if state in ('DONE','COMPLETED') else ('failed' if finished else 'pending')
            record['last_collected_at']=_now()
            if finished and 'finished_observed_epoch' not in record:
                record['finished_observed_epoch']=time.time()
            record['submission_to_last_observation_seconds']=time.time()-record['submitted_epoch']
            if finished:
                record['submission_to_completion_observation_seconds']=record['finished_observed_epoch']-record['submitted_epoch']
                try:
                    metrics=job.metrics()
                    _write_json(Path(record['output_dir'])/'job_metrics.json',metrics)
                except Exception:
                    metrics={}
                try:
                    used=float(job.usage())
                except Exception:
                    used=None
                if used is not None and used>0:
                    record['accounted_seconds']=used
                    record['reported_usage_seconds']=used
                    record['usage_reservation_retained']=False
                else:
                    record['usage_reservation_retained']=True
                if state in ('DONE','COMPLETED'):
                    result=job.result()
                    counts={str(k):int(v) for k,v in result[0].data.mask.get_counts().items()}
                    if sum(counts.values())!=record['shots'] or any(not re.fullmatch('[01]{8}',k) for k in counts):
                        raise AssertionError('Unexpected mask register width or shot count.')
                    payload={'job_id':job_id,'backend':self.backend.name,'counts':counts,
                        'logical_mask_bits':8,'bit_order':'Qiskit c7..c0; neuron group i is bit i of int(key,2)',
                        'reported_usage_seconds':record.get('reported_usage_seconds'),
                        'wallclock_seconds':record['submission_to_completion_observation_seconds'],
                        'test_mode':self.test_mode}
                    _write_json(Path(record['output_dir'])/'result.json',payload)
                    record['result_file']=str(Path(record['output_dir'])/'result.json')
            _write_json(self.ledger_path,ledger)
            response=dict(record)
            if record.get('result_file'):
                response.update(json.loads(Path(record['result_file']).read_text(encoding='utf-8')))
            return response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    setup=subs.add_parser('configure',help='Enter API key locally with hidden input; verify/save Open account.')
    setup.add_argument('--instance',help='Exact IBM Open Plan instance CRN; prompted if omitted. Not an API key.')
    setup.add_argument('--account-name',default='qismet-open')
    setup.add_argument('--overwrite',action='store_true')
    check=subs.add_parser('check',help='Verify credentials, Open Plan and backend access; submit no jobs.')
    check.add_argument('--account-name',default='qismet-open')
    check.add_argument('--backend')
    args=parser.parse_args()
    if args.command=='configure':
        outcome=configure_account(args.instance,args.account_name,args.overwrite)
        print(json.dumps(outcome,indent=2))
    elif args.command=='check':
        service,backend=connect(args.account_name,args.backend)
        print(json.dumps({'connected':True,'plan':'open','backend':backend.name,'jobs_submitted':0},indent=2))


if __name__=='__main__':
    main()
