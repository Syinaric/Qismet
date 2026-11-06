"""Offline credential loading and Git-ignore checks; no IBM calls."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs'/'hardware_v3'))
import ibm_backend as ibm

with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
    temp=Path(temp)
    credential=temp/'credentials.json'
    dummy={'api_key':'offline-test-key','crn':'crn:offline-test-instance'}
    credential.write_text(json.dumps(dummy))
    calls=[]
    backend=SimpleNamespace(name='offline-test-backend',num_qubits=8,is_mock=False,
        status=lambda:SimpleNamespace(operational=True),
        configuration=lambda:SimpleNamespace(simulator=False))
    class Service:
        def __init__(self,**kwargs): calls.append(kwargs)
        @staticmethod
        def saved_accounts():
            return {'qismet-open':{'instance':dummy['crn']}}
        def instances(self): return [{'crn':dummy['crn'],'plan':'open'}]
        def active_instance(self): return dummy['crn']
        def least_busy(self,**kwargs): return backend
        def backend(self,name,instance):
            assert name==backend.name and instance==dummy['crn']
            return backend
    with patch.object(ibm,'LOCAL_CREDENTIALS',credential),patch('qiskit_ibm_runtime.QiskitRuntimeService',Service):
        # Files are resolved independently of the current working directory.
        previous=Path.cwd()
        try:
            os.chdir(temp)
            assert ibm.connect()[1] is backend
        finally:
            os.chdir(previous)
        assert calls[-1]==dict(channel='ibm_quantum_platform',token=dummy['api_key'],instance=dummy['crn'])
        assert ibm.connect(backend_name=backend.name)[1] is backend
        for content in ['{ "api_key": "offline-test-key", BROKEN', '[]',
                        '{"api_key":"PASTE_YOUR_IBM_API_KEY_HERE","crn":"crn:test"}',
                        '{"api_key":"offline-test-key","crn":"invalid"}']:
            credential.write_text(content)
            before=len(calls)
            try: ibm.connect()
            except ValueError as exc:
                assert dummy['api_key'] not in str(exc)
            else: raise AssertionError('Invalid credentials accepted')
            assert len(calls)==before
        credential.unlink()
        assert ibm.connect()[1] is backend
        assert calls[-1]==dict(name='qismet-open',instance=dummy['crn'])

    # Verify actual Git behavior in an isolated repository, leaving the project alone.
    subprocess.run(['git','init','--quiet',str(temp)],check=True,capture_output=True)
    (temp/'.gitignore').write_text((ROOT/'.gitignore').read_text())
    folder=temp/'outputs'/'hardware_v3'; folder.mkdir(parents=True)
    (folder/'.gitignore').write_text((ROOT/'outputs'/'hardware_v3'/'.gitignore').read_text())
    result=subprocess.run(['git','-C',str(temp),'check-ignore','outputs/hardware_v3/ibm_credentials.local.json'],capture_output=True,text=True)
    assert result.returncode==0
    (temp/'.gitignore').unlink()
    result=subprocess.run(['git','-C',str(temp),'check-ignore','outputs/hardware_v3/ibm_credentials.local.json'],capture_output=True,text=True)
    assert result.returncode==0
print('PASS: local file priority, cwd independence, backend resume, validation/redaction, named-account fallback, both Git-ignore rules. No network or QPU calls.')
