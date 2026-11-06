"""Save IBM Quantum credentials to ~/.qiskit/qiskit-ibm.json (outside this repo).

Run:  python outputs/ibm_setup.py
The API key is read without echoing and is never written to this project.
"""
import getpass

from qiskit_ibm_runtime import QiskitRuntimeService

token = getpass.getpass('Paste your IBM Cloud API key (hidden), then press Enter: ').strip()
crn = input('Paste your instance CRN, then press Enter: ').strip()

QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=token, instance=crn,
                                  set_as_default=True, overwrite=True)
print('Saved. Checking the connection...')

service = QiskitRuntimeService()
names = [b.name for b in service.backends()]
print('Connected. Backends you can use:', ', '.join(names) if names else '(none listed)')
