"""Wait locally for completed outputs, then build the report. Submits no jobs."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
folder=root/'outputs'/'hardware_v3'
state_path=folder/'runs'/'hardware'/'state.json'
while True:
    if state_path.exists():
        try:
            state=json.loads(state_path.read_text())
            if state['complete'] and (folder/'runs'/'hardware'/'timing_scope.json').exists():
                break
        except (OSError,ValueError):
            pass
    time.sleep(20)
for script in ['build_hardware_results.py','build_hardware_notebook.py']:
    subprocess.run([sys.executable,str(root/'work'/script)],cwd=root,check=True)
readme=folder/'README.md'
text=readme.read_text(encoding='utf-8')
lines=text.splitlines()
for i,line in enumerate(lines):
    if line.startswith('**Current status:'):
        lines[i]='**Current status: the real IBM hardware experiment is complete.** Open `Hardware_Results.md` for results and job IDs. The original commands below document how it was run; the saved ledger prevents duplicate execution.'
text='\n'.join(lines)+'\n'
readme.write_text(text,encoding='utf-8')
print('Final report and executed walkthrough updated with real hardware results.',flush=True)
