"""Resume the authorized experiment, preserving the original ledger and limits."""
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'outputs'/'hardware_v3'))
import workflow

while True:
    state=workflow.advance('hardware',backend_name='ibm_kingston')
    if state['complete']:
        workflow.evaluate_selected('hardware')
        print('Hardware experiment and final retraining complete.',flush=True)
        break
    time.sleep(20)
