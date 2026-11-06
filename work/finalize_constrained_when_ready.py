"""Generate the final report after the saved hardware evaluation completes."""
from pathlib import Path
import subprocess,sys,time
root=Path(__file__).resolve().parents[1]
while not (root/'outputs'/'constrained_v4'/'hardware'/'summary.json').exists():
    time.sleep(20)
subprocess.run([sys.executable,str(root/'work'/'build_constrained_report.py')],cwd=root,check=True)
print('Final constrained-QAOA report and notebook ready.',flush=True)
