from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root=Path(__file__).resolve().parents[1]
out=root/'outputs'/'hardware_v3'
hardware_state_path=out/'runs'/'hardware'/'state.json'
hardware_status='Local rehearsal only. No IBM quantum hardware job has been submitted.'
if hardware_state_path.exists():
    import json
    hardware_state=json.loads(hardware_state_path.read_text())
    hardware_status=('Real IBM hardware experiment completed. See the hardware evidence in section 6 and Hardware_Results.md.'
                     if hardware_state['complete'] else 'Real IBM hardware experiment is in progress. See section 6 for saved progress.')
md=nbformat.v4.new_markdown_cell; code=nbformat.v4.new_code_cell
cells=[md('''# Qismet: from a pruning score to an IBM quantum circuit

**HARDWARE_STATUS_PLACEHOLDER**

The existing classifier receives 784 image pixels and predicts ten clothing classes.
Eight mask bits each retain/remove two hidden neurons. We keep exactly four groups in
this first hardware experiment. The 4-percentage-point tolerance is assessed on validation
accuracy; final test results do not choose the mask.

The quantum circuit proposes masks. PyTorch trains the neural networks on the CPU.
Follow these cells in order, then use README.md to configure IBM access locally.
'''),code('''from pathlib import Path
import json, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from circuit_core import *
from workflow import problem_and_warmstart, PROTOCOL
BASE=Path('.').resolve()
problem,initial_angles,prior=problem_and_warmstart()
display(pd.DataFrame([{'group':i,'neurons':f'{2*i}, {2*i+1}','linear coefficient':problem.linear[i]} for i in range(8)]))
print('Mask bit 1 keeps the group; bit 0 removes it.')
'''),md(r'''## 1. Define the expensive objective and its approximation

$g(m)=L_{val}(Train(W_0\text{ restricted by }m))$ requires retraining.
The frozen quadratic approximation is
$f(m)=c+\sum_i a_i m_i+\sum_{i<j}b_{ij}m_i m_j$.
This initial hardware run reuses the original fitted coefficients; it does not improve the surrogate.
Earlier mean held-out R² was negative, so we cannot call the approximation accurate in absolute loss.
'''),code('''audit=pd.read_csv(BASE.parent/'fashion_v2'/'surrogate_summary.csv')
display(audit)
display(Image(filename=str(BASE.parent/'fashion_v2'/'approximation_check'/'qubo_loss_fidelity.png')))
'''),md(r'''## 2. Encode a cost exactly

Add $\lambda(\sum m_i-4)^2$ for the required group count and replace $m_i$ by $(I-Z_i)/2$.
The resulting diagonal operator satisfies $H_C|m\rangle=C(m)|m\rangle$.
This is exact algebra, not another learned approximation. Energy scaling only changes how angles
are expressed; the constant offset contributes a global phase and is omitted from circuit gates.

The following three-bit example is small enough to check by hand. Enumeration here is a diagnostic;
hardware scoring evaluates only observed bitstrings.
'''),code('''toy=toy_data()
display(pd.DataFrame(toy['basis_table']))
print(qubo_to_ising(toy_problem()))
print('Actual eight-bit Ising operator:')
print(qubo_to_ising(problem))
'''),md(r'''## 3. Build the QAOA circuit

The Hadamard layer prepares a superposition. Each cost layer uses RZ and RZZ gates determined
by the Ising coefficients. RX gates implement the X mixer. Two repetitions use four classical
angle parameters. Cost evolution changes phases; subsequent mixing can change measured probabilities.

The code printed below is the actual circuit builder used by this package. The first diagram is
a three-qubit teaching example; the second is the actual eight-variable pruning circuit.
'''),code('''print(inspect.getsource(build_qaoa))
for name,p,reps in [('teaching_circuit',toy_problem(),1),('pruning_circuit',problem,2)]:
    qc=build_qaoa(p,reps=reps,measure=True,barriers=True)
    fig=qc.draw(output='mpl',fold=35)
    fig.savefig(BASE/(name+'.png'),dpi=130,bbox_inches='tight')
    display(fig); plt.close(fig)
'''),md(r'''## 4. Verify that the refactor preserves the original experiment

Gamma/beta binding order and little-endian mask interpretation are easy sources of errors.
Here the old saved probability vector and the newly built circuit must agree.
'''),code('''probabilities=ideal_probabilities(problem,initial_angles,reps=2)
error=np.max(np.abs(probabilities-np.array(prior['probabilities'])))
assert error<1e-12
print('Maximum probability difference:',error)
print('PASS: the new circuit reproduces the original saved Qiskit simulation.')
'''),md(r'''## 5. Understand the measured-cost feedback loop

The classical optimizer sends angles to the circuit. Measurements return masks, whose costs
are computed classically. Their average estimates the circuit's expected cost.

This budgeted run uses three SPSA steps. Each step evaluates two angle perturbations,
estimates a gradient, then updates the angles. It starts from the saved simulator-optimized
angles. The best *observed* parameter setting gets a separate final measurement.
Three updates are a demonstration budget, not evidence of convergence or superiority.

The executed outputs below are an ideal finite-shot rehearsal of the same control flow.
They are explicitly not hardware results.
'''),code('''state=json.loads((BASE/'runs'/'dry-run'/'state.json').read_text())
assert state['mode']=='dry-run' and state['complete']
trace=pd.DataFrame([{'phase':h['phase'],'iteration':h['iteration'],'scaled mean cost':h['scaled_mean_cost'],
                    'shots':h['shots'],'feasible fraction':h['stats']['feasible_fraction']} for h in state['history']])
display(trace)
print('Dry-run selected mask:',state['selected_mask'])
print('Final selection uses 32 observed shots; the full 2048 estimate the distribution.')
display(pd.read_csv(BASE/'runs'/'dry-run'/'final_comparison.csv'))
'''),md(r'''### How close does QAOA get to the QUBO optimum?

This is a separate question from whether the QUBO predicts retrained loss.
For this small diagnostic only, enumerate the 70 masks retaining four groups.
Compare their exact QUBO costs with the selected mask's cost. A zero gap means
the QUBO optimum was found; it does not mean the true retraining optimum was found.
The probability graph conditions on feasible outcomes; infeasible shots still cost resources.
'''),code('''masks=np.array([[(i>>j)&1 for j in range(8)] for i in range(256)])
feasible=np.flatnonzero(masks.sum(axis=1)==4)
values=np.array([cost(problem,masks[i],penalized=False) for i in feasible])
order=np.argsort(values)
final_angles=state['history'][-1]['parameters']
p_final=ideal_probabilities(problem,final_angles,reps=2)
mass=float(p_final[feasible].sum())
selected_value=cost(problem,state['selected_mask'],penalized=False)
display(pd.DataFrame([{'exact feasible QUBO minimum':values.min(),
                      'selected QUBO cost':selected_value,
                      'selected cost gap':selected_value-values.min(),
                      'ideal feasible probability':mass}]))
fig,ax=plt.subplots(figsize=(8,3.5))
ax.plot(np.arange(1,len(feasible)+1),p_final[feasible[order]]/mass,label='Ideal QAOA, conditional on feasible')
ax.axhline(1/len(feasible),color='gray',linestyle='--',label='Uniform feasible masks')
ax.set(xlabel='Feasible masks ranked by QUBO cost (best first)',ylabel='Conditional probability',
       title='Local simulation: does QAOA favor lower-cost masks?')
ax.set_ylim(0,2/len(feasible))
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(BASE/'qaoa_diagnostics.png',dpi=160)
display(fig); plt.close(fig)
print('Only a future hardware run can provide hardware probabilities and job timings.')
'''),md('''## 6. Switch execution to IBM hardware

Fill in the gitignored `ibm_credentials.local.json` beside this notebook, then run the
connection check in README.md. The adapter loads it automatically and verifies an explicitly selected
Open Plan instance, transpiles to a real QPU, adds measurements to the logical mask register,
and saves the hardware circuit and job identifiers.

Use `workflow.py hardware` to advance one job or collect the saved pending job. Eight jobs
maximum: one replay, six optimizer evaluations, one final measurement. The local allowance is
600 seconds with 120 reserved; each job requests a 60-second server execution ceiling.
QPU usage, queue-inclusive observed time, and classical timings are distinct quantities.
'''),code('''hardware_path=BASE/'runs'/'hardware'/'state.json'
if not hardware_path.exists():
    print('HARDWARE STATUS: not started. No hardware result is being claimed.')
else:
    hardware=json.loads(hardware_path.read_text())
    print('Hardware phase:',hardware['phase'],'complete:',hardware['complete'])
    display(pd.DataFrame(hardware['history']))
    comparison_path=BASE/'runs'/'hardware'/'final_comparison.csv'
    if comparison_path.exists():
        display(pd.read_csv(comparison_path))
    if (BASE/'hardware_results.png').exists():
        display(Image(filename=str(BASE/'hardware_results.png')))
    ledger=json.loads((BASE/'runs'/'hardware'/'ledger.json').read_text())
    display(pd.DataFrame([{'stage':j['tag'],'job ID':j['job_id'],'state':j['state'],
                          'reported QPU seconds':j.get('reported_usage_seconds')} for j in ledger['jobs']]))
'''),md('''## 7. Present only what the evidence supports

1. Circuit correctness: exact encoding, parameter binding and bit-order checks.
2. Hardware execution: device/job IDs, transpiled circuits, counts and usage from actual jobs.
3. Useful pruning: validation retention, final test accuracy and retained parameters after rewinding.
4. Search effectiveness: compare against exact QUBO and matched best-of-random selection.
5. Speed: distinguish smaller-network inference speed from end-to-end pruning-search speed.

Current saved labels were collected before this hardware experiment. Their historical runtime
was not broken into every stage. This rehearsal and a future hardware run alone cannot establish
an exact end-to-end quantum speedup. The short optimizer budget, weak loss calibration and one
chosen initialization must be disclosed.

The helper files are `circuit_core.py` (circuit/scoring), `ibm_backend.py` (actual IBM jobs), and
`workflow.py` (optimization/retraining). See README.md for the executable commands.
''')]
cells[0].source=cells[0].source.replace('HARDWARE_STATUS_PLACEHOLDER',hardware_status)
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (Qismet)','language':'python'}})
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
nbformat.write(nb,out/'Hardware_Walkthrough.ipynb')
body,_=HTMLExporter().from_notebook_node(nb)
(out/'Hardware_Walkthrough.html').write_text(body,encoding='utf-8')
print('Executed walkthrough created; no network or QPU calls.')
