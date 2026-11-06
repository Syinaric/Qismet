"""Build an evidence-backed report from completed IBM jobs, never credentials."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]/'outputs'/'hardware_v3'
sys.path.insert(0,str(HERE))
from workflow import problem_and_warmstart
from circuit_core import cost,ideal_probabilities

folder=HERE/'runs'/'hardware'
state=json.loads((folder/'state.json').read_text())
ledger=json.loads((folder/'ledger.json').read_text())
assert state['mode']=='hardware' and state['complete'] and not ledger['test_mode']
assert len(ledger['jobs'])==8 and all(j['state']=='complete' for j in ledger['jobs'])
comparison=pd.read_csv(folder/'final_comparison.csv')
problem,_,_=problem_and_warmstart()
counts=state['final_counts']; shots=sum(counts.values())
observed=np.zeros(256)
for bits,count in counts.items(): observed[int(bits,2)]=count/shots
ideal=ideal_probabilities(problem,state['history'][-1]['parameters'],reps=2)
masks=np.array([[(i>>j)&1 for j in range(8)] for i in range(256)])
feasible=np.flatnonzero(masks.sum(1)==4)
costs=np.array([cost(problem,masks[i],penalized=False) for i in feasible])
order=np.argsort(costs)
selected=cost(problem,state['selected_mask'],penalized=False)
usage=sum(j['reported_usage_seconds'] for j in ledger['jobs'])
elapsed=ledger['jobs'][-1]['finished_observed_epoch']-ledger['jobs'][0]['submitted_epoch']
jobs=pd.DataFrame([{'tag':j['tag'],'job_id':j['job_id'],'status':j['remote_status'],
                   'shots':j['shots'],'qpu_seconds':j['reported_usage_seconds']} for j in ledger['jobs']])
jobs.to_csv(folder/'job_summary.csv',index=False)
diagnostics={'backend':ledger['backend'],'jobs':len(jobs),'shots':int(jobs.shots.sum()),
             'qpu_seconds':usage,'first_submission_to_last_completion_observation_seconds':elapsed,
             'final_feasible_fraction':float(observed[feasible].sum()),
             'ideal_final_feasible_probability':float(ideal[feasible].sum()),
             'final_distribution_total_variation_from_ideal':float(np.abs(observed-ideal).sum()/2),
             'selected_mask':state['selected_mask'],'selected_qubo_cost':selected,
             'exact_feasible_qubo_cost':float(costs.min()),'selected_qubo_gap':float(selected-costs.min()),
             'quantum_advantage_established':False,
             'scope':'One initialization, simulator warm start and three hardware SPSA steps. Finite-shot TVD combines sampling error with device/transpilation differences.'}
(folder/'hardware_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(np.arange(len(comparison)),comparison.test_accuracy*100,color=['#777777','#2563eb','#059669','#d97706'])
axes[0].set(xticks=np.arange(len(comparison)),xticklabels=['Dense','QAOA','Exact QUBO','Best 32 random'],
            ylim=(0,100),ylabel='Test accuracy (%)',title='Real hardware selects a mask; CPU retrains it')
for i,row in comparison.iterrows(): axes[0].text(i,row.test_accuracy*100+1,f'{row.test_accuracy*100:.2f}%',ha='center',fontsize=9)
mass=observed[feasible].sum(); imass=ideal[feasible].sum()
axes[1].plot(np.arange(1,71),observed[feasible[order]]/mass,label='IBM final samples',alpha=.8)
axes[1].plot(np.arange(1,71),ideal[feasible[order]]/imass,label='Ideal at same angles')
axes[1].axhline(1/70,color='gray',ls='--',label='Uniform feasible')
axes[1].set(xlabel='Feasible masks ranked by QUBO cost (best first)',ylabel='Conditional probability',
            title='Final distribution; finite-shot noise is included',ylim=(0,max(.04,observed[feasible].max()/mass*1.1)))
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE/'hardware_results.png',dpi=170); plt.close(fig)

table='| Method | Parameters | Validation accuracy | Test accuracy | Validation drop |\n|---|---:|---:|---:|---:|\n'
for _,row in comparison.iterrows():
    table+=f"| {row.method} | {row.parameters:,} | {row.val_accuracy*100:.2f}% | {row.test_accuracy*100:.2f}% | {row.validation_drop_pp:.2f} pp |\n"
jobtable='| Stage | IBM job ID | Shots | QPU seconds |\n|---|---|---:|---:|\n'
for _,row in jobs.iterrows(): jobtable+=f'| {row.tag} | `{row.job_id}` | {row.shots} | {row.qpu_seconds:g} |\n'
text=f'''# Qismet: completed IBM hardware experiment

This experiment ran eight real jobs on **{ledger['backend']}**, using **{usage:g} seconds** of reported QPU usage and **{int(jobs.shots.sum()):,} shots**. No simulator substituted for hardware. IBM job IDs and raw results are saved below.

The classifier is the existing Fashion-MNIST 784→16→10 network. Each of eight qubits represents a pair of hidden neurons. This run keeps four pairs, uses the original initialization, and retrains each final network for the same 20 epochs on the CPU. Only seed 11 is used here.

{table}

![Hardware results](hardware_results.png)

## How the quantum algorithm was used

The fitted QUBO receives a fixed-size penalty, then maps exactly to an Ising cost operator. Two alternating cost/mixer layers form QAOA. The first hardware job replays saved simulator-optimized angles. Three SPSA steps use six additional hardware evaluations. A separate final job samples the best observed angles. The primary mask is selected by QUBO score from 32 seeded draws from those final measurements. Classical best-of-32 draws from feasible masks; QAOA's 32 draws can include infeasible outcomes. All 2,048 final samples remain available for diagnostics.

The selected mask is `{state['selected_mask']}` (1 retains a pair). Its QUBO cost is **{selected:.6f}**, versus an exact feasible minimum of **{costs.min():.6f}**, giving a gap of **{selected-costs.min():.6f}**. This gap measures QUBO optimization only. Earlier held-out surrogate tests showed poor absolute loss calibration, so the QUBO must not be described as an accurate loss predictor.

Final feasible fraction: **{mass:.2%}** on hardware, versus **{imass:.2%}** in the ideal circuit at the same angles. Full-distribution total variation is **{diagnostics['final_distribution_total_variation_from_ideal']:.3f}**; finite-shot sampling contributes to this number. It is not a pure device-error measurement.

## Timing and claims

Reported QPU usage totals **{usage:g} seconds**. First submission to last observed completion took **{elapsed:.1f} seconds**, including queue, client preparation between jobs and polling delay. The original simulator warm start and historical surrogate-label generation happened earlier and are excluded from this elapsed interval. Their costs must be included in any new end-to-end benchmark.

This establishes hardware execution and measures pruning quality for one chosen initialization. It does **not establish quantum speedup or superiority over classical selection**. Classical exact QUBO search has only 70 feasible masks in this experiment. Raw CPU inference/retraining timings and classical search timings are in `runs/hardware/final_comparison.csv` and `timing_scope.json`; tiny CPU timings are noisy and separate from pruning-search speed.

## Verifiable hardware evidence

{jobtable}

Each job directory in `runs/hardware/` contains logical/transpiled circuits, a measurement map, counts, IBM metrics and reported usage. `ledger.json` preserves budget accounting. The 600-second allowance retained at least a 120-second reservation buffer; no additional jobs were submitted beyond this protocol.
'''
(HERE/'Hardware_Results.md').write_text(text,encoding='utf-8')
print(json.dumps(diagnostics,indent=2)); print(comparison.to_string(index=False))
