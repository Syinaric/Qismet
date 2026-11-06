"""Local search timing on the exact revised QUBO. No IBM calls or retraining."""
from pathlib import Path
from itertools import combinations
import json,time,platform
import numpy as np

out=Path(__file__).resolve().parents[1]/'outputs'/'constrained_v4'
a=np.load(out/'pilot_p3_seed_11.npz')
linear=a['linear']; pair=np.zeros((8,8)); pair[np.triu_indices(8,1)]=a['pairs']; constant=float(a['constant'])

def exact():
    # Include constructing all valid masks, evaluating them and finding the winner.
    masks=np.zeros((70,8),dtype=int)
    for row,indices in enumerate(combinations(range(8),4)): masks[row,list(indices)]=1
    scores=constant+masks@linear+np.einsum('bi,ij,bj->b',masks,pair,masks)
    best=int(scores.argmin())
    return int(masks[best]@(2**np.arange(8))),float(scores[best])

rng=np.random.default_rng(20261108)
def random32():
    # Independent uniform feasible draws, with replacement; include generation/scoring.
    choices=np.argsort(rng.random((32,8)),axis=1)[:,:4]
    masks=np.zeros((32,8),dtype=int); masks[np.arange(32)[:,None],choices]=1
    scores=constant+masks@linear+np.einsum('bi,ij,bj->b',masks,pair,masks)
    return int(masks[scores.argmin()]@(2**np.arange(8)))

results={}
for name,fn in [('exact_70_feasible_masks',exact),('best_of_32_random_feasible_masks',random32)]:
    for _ in range(10): fn()
    samples=[]
    for _ in range(200):
        start=time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns()-start)/1e6)
    results[name]={'median_ms':float(np.median(samples)),'p10_ms':float(np.quantile(samples,.1)),
                   'p90_ms':float(np.quantile(samples,.9)),'repetitions':len(samples)}
state,score=exact()
assert state==60
ledger=json.loads((out/'hardware'/'ledger.json').read_text())
summary=json.loads((out/'hardware'/'summary.json').read_text())
selected=next(j for j in ledger['jobs'] if j['tag']==f"constrained_p{summary['chosen_layers']}")
results.update({'same_qubo_exact_state':state,'same_qubo_exact_cost':score,
                'selected_hardware_job_qpu_seconds':selected['reported_usage_seconds'],
                'all_four_followup_jobs_qpu_seconds':sum(j['reported_usage_seconds'] for j in ledger['jobs']),
                'followup_submission_to_last_completion_observation_seconds':ledger['jobs'][-1]['finished_observed_epoch']-ledger['jobs'][0]['submitted_epoch'],
                'python':platform.python_version(),
                'scope':'CPU search includes valid-mask generation, score evaluation and selection. Both methods reuse the same fitted QUBO. Hardware QPU usage is a different clock from CPU elapsed time; no controlled scaling or end-to-end quantum advantage is claimed. Quantum simulator angle optimization and all shared label generation are excluded from this search-only comparison.'})
(out/'hardware'/'search_timing_benchmark.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
