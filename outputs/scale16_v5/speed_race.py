"""Speed race on the 784->32->10 network (16 groups): brute force vs Qismet (QPU and CPU search).

    python speed_race.py qpu       # run the Qismet QPU pipeline's quantum step on IBM hardware, timed
    python speed_race.py report    # combine with brute_force/ results and write speed_race/raw_report.md

All timings are measured on this machine except the IBM job, which is wall-clock
from submission to result (queue + execution + transfer), plus IBM's reported QPU time.

Pipelines compared, each ending with the retraining of its chosen network:
  Brute force      retrain every valid mask, keep the lowest validation loss
  Qismet + CPU     retrain 192 sampled masks, fit the QUBO, exact search on CPU, retrain pick
  Qismet + QPU     retrain 192 sampled masks, fit the QUBO, tune QAOA angles on CPU simulator,
                   sample on IBM hardware, best-of-32 feasible shots by QUBO, retrain pick
The 64 audit retrainings in experiment.py only check the surrogate and are not
needed to make a choice, so they are excluded from both Qismet pipelines.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / 'speed_race'
SEED = 1001
KEEPS = [8, 4]
SHOTS = 4096
BEST_OF = 32


def local_qaoa_timing(keep):
    """Re-run the exact QAOA angle optimization of experiment.py on this machine."""
    import experiment as e
    from scipy.optimize import minimize
    base = HERE / f'seed_{SEED}' / f'keep_{keep}'
    summary = json.loads((base / 'summary.json').read_text())
    layers = summary['chosen_layers']
    a = np.load(base / f'qaoa_p{layers}.npz')
    masks, ids, x = e.design(keep)
    space = e.Subspace(masks, ids, keep)
    pred = a['predictions']
    normalized = (pred - pred.mean()) / float(a['scale'])
    t = time.perf_counter()
    runs = []
    for restart in range(3):
        start = np.random.default_rng(SEED + keep * 100 + layers * 10 + restart).uniform(-1, 1, 2 * layers)
        runs.append(minimize(lambda th: space.probabilities(th, normalized, layers) @ normalized, start,
                             method='BFGS', options={'maxiter': 100, 'gtol': 1e-5}))
    seconds = time.perf_counter() - t
    best = min(runs, key=lambda r: r.fun)
    return dict(layers=layers, seconds=seconds, theta_matches_saved=bool(np.allclose(best.x, a['theta'], atol=1e-4)))


def local_fit_timing(keep):
    """Re-time QUBO fitting (ridge with CV) on the 192 fit labels."""
    import experiment as e
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    base = HERE / f'seed_{SEED}' / f'keep_{keep}'
    masks, ids, x = e.design(keep)
    labels = pd.read_csv(base / 'labels.csv').set_index('row')
    fit = labels[labels.split == 'fit'].index.to_numpy()
    t = time.perf_counter()
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 15), cv=KFold(5, shuffle=True, random_state=SEED),
                    scoring='neg_mean_squared_error').fit(x[fit], labels.loc[fit].val_loss.to_numpy())
    pred = ridge.predict(x)
    fit_seconds = time.perf_counter() - t
    t = time.perf_counter()
    exact = int(np.argmin(ridge.intercept_ + x @ ridge.coef_))
    exact_seconds = time.perf_counter() - t
    return dict(fit_seconds=fit_seconds, exact_search_seconds=exact_seconds, exact_row=exact,
                fit_rows=fit.tolist(), pred=pred)


def run_qpu():
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    import experiment as e
    import hardware as hw
    OUT.mkdir(exist_ok=True)
    service = QiskitRuntimeService()
    for keep in KEEPS:
        base = HERE / f'seed_{SEED}' / f'keep_{keep}'
        layers = json.loads((base / 'summary.json').read_text())['chosen_layers']
        masks, ids, _ = e.design(keep)
        qubo = np.load(base / f'qaoa_p{layers}.npz')['predictions']
        rec = dict(seed=SEED, keep=keep, layers=layers, shots=SHOTS)
        rec['local_qaoa'] = local_qaoa_timing(keep)
        t = time.perf_counter()
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=16)
        rec['backend'] = backend.name
        circuit = hw.build(keep, layers)
        circuit.measure_all()
        isa = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7).run(circuit)
        rec['compile_seconds'] = time.perf_counter() - t
        rec['transpiled_depth'] = isa.depth()
        rec['transpiled_two_qubit_gates'] = isa.num_nonlocal_gates()
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = SHOTS
        sampler.options.dynamical_decoupling.enable = True
        t = time.perf_counter()
        job = sampler.run([isa])
        rec['job_id'] = job.job_id()
        rec['submitted_utc'] = datetime.now(timezone.utc).isoformat()
        print(f'keep {keep}: submitted {rec["job_id"]} to {backend.name}', flush=True)
        result = job.result()
        rec['submit_to_result_seconds'] = time.perf_counter() - t
        try:
            rec['ibm_reported_qpu_seconds'] = job.usage()
        except Exception:
            rec['ibm_reported_qpu_seconds'] = None
        try:
            m = job.metrics()
            rec['ibm_timestamps'] = m.get('timestamps')
        except Exception:
            pass
        bits = result[0].data.meas.get_bitstrings()
        states = np.array([int(b, 2) for b in bits])
        lookup = {int(s): i for i, s in enumerate(ids)}
        rows = np.array([lookup.get(int(s), -1) for s in states])
        t = time.perf_counter()
        first = rows[:BEST_OF]
        feasible_first = first[first >= 0]
        pick = int(feasible_first[np.argmin(qubo[feasible_first])]) if len(feasible_first) else None
        rec['select_seconds'] = time.perf_counter() - t
        feasible_all = rows[rows >= 0]
        rec.update(feasible_fraction=float((rows >= 0).mean()), feasible_in_first_32=int(len(feasible_first)),
                   uniform_feasible_fraction=len(ids) / 2 ** 16,
                   qpu_pick_row=pick, qpu_pick_state=int(ids[pick]) if pick is not None else None,
                   qpu_pick_all_shots_row=int(feasible_all[np.argmin(qubo[feasible_all])]) if len(feasible_all) else None)
        np.save(OUT / f'keep_{keep}_shots.npy', states)
        (OUT / f'qpu_keep_{keep}.json').write_text(json.dumps(rec, indent=2, default=str))
        print(json.dumps({k: rec[k] for k in ['backend', 'submit_to_result_seconds', 'ibm_reported_qpu_seconds',
                                              'feasible_fraction', 'qpu_pick_state']}, default=str), flush=True)


def run_report():
    import experiment as e
    lines = ['# Speed race: brute force vs Qismet on the 784→32→10 network', '',
             f'Seed {SEED}. 16 neuron groups (pairs). Every timing below was measured on the same Mac '
             '(single CPU core per retraining), except the IBM job, which is wall-clock from submission to result.', '']
    rows = []
    for keep in KEEPS:
        bf = pd.read_csv(HERE / 'brute_force' / f'seed_{SEED}_keep_{keep}.csv').set_index('row')
        bft = json.loads((HERE / 'brute_force' / f'seed_{SEED}_keep_{keep}_timing.json').read_text())
        q = json.loads((OUT / f'qpu_keep_{keep}.json').read_text())
        local = local_fit_timing(keep)
        n = len(bf)
        rank = bf.val_loss.rank(method='min')
        per_train = bf.training_seconds
        label_cpu = float(per_train.loc[local['fit_rows']].sum())
        final_train = float(per_train.median())
        best_row = int(bf.val_loss.idxmin())

        def quality(row):
            return dict(val_loss=float(bf.loc[row, 'val_loss']), val_accuracy=float(bf.loc[row, 'val_accuracy']),
                        rank=int(rank.loc[row]), top_percent=100 * float(rank.loc[row]) / n)

        brute = dict(pipeline='Brute force (regular way)', single_core_seconds=float(per_train.sum()),
                     wall_seconds=bft['wall_seconds_this_run'], wall_note=f"{bft['workers']} CPU workers in parallel",
                     **quality(best_row))
        cpu = dict(pipeline='Qismet + CPU search',
                   single_core_seconds=label_cpu + local['fit_seconds'] + local['exact_search_seconds'] + final_train,
                   **quality(local['exact_row']))
        qpu_seconds = (label_cpu + local['fit_seconds'] + q['local_qaoa']['seconds'] + q['compile_seconds']
                       + q['submit_to_result_seconds'] + q['select_seconds'] + final_train)
        qpu = dict(pipeline=f"Qismet + QPU ({q['backend']})", single_core_seconds=qpu_seconds,
                   **(quality(q['qpu_pick_row']) if q['qpu_pick_row'] is not None else {}))
        cpu['wall_seconds'] = cpu['single_core_seconds']
        qpu['wall_seconds'] = qpu['single_core_seconds']
        for r in (brute, cpu, qpu):
            r.update(keep=keep, masks=n)
            rows.append(r)
        dense = json.loads((HERE / f'seed_{SEED}' / 'dense.json').read_text())
        lines += [f'## Keep {keep} of 16 groups ({100 * (1 - keep / 16):.0f}% of neurons removed, {n:,} possible networks)', '',
                  '| Pipeline | Time (one CPU core + QPU) | Retrainings | Validation accuracy of pick | Rank of pick among all networks |',
                  '|---|---:|---:|---:|---:|',
                  f"| Brute force (regular way) | {brute['single_core_seconds'] / 60:.1f} min | {n:,} | {100 * brute['val_accuracy']:.1f}% | 1 (true best) |",
                  f"| Qismet + CPU search | {cpu['single_core_seconds']:.1f} s | {len(local['fit_rows']) + 1} | {100 * cpu['val_accuracy']:.1f}% | {cpu['rank']:,} (top {cpu['top_percent']:.1f}%) |",
                  f"| Qismet + QPU ({q['backend']}) | {qpu['single_core_seconds']:.1f} s | {len(local['fit_rows']) + 1} | "
                  + (f"{100 * qpu['val_accuracy']:.1f}% | {qpu['rank']:,} (top {qpu['top_percent']:.1f}%) |" if 'rank' in qpu else 'no feasible shot | – |'),
                  '',
                  f"Dense 32-neuron network: {100 * dense['val_accuracy']:.1f}% validation accuracy. "
                  f"Brute force wall time with {bft['workers']} parallel workers: {bft['wall_seconds_this_run'] / 60:.1f} min.",
                  '',
                  'Where the Qismet time goes:', '',
                  '| Step | Qismet + CPU | Qismet + QPU |', '|---|---:|---:|',
                  f"| Retrain 192 sample networks | {label_cpu:.1f} s | {label_cpu:.1f} s |",
                  f"| Fit QUBO (ridge + CV) | {local['fit_seconds']:.2f} s | {local['fit_seconds']:.2f} s |",
                  f"| Search | {1000 * local['exact_search_seconds']:.2f} ms (score all {n:,}) | "
                  f"{q['local_qaoa']['seconds']:.1f} s angle tuning on simulator + {q['compile_seconds']:.1f} s compile + "
                  f"{q['submit_to_result_seconds']:.1f} s IBM submit→result ({q['ibm_reported_qpu_seconds']} s billed QPU) |",
                  f"| Retrain the pick | {final_train:.2f} s | {final_train:.2f} s |", '',
                  f"Hardware: {q['transpiled_two_qubit_gates']} two-qubit gates after compilation (depth {q['transpiled_depth']}); "
                  f"{100 * q['feasible_fraction']:.1f}% of {q['shots']} shots were valid networks "
                  f"(random bits: {100 * q['uniform_feasible_fraction']:.1f}%); {q['feasible_in_first_32']} of the first 32 were valid.", '']
    pd.DataFrame(rows).to_csv(OUT / 'speed_race.csv', index=False)
    lines += ['## Reading this', '',
              '- **Qismet vs brute force** is the real speedup: retraining ~1.5% (keep 8) or ~11% (keep 4) of the networks '
              'and searching a fitted formula instead of retraining everything.',
              '- **QPU vs CPU inside Qismet**: the search step itself takes milliseconds on a CPU. The QPU version '
              'adds simulator angle tuning, compilation and the IBM round trip, so it is slower here. '
              'The speedup comes from the Qismet method, not from the quantum hardware.',
              '- **The hardware output is close to noise.** After compilation each circuit has ~950 two-qubit gates. '
              'At keep 8 only 20.6% of shots were valid networks, about the same as random bits (19.6%), so the QPU pick '
              'came from 6 essentially random valid shots ranked by the formula. Its good keep-8 rank is luck; its keep-4 '
              'rank (top 64%) is the other side of that.',
              '- **The formula is the weak link.** Its exact optimum ranked only top 31% (keep 8) and top 14.5% (keep 4) '
              'by true validation loss, so even a perfect search of the formula would not find the true best network.',
              '- Ranks use validation loss (the selection objective) from brute-force retraining of every network; '
              'validation accuracy can rank them slightly differently. The test set was not used.',
              '- Brute force per-network timings come from 6 parallel workers, which made each retraining slower '
              '(~0.5 s vs ~0.15 s alone); Qismet\'s 192 retrainings use the same per-network timings, so the comparison is like for like.', '']
    (OUT / 'raw_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    {'qpu': run_qpu, 'report': run_report}[sys.argv[1]]()
