"""Run Qismet's final QAOA sampling circuits on real IBM quantum hardware.

The QAOA angles were already optimized on the noiseless simulator
(results/seed_*/qaoa_keep_*.json). Optimizing them on hardware would need
hundreds of queued jobs, so this script runs only the final sampling step:
the same 8-qubit circuit with the same optimized angles, measured on a device.

For each of 5 seeds x 2 pruning budgets (10 circuits, one job) it compares the
hardware distribution with the ideal simulated one, and repeats the original
selection rule (best surrogate among feasible masks in the first 32 shots).
Every feasible mask already has a measured retraining label in labels.csv, so
the hardware pick's real validation loss/accuracy is reported without retraining.

    python run_on_ibm.py                 # submit, wait, analyze
    python run_on_ibm.py --job-id <id>   # re-fetch a finished job and analyze
    python run_on_ibm.py --backend ibm_fez --shots 4096

Credentials come from ~/.qiskit/qiskit-ibm.json (see ../../ibm_setup.py).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
SEEDS = [11, 22, 33, 44, 55]
KEEPS = [4, 2]
PROTOCOL_SHOTS = 32  # shots used for selection in the original experiment


def build_circuit(q, seed):
    """Rebuild exactly the circuit in qaoa_utils.solve_qaoa with its optimized angles."""
    n, reps, scale = q['n'], q['reps'], q['scale']
    h = np.array(q['ising']['h'])
    J = np.array(q['ising']['J'])
    params = np.array(q['parameters'])
    gammas, betas = params[:reps], params[reps:]
    qc = QuantumCircuit(n, name=f"seed{seed}_keep{q['keep']}")
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n):
            qc.rz(2 * gamma * h[i] / scale, i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * gamma * J[i, j] / scale, i, j)
        qc.rx(2 * beta, range(n))
    qc.measure_all()
    return qc


def load_problems():
    problems = []
    for seed in SEEDS:
        labels = pd.read_csv(RESULTS / f'seed_{seed}' / 'labels.csv').set_index('state')
        for keep in KEEPS:
            q = json.loads((RESULTS / f'seed_{seed}' / f'qaoa_keep_{keep}.json').read_text())
            problems.append(dict(seed=seed, keep=keep, q=q, labels=labels, circuit=build_circuit(q, seed)))
    return problems


def analyze(problems, sample_lists, meta):
    rows = []
    for p, samples in zip(problems, sample_lists):
        q, labels, keep = p['q'], p['labels'], p['keep']
        n = q['n']
        masks = np.array(q['basis_masks'])
        feasible = masks.sum(1) == keep
        surrogate = np.array(q['surrogate_energies'])
        ideal = np.array(q['probabilities'])
        idx = np.array([int(b, 2) for b in samples])  # Qiskit bitstring -> basis index (qubit i = bit i)
        counts = np.bincount(idx, minlength=2 ** n)
        hw = counts / counts.sum()
        exact_min = surrogate[feasible].min()
        optimum = feasible & np.isclose(surrogate, exact_min, atol=1e-10)
        tail_thr = np.quantile(surrogate[feasible], 0.1)
        tail = feasible & (surrogate <= tail_thr + 1e-12)
        feas_mass = hw[feasible].sum()
        # Original selection rule applied to the first 32 hardware shots.
        first = idx[:PROTOCOL_SHOTS]
        feas_first = np.unique(first[feasible[first]])
        pick32 = int(feas_first[np.argmin(surrogate[feas_first])]) if len(feas_first) else None
        # Same rule with every hardware shot.
        feas_all = np.flatnonzero((counts > 0) & feasible)
        pick_all = int(feas_all[np.argmin(surrogate[feas_all])]) if len(feas_all) else None
        exact_state = int(np.flatnonzero(optimum)[0])
        sim_state = int(q['selected_index'])

        def label(state, key):
            return float(labels.loc[state, key]) if state is not None and state in labels.index else np.nan

        rows.append(dict(
            seed=p['seed'], keep_groups=keep, pruning=100 * (1 - keep / n), shots=int(counts.sum()),
            ideal_feasible_probability=float(ideal[feasible].sum()),
            hardware_feasible_probability=float(feas_mass),
            uniform_feasible_probability=float(feasible.mean()),
            ideal_optimum_probability=float(ideal[optimum].sum()),
            hardware_optimum_probability=float(hw[optimum].sum()),
            uniform_optimum_probability=float(optimum.mean()),
            hardware_top_decile_conditional=float(hw[tail].sum() / feas_mass) if feas_mass else np.nan,
            ideal_top_decile_conditional=float(ideal[tail].sum() / ideal[feasible].sum()),
            uniform_top_decile_conditional=float(tail.sum() / feasible.sum()),
            total_variation_ideal_vs_hardware=float(0.5 * np.abs(hw - ideal).sum()),
            feasible_in_first_32=int(feasible[first].sum()),
            hardware_pick_32=pick32, hardware_pick_all=pick_all, simulator_pick=sim_state, exact_qubo=exact_state,
            hardware_pick_32_mask=''.join(map(str, masks[pick32])) if pick32 is not None else '',
            hardware_pick_32_matches_simulator=pick32 == sim_state,
            hardware_pick_32_is_exact=pick32 == exact_state,
            hardware_pick_32_surrogate_gap=float(surrogate[pick32] - exact_min) if pick32 is not None else np.nan,
            simulator_surrogate_gap=float(surrogate[sim_state] - exact_min),
            hardware_pick_32_val_loss=label(pick32, 'val_loss'),
            hardware_pick_32_val_accuracy=label(pick32, 'val_accuracy'),
            simulator_pick_val_loss=label(sim_state, 'val_loss'),
            simulator_pick_val_accuracy=label(sim_state, 'val_accuracy'),
            exact_qubo_val_loss=label(exact_state, 'val_loss'),
            exact_qubo_val_accuracy=label(exact_state, 'val_accuracy'),
            best_possible_val_loss=float(labels.loc[np.flatnonzero(feasible)].val_loss.min()),
        ))
        (HERE / 'counts').mkdir(exist_ok=True)
        (HERE / 'counts' / f"seed_{p['seed']}_keep_{keep}.json").write_text(json.dumps(dict(
            seed=p['seed'], keep=keep, backend=meta['backend'], job_id=meta['job_id'],
            shot_order_basis_indices=idx.tolist(),
            counts={format(i, f'0{n}b'): int(c) for i, c in enumerate(counts) if c},
            ideal_probabilities=ideal.tolist()), indent=1))
    table = pd.DataFrame(rows)
    table.to_csv(HERE / 'hardware_vs_simulator.csv', index=False)
    write_report(table, meta)
    return table


def write_report(t, meta):
    g = t.groupby('pruning')
    lines = [
        '# QAOA on IBM quantum hardware',
        '',
        f"Backend **{meta['backend']}** ({meta['num_qubits']} qubits), job `{meta['job_id']}`, "
        f"{meta['shots']} shots per circuit, run {meta['submitted_utc']} UTC.",
        '',
        'The QAOA angles come from the original noiseless simulator run. Only the final sampling '
        'circuits (8 qubits, 2 QAOA layers, 56 RZZ gates before compilation) ran on hardware: one job, 10 circuits '
        '(5 seeds × 2 pruning levels). Selection repeats the original rule: the lowest-surrogate '
        'feasible mask among the first 32 shots. Validation numbers come from the existing '
        'retraining labels in `results/seed_*/labels.csv`; no network was retrained and the test set was not used.',
        '',
        '## Sampling quality (mean of 5 seeds)',
        '',
        '| Neurons removed | Feasible prob. ideal | hardware | uniform | Optimum prob. ideal | hardware | uniform | TV distance ideal vs hardware |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for pr, d in g:
        lines.append(f"| {pr:.0f}% | {d.ideal_feasible_probability.mean():.3f} | {d.hardware_feasible_probability.mean():.3f} | "
                     f"{d.uniform_feasible_probability.mean():.3f} | {d.ideal_optimum_probability.mean():.3f} | "
                     f"{d.hardware_optimum_probability.mean():.3f} | {d.uniform_optimum_probability.mean():.3f} | "
                     f"{d.total_variation_ideal_vs_hardware.mean():.3f} |")
    lines += [
        '',
        '"Feasible" means the measured mask keeps exactly the required number of groups. '
        '"Uniform" is what random guessing over all 256 bitstrings would give.',
        '',
        '## Selected masks (32-shot rule)',
        '',
        '| Neurons removed | Seed | Hardware pick | = simulator pick | = exact QUBO | Surrogate gap hw / sim | Val. loss hw / sim / exact / best possible |',
        '|---:|---:|---|:-:|:-:|---:|---:|',
    ]
    for _, r in t.sort_values(['pruning', 'seed']).iterrows():
        lines.append(f"| {r.pruning:.0f}% | {r.seed} | `{r.hardware_pick_32_mask}` | {'yes' if r.hardware_pick_32_matches_simulator else 'no'} | "
                     f"{'yes' if r.hardware_pick_32_is_exact else 'no'} | {r.hardware_pick_32_surrogate_gap:.4f} / {r.simulator_surrogate_gap:.4f} | "
                     f"{r.hardware_pick_32_val_loss:.4f} / {r.simulator_pick_val_loss:.4f} / {r.exact_qubo_val_loss:.4f} / {r.best_possible_val_loss:.4f} |")
    lines += ['', '## Mean validation accuracy of the selected mask', '',
              '| Neurons removed | Hardware pick | Simulator pick | Exact QUBO |', '|---:|---:|---:|---:|']
    for pr, d in g:
        lines.append(f"| {pr:.0f}% | {100 * d.hardware_pick_32_val_accuracy.mean():.2f}% | "
                     f"{100 * d.simulator_pick_val_accuracy.mean():.2f}% | {100 * d.exact_qubo_val_accuracy.mean():.2f}% |")
    lines += ['', 'Mask strings list groups G1…G8 left to right (1 = kept). This is one hardware run; '
              'noise, compilation and calibration vary between runs and devices. It is not evidence of quantum advantage: '
              'with only 70 or 28 feasible masks, exhaustive classical search is instant.', '']
    (HERE / 'README.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', help='default: least busy operational device')
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--job-id', help='analyze an already submitted job')
    args = ap.parse_args()

    problems = load_problems()
    service = QiskitRuntimeService()
    if args.job_id:
        job = service.job(args.job_id)
        meta = json.loads((HERE / 'job_info.json').read_text())
    else:
        backend = service.backend(args.backend) if args.backend else service.least_busy(operational=True, simulator=False)
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
        isa = [pm.run(p['circuit']) for p in problems]
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = args.shots
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = 'XY4'
        job = sampler.run(isa)
        meta = dict(job_id=job.job_id(), backend=backend.name, num_qubits=backend.num_qubits, shots=args.shots,
                    submitted_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),
                    transpiled=[dict(name=c.name, depth=c.depth(), two_qubit_gates=c.num_nonlocal_gates(),
                                     physical_qubits=sorted({c.find_bit(q).index for inst in c.data for q in inst.qubits
                                                             if inst.operation.name != 'barrier'}))
                                for c in isa])
        (HERE / 'job_info.json').write_text(json.dumps(meta, indent=1))
        print(f"Submitted job {meta['job_id']} to {meta['backend']}; saved job_info.json", flush=True)
    t0 = time.time()
    while True:
        status = job.status()
        status = getattr(status, 'name', status)
        if status in ('DONE', 'ERROR', 'CANCELLED'):
            break
        print(f'  status {status} ({time.time() - t0:.0f}s)', flush=True)
        time.sleep(15)
    if status != 'DONE':
        raise SystemExit(f'Job ended with status {status}: {job.error_message() if hasattr(job, "error_message") else ""}')
    result = job.result()
    sample_lists = [r.data.meas.get_bitstrings() for r in result]
    try:
        meta['quantum_seconds'] = job.usage()
    except Exception:
        pass
    (HERE / 'job_info.json').write_text(json.dumps(meta, indent=1))
    table = analyze(problems, sample_lists, meta)
    print(table[['seed', 'pruning', 'hardware_feasible_probability', 'ideal_feasible_probability',
                 'hardware_pick_32_mask', 'hardware_pick_32_matches_simulator', 'hardware_pick_32_val_accuracy']].to_string())
    print('wrote', HERE / 'README.md')


if __name__ == '__main__':
    main()
