"""Pre-registered 100-qubit test: does QAOA on IBM hardware find better pruned networks?

Network: Fashion-MNIST, 784 -> 400 ReLU -> 10, same data split and training recipe as
fashion_v2 (20 epochs, Adam 0.003, batch 256). The 400 hidden neurons form 200 groups
of 2. Qubit q chooses between group 2q (bit 0) and group 2q+1 (bit 1), so EVERY
100-bit string is a valid network that keeps exactly half the neurons (2^100 options).

The QUBO only couples qubits that are physically wired together on the chip, so the
QAOA circuit needs no routing (SWAP) gates. QAOA depth p=1; angles are tuned with an
exact light-cone calculation (no 100-qubit simulation needed).

Stages (run in order; each saves to disk and resumes):
    python hw100.py layout      # choose 100 physical qubits on ibm_fez, freeze qubit graph
    python hw100.py verify      # check light-cone math against full statevectors
    python hw100.py prepare     # per seed: dense net, 400 fit + 100 audit retrainings, fit QUBO, tune angles
    python hw100.py hardware    # one 4096-shot job per seed on ibm_fez
    python hw100.py baselines   # random sampling, simulated annealing (matched and long)
    python hw100.py evaluate    # retrain each method's top-10 picks, frozen before test use
    python hw100.py report      # tables, figures, verdict per PREREGISTRATION.md
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'fashion_v2'))

PROTOCOL = {
    'question': 'At an equal candidate budget, do QAOA samples from IBM hardware lead to better retrained '
                'pruned networks than classical candidate generators?',
    'dataset': 'fashion_v2 split: 5000 train / 1000 validation / 10000 test',
    'hidden': 400, 'group_size': 2, 'groups': 200, 'qubits': 100,
    'encoding': 'qubit q keeps group 2q if bit=0 else group 2q+1; all 2^100 bitstrings are valid; 50% of neurons kept',
    'training': 'fashion_v2.train: 20 epochs, Adam lr 0.003, batch 256, paired shuffles; rewind to original init',
    'seeds': [11, 22, 33],
    'fit_samples': 400, 'audit_samples': 100,
    'qubo': 'ridge (RidgeCV alphas logspace(-4,3,15), 5-fold) on 100 linear + chip-edge pairwise features; target = validation loss',
    'backend': 'ibm_fez', 'qaoa_layers': 1, 'shots': 4096,
    'angle_tuning': 'exact p=1 light-cone expectation; 41x41 grid then Nelder-Mead',
    'candidate_budget': 4096,
    'methods': {
        'QAOA (ibm_fez)': '4096 hardware shots',
        'Random sampling': '4096 uniform random bitstrings',
        'Simulated annealing (matched)': '16 chains x 256 single-flip steps = 4096 QUBO evaluations; all visited states are candidates',
        'Simulated annealing (long)': '64 chains x 20000 steps = 1,280,000 QUBO evaluations (reference ceiling); candidates = states in the final 2000 steps',
    },
    'picks_per_method': 10,
    'pick_rule': 'the 10 distinct candidates with the lowest QUBO score',
    'primary_metric': 'best-of-10 validation loss after retraining (lower is better)',
    'win_rule': 'QAOA beats a method if its best-of-10 validation loss is lower in at least 2 of 3 seeds '
                'AND its mean best-of-10 validation loss over the 3 seeds is lower',
    'secondary': ['mean validation loss of the 10 picks', 'mean QUBO score of all candidates',
                  'share of candidates better than the best of 4096 random samples',
                  'hardware vs ideal expected QUBO score', 'test accuracy of each method\'s best pick (reported once, never used to choose)'],
    'everything_reported': 'all seeds, all methods, all jobs, including failures',
    'known_caveats': [
        'The native 100-qubit region has 100 couplings (nearly a tree), so the QUBO is easy for classical solvers; '
        'the test is about candidate quality at equal budget, not about beating the best classical solver.',
        'Restricting each qubit to a choice between two fixed groups searches 2^100 of the C(200,100) possible half-size networks.',
        'p=1 QAOA is the shallowest version; it was chosen so angles can be tuned exactly without simulating 100 qubits.',
        'Three seeds and one fixed data split: a demonstration-scale test, not a general benchmark.',
    ],
}
SEEDS = PROTOCOL['seeds']
NQ = 100
_G = {}


def out(*parts):
    p = HERE.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def load(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- qubit graph
def stage_layout():
    from qiskit_ibm_runtime import QiskitRuntimeService
    path = out('layout.json')
    if path.exists():
        print('layout already frozen'); return
    backend = QiskitRuntimeService().backend(PROTOCOL['backend'])
    t = backend.target
    ro = {q: t['measure'][(q,)].error for q in range(backend.num_qubits)}
    cz = {}
    for (a, b), props in t['cz'].items():
        e = tuple(sorted((a, b)))
        cz[e] = min(cz.get(e, 1.0), props.error if props and props.error is not None else 1.0)
    good_edge = {e: v for e, v in cz.items() if v < 0.02 and ro[e[0]] < 0.06 and ro[e[1]] < 0.06}
    nbr = {}
    for a, b in good_edge:
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    best = None
    for start in sorted(nbr, key=lambda q: ro[q])[:20]:
        chosen = {start}
        while len(chosen) < NQ:
            frontier = {n for q in chosen for n in nbr[q]} - chosen
            if not frontier:
                break
            def cost(n):
                links = [good_edge[tuple(sorted((n, m)))] for m in nbr[n] if m in chosen]
                return ro[n] + min(links) - 0.002 * len(links)
            chosen.add(min(frontier, key=cost))
        if len(chosen) == NQ:
            edges = [e for e in good_edge if e[0] in chosen and e[1] in chosen]
            score = sum(ro[q] for q in chosen) + sum(good_edge[e] for e in edges)
            if best is None or score < best[0]:
                best = (score, sorted(chosen), edges)
    _, physical, edges = best
    index = {p: i for i, p in enumerate(physical)}
    virtual_edges = sorted(tuple(sorted((index[a], index[b]))) for a, b in edges)
    save(path, dict(backend=backend.name, physical_qubits=physical, virtual_edges=virtual_edges,
                    readout_error={str(p): ro[p] for p in physical},
                    cz_error={f'{a}-{b}': good_edge[(a, b)] for a, b in edges},
                    frozen_utc=datetime.now(timezone.utc).isoformat()))
    print(f'froze {len(physical)} qubits, {len(virtual_edges)} native couplings; '
          f'median readout {np.median([ro[p] for p in physical]):.4f}, median CZ {np.median([good_edge[e] for e in edges]):.4f}')


def graph():
    L = load(HERE / 'layout.json')
    edges = [tuple(e) for e in L['virtual_edges']]
    nbr = [[] for _ in range(NQ)]
    for a, b in edges:
        nbr[a].append(b)
        nbr[b].append(a)
    return L, edges, nbr


# ---------------------------------------------------------------- QUBO + Ising
def features(X, edges):
    X = np.asarray(X, dtype=float)
    return np.column_stack([X] + [X[:, a] * X[:, b] for a, b in edges])


def qubo_score(X, model, edges):
    X = np.asarray(X, dtype=float)
    lin, pair = model['linear'], model['pairs']
    s = model['constant'] + X @ lin
    for k, (a, b) in enumerate(edges):
        s = s + pair[k] * X[:, a] * X[:, b]
    return s


def to_ising(model, edges):
    """x = (1 - z) / 2.  C(x) = offset + sum h_i z_i + sum J_ab z_a z_b."""
    a, b = np.asarray(model['linear']), np.asarray(model['pairs'])
    h = -a / 2
    J = b / 4
    offset = model['constant'] + a.sum() / 2 + b.sum() / 4
    for k, (i, j) in enumerate(edges):
        h[i] -= b[k] / 4
        h[j] -= b[k] / 4
    return offset, h, J


# ---------------------------------------------------------------- exact p=1 light cone
def _apply_rx_all(psi, n, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * n)
    for ax in range(n):
        a0 = np.take(psi, 0, axis=ax)
        a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


def _cone(sites, terms_h, terms_J, gamma, beta, observe):
    """Statevector on a small set of sites. terms indexed by local position."""
    n = len(sites)
    bits = np.array(list(product([0, 1], repeat=n)))  # first site = most significant
    z = 1 - 2 * bits
    phase = np.zeros(len(bits))
    for i, hv in terms_h:
        phase += hv * z[:, i]
    for i, j, jv in terms_J:
        phase += jv * z[:, i] * z[:, j]
    psi = np.exp(-1j * gamma * phase) / np.sqrt(2 ** n)
    psi = _apply_rx_all(psi, n, beta)
    p = np.abs(psi) ** 2
    obs = np.ones(len(bits))
    for i in observe:
        obs = obs * z[:, i]
    return float(p @ obs)


def expectations(h, J, edges, nbr, gamma, beta):
    """<Z_i> for all i and <Z_a Z_b> for every coupling, exact for p=1 QAOA."""
    eid = {e: k for k, e in enumerate(edges)}
    Jd = {e: J[k] for k, e in enumerate(edges)}

    def jv(a, b):
        return Jd[tuple(sorted((a, b)))]

    ez = np.zeros(NQ)
    for i in range(NQ):
        sites = [i] + nbr[i]
        pos = {q: k for k, q in enumerate(sites)}
        ez[i] = _cone(sites, [(0, h[i])], [(0, pos[j], jv(i, j)) for j in nbr[i]], gamma, beta, [0])
    ezz = np.zeros(len(edges))
    for (a, b), k in eid.items():
        sites = sorted(set([a, b] + nbr[a] + nbr[b]))
        pos = {q: m for m, q in enumerate(sites)}
        th = [(pos[a], h[a]), (pos[b], h[b])]
        tj = {}
        for c in (a, b):
            for d in nbr[c]:
                e = tuple(sorted((c, d)))
                tj[e] = (pos[e[0]], pos[e[1]], Jd[e])
        ezz[k] = _cone(sites, th, list(tj.values()), gamma, beta, [pos[a], pos[b]])
    return ez, ezz


def expected_energy(offset, h, J, edges, nbr, gamma, beta):
    ez, ezz = expectations(h, J, edges, nbr, gamma, beta)
    return offset + h @ ez + J @ ezz


def build_circuit(h, J, edges, gamma, beta):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(NQ)
    qc.h(range(NQ))
    for i in range(NQ):
        qc.rz(2 * gamma * h[i], i)
    for k, (a, b) in enumerate(edges):
        qc.rzz(2 * gamma * J[k], a, b)
    qc.rx(2 * beta, range(NQ))
    qc.measure_all()
    return qc


def stage_verify():
    """Light cone vs exact statevector on connected 10-14 qubit pieces of the frozen graph."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    global NQ
    L, edges, nbr = graph()
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(6):
        start = int(rng.integers(NQ))
        sub = [start]
        while len(sub) < 10 + 2 * (trial % 3):
            cand = sorted({n for q in sub for n in nbr[q]} - set(sub))
            sub.append(int(rng.choice(cand)))
        idx = {q: k for k, q in enumerate(sub)}
        se = [(idx[a], idx[b]) for a, b in edges if a in idx and b in idx]
        se = [tuple(sorted(e)) for e in se]
        sn = [[] for _ in sub]
        for a, b in se:
            sn[a].append(b); sn[b].append(a)
        n = len(sub)
        h = rng.normal(size=n); J = rng.normal(size=len(se)); g, bta = rng.uniform(-1, 1, 2)
        saved, NQ = NQ, n
        try:
            ez, ezz = expectations(h, J, se, sn, g, bta)
            qc = QuantumCircuit(n); qc.h(range(n))
            for i in range(n): qc.rz(2 * g * h[i], i)
            for k, (a, b) in enumerate(se): qc.rzz(2 * g * J[k], a, b)
            qc.rx(2 * bta, range(n))
            p = Statevector.from_instruction(qc).probabilities()
            ids = np.arange(2 ** n)
            Z = 1 - 2 * ((ids[:, None] >> np.arange(n)) & 1)  # qubit i = bit i
            err = max(np.abs(p @ Z - ez).max(), max(abs(p @ (Z[:, a] * Z[:, b]) - ezz[k]) for k, (a, b) in enumerate(se)))
        finally:
            NQ = saved
        worst = max(worst, float(err))
        print(f'  {n}-qubit piece, {len(se)} couplings: max error {err:.2e}')
    assert worst < 1e-10, worst
    save(out('verify.json'), dict(max_error=worst, passed=True))
    print('light-cone verification passed')


# ---------------------------------------------------------------- training
def kept_neurons(x):
    x = np.asarray(x, dtype=int)
    base = 4 * np.arange(NQ) + 2 * x
    return np.sort(np.concatenate([base, base + 1]))


def _init(seed):
    import torch
    import fashion_experiment as f
    torch.set_num_threads(1)
    arrays = np.load(ROOT / 'fashion_v2' / 'dataset_subset.npz')
    data = {s: (torch.tensor(arrays[s + '_images'].reshape(-1, 784) / 255., dtype=torch.float32),
                torch.tensor(arrays[s + '_labels'], dtype=torch.long)) for s in ['train', 'val', 'test']}
    initial = torch.load(HERE / f'seed_{seed}' / 'original_initialization.pt', weights_only=True)
    _G.update(f=f, data=data, initial=initial, seed=seed, torch=torch)


def _retrain(job):
    key, x, with_test = job
    f, torch, init = _G['f'], _G['torch'], _G['initial']
    ix = torch.tensor(kept_neurons(x))
    net = f.FashionMLP(len(ix))
    net.load_state_dict({'fc1.weight': init['fc1.weight'][ix], 'fc1.bias': init['fc1.bias'][ix],
                         'fc2.weight': init['fc2.weight'][:, ix], 'fc2.bias': init['fc2.bias']})
    t = time.perf_counter()
    f.train(net, _G['data'], _G['seed'])
    sec = time.perf_counter() - t
    v = f.evaluate(net, _G['data']['val'])
    rec = dict(key=key, bits=''.join(map(str, x)), val_loss=v['loss'], val_accuracy=v['accuracy'], training_seconds=sec)
    if with_test:
        te = f.evaluate(net, _G['data']['test'])
        rec.update(test_loss=te['loss'], test_accuracy=te['accuracy'])
    return rec


def retrain_many(seed, jobs, workers=6):
    with Pool(workers, initializer=_init, initargs=(seed,)) as pool:
        return list(pool.imap(_retrain, jobs, chunksize=2))


def stage_prepare(workers=6):
    import torch
    import fashion_experiment as f
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from scipy.optimize import minimize
    torch.set_num_threads(4)
    L, edges, nbr = graph()
    arrays = np.load(ROOT / 'fashion_v2' / 'dataset_subset.npz')
    data = {s: (torch.tensor(arrays[s + '_images'].reshape(-1, 784) / 255., dtype=torch.float32),
                torch.tensor(arrays[s + '_labels'], dtype=torch.long)) for s in ['train', 'val', 'test']}
    for seed in SEEDS:
        folder = out(f'seed_{seed}', 'x').parent
        if (folder / 'qubo.json').exists():
            print('seed', seed, 'prepared'); continue
        timing = {}
        if not (folder / 'dense.json').exists():
            torch.manual_seed(seed)
            dense = f.FashionMLP(PROTOCOL['hidden'])
            torch.save(f.clone_state(dense), folder / 'original_initialization.pt')
            t = time.perf_counter(); f.train(dense, data, seed); sec = time.perf_counter() - t
            v, te = f.evaluate(dense, data['val']), f.evaluate(dense, data['test'])
            save(folder / 'dense.json', dict(training_seconds=sec, val_loss=v['loss'], val_accuracy=v['accuracy'],
                                             test_accuracy=te['accuracy'], parameters=sum(p.numel() for p in dense.parameters())))
        rng = np.random.default_rng(seed * 7 + 1)
        X = rng.integers(0, 2, (PROTOCOL['fit_samples'] + PROTOCOL['audit_samples'], NQ))
        lab = folder / 'labels.csv'
        if not lab.exists():
            t = time.perf_counter()
            recs = retrain_many(seed, [(i, X[i].tolist(), False) for i in range(len(X))], workers)
            timing['label_wall_seconds'] = time.perf_counter() - t
            df = pd.DataFrame(recs)
            df['split'] = np.where(df.key < PROTOCOL['fit_samples'], 'fit', 'audit')
            df.to_csv(lab, index=False)
        df = pd.read_csv(lab)
        fit, aud = df[df.split == 'fit'], df[df.split == 'audit']
        Xf, Xa = X[fit.key.to_numpy()], X[aud.key.to_numpy()]
        t = time.perf_counter()
        ridge = RidgeCV(alphas=np.logspace(-4, 3, 15), cv=KFold(5, shuffle=True, random_state=seed),
                        scoring='neg_mean_squared_error').fit(features(Xf, edges), fit.val_loss)
        timing['fit_seconds'] = time.perf_counter() - t
        model = dict(constant=float(ridge.intercept_), linear=ridge.coef_[:NQ].tolist(), pairs=ridge.coef_[NQ:].tolist(),
                     alpha=float(ridge.alpha_))
        pa = qubo_score(Xa, {k: np.asarray(v) if isinstance(v, list) else v for k, v in model.items()}, edges)
        audit = dict(r2=float(r2_score(aud.val_loss, pa)), spearman=float(spearmanr(aud.val_loss, pa).statistic),
                     mae=float(np.mean(np.abs(aud.val_loss - pa))))
        # angle tuning (exact, light cone)
        m = {k: np.asarray(v) if isinstance(v, list) else v for k, v in model.items()}
        offset, h, J = to_ising(m, edges)
        scale = float(np.sqrt((h ** 2).sum() + (J ** 2).sum()))
        hs, Js = h / scale, J / scale
        t = time.perf_counter()
        grid = [(g, b) for g in np.linspace(-3, 3, 41) for b in np.linspace(-np.pi / 2, np.pi / 2, 41)]
        vals = [expected_energy(0, hs, Js, edges, nbr, g, b) for g, b in grid]
        g0, b0 = grid[int(np.argmin(vals))]
        res = minimize(lambda th: expected_energy(0, hs, Js, edges, nbr, th[0], th[1]), [g0, b0], method='Nelder-Mead',
                       options=dict(xatol=1e-6, fatol=1e-10, maxiter=400))
        timing['angle_tuning_seconds'] = time.perf_counter() - t
        gamma, beta = float(res.x[0]), float(res.x[1])
        ez, ezz = expectations(hs, Js, edges, nbr, gamma, beta)
        ideal = float(offset + h @ ez + J @ ezz)
        uniform = float(offset)
        save(folder / 'qubo.json', dict(model=model, audit=audit, ising=dict(offset=offset, h=h, J=J, scale=scale),
                                        angles=dict(gamma=gamma, beta=beta, scaled_by=scale),
                                        ideal_expected_qubo=ideal, uniform_expected_qubo=uniform, timing=timing))
        print(f'seed {seed}: audit R2 {audit["r2"]:.3f} spearman {audit["spearman"]:.3f}; '
              f'ideal QAOA expected QUBO {ideal:.4f} vs uniform {uniform:.4f}; gamma {gamma:.3f} beta {beta:.3f}', flush=True)


def seed_model(seed):
    q = load(HERE / f'seed_{seed}' / 'qubo.json')
    m = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in q['model'].items()}
    return q, m


# ---------------------------------------------------------------- hardware
def stage_hardware():
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    L, edges, nbr = graph()
    service = QiskitRuntimeService()
    backend = service.backend(L['backend'])
    for seed in SEEDS:
        folder = HERE / f'seed_{seed}'
        path = folder / 'hardware.json'
        if path.exists() and load(path).get('state') == 'complete':
            print('seed', seed, 'hardware done'); continue
        if path.exists() and load(path).get('job_id'):
            raise SystemExit(f"seed {seed}: job {load(path)['job_id']} was submitted but not collected; "
                             "collect it (do not resubmit) before continuing")
        q, _ = seed_model(seed)
        s = q['angles']['scaled_by']
        h = np.asarray(q['ising']['h']) / s
        J = np.asarray(q['ising']['J']) / s
        qc = build_circuit(h, J, edges, q['angles']['gamma'], q['angles']['beta'])
        t = time.perf_counter()
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=L['physical_qubits'],
                                          seed_transpiler=7)
        isa = pm.run(qc)
        compile_s = time.perf_counter() - t
        ops = isa.count_ops()
        rec = dict(seed=seed, backend=backend.name, compile_seconds=compile_s, depth=isa.depth(),
                   two_qubit_gates=isa.num_nonlocal_gates(), swaps=int(ops.get('swap', 0)), shots=PROTOCOL['shots'])
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = PROTOCOL['shots']
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = 'XY4'
        t = time.perf_counter()
        job = sampler.run([isa])
        rec.update(job_id=job.job_id(), submitted_utc=datetime.now(timezone.utc).isoformat(), state='submitted')
        save(path, rec)
        print(f'seed {seed}: job {rec["job_id"]} ({rec["two_qubit_gates"]} two-qubit gates, depth {rec["depth"]}, swaps {rec["swaps"]})', flush=True)
        result = job.result()
        rec['submit_to_result_seconds'] = time.perf_counter() - t
        bits = result[0].data.meas.get_bitstrings()
        X = np.array([[int(b[-1 - i]) for i in range(NQ)] for b in bits], dtype=np.int8)  # qubit i = clbit i
        np.save(folder / 'hardware_shots.npy', X)
        try:
            m = job.metrics()
            rec['usage'] = m.get('usage')
            rec['timestamps'] = m.get('timestamps')
        except Exception as e:
            rec['metrics_error'] = str(e)
        rec['state'] = 'complete'
        save(path, rec)
        print(f'seed {seed}: complete in {rec["submit_to_result_seconds"]:.1f}s, usage {rec.get("usage")}', flush=True)


# ---------------------------------------------------------------- baselines
def anneal(model, edges, nbr, chains, steps, rng):
    lin = model['linear']
    pair = {}
    for k, (a, b) in enumerate(edges):
        pair[(a, b)] = pair[(b, a)] = model['pairs'][k]
    W = np.zeros((NQ, NQ))
    for (a, b), v in pair.items():
        W[a, b] = v
    x = rng.integers(0, 2, (chains, NQ)).astype(np.int8)
    cur = qubo_score(x, model, edges)
    visited = [x.copy()]
    spread = float(np.std(qubo_score(rng.integers(0, 2, (512, NQ)), model, edges)))
    temps = np.geomspace(spread, spread * 1e-3, steps)
    for T in temps:
        i = rng.integers(0, NQ, chains)
        xi = x[np.arange(chains), i]
        delta = (1 - 2 * xi) * (lin[i] + (W[i] * x).sum(1))
        acc = (delta < 0) | (rng.random(chains) < np.exp(-np.maximum(delta, 0) / T))
        x[np.arange(chains)[acc], i[acc]] ^= 1
        cur = cur + np.where(acc, delta, 0)
        visited.append(x.copy())
    return np.concatenate(visited)


def stage_baselines():
    L, edges, nbr = graph()
    for seed in SEEDS:
        folder = HERE / f'seed_{seed}'
        if (folder / 'candidates.npz').exists():
            print('seed', seed, 'baselines done'); continue
        _, m = seed_model(seed)
        rng = np.random.default_rng(seed + 424242)
        pools, timing = {}, {}
        pools['QAOA (ibm_fez)'] = np.load(folder / 'hardware_shots.npy')
        t = time.perf_counter(); pools['Random sampling'] = rng.integers(0, 2, (PROTOCOL['candidate_budget'], NQ)).astype(np.int8)
        timing['Random sampling'] = time.perf_counter() - t
        t = time.perf_counter(); pools['Simulated annealing (matched)'] = anneal(m, edges, nbr, 16, 255, rng)
        timing['Simulated annealing (matched)'] = time.perf_counter() - t
        t = time.perf_counter(); pools['Simulated annealing (long)'] = anneal(m, edges, nbr, 64, 20000, rng)[-64 * 2000:]
        timing['Simulated annealing (long)'] = time.perf_counter() - t
        picks, stats = {}, {}
        best_random = qubo_score(pools['Random sampling'], m, edges).min()
        for name, X in pools.items():
            u, idx = np.unique(X, axis=0, return_index=True)
            s = qubo_score(u, m, edges)
            order = np.argsort(s)[:PROTOCOL['picks_per_method']]
            picks[name] = u[order]
            allscore = qubo_score(X, m, edges)
            stats[name] = dict(candidates=int(len(X)), distinct=int(len(u)), mean_qubo=float(allscore.mean()),
                               best_qubo=float(s.min()), share_better_than_best_random=float((allscore < best_random).mean()),
                               pick_qubo=s[order].tolist(), generation_seconds=timing.get(name))
        np.savez(folder / 'candidates.npz', **{k.replace(' ', '_'): v for k, v in picks.items()})
        save(folder / 'candidate_stats.json', stats)
        print(f'seed {seed}:', {k: round(v['best_qubo'], 4) for k, v in stats.items()}, flush=True)


# ---------------------------------------------------------------- evaluate
def stage_evaluate(workers=6):
    L, edges, nbr = graph()
    names = list(PROTOCOL['methods'])
    for seed in SEEDS:
        folder = HERE / f'seed_{seed}'
        if (folder / 'evaluation.csv').exists():
            print('seed', seed, 'evaluated'); continue
        c = np.load(folder / 'candidates.npz')
        jobs, seen = [], {}
        for name in names:
            for r, x in enumerate(c[name.replace(' ', '_')]):
                b = ''.join(map(str, x))
                if b not in seen:
                    seen[b] = len(jobs)
                    jobs.append((b, x.tolist(), True))
        recs = {r['key']: r for r in retrain_many(seed, jobs, workers)}
        rows = []
        _, m = seed_model(seed)
        for name in names:
            for r, x in enumerate(c[name.replace(' ', '_')]):
                b = ''.join(map(str, x))
                rows.append(dict(seed=seed, method=name, pick=r, qubo=float(qubo_score(x[None], m, edges)[0]), **recs[b]))
        pd.DataFrame(rows).to_csv(folder / 'evaluation.csv', index=False)
        print('seed', seed, 'evaluated', len(jobs), 'distinct networks', flush=True)


# ---------------------------------------------------------------- report
def stage_report():
    names = list(PROTOCOL['methods'])
    ev = pd.concat([pd.read_csv(HERE / f'seed_{s}' / 'evaluation.csv') for s in SEEDS])
    best = ev.loc[ev.groupby(['seed', 'method']).val_loss.idxmin()].set_index(['seed', 'method'])
    mean10 = ev.groupby(['seed', 'method']).val_loss.mean()
    lines = ['# Pre-registered 100-qubit result', '']
    verdict = {}
    q = 'QAOA (ibm_fez)'
    for other in names[1:]:
        wins = sum(best.loc[(s, q), 'val_loss'] < best.loc[(s, other), 'val_loss'] for s in SEEDS)
        mq = np.mean([best.loc[(s, q), 'val_loss'] for s in SEEDS])
        mo = np.mean([best.loc[(s, other), 'val_loss'] for s in SEEDS])
        verdict[other] = dict(seed_wins=int(wins), qaoa_mean=float(mq), other_mean=float(mo), qaoa_wins=bool(wins >= 2 and mq < mo))
    save(HERE / 'verdict.json', dict(rule=PROTOCOL['win_rule'], against=verdict))
    tab = []
    for s in SEEDS:
        stats = load(HERE / f'seed_{s}' / 'candidate_stats.json')
        hw = load(HERE / f'seed_{s}' / 'hardware.json')
        qb = load(HERE / f'seed_{s}' / 'qubo.json')
        dense = load(HERE / f'seed_{s}' / 'dense.json')
        X = np.load(HERE / f'seed_{s}' / 'hardware_shots.npy')
        L, edges, nbr = graph()
        _, m = seed_model(s)
        for name in names:
            b = best.loc[(s, name)]
            tab.append(dict(seed=s, method=name, best_val_loss=b.val_loss, best_val_accuracy=b.val_accuracy,
                            best_test_accuracy=b.test_accuracy, mean10_val_loss=mean10.loc[(s, name)],
                            mean_qubo=stats[name]['mean_qubo'], best_qubo=stats[name]['best_qubo'],
                            share_better_than_best_random=stats[name]['share_better_than_best_random'],
                            hardware_mean_qubo=float(qubo_score(X, m, edges).mean()) if name == q else np.nan,
                            ideal_mean_qubo=qb['ideal_expected_qubo'] if name == q else np.nan,
                            uniform_mean_qubo=qb['uniform_expected_qubo'], audit_r2=qb['audit']['r2'],
                            audit_spearman=qb['audit']['spearman'], dense_val_loss=dense['val_loss'],
                            dense_test_accuracy=dense['test_accuracy'], two_qubit_gates=hw['two_qubit_gates'],
                            swaps=hw['swaps'], job_id=hw['job_id']))
    t = pd.DataFrame(tab)
    t.to_csv(HERE / 'results.csv', index=False)
    print(t.to_string())
    print(json.dumps(verdict, indent=2))


def stage_preregister():
    path = HERE / 'PREREGISTRATION.md'
    if path.exists():
        print('already pre-registered'); return
    save(HERE / 'protocol.json', PROTOCOL)
    code_hash = hashlib.sha256((HERE / 'hw100.py').read_bytes()).hexdigest()
    proto_hash = hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest()
    m = PROTOCOL
    lines = ['# Pre-registration: 100-qubit QAOA pruning test', '',
             f"Written {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}, before any network in this experiment was trained, "
             'before any hardware job, and before any result was seen.', '',
             f"- `protocol.json` SHA-256: `{proto_hash}`", f"- `hw100.py` SHA-256: `{code_hash}`", '',
             '## Question', '', m['question'], '',
             '## Setup', '',
             f"- Network: Fashion-MNIST, 784 → {m['hidden']} → 10; {m['groups']} groups of {m['group_size']} neurons; {m['qubits']} qubits.",
             f"- Encoding: {m['encoding']}.", f"- Training: {m['training']}.",
             f"- Seeds: {m['seeds']}. QUBO: {m['qubo']}; {m['fit_samples']} fit + {m['audit_samples']} audit retrainings per seed.",
             f"- Hardware: {m['backend']}, QAOA p={m['qaoa_layers']}, {m['shots']} shots, couplings only on native chip edges. Angles: {m['angle_tuning']}.", '',
             '## Methods (equal 4,096-candidate budget except the reference ceiling)', '']
    lines += [f'- **{k}**: {v}' for k, v in m['methods'].items()]
    lines += ['', '## Decision rule', '',
              f"- Each method's picks: {m['pick_rule']} ({m['picks_per_method']} per method per seed), each retrained from the original initialization.",
              f"- Primary metric: {m['primary_metric']}.", f"- **Win rule:** {m['win_rule']}.",
              '- Secondary (reported, not decisive): ' + '; '.join(m['secondary']) + '.',
              f"- {m['everything_reported'].capitalize()}.", '', '## Known caveats (stated in advance)', '']
    lines += [f'- {c}' for c in m['known_caveats']]
    lines += ['']
    path.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    stage = sys.argv[1]
    {'preregister': stage_preregister, 'layout': stage_layout, 'verify': stage_verify, 'prepare': stage_prepare, 'hardware': stage_hardware,
     'baselines': stage_baselines, 'evaluate': stage_evaluate, 'report': stage_report}[stage]()
