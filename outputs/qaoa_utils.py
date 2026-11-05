"""Small, auditable Qiskit QAOA solver for a binary quadratic surrogate.

This is a noiseless CPU statevector simulation, not a quantum-hardware run.
The exhaustive arrays below are diagnostic conveniences for tiny examples. They
are also used to evaluate the diagonal expectation exactly; they do not select
the returned mask. Selection is restricted to finite-shot measurement samples.

Qiskit bit order: mask[i] is qubit i, the i-th least-significant bit of a basis
index. Printed Qiskit bitstrings have the reverse order from the mask list.

References:
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparsePauliOp
https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm
"""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def solve_qaoa(linear, pairwise, constant, keep, seed=17, reps=2,
               restarts=3, maxiter=120, shots=32):
    """Minimize c + sum(a_i m_i) + sum_{i<j}(b_ij m_i m_j), sum(m)=keep.

    ``pairwise`` must be strictly upper triangular, with each interaction given
    ONCE. A symmetric matrix would double-count its off-diagonal entries and is
    rejected. Width is restricted to 1..14 for this exhaustive teaching demo.

    Add lambda*(sum(m)-keep)**2, where lambda=1+sum(abs(a))+sum(abs(b)).
    The unpenalized objective's range is at most the sum of absolute nonconstant
    coefficients, so every infeasible mask costs more than every feasible mask.
    This sufficient penalty can dominate the useful surrogate signal; the
    diagnostics explicitly expose whether QAOA learns anything beyond sparsity.

    The circuit starts in |+>^n, applies p alternating exp(-i gamma H/scale) and
    exp(-i beta sum(X)) layers, and uses COBYLA with seeded independent starts.
    scale is the RMS nonconstant Ising energy under uniform basis sampling,
    sqrt(sum(h**2)+sum(J**2)). The identity offset is omitted from the circuit.
    Optimization uses exact statevector expectations; only final measurement
    uses ``shots``. Restart selection uses penalized expectation, not exact
    optimal-mask knowledge. Selected mask minimizes surrogate among feasible
    observed samples. No feasible sample raises RuntimeError, with no oracle
    fallback or projection to a mask that was not measured.

    Returns a JSON-serializable dict with these keys:
      selected_mask / selected_index / selected_qubit_bitstring: sampled result.
      selected_surrogate / selected_penalized_energy: cost of that result.
      selected_sample_count: how often that exact basis state was measured.
      n / keep / seed / reps / restarts / maxiter / shots: experiment settings.
      penalty / scale: cardinality coefficient and circuit energy scaling.
      ising: offset, linear Z and upper-triangular ZZ coefficients (unscaled).
      pauli_terms: equivalent full unscaled SparsePauliOp as label/coefficient.
      basis_masks / surrogate_energies / energies / probabilities: full arrays
        indexed by little-endian computational-basis integer.
      sampled_indices / counts / feasible_shots / unique_feasible_samples:
        raw finite-shot outcomes and statistics.
      feasible_probability / uniform_feasible_probability: actual and uniform
        probability of satisfying the cardinality constraint.
      exact_feasible_minimum / optimum_feasible_probability: reference optimum
        of the SURROGATE and probability of measuring any tied optimum.
      uniform_optimum_probability / conditional_optimum_probability: optimum
        probability before and after restricting to feasible QAOA samples.
      uniform_conditional_optimum_probability: optimum fraction among feasible
        masks under uniform sampling (a fairer sparsity-controlled comparison).
      feasible_top_decile_threshold / feasible_top_decile_probability /
        conditional_top_decile_probability / uniform_conditional_top_decile_probability:
        low-surrogate tail diagnostics. Include ties at the 10th percentile.
      conditional_surrogate_mean / uniform_feasible_surrogate_mean: expected
        surrogate conditioned on feasibility and its uniform feasible baseline.
      expected_penalized_energy / uniform_penalized_energy: expectation values.
      parameters / optimizer_records / total_evaluations: optimizer evidence.
      checks: maximum encoding error and state expectation agreement, with
        assertion failures raised if either check fails.
      circuit_depth / circuit_gate_counts / elapsed_seconds: resource metadata.

    Exact optimum probabilities are diagnostic, never optimizer objectives.
    """
    started = time.perf_counter()
    a = np.asarray(linear, dtype=float).copy()
    b = np.asarray(pairwise, dtype=float).copy()
    if a.ndim != 1 or not 1 <= len(a) <= 14:
        raise ValueError("linear must have 1..14 entries for this small demo")
    n = len(a)
    if b.shape != (n, n) or not np.allclose(np.tril(b), 0, atol=1e-14):
        raise ValueError("pairwise must be a strictly upper-triangular n*n matrix")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(constant):
        raise ValueError("All surrogate coefficients must be finite")
    if int(keep) != keep or not 0 <= keep <= n:
        raise ValueError("keep must be an integer in [0,n]")
    for name, value in [("reps", reps), ("restarts", restarts),
                        ("maxiter", maxiter), ("shots", shots)]:
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    keep, reps, restarts, maxiter, shots = map(int, (keep, reps, restarts, maxiter, shots))
    constant = float(constant)
    ids = np.arange(2**n)
    masks = ((ids[:, None] >> np.arange(n)) & 1).astype(int)
    feasible = masks.sum(axis=1) == keep
    surrogate = constant + masks @ a + np.einsum("bi,ij,bj->b", masks, b, masks)
    penalty = float(1 + np.abs(a).sum() + np.abs(b).sum())
    energies = surrogate + penalty * (masks.sum(axis=1) - keep)**2
    ap = a + penalty * (1 - 2*keep)
    bp = b + 2*penalty*np.triu(np.ones((n, n)), 1)
    cp = constant + penalty*keep**2
    h = -ap/2 - (bp.sum(axis=0) + bp.sum(axis=1))/4
    J = bp/4
    offset = float(cp + ap.sum()/2 + bp.sum()/4)
    scale = float(max(np.sqrt(np.dot(h, h) + np.sum(J**2)), 1e-12))

    terms = [("I"*n, offset)]
    for i in range(n):
        label = ["I"]*n
        label[n-1-i] = "Z"
        terms.append(("".join(label), float(h[i])))
    for i in range(n):
        for j in range(i+1, n):
            label = ["I"]*n
            label[n-1-i] = label[n-1-j] = "Z"
            terms.append(("".join(label), float(J[i, j])))
    operator = SparsePauliOp.from_list(terms)
    # A sparse matrix avoids allocating a dense 2^n by 2^n verification matrix.
    encoded = operator.to_matrix(sparse=True)
    diagonal = np.real(encoded.diagonal())
    encoding_error = float(np.max(np.abs(diagonal - energies)))
    np.testing.assert_allclose(diagonal, energies, atol=1e-9, rtol=1e-10)
    rows, cols = encoded.nonzero()
    assert np.all(rows == cols), "Cost Hamiltonian must be diagonal"
    if feasible.any() and (~feasible).any():
        assert energies[~feasible].min() > energies[feasible].max(), "Penalty guarantee failed"

    def circuit(parameters):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        gammas, betas = parameters[:reps], parameters[reps:]
        for gamma, beta in zip(gammas, betas):
            for i in range(n):
                qc.rz(2*gamma*h[i]/scale, i)
            for i in range(n):
                for j in range(i+1, n):
                    qc.rzz(2*gamma*J[i, j]/scale, i, j)
            qc.rx(2*beta, range(n))
        return qc

    scaled_energies = (energies - offset)/scale
    rng = np.random.default_rng(seed)
    records = []
    global_best_value = np.inf
    global_best_parameters = None
    total_evaluations = 0
    for restart in range(restarts):
        x0 = np.r_[rng.uniform(-np.pi, np.pi, reps), rng.uniform(-np.pi/2, np.pi/2, reps)]
        trace = []
        run_best_value = np.inf
        run_best_parameters = None

        def objective(parameters):
            nonlocal run_best_value, run_best_parameters, total_evaluations
            state = Statevector.from_instruction(circuit(parameters))
            value = float(state.probabilities() @ scaled_energies)
            total_evaluations += 1
            trace.append(value)
            if value < run_best_value:
                run_best_value = value
                run_best_parameters = np.array(parameters, copy=True)
            return value

        result = minimize(objective, x0, method="COBYLA",
                          options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-5})
        records.append({"restart": restart, "initial_parameters": x0.tolist(),
                        "best_parameters": run_best_parameters.tolist(),
                        "best_scaled_expectation": float(run_best_value),
                        "best_penalized_expectation": float(run_best_value*scale + offset),
                        "evaluations": len(trace), "success": bool(result.success),
                        "message": str(result.message), "scaled_trace": trace})
        if run_best_value < global_best_value:
            global_best_value = run_best_value
            global_best_parameters = run_best_parameters.copy()

    final_circuit = circuit(global_best_parameters)
    final_state = Statevector.from_instruction(final_circuit)
    probabilities = final_state.probabilities()
    expected = float(probabilities @ energies)
    state_expectation = float(np.real(final_state.expectation_value(operator)))
    expectation_error = abs(expected - state_expectation)
    np.testing.assert_allclose(expected, state_expectation, atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-10)
    # Statevector.sample_memory samples the actual circuit distribution, in
    # ordinary Qiskit bitstring order. Parsing int(bits,2) recovers basis ids.
    final_state.seed(seed + 10000)
    bitstrings = final_state.sample_memory(shots)
    sampled_indices = np.array([int(bits, 2) for bits in bitstrings], dtype=int)
    counts = np.bincount(sampled_indices, minlength=2**n)
    feasible_observed = np.flatnonzero((counts > 0) & feasible)
    if not len(feasible_observed):
        raise RuntimeError(f"No feasible mask in {shots} QAOA shots; do not report an oracle fallback")
    # Only sampled states participate in selecting the proposed mask.
    selected = int(feasible_observed[np.argmin(surrogate[feasible_observed])])
    feasible_mass = float(probabilities[feasible].sum())
    exact_min = float(surrogate[feasible].min())
    optimum = feasible & np.isclose(surrogate, exact_min, atol=1e-10, rtol=1e-10)
    optimum_mass = float(probabilities[optimum].sum())
    tail_threshold = float(np.quantile(surrogate[feasible], 0.1))
    tail = feasible & (surrogate <= tail_threshold + 1e-12)
    tail_mass = float(probabilities[tail].sum())
    return {
        "selected_mask": masks[selected].tolist(), "selected_index": selected,
        "selected_qubit_bitstring": format(selected, f"0{n}b"),
        "selected_surrogate": float(surrogate[selected]),
        "selected_penalized_energy": float(energies[selected]),
        "selected_sample_count": int(counts[selected]),
        "n": n, "keep": keep, "seed": int(seed), "reps": reps,
        "restarts": restarts, "maxiter": maxiter, "shots": shots,
        "penalty": penalty, "scale": scale,
        "ising": {"offset": offset, "h": h.tolist(), "J": J.tolist()},
        "pauli_terms": [[label, coefficient] for label, coefficient in terms],
        "basis_masks": masks.tolist(), "surrogate_energies": surrogate.tolist(),
        "energies": energies.tolist(), "probabilities": probabilities.tolist(),
        "sampled_indices": sampled_indices.tolist(),
        "counts": counts.tolist(), "feasible_shots": int(counts[feasible].sum()),
        "unique_feasible_samples": int(len(feasible_observed)),
        "feasible_probability": feasible_mass,
        "uniform_feasible_probability": float(feasible.mean()),
        "exact_feasible_minimum": exact_min,
        "optimum_feasible_probability": optimum_mass,
        "uniform_optimum_probability": float(optimum.mean()),
        "conditional_optimum_probability": optimum_mass / feasible_mass,
        "uniform_conditional_optimum_probability": float(optimum.sum()/feasible.sum()),
        "feasible_top_decile_threshold": tail_threshold,
        "feasible_top_decile_probability": tail_mass,
        "conditional_top_decile_probability": tail_mass / feasible_mass,
        "uniform_conditional_top_decile_probability": float(tail.sum()/feasible.sum()),
        "conditional_surrogate_mean": float(probabilities[feasible] @ surrogate[feasible]/feasible_mass),
        "uniform_feasible_surrogate_mean": float(surrogate[feasible].mean()),
        "expected_penalized_energy": expected,
        "uniform_penalized_energy": float(energies.mean()),
        "parameters": global_best_parameters.tolist(), "optimizer_records": records,
        "total_evaluations": total_evaluations,
        "checks": {"maximum_basis_encoding_error": encoding_error,
                   "state_expectation_error": expectation_error,
                   "penalty_guarantee": True, "selected_mask_was_sampled": True},
        "circuit_depth": final_circuit.depth(),
        "circuit_gate_counts": {str(k): int(v) for k, v in final_circuit.count_ops().items()},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
