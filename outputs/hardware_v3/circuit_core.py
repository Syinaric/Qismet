"""Transparent QUBO, circuit, and measured-count utilities for Qismet.

The hardware path constructs O(n**2) gates and scores only observed bitstrings.
It never enumerates 2**n masks. Functions named ``ideal_*`` and ``toy_data`` are
explicit local statevector/exhaustive teaching diagnostics, not hardware runs.

Mask convention: m[i] corresponds to qubit i and classical bit mask[i]. Qiskit
prints the highest-index classical bit at the left: '001' means m=[1,0,0].
Parameter convention: legacy saved arrays are [gamma_0,...,gamma_p-1,
beta_0,...,beta_p-1], which differs from Qiskit's alphabetical parameter order.
Use bind_qaoa() or parameter_values(); never bind saved arrays positionally.

Qiskit API references:
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparsePauliOp
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass(frozen=True)
class QuboProblem:
    """Unscaled coefficients; pairwise and J use each i<j interaction once."""

    linear: np.ndarray
    pairwise: np.ndarray
    constant: float
    keep: int
    penalty: float
    h: np.ndarray
    J: np.ndarray
    offset: float
    scale: float
    source: str | None = None

    @property
    def n(self):
        return len(self.linear)

    def as_dict(self):
        """Serializable coefficients and conventions, without any enumeration."""
        return {"n": self.n, "linear": self.linear.tolist(),
                "pairwise": self.pairwise.tolist(), "constant": self.constant,
                "keep": self.keep, "penalty": self.penalty, "h": self.h.tolist(),
                "J": self.J.tolist(), "offset": self.offset, "scale": self.scale,
                "source": self.source,
                "mask_convention": "mask[i] is qubit i; reverse displayed bitstrings",
                "parameter_convention": "gammas first, then betas",
                "scale_convention": "sqrt(sum(h**2)+sum(J**2)); identity excluded"}


def make_problem(linear, pairwise, constant, keep, *, source=None):
    """Add a sufficient cardinality penalty and map m=(1-Z)/2 exactly.

    f(m)=c+a.m+sum_{i<j}b_ij*m_i*m_j. The penalized cost is
    C(m)=f(m)+lambda*(sum(m)-keep)**2. Its unpenalized range is at most
    sum(abs(a))+sum(abs(b)); lambda=1+that_bound therefore places EVERY
    infeasible mask above EVERY feasible mask. This conservative penalty can
    dominate the ranking signal. It matches the original qaoa_utils.py exactly.

    H=offset*I+sum(h_i Z_i)+sum(J_ij Z_i Z_j). The circuit implements
    (H-offset*I)/scale; removing identity changes only a global phase.
    """
    a = np.asarray(linear, dtype=float).copy()
    b = np.asarray(pairwise, dtype=float).copy()
    if a.ndim != 1 or not len(a):
        raise ValueError("linear must be a nonempty one-dimensional array")
    n = len(a)
    if b.shape != (n, n) or not np.allclose(np.tril(b), 0, atol=1e-14, rtol=0):
        raise ValueError("pairwise must be strictly upper triangular; each pair appears once")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(constant):
        raise ValueError("Every coefficient must be finite")
    if isinstance(keep, (bool, np.bool_)) or int(keep) != keep or not 0 <= keep <= n:
        raise ValueError("keep must be an integer in [0,n]")
    keep = int(keep)
    c = float(constant)
    penalty = float(1 + np.abs(a).sum() + np.abs(b).sum())
    ap = a + penalty * (1 - 2*keep)
    bp = b + 2*penalty*np.triu(np.ones((n, n)), 1)
    cp = c + penalty*keep**2
    h = -ap/2 - (bp.sum(axis=0) + bp.sum(axis=1))/4
    J = bp/4
    offset = float(cp + ap.sum()/2 + bp.sum()/4)
    scale = float(max(np.sqrt(np.dot(h, h) + np.sum(J**2)), 1e-12))
    for array in (a, b, h, J):
        array.setflags(write=False)
    return QuboProblem(a, b, c, keep, penalty, h, J, offset, scale,
                       str(source) if source is not None else None)


def load_problem(npz_path, keep):
    """Load a Fashion v2 surrogates.npz (or original linear/pairwise/constant NPZ)."""
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        if {"quadratic_linear", "quadratic_pairs", "quadratic_constant"} <= set(data.files):
            keys = ("quadratic_linear", "quadratic_pairs", "quadratic_constant")
        elif {"linear", "pairwise", "constant"} <= set(data.files):
            keys = ("linear", "pairwise", "constant")
        else:
            raise ValueError("NPZ must contain quadratic_linear/quadratic_pairs/quadratic_constant")
        return make_problem(*(data[key] for key in keys), keep, source=path)


def qubo_to_ising(problem, include_identity=True, normalized=False):
    """Return diagonal SparsePauliOp; construction is polynomial, not exhaustive."""
    factor = 1/problem.scale if normalized else 1.0
    terms = []
    if include_identity:
        terms.append(("I"*problem.n, problem.offset*factor))
    for i in range(problem.n):
        label = ["I"]*problem.n
        label[problem.n-1-i] = "Z"
        terms.append(("".join(label), float(problem.h[i]*factor)))
    for i in range(problem.n):
        for j in range(i+1, problem.n):
            label = ["I"]*problem.n
            label[problem.n-1-i] = label[problem.n-1-j] = "Z"
            terms.append(("".join(label), float(problem.J[i, j]*factor)))
    return SparsePauliOp.from_list(terms)


def cost_layer(problem, gamma):
    """RZ/RZZ gates for exp[-i*gamma*(H-offset*I)/scale], without global phase.

    RZ(theta)=exp(-i theta Z/2), hence theta=2*gamma*h_i/scale.
    RZZ(theta)=exp(-i theta ZiZj/2), hence theta=2*gamma*J_ij/scale.
    gamma can be a float, Parameter, or ParameterExpression.
    """
    qc = QuantumCircuit(problem.n, name="cost")
    for i in range(problem.n):
        qc.rz(2*gamma*problem.h[i]/problem.scale, i)
    for i in range(problem.n):
        for j in range(i+1, problem.n):
            qc.rzz(2*gamma*problem.J[i, j]/problem.scale, i, j)
    return qc


def build_qaoa(problem, reps=2, measure=False, barriers=False):
    """Build standard X-mixer QAOA with ParameterVector('gamma'/'beta').

    Start with H on every qubit. Each layer applies the diagonal cost phase,
    then RX(2*beta) on every qubit, i.e. exp(-i*beta*sum(X)).
    When measured, exactly one classical register named 'mask' is created;
    use result.data.mask.get_counts() with Runtime SamplerV2. An unmeasured
    circuit has no classical registers. Barrier insertion is for diagrams only.
    """
    if isinstance(reps, bool) or int(reps) != reps or reps < 1:
        raise ValueError("reps must be a positive integer")
    reps = int(reps)
    gamma, beta = ParameterVector("gamma", reps), ParameterVector("beta", reps)
    q = QuantumRegister(problem.n, "q")
    qc = QuantumCircuit(q, name=f"Qismet_QAOA_p{reps}")
    qc.h(range(problem.n))
    if barriers:
        qc.barrier(label="initial |+>")
    for layer in range(reps):
        qc.compose(cost_layer(problem, gamma[layer]), inplace=True)
        if barriers:
            qc.barrier(label=f"cost {layer+1}")
        qc.rx(2*beta[layer], range(problem.n))
        if barriers:
            qc.barrier(label=f"mixer {layer+1}")
    if measure:
        mask = ClassicalRegister(problem.n, "mask")
        qc.add_register(mask)
        qc.measure(q, mask)
    qc.metadata = {"qismet_reps": reps, "qismet_n": problem.n,
                   "qismet_parameter_order": "gammas then betas",
                   "qismet_cost_scale": problem.scale,
                   "qismet_omitted_identity_offset": problem.offset}
    return qc


def parameter_values(circuit, parameters):
    """Convert legacy [gamma..., beta...] to the circuit's actual parameter order.

    Works for logical or transpiled parameterized circuits. Runtime PUBs may
    use the returned array as their positional parameter-values field.
    """
    values = np.asarray(parameters, dtype=float)
    if values.ndim != 1 or len(values) < 2 or len(values) % 2 or not np.isfinite(values).all():
        raise ValueError("parameters must be a finite even-length gamma-then-beta vector")
    reps = len(values)//2
    names = {f"gamma[{i}]": values[i] for i in range(reps)}
    names.update({f"beta[{i}]": values[reps+i] for i in range(reps)})
    circuit_names = [parameter.name for parameter in circuit.parameters]
    if set(circuit_names) != set(names):
        raise ValueError(f"Circuit parameters {circuit_names} do not match {reps} QAOA layers")
    return np.array([names[name] for name in circuit_names], dtype=float)


def bind_qaoa(circuit, parameters):
    """Bind by Parameter identity after safely translating saved array order."""
    values = parameter_values(circuit, parameters)
    return circuit.assign_parameters(dict(zip(circuit.parameters, values)), inplace=False)


def bitstring_to_mask(bitstring, n):
    """Decode Qiskit displayed bits to m[i]; accepts padded bits or integer ids."""
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    if isinstance(bitstring, (int, np.integer)) and not isinstance(bitstring, (bool, np.bool_)):
        index = int(bitstring)
    elif isinstance(bitstring, str):
        text = bitstring.replace(" ", "")
        if text.startswith("0x"):
            index = int(text, 16)
        elif text.startswith("0b"):
            index = int(text, 2)
        elif len(text) == n and set(text) <= {"0", "1"}:
            index = int(text, 2)
        else:
            raise ValueError(f"Expected exactly {n} displayed binary bits, got {bitstring!r}")
    else:
        raise ValueError("Count keys must be bitstrings or integer basis indices")
    if not 0 <= index < 2**n:
        raise ValueError(f"Basis index {index} does not fit {n} bits")
    return np.array([(index >> i) & 1 for i in range(n)], dtype=int)


def cost(problem, mask, penalized=True):
    """Score one observed mask in O(n**2); no enumeration or neural retraining."""
    m = np.asarray(mask)
    if m.shape != (problem.n,) or not np.isin(m, [0, 1]).all():
        raise ValueError(f"mask must contain exactly {problem.n} binary entries")
    value = float(problem.constant + m @ problem.linear + m @ problem.pairwise @ m)
    if penalized:
        value += problem.penalty * (int(m.sum()) - problem.keep)**2
    return float(value)


def _mean_and_se(values, weights):
    total = int(sum(weights))
    mean = float(np.dot(values, weights)/total)
    # Unbiased sample variance / sample size; covariance between individual
    # shots is assumed zero. This does not capture hardware drift/systematics.
    variance = float(np.dot((np.asarray(values)-mean)**2, weights)/(total-1)) if total > 1 else None
    return mean, math.sqrt(max(variance, 0)/total) if variance is not None else None


def counts_stats(problem, counts):
    """Summarize measured counts using only observed outcomes, in O(unique*n**2).

    Returns raw/normalized energy means and standard errors; feasibility and
    distinct counts; best observed feasible mask by surrogate; and a per-outcome
    table. No feasible outcomes gives best_feasible_* and conditional metrics
    equal to None: there is no replacement mask, projection, or re-roll.

    Standard errors describe independent-shot sampling uncertainty only, not
    readout bias, gate errors, or temporal drift. One shot gives None for SE.
    Input must be nonnegative integer counts, not quasi-probabilities.
    """
    if not isinstance(counts, Mapping):
        raise ValueError("counts must map measured bitstrings to integer frequencies")
    canonical = {}
    for key, frequency in counts.items():
        if isinstance(frequency, (bool, np.bool_)) or not isinstance(frequency, (int, np.integer)) or frequency < 0:
            raise ValueError("Every count must be a nonnegative integer")
        mask = bitstring_to_mask(key, problem.n)
        displayed = "".join(str(int(bit)) for bit in mask[::-1])
        canonical[displayed] = canonical.get(displayed, 0) + int(frequency)
    canonical = {key: value for key, value in sorted(canonical.items()) if value > 0}
    if not canonical:
        raise ValueError("At least one measured shot is required")
    rows = []
    for bits, count in canonical.items():
        mask = bitstring_to_mask(bits, problem.n)
        rows.append({"bitstring": bits, "index": int(bits, 2), "mask": mask.tolist(),
                     "count": count, "feasible": bool(mask.sum() == problem.keep),
                     "surrogate": cost(problem, mask, penalized=False),
                     "penalized_cost": cost(problem, mask)})
    shots = int(sum(canonical.values()))
    feasible = [row for row in rows if row["feasible"]]
    feasible_shots = int(sum(row["count"] for row in feasible))
    mean, se = _mean_and_se([row["penalized_cost"] for row in rows], [row["count"] for row in rows])
    best = min(feasible, key=lambda row: (row["surrogate"], row["index"])) if feasible else None
    conditional_mean, conditional_se = (_mean_and_se([row["surrogate"] for row in feasible],
                                                   [row["count"] for row in feasible])
                                       if feasible else (None, None))
    return {"shots": shots, "counts": canonical, "unique_samples": len(rows),
            "feasible_shots": feasible_shots, "unique_feasible_samples": len(feasible),
            "feasible_fraction": feasible_shots/shots,
            "penalized_mean": mean, "mean_penalized_energy": mean,
            "penalized_standard_error": se, "standard_error_penalized_energy": se,
            "normalized_mean": (mean-problem.offset)/problem.scale,
            "normalized_standard_error": se/problem.scale if se is not None else None,
            "conditional_surrogate_mean": conditional_mean,
            "conditional_surrogate_standard_error": conditional_se,
            "best_feasible_observed_mask": best["mask"] if best else None,
            "best_feasible_index": best["index"] if best else None,
            "best_feasible_bitstring": best["bitstring"] if best else None,
            "best_feasible_surrogate": best["surrogate"] if best else None,
            "best_feasible_count": best["count"] if best else None,
            "observed_outcomes": rows,
            "standard_error_scope": "Independent-shot sampling only; excludes device bias and drift"}


def ideal_probabilities(problem, parameters, reps=2):
    """LOCAL noiseless exact probabilities indexed by little-endian basis id."""
    if problem.n > 20:
        raise ValueError("Local ideal diagnostics are restricted to at most 20 qubits")
    bound = bind_qaoa(build_qaoa(problem, reps=reps), parameters)
    return Statevector.from_instruction(bound).probabilities()


def ideal_exact(problem, parameters, reps=2):
    """LOCAL exhaustive diagnostic; do not call this the quantum hardware path."""
    probs = ideal_probabilities(problem, parameters, reps=reps)
    energies = np.array([cost(problem, bitstring_to_mask(i, problem.n)) for i in range(len(probs))])
    feasible = np.array([i.bit_count() == problem.keep for i in range(len(probs))])
    mass = float(probs[feasible].sum())
    mean = float(probs @ energies)
    return {"execution": "local_noiseless_exact_statevector", "probabilities": probs.tolist(),
            "penalized_mean": mean, "normalized_mean": (mean-problem.offset)/problem.scale,
            "penalized_variance": float(probs @ (energies-mean)**2),
            "feasible_probability": mass,
            "conditional_surrogate_mean": float(probs[feasible] @ energies[feasible]/mass) if mass else None}


def ideal_sample(problem, parameters, shots=1024, seed=17, reps=2):
    """LOCAL ideal finite-shot simulation; returns counts_stats plus execution label."""
    if isinstance(shots, bool) or int(shots) != shots or shots < 1:
        raise ValueError("shots must be a positive integer")
    if problem.n > 20:
        raise ValueError("Local ideal diagnostics are restricted to at most 20 qubits")
    state = Statevector.from_instruction(bind_qaoa(build_qaoa(problem, reps=reps), parameters))
    state.seed(seed)
    result = counts_stats(problem, state.sample_counts(int(shots)))
    result.update({"execution": "local_noiseless_shot_simulation", "seed": int(seed)})
    return result


def toy_problem():
    """Three-neuron explanatory example: keep two, one binary qubit per neuron."""
    return make_problem([-1.2, -0.7, -0.9],
                        [[0, 0.8, 0.3], [0, 0, -0.4], [0, 0, 0]], 0.4, 2,
                        source="three-qubit explanatory toy; not fitted neural-network data")


def toy_data():
    """Eight-state explanatory table, coefficients, and phase-gate descriptions."""
    problem = toy_problem()
    rows = []
    for index in range(8):
        mask = bitstring_to_mask(index, 3)
        rows.append({"bitstring_q2q1q0": format(index, "03b"), "mask_m0m1m2": mask.tolist(),
                     "surrogate": cost(problem, mask, False),
                     "penalized_cost": cost(problem, mask), "feasible": bool(mask.sum() == 2)})
    return {"problem": problem.as_dict(), "basis_table": rows,
            "cost_gate_rule": "RZ_i(2*gamma*h_i/scale), RZZ_ij(2*gamma*J_ij/scale)",
            "mixer_gate_rule": "RX_i(2*beta)",
            "identity_rule": "offset*I contributes global phase only; no gate is needed",
            "bit_order_example": "Displayed '011' means m=[1,1,0]"}
