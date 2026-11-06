import sys, json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Operator, Statevector
sys.path.insert(0, str(Path('outputs/hardware_v3').resolve()))
from circuit_core import *

record = {'execution': 'local_verification_only_no_hardware_jobs'}
p = load_problem('outputs/fashion_v2/results/seed_11/surrogates.npz', 4)
saved = json.loads(Path('outputs/fashion_v2/results/seed_11/qaoa_keep_4.json').read_text())
params = saved['parameters']
probs = ideal_probabilities(p, params, reps=saved['reps'])
record['saved_probability_max_absolute_error'] = float(np.max(np.abs(probs-saved['probabilities'])))
np.testing.assert_allclose(probs, saved['probabilities'], atol=2e-14, rtol=1e-12)
np.testing.assert_allclose(p.h, saved['ising']['h'], rtol=0, atol=0)
np.testing.assert_allclose(p.J, saved['ising']['J'], rtol=0, atol=0)
assert p.scale == saved['scale'] and p.penalty == saved['penalty']
encoded = qubo_to_ising(p).to_matrix(sparse=True).diagonal().real
direct = np.array([cost(p, bitstring_to_mask(i, p.n)) for i in range(2**p.n)])
record['fashion_max_basis_encoding_error'] = float(np.max(np.abs(encoded-direct)))
np.testing.assert_allclose(encoded, direct, atol=1e-12, rtol=1e-12)
np.testing.assert_allclose(direct, saved['energies'], atol=1e-12, rtol=1e-12)

sample = ideal_sample(p, params, shots=saved['shots'], seed=saved['seed']+10000, reps=saved['reps'])
expected_counts = {format(i, '08b'): int(c) for i,c in enumerate(saved['counts']) if c}
assert sample['counts'] == expected_counts
assert sample['best_feasible_index'] == saved['selected_index']
record['saved_finite_shot_counts_identical'] = True
record['saved_selected_mask_identical'] = True

logical = build_qaoa(p, measure=True)
assert [reg.name for reg in logical.cregs] == ['mask']
assert logical.num_clbits == p.n
assert build_qaoa(p).num_clbits == 0
np.testing.assert_allclose(parameter_values(logical, params), params[2:]+params[:2], atol=0, rtol=0)
assert len(bind_qaoa(logical, params).parameters) == 0
record['parameter_and_register_mapping'] = True

toy = toy_problem()
gamma, beta = 0.73, -0.21
diag = qubo_to_ising(toy).to_matrix()
expected_u = expm(-1j*gamma*(diag-toy.offset*np.eye(8))/toy.scale)
gate_u = Operator(cost_layer(toy, gamma)).data
record['toy_cost_unitary_error'] = float(np.max(np.abs(gate_u-expected_u)))
np.testing.assert_allclose(gate_u, expected_u, atol=2e-14, rtol=1e-12)
X = np.array([[0,1],[1,0]])
mixer = np.kron(np.eye(4), X)+np.kron(np.eye(2),np.kron(X,np.eye(2)))+np.kron(X,np.eye(4))
expected_state = expm(-1j*beta*mixer) @ expected_u @ (np.ones(8)/np.sqrt(8))
actual_state = Statevector.from_instruction(bind_qaoa(build_qaoa(toy,reps=1),[gamma,beta])).data
record['toy_full_layer_state_error'] = float(np.max(np.abs(expected_state-actual_state)))
np.testing.assert_allclose(actual_state, expected_state, atol=2e-14, rtol=1e-12)

assert bitstring_to_mask('001',3).tolist() == [1,0,0]
assert bitstring_to_mask('110',3).tolist() == [0,1,1]
counts = {'001': 2, '110': 3, '000': 1}
stats = counts_stats(toy, counts)
expanded = np.array([cost(toy,bitstring_to_mask(bits,3)) for bits,c in counts.items() for _ in range(c)])
np.testing.assert_allclose(stats['penalized_mean'], expanded.mean(), atol=1e-14)
np.testing.assert_allclose(stats['penalized_standard_error'], expanded.std(ddof=1)/np.sqrt(len(expanded)), atol=1e-14)
assert stats['best_feasible_observed_mask'] == [0,1,1]
assert stats['feasible_shots'] == 3
assert counts_stats(toy, {'000': 2})['best_feasible_observed_mask'] is None
assert counts_stats(toy, {'000': 1})['penalized_standard_error'] is None
record['count_statistics_bit_order_and_no_feasible_behavior'] = True

large = make_problem(np.ones(32), np.zeros((32,32)),0,16)
qc32 = build_qaoa(large,reps=1,measure=True)
assert qc32.num_qubits == 32
assert counts_stats(large, {'0'*16+'1'*16: 3})['feasible_fraction'] == 1
record['hardware_path_no_enumeration_32_qubit_smoke'] = True
record['all_checks_passed'] = True
Path('work/hardware_v3_core_verification.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
