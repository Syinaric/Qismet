# Speed race: brute force vs Qismet on the 784→32→10 network

Seed 1001. 16 neuron groups (pairs). Every timing below was measured on the same Mac (single CPU core per retraining), except the IBM job, which is wall-clock from submission to result.

## Keep 8 of 16 groups (50% of neurons removed, 12,870 possible networks)

| Pipeline | Time (one CPU core + QPU) | Retrainings | Validation accuracy of pick | Rank of pick among all networks |
|---|---:|---:|---:|---:|
| Brute force (regular way) | 114.9 min | 12,870 | 82.4% | 1 (true best) |
| Qismet + CPU search | 104.1 s | 193 | 82.9% | 4,016 (top 31.2%) |
| Qismet + QPU (ibm_fez) | 123.1 s | 193 | 83.3% | 90 (top 0.7%) |

Dense 32-neuron network: 82.6% validation accuracy. Brute force wall time with 6 parallel workers: 19.3 min.

Where the Qismet time goes:

| Step | Qismet + CPU | Qismet + QPU |
|---|---:|---:|
| Retrain 192 sample networks | 103.6 s | 103.6 s |
| Fit QUBO (ridge + CV) | 0.05 s | 0.05 s |
| Search | 0.26 ms (score all 12,870) | 0.9 s angle tuning on simulator + 2.4 s compile + 15.7 s IBM submit→result (3 s billed QPU) |
| Retrain the pick | 0.52 s | 0.52 s |

Hardware: 935 two-qubit gates after compilation (depth 986); 20.6% of 4096 shots were valid networks (random bits: 19.6%); 6 of the first 32 were valid.

## Keep 4 of 16 groups (75% of neurons removed, 1,820 possible networks)

| Pipeline | Time (one CPU core + QPU) | Retrainings | Validation accuracy of pick | Rank of pick among all networks |
|---|---:|---:|---:|---:|
| Brute force (regular way) | 9.0 min | 1,820 | 82.9% | 1 (true best) |
| Qismet + CPU search | 56.3 s | 193 | 80.4% | 264 (top 14.5%) |
| Qismet + QPU (ibm_fez) | 65.9 s | 193 | 78.9% | 1,159 (top 63.7%) |

Dense 32-neuron network: 82.6% validation accuracy. Brute force wall time with 6 parallel workers: 1.6 min.

Where the Qismet time goes:

| Step | Qismet + CPU | Qismet + QPU |
|---|---:|---:|
| Retrain 192 sample networks | 56.0 s | 56.0 s |
| Fit QUBO (ridge + CV) | 0.04 s | 0.04 s |
| Search | 0.05 ms (score all 1,820) | 0.1 s angle tuning on simulator + 1.4 s compile + 8.0 s IBM submit→result (3 s billed QPU) |
| Retrain the pick | 0.29 s | 0.29 s |

Hardware: 979 two-qubit gates after compilation (depth 1104); 7.6% of 4096 shots were valid networks (random bits: 2.8%); 2 of the first 32 were valid.

## Reading this

- **Qismet vs brute force** is the real speedup: retraining ~1.5% (keep 8) or ~11% (keep 4) of the networks and searching a fitted formula instead of retraining everything.
- **QPU vs CPU inside Qismet**: the search step itself takes milliseconds on a CPU. The QPU version adds simulator angle tuning, compilation and the IBM round trip, so it is slower here. The speedup comes from the Qismet method, not from the quantum hardware.
- **The hardware output is close to noise.** After compilation each circuit has ~950 two-qubit gates. At keep 8 only 20.6% of shots were valid networks, about the same as random bits (19.6%), so the QPU pick came from 6 essentially random valid shots ranked by the formula. Its good keep-8 rank is luck; its keep-4 rank (top 64%) is the other side of that.
- **The formula is the weak link.** Its exact optimum ranked only top 31% (keep 8) and top 14.5% (keep 4) by true validation loss, so even a perfect search of the formula would not find the true best network.
- Ranks use validation loss (the selection objective) from brute-force retraining of every network; validation accuracy can rank them slightly differently. The test set was not used.
- Brute force per-network timings come from 6 parallel workers, which made each retraining slower (~0.5 s vs ~0.15 s alone); Qismet's 192 retrainings use the same per-network timings, so the comparison is like for like.
