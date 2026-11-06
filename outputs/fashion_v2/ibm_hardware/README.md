# QAOA on IBM quantum hardware

Backend **ibm_fez** (156 qubits), job `dbmiv0674qds73dsca4g`, 4096 shots per circuit, run 2026-11-06 01:38 UTC.

The QAOA angles come from the original noiseless simulator run. Only the final sampling circuits (8 qubits, 2 QAOA layers, 56 RZZ gates before compilation) ran on hardware: one job, 10 circuits (5 seeds × 2 pruning levels). Selection repeats the original rule: the lowest-surrogate feasible mask among the first 32 shots. Validation numbers come from the existing retraining labels in `results/seed_*/labels.csv`; no network was retrained and the test set was not used.

## Sampling quality (mean of 5 seeds)

| Neurons removed | Feasible prob. ideal | hardware | uniform | Optimum prob. ideal | hardware | uniform | TV distance ideal vs hardware |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 0.768 | 0.471 | 0.273 | 0.011 | 0.008 | 0.004 | 0.311 |
| 75% | 0.530 | 0.352 | 0.109 | 0.019 | 0.011 | 0.004 | 0.291 |

"Feasible" means the measured mask keeps exactly the required number of groups. "Uniform" is what random guessing over all 256 bitstrings would give.

## Selected masks (32-shot rule)

| Neurons removed | Seed | Hardware pick | = simulator pick | = exact QUBO | Surrogate gap hw / sim | Val. loss hw / sim / exact / best possible |
|---:|---:|---|:-:|:-:|---:|---:|
| 50% | 11 | `10000111` | no | no | 0.0045 / 0.0002 | 0.5750 / 0.5654 / 0.5362 / 0.5362 |
| 50% | 22 | `00011110` | no | no | 0.0205 / 0.0119 | 0.5968 / 0.5533 / 0.5623 / 0.5305 |
| 50% | 33 | `11010010` | no | yes | 0.0000 / 0.0275 | 0.5487 / 0.5606 / 0.5487 / 0.5289 |
| 50% | 44 | `00100111` | no | no | 0.0092 / 0.0000 | 0.5576 / 0.5583 / 0.5583 / 0.5287 |
| 50% | 55 | `01001011` | no | no | 0.0285 / 0.0191 | 0.5662 / 0.5699 / 0.5684 / 0.5584 |
| 75% | 11 | `00011000` | no | yes | 0.0000 / 0.0024 | 0.6519 / 0.6426 / 0.6519 / 0.6368 |
| 75% | 22 | `00011000` | yes | yes | 0.0000 / 0.0000 | 0.6618 / 0.6618 / 0.6618 / 0.6351 |
| 75% | 33 | `10000001` | no | yes | 0.0000 / 0.0124 | 0.6852 / 0.7007 / 0.6852 / 0.6610 |
| 75% | 44 | `01000010` | no | no | 0.0202 / 0.0000 | 0.7081 / 0.6691 / 0.6691 / 0.6349 |
| 75% | 55 | `10001000` | yes | no | 0.0321 / 0.0321 | 0.6610 / 0.6610 / 0.6531 / 0.6531 |

## Mean validation accuracy of the selected mask

| Neurons removed | Hardware pick | Simulator pick | Exact QUBO |
|---:|---:|---:|---:|
| 50% | 79.74% | 80.36% | 80.92% |
| 75% | 74.74% | 76.00% | 75.24% |

Mask strings list groups G1…G8 left to right (1 = kept). This is one hardware run; noise, compilation and calibration vary between runs and devices. It is not evidence of quantum advantage: with only 70 or 28 feasible masks, exhaustive classical search is instant.
