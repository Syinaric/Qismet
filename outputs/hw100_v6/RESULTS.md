# Result: pre-registered 100-qubit test

Rules: [PREREGISTRATION.md](PREREGISTRATION.md) (pushed in commit `bdb0d15` before any training or hardware run). Changes: [DEVIATIONS.md](DEVIATIONS.md).

## Verdict (primary metric: best-of-10 validation loss after retraining)

| QAOA on ibm_fez vs. | Seeds won | QAOA mean | Other mean | Pre-registered verdict |
|---|:-:|---:|---:|:-:|
| Random sampling | 2 of 3 | 0.5014 | 0.5085 | **QAOA wins** |
| Simulated annealing (matched) | 1 of 3 | 0.5014 | 0.5014 | not a win |
| Simulated annealing (long) | 0 of 3 | 0.5014 | 0.4976 | not a win |

## Per seed

| Seed | Method | Best val. loss | Best val. acc. | Test acc. of best pick | Mean val. loss of 10 | Share of candidates beating best random |
|---:|---|---:|---:|---:|---:|---:|
| 11 | QAOA (ibm_fez) | 0.5044 | 83.7% | 84.37% | 0.5194 | 83% |
| 11 | Random sampling | 0.5208 | 83.4% | 83.37% | 0.5294 | 0% |
| 11 | Simulated annealing (matched) | 0.5032 | 84.2% | 83.84% | 0.5197 | 54% |
| 11 | Simulated annealing (long) | 0.5041 | 83.6% | 84.04% | 0.5146 | 100% |
| 22 | QAOA (ibm_fez) | 0.5006 | 83.7% | 84.06% | 0.5166 | 84% |
| 22 | Random sampling | 0.5001 | 83.7% | 83.80% | 0.5284 | 0% |
| 22 | Simulated annealing (matched) | 0.5038 | 83.6% | 83.49% | 0.5141 | 50% |
| 22 | Simulated annealing (long) | 0.4952 | 83.5% | 83.92% | 0.5101 | 100% |
| 33 | QAOA (ibm_fez) | 0.4994 | 85.0% | 83.96% | 0.5117 | 82% |
| 33 | Random sampling | 0.5048 | 84.7% | 83.69% | 0.5217 | 0% |
| 33 | Simulated annealing (matched) | 0.4971 | 84.8% | 83.64% | 0.5119 | 57% |
| 33 | Simulated annealing (long) | 0.4934 | 84.8% | 84.27% | 0.5066 | 100% |

## Hardware

| Seed | Job | Two-qubit gates | Routing swaps | Hardware mean QUBO | Ideal (noiseless) mean QUBO | Random mean QUBO |
|---:|---|---:|---:|---:|---:|---:|
| 11 | `dbmn0l9kbkrc73cprbl0` | 200 | 0 | 0.5347 | 0.5332 | 0.5447 |
| 22 | `dbmvgcl8mdtc73d8nl7g` | 200 | 0 | 0.5224 | 0.5187 | 0.5479 |
| 33 | `dbmvnot8ft8s738vkd50` | 200 | 0 | 0.5186 | 0.5167 | 0.5353 |

Dense 400-neuron network test accuracy: seed 11 83.17%, seed 22 82.66%, seed 33 82.40%.
Each hardware job: 4,096 shots, 3 s billed QPU time. Formula held-out R²: 0.09, 0.03, 0.14.
