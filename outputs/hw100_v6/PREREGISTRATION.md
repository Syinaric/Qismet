# Pre-registration: 100-qubit QAOA pruning test

Written 2026-11-06 04:59 UTC, before any network in this experiment was trained, before any hardware job, and before any result was seen.

- `protocol.json` SHA-256: `54f60190ff7c519e557b522797b3485c6d7dfd2106e52d73d38c6d898c324286`
- `hw100.py` SHA-256: `e5912916eadc78195f8ccde0b3ba8ddd56bae1f371d58131613da8629212b2ce`

## Question

At an equal candidate budget, do QAOA samples from IBM hardware lead to better retrained pruned networks than classical candidate generators?

## Setup

- Network: Fashion-MNIST, 784 → 400 → 10; 200 groups of 2 neurons; 100 qubits.
- Encoding: qubit q keeps group 2q if bit=0 else group 2q+1; all 2^100 bitstrings are valid; 50% of neurons kept.
- Training: fashion_v2.train: 20 epochs, Adam lr 0.003, batch 256, paired shuffles; rewind to original init.
- Seeds: [11, 22, 33]. QUBO: ridge (RidgeCV alphas logspace(-4,3,15), 5-fold) on 100 linear + chip-edge pairwise features; target = validation loss; 400 fit + 100 audit retrainings per seed.
- Hardware: ibm_fez, QAOA p=1, 4096 shots, couplings only on native chip edges. Angles: exact p=1 light-cone expectation; 41x41 grid then Nelder-Mead.

## Methods (equal 4,096-candidate budget except the reference ceiling)

- **QAOA (ibm_fez)**: 4096 hardware shots
- **Random sampling**: 4096 uniform random bitstrings
- **Simulated annealing (matched)**: 16 chains x 256 single-flip steps = 4096 QUBO evaluations; all visited states are candidates
- **Simulated annealing (long)**: 64 chains x 20000 steps = 1,280,000 QUBO evaluations (reference ceiling); candidates = states in the final 2000 steps

## Decision rule

- Each method's picks: the 10 distinct candidates with the lowest QUBO score (10 per method per seed), each retrained from the original initialization.
- Primary metric: best-of-10 validation loss after retraining (lower is better).
- **Win rule:** QAOA beats a method if its best-of-10 validation loss is lower in at least 2 of 3 seeds AND its mean best-of-10 validation loss over the 3 seeds is lower.
- Secondary (reported, not decisive): mean validation loss of the 10 picks; mean QUBO score of all candidates; share of candidates better than the best of 4096 random samples; hardware vs ideal expected QUBO score; test accuracy of each method's best pick (reported once, never used to choose).
- All seeds, all methods, all jobs, including failures.

## Known caveats (stated in advance)

- The native 100-qubit region has 100 couplings (nearly a tree), so the QUBO is easy for classical solvers; the test is about candidate quality at equal budget, not about beating the best classical solver.
- Restricting each qubit to a choice between two fixed groups searches 2^100 of the C(200,100) possible half-size networks.
- p=1 QAOA is the shallowest version; it was chosen so angles can be tuned exactly without simulating 100 qubits.
- Three seeds and one fixed data split: a demonstration-scale test, not a general benchmark.
