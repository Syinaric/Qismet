# Deviations from the pre-registration

## 1. Backend changed from ibm_fez to ibm_marrakesh

Recorded 2026-11-06 06:13 UTC, before any hardware result in this experiment existed.

**Why:** the seed-11 job on ibm_fez (`dbmm2vd8ft8s738v4n60`, submitted 05:11 UTC) sat in a queue
of 74–96 jobs. ibm_marrakesh had 1 queued job. The job was cancelled while still QUEUED, so it never
ran and returned no data. Seeds 22 and 33 had not been submitted.

**Why a new layout was needed:** marrakesh has the same coupling map as fez, but on the 100 qubits chosen
for fez it has one qubit with 48% readout error and three couplings above 2% CZ error. Running the fez
layout unchanged would have handicapped the quantum method.

**What changed:**
- `layout` re-run on ibm_marrakesh with the identical selection rule. The 100 qubits, and therefore the native
  couplings, change.
- The QUBO for each seed was refit with the identical procedure on the new coupling set, and the QAOA
  angles were retuned with the identical light-cone procedure. (The pairwise features are the chip couplings,
  so they must follow the layout.)

**What did not change:** seeds, the dense networks, the 500 retrained sample networks per seed (random
bitstrings drawn before any QUBO was fit, so independent of the layout), every method, the budgets, the
pick rule, the primary metric and the win rule. All methods are compared on the same refit QUBO.

**Superseded files:** kept in `superseded_fez/` (fez layout, the fez-fitted QUBOs, the cancelled job record).

## 2. Hardware step made resumable (process only)

Recorded 2026-11-06 07:17 UTC, before any hardware result existed.

While waiting for the resubmitted seed-11 job (`dbmn0l9kbkrc73cprbl0`), the laptop's internet connection dropped and the
script stopped. The job was unaffected and stayed QUEUED on IBM. `hw100.py` was changed so the hardware stage
collects an already-submitted job by ID instead of stopping, and retries through network errors. Nothing about the circuits,
data, methods or decision rule changed, but the file's SHA-256 no longer matches the pre-registered one. The diff against
commit `bdb0d15` shows exactly what changed (`git diff bdb0d15 -- outputs/hw100_v6/hw100.py`). The submit-to-result time
is now measured from the recorded submission timestamp, so it includes all queue time.

## 3. Report indexing bug fixed (analysis code only)

Recorded 2026-11-06 17:26 UTC. The report stage crashed before computing anything because the three seeds'
evaluation tables were concatenated with overlapping row numbers. Fixed by `ignore_index=True`. The metric and win rule are
unchanged, and the verdict is computed exactly as pre-registered.
