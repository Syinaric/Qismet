# Start here: the IBM hardware experiment

**Current status: the first real IBM hardware experiment is complete.** Open `Hardware_Results.md` for its results and job IDs. The revised constraint-preserving experiment is in the sibling `constrained_v4/` folder.

Open **Hardware_Walkthrough.ipynb** (or its HTML export) to see the problem, cost encoding, circuit, measured-cost feedback loop, and dry-run results. The first experiment retains the existing Fashion-MNIST fully connected model: 784→16→10, with eight groups of two hidden neurons. It retains four groups. This is one preselected initialization and does not establish statistical or quantum advantage.

The target is at most **four percentage points** of validation-accuracy loss relative to the dense network, with three preferred. Choosing the globally smallest qualifying network needs an additional size sweep. Test accuracy is reported only after selection.

## Three source files

| File | Responsibility |
|---|---|
| `circuit_core.py` | QUBO-to-Ising encoding, transparent circuit builder, bit order and measured cost |
| `ibm_backend.py` | Secure local account setup, actual IBM submission, transpilation and resumable job ledger |
| `workflow.py` | Classical optimizer using measured costs, mask selection, final retraining and timings |

The previous experiment's model and saved QUBO remain in the sibling `fashion_v2/` folder.

## Add your IBM credentials locally

Open `ibm_credentials.local.json` beside this README. Replace the two placeholder strings with your API key and full Open Plan instance CRN. Keep the quotes. Both the project and this folder's `.gitignore` exclude this file. This workspace currently has no Git repository; the rules apply when it is placed under Git. Keep this file out of shared ZIPs too.

The hardware runner loads this file automatically, regardless of your terminal's current directory. It takes priority over saved IBM accounts. An incomplete file stops execution with an actionable message. No `configure` step or second copy of the key is required.

To verify authentication, Open Plan access and device availability without submitting a quantum job, open PowerShell in `C:\Users\kpush\Documents\Codex\2026-11-04\i-w` and run:

```powershell
& .\work\venv\Scripts\python.exe .\outputs\hardware_v3\ibm_backend.py check
```

Find the instance CRN on IBM's Instances page. The check prints the verified plan and selected backend, without printing the key. A live connection can only be verified after real credentials are entered.

Tell Codex when you have filled in the file so it can continue the authorized run. Alternatively, the commands below advance it yourself. The optional `configure` command can still save credentials to IBM's account store; if the JSON file is removed, the runner falls back to the named account `qismet-open`.

## Run and resume hardware jobs

```powershell
& .\work\venv\Scripts\python.exe .\outputs\hardware_v3\workflow.py hardware
```

Each invocation submits at most one new job or collects an existing job. Repeat the same command while a job is pending; its saved ID prevents resubmission. The backend is pinned after the first submission. A failed or ambiguous submission stops rather than retrying automatically. Read its saved ledger and IBM Workloads before recovery. Do not delete the ledger to restart: it tracks usage already spent.

The sequence is:

1. Replay simulator-optimized angles on hardware (1,024 shots).
2. Three SPSA optimizer steps; each estimates a direction from two perturbed hardware evaluations (6 jobs, 1,024 shots each).
3. Independently measure the best observed parameters (1 job, 2,048 shots).

Hardware results determine optimizer updates. The simulator warm start and its prior classical work are explicitly recorded. This is a small-budget experiment, not a convergence guarantee. To keep final candidate selection comparable with best-of-32 random, the primary mask is chosen from a seeded random subset of 32 actual final shots; all 2,048 are retained for distribution diagnostics.

The allowance is **600 seconds**, with **120 seconds held in reserve**, at most **eight jobs**, and a **60-second server execution ceiling per job**. Pending/uncertain usage retains a reservation. Actual job usage and total elapsed time are recorded separately. These are conservative client guards, not an IBM billing guarantee; other jobs/users and already-spent account quota must be considered. Open Plan job mode can have queue latency that extends beyond your presentation deadline.

Once the state says complete:

```powershell
& .\work\venv\Scripts\python.exe .\outputs\hardware_v3\workflow.py evaluate-hardware
```

This retrains the observed mask from original weights, checks validation retention, and compares test accuracy and CPU inference latency against dense, exact-QUBO and best-of-random models. Smaller-model inference speed is distinct from a faster pruning search. Historical label-generation stages were not separately timed, so this package does **not** claim a precise end-to-end speed advantage. A new controlled timing experiment is required for that claim.

## Offline rehearsal

```powershell
& .\work\venv\Scripts\python.exe .\outputs\hardware_v3\workflow.py dry-run
& .\work\venv\Scripts\python.exe .\outputs\hardware_v3\workflow.py evaluate-dry-run
```

Dry-run outputs are saved under `runs/dry-run`; hardware evidence is written only under `runs/hardware`. The hardware adapter does not silently substitute a simulator. The notebook will display a pending status until actual hardware evidence exists.

Tested packages include Qiskit 2.6.1 and qiskit-ibm-runtime 0.51.0. The current client-side Sampler is used. No paid instance is selected automatically.

Sources: [IBM Sampler](https://quantum.cloud.ibm.com/docs/en/guides/get-started-with-sampler), [Runtime release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/release-notes), [job metrics](https://quantum.cloud.ibm.com/docs/en/guides/monitor-job).
