"""Brute-force ground truth: retrain EVERY valid mask for one seed and budget.

This is the "regular way" of finding the best pruned network: no surrogate, no
search, just retrain all C(16, keep) subnetworks with the exact protocol of
experiment.py and keep the one with the lowest validation loss.

    python brute_force.py --seed 1001 --keep 8 --workers 6

Writes brute_force/seed_<seed>_keep_<keep>.csv (one row per mask, resumable) and
brute_force/seed_<seed>_keep_<keep>_timing.json. Test data is never evaluated here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'brute_force'
_G = {}


def _init(seed):
    sys.path.insert(0, str(HERE))
    import torch
    import experiment as e
    torch.set_num_threads(1)
    arrays = np.load(HERE.parent / 'fashion_v2' / 'dataset_subset.npz')
    data = {s: (torch.tensor(arrays[s + '_images'].reshape(-1, 784) / 255., dtype=torch.float32),
                torch.tensor(arrays[s + '_labels'], dtype=torch.long)) for s in ['train', 'val']}
    initial = torch.load(HERE / f'seed_{seed}' / 'original_initialization.pt', weights_only=True)
    _G.update(e=e, data=data, initial=initial, seed=seed)


def _train(job):
    row, mask = job
    e, data = _G['e'], _G['data']
    net = e.rewind(_G['initial'], np.array(mask))
    t = time.perf_counter()
    e.f.train(net, data, _G['seed'])
    seconds = time.perf_counter() - t
    m = e.f.evaluate(net, data['val'])
    return dict(row=row, state=int(np.array(mask) @ (2 ** np.arange(16))), val_loss=m['loss'],
                val_accuracy=m['accuracy'], training_seconds=seconds, pid=os.getpid())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--keep', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import experiment as e
    OUT.mkdir(exist_ok=True)
    masks, ids, _ = e.design(args.keep)
    path = OUT / f'seed_{args.seed}_keep_{args.keep}.csv'
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    seen = set(done.row) if len(done) else set()
    jobs = [(int(r), masks[r].tolist()) for r in range(len(masks)) if r not in seen]
    print(f'seed {args.seed} keep {args.keep}: {len(masks)} masks, {len(jobs)} left, {args.workers} workers', flush=True)
    records = done.to_dict('records') if len(done) else []
    wall0 = time.perf_counter()
    with Pool(args.workers, initializer=_init, initargs=(args.seed,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_train, jobs, chunksize=4), 1):
            records.append(rec)
            if i % 200 == 0 or i == len(jobs):
                pd.DataFrame(records).to_csv(path, index=False)
                print(f'  {len(records)}/{len(masks)} retrained ({time.perf_counter() - wall0:.0f}s wall)', flush=True)
    wall = time.perf_counter() - wall0
    table = pd.DataFrame(records).sort_values('val_loss')
    table.to_csv(path, index=False)
    timing = dict(seed=args.seed, keep=args.keep, masks=len(masks), workers=args.workers,
                  wall_seconds_this_run=wall, retrained_this_run=len(jobs),
                  total_training_cpu_seconds=float(table.training_seconds.sum()),
                  best_state=int(table.iloc[0].state), best_val_loss=float(table.iloc[0].val_loss),
                  best_val_accuracy=float(table.iloc[0].val_accuracy))
    (OUT / f'seed_{args.seed}_keep_{args.keep}_timing.json').write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))


if __name__ == '__main__':
    main()
