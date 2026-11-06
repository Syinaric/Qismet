"""Small Lottery Ticket check on the Qismet Fashion-MNIST setup.

Same data split, network (784 -> 16 ReLU -> 10) and training recipe as
outputs/fashion_v2/fashion_experiment.py. One-shot, layer-wise magnitude
pruning of individual weights. For each sparsity we train the same pruned
connection pattern three ways:

  winning ticket : survivors reset to the ORIGINAL initial weights
  random reinit  : same mask, fresh random initial weights
  random mask    : random mask of the same size, original initial weights

Then iterative magnitude pruning, the procedure from Frankle & Carbin (2019):
train, prune 20% of the remaining fc1 weights and 10% of fc2, rewind the
survivors to the original initialization, and repeat. Each round's mask is
also trained from a fresh random initialization for comparison.

Results are written to lottery_ticket_results.json / *.csv and used by the video.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'outputs' / 'fashion_v2' / 'dataset_subset.npz'
SEEDS = [11, 22, 33, 44, 55]
SPARSITIES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
EPOCHS, BATCH, LR = 20, 256, 0.003
torch.set_num_threads(4)

d = np.load(DATA)
split = {s: (torch.tensor(d[s + '_images'].reshape(-1, 784) / 255., dtype=torch.float32),
             torch.tensor(d[s + '_labels'], dtype=torch.long)) for s in ['train', 'val', 'test']}


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train(net, seed, masks=None):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 100)
    x, y = split['train']
    curve = []
    for _ in range(EPOCHS):
        net.train()
        for idx in torch.randperm(len(y), generator=g).split(BATCH):
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(net(x[idx]), y[idx]).backward()
            opt.step()
            if masks:
                with torch.no_grad():
                    for name, m in masks.items():
                        getattr(net, name).weight.mul_(m)
        curve.append(accuracy(net, 'val'))
    return curve


def accuracy(net, s):
    net.eval()
    with torch.no_grad():
        return float((net(split[s][0]).argmax(1) == split[s][1]).float().mean())


def build(init, masks):
    net = MLP()
    net.load_state_dict(init)
    with torch.no_grad():
        for name, m in masks.items():
            getattr(net, name).weight.mul_(m)
    return net


rows = []
for seed in SEEDS:
    torch.manual_seed(seed)
    dense = MLP()
    init = {k: v.clone() for k, v in dense.state_dict().items()}
    train(dense, seed)
    trained = {k: v.clone() for k, v in dense.state_dict().items()}
    for p in SPARSITIES:
        masks, random_masks = {}, {}
        rng = torch.Generator().manual_seed(seed * 7 + int(p * 1000))
        for name in ['fc1', 'fc2']:
            w = trained[name + '.weight'].abs()
            k = int(round(p * w.numel()))
            m = torch.ones_like(w)
            if k:
                m.view(-1)[w.view(-1).argsort()[:k]] = 0
            masks[name] = m
            r = torch.ones(w.numel())
            if k:
                r[torch.randperm(w.numel(), generator=rng)[:k]] = 0
            random_masks[name] = r.view_as(w)
        torch.manual_seed(seed + 9999)
        fresh = MLP().state_dict()
        for variant, start, mk in [('winning ticket', init, masks),
                                   ('random reinit', fresh, masks),
                                   ('random mask', init, random_masks)]:
            net = build(start, mk)
            curve = train(net, seed, mk)
            rows.append(dict(seed=seed, sparsity=p, variant=variant,
                             test_accuracy=accuracy(net, 'test'), val_curve=curve))
            print(seed, p, variant, round(rows[-1]['test_accuracy'], 4), flush=True)

imp_rows = []
ROUNDS = 12
for seed in SEEDS:
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in MLP().state_dict().items()}
    torch.manual_seed(seed + 9999)
    fresh = MLP().state_dict()
    masks = {'fc1': torch.ones(16, 784), 'fc2': torch.ones(10, 16)}
    for rnd in range(ROUNDS + 1):
        sparsity = 1 - float(masks['fc1'].sum() + masks['fc2'].sum()) / (16 * 784 + 160)
        for variant, start in [('winning ticket', init), ('random reinit', fresh)]:
            net = build(start, masks)
            curve = train(net, seed, masks)
            imp_rows.append(dict(seed=seed, round=rnd, sparsity=sparsity, variant=variant,
                                 test_accuracy=accuracy(net, 'test'), val_curve=curve))
            if variant == 'winning ticket':
                trained = {k: v.clone() for k, v in net.state_dict().items()}
        print('IMP', seed, rnd, round(sparsity, 3),
              [round(r['test_accuracy'], 4) for r in imp_rows[-2:]], flush=True)
        # Prune the smallest surviving weights of the trained ticket.
        for name, rate in [('fc1', 0.2), ('fc2', 0.1)]:
            w = trained[name + '.weight'].abs().reshape(-1)
            m = masks[name].reshape(-1).clone()
            alive = m.nonzero().view(-1)
            drop = alive[w[alive].argsort()[:int(round(rate * len(alive)))]]
            m[drop] = 0
            masks[name] = m.view_as(masks[name])

with open(HERE / 'lottery_ticket_iterative.csv', 'w') as f:
    f.write('seed,round,sparsity,variant,test_accuracy\n')
    for r in imp_rows:
        f.write(f"{r['seed']},{r['round']},{r['sparsity']:.4f},{r['variant']},{r['test_accuracy']:.4f}\n")
isum = {}
for r in imp_rows:
    isum.setdefault((r['round'], r['variant']), []).append((r['sparsity'], r['test_accuracy']))
for (rnd, v), a in sorted(isum.items()):
    print(f'IMP round {rnd:2d} sparsity {np.mean([x[0] for x in a]):.3f} {v:15s} '
          f'{100*np.mean([x[1] for x in a]):.2f} +- {100*np.std([x[1] for x in a]):.2f}')

(HERE / 'lottery_ticket_results.json').write_text(json.dumps({'one_shot': rows, 'iterative': imp_rows}))
with open(HERE / 'lottery_ticket_results.csv', 'w') as f:
    f.write('seed,sparsity,variant,test_accuracy\n')
    for r in rows:
        f.write(f"{r['seed']},{r['sparsity']},{r['variant']},{r['test_accuracy']:.4f}\n")
summary = {}
for r in rows:
    summary.setdefault((r['sparsity'], r['variant']), []).append(r['test_accuracy'])
for (p, v), a in sorted(summary.items()):
    print(f'{p:.2f} {v:15s} {100*np.mean(a):.2f} +- {100*np.std(a):.2f}')
