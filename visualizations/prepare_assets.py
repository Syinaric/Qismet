"""Extract real numbers from the Qismet run for the explainer video.

Uses seed 11's trained dense network (results/seed_11/selected_state_255.pt),
its original initialization, the saved data split and the aggregate results.
Writes video_assets.npz and video_assets.json next to this file.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'outputs' / 'fashion_v2'
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

d = np.load(RUN / 'dataset_subset.npz')
dense = torch.load(RUN / 'results' / 'seed_11' / 'selected_state_255.pt')['state_dict']
init = torch.load(RUN / 'results' / 'seed_11' / 'original_initialization.pt')
W1, b1 = dense['fc1.weight'].numpy(), dense['fc1.bias'].numpy()
W2, b2 = dense['fc2.weight'].numpy(), dense['fc2.bias'].numpy()

x = d['test_images'].reshape(-1, 784) / 255.
hidden = np.maximum(x @ W1.T + b1, 0)
logits = hidden @ W2.T + b2
probs = np.exp(logits - logits.max(1, keepdims=True))
probs /= probs.sum(1, keepdims=True)
pred = probs.argmax(1)
labels = d['test_labels']
acc = float((pred == labels).mean())

# One confidently-correct test example per class for the gallery / forward pass.
gallery = []
for c in range(10):
    ids = np.flatnonzero((labels == c) & (pred == c))
    gallery.append(int(ids[np.argsort(-probs[ids, c])[len(ids) // 4]]))
# A grid of many training images for the opening shot.
rng = np.random.default_rng(3)
grid_ids = rng.choice(len(d['train_labels']), 96, replace=False)

curves = pd.read_csv(RUN / 'results' / 'learning_curves.csv')
dense_curve = (curves[(curves.method == 'Dense') & (curves.pruning == 50.0)]
               .groupby('epoch').val_accuracy.mean().values)
comp = pd.read_csv(RUN / 'results' / 'comparison.csv')
table = (comp.groupby(['pruning', 'method']).test_accuracy.mean() * 100).round(2)
table_sd = (comp.groupby(['pruning', 'method']).test_accuracy.std() * 100).round(2)

# Accuracy of the untrained network (original random weights) on validation data.
W1i, b1i = init['fc1.weight'].numpy(), init['fc1.bias'].numpy()
W2i, b2i = init['fc2.weight'].numpy(), init['fc2.bias'].numpy()
xv = d['val_images'].reshape(-1, 784) / 255.
init_val_accuracy = float(((np.maximum(xv @ W1i.T + b1i, 0) @ W2i.T + b2i).argmax(1)
                           == d['val_labels']).mean())

lt = json.loads((HERE / 'lottery_ticket_results.json').read_text())


def summarize(rows, key):
    out = {}
    for r in rows:
        out.setdefault((r[key], r['variant']), []).append(r)
    res = []
    for (k, v), rs in sorted(out.items()):
        res.append(dict(key=k, variant=v,
                        sparsity=float(np.mean([r['sparsity'] for r in rs])),
                        mean=float(np.mean([r['test_accuracy'] for r in rs]) * 100),
                        sd=float(np.std([r['test_accuracy'] for r in rs]) * 100)))
    return res


np.savez_compressed(HERE / 'video_assets.npz',
                    W1=W1, b1=b1, W2=W2, b2=b2,
                    W1_init=init['fc1.weight'].numpy(), W2_init=init['fc2.weight'].numpy(),
                    gallery_images=d['test_images'][gallery], gallery_labels=labels[gallery],
                    gallery_hidden=hidden[gallery], gallery_probs=probs[gallery],
                    grid_images=d['train_images'][grid_ids], grid_labels=d['train_labels'][grid_ids])
meta = dict(classes=CLASSES, dense_test_accuracy=acc, dense_val_curve=dense_curve.tolist(),
            init_val_accuracy=init_val_accuracy,
            results={f'{p}|{m}': float(v) for (p, m), v in table.items()},
            results_sd={f'{p}|{m}': float(v) for (p, m), v in table_sd.items()},
            lottery_one_shot=summarize(lt['one_shot'], 'sparsity'),
            lottery_iterative=summarize(lt['iterative'], 'round'))
(HERE / 'video_assets.json').write_text(json.dumps(meta, indent=1))
print('dense test accuracy', acc)
print(table)
