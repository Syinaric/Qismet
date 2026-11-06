"""Small, deterministic structured-pruning experiment for the Qismet notebook."""
import copy
import json
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

# %% Milestone 1: data, training, structural pruning, and exact weight rewinding
class TinyMLP(nn.Module):
    """Supported model: two inputs, one tanh hidden layer, two class logits."""
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def prepare_data(seed=2026, samples=1500):
    x, y = make_moons(n_samples=samples, noise=0.20, random_state=seed)
    xt, xrest, yt, yrest = train_test_split(x, y, test_size=0.4, stratify=y, random_state=seed)
    xv, xe, yv, ye = train_test_split(xrest, yrest, test_size=0.5, stratify=yrest, random_state=seed+1)
    scaler = StandardScaler().fit(xt)  # Never fit preprocessing on validation/test.
    return {name: (torch.tensor(scaler.transform(xx), dtype=torch.float32), torch.tensor(yy, dtype=torch.long))
            for name, xx, yy in [('train', xt, yt), ('val', xv, yv), ('test', xe, ye)]}


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def rewind_subnetwork(initial, mask):
    mask = np.asarray(mask)
    if mask.ndim != 1 or not np.isin(mask, [0, 1]).all() or len(mask) != initial['fc1.bias'].numel():
        raise ValueError('mask must contain one binary value per hidden neuron')
    indices = torch.tensor(np.flatnonzero(mask), dtype=torch.long)
    if len(indices) == 0:
        raise ValueError('Keep at least one neuron')
    # Constructor randomness is completely overwritten by original surviving weights.
    model = TinyMLP(len(indices))
    expected = {'fc1.weight': initial['fc1.weight'][indices], 'fc1.bias': initial['fc1.bias'][indices],
                'fc2.weight': initial['fc2.weight'][:, indices], 'fc2.bias': initial['fc2.bias']}
    model.load_state_dict(expected)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())
    return model


def train(model, data, epochs=200, lr=0.03):
    # Full-batch CPU training: identical examples/order/budget and a fresh optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    x, y = data['train']
    history = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def evaluate(model, split):
    model.eval()
    with torch.no_grad():
        x, y = split
        logits = model(x)
        return {'loss': float(nn.functional.cross_entropy(logits, y)),
                'accuracy': float((logits.argmax(1) == y).float().mean())}


def magnitude_mask(model, keep):
    # Rank trained hidden units by outgoing-weight L2 norm (a simple baseline).
    scores = model.fc2.weight.detach().square().sum(dim=0).sqrt().numpy()
    mask = np.zeros(len(scores), dtype=int)
    mask[np.argsort(scores, kind='stable')[-keep:]] = 1
    return mask


def milestone_one(data, seed=11, hidden=8, keep=4, epochs=200):
    torch.manual_seed(seed)
    dense = TinyMLP(hidden)
    initial = clone_state(dense)
    history = train(dense, data, epochs)
    mask = magnitude_mask(dense, keep)
    sparse = rewind_subnetwork(initial, mask)
    # Structural slicing must equal masking the original dense network's activations.
    original = TinyMLP(hidden)
    original.load_state_dict(initial)
    with torch.no_grad():
        x = data['train'][0][:20]
        gated = original.fc2(torch.tanh(original.fc1(x)) * torch.tensor(mask))
        torch.testing.assert_close(sparse(x), gated)
    sparse_history = train(sparse, data, epochs)
    assert all(torch.equal(initial[k], original.state_dict()[k]) for k in initial)
    rows = []
    for name, net in [('Dense', dense), ('Magnitude + rewind', sparse)]:
        rows.append({'method': name, 'parameters': sum(p.numel() for p in net.parameters()),
                     **{f'val_{k}': v for k, v in evaluate(net, data['val']).items()}})
    # Test remains untouched here. Final test reporting happens after all selections.
    return pd.DataFrame(rows), initial, mask, history, sparse_history

# %% Milestones 2 and 3: independent mask holdout and quadratic retraining surrogate
def all_masks(hidden):
    return ((np.arange(2**hidden)[:, None] >> np.arange(hidden)) & 1).astype(int)


def quadratic_features(masks):
    masks = np.atleast_2d(masks)
    pairs = list(combinations(range(masks.shape[1]), 2))
    return np.column_stack([masks] + [masks[:, i] * masks[:, j] for i, j in pairs])


def label_design(hidden, keep, seed):
    """128 fit masks, up to 48 audit masks, including half of feasible masks each.

    Default n=8,k=4: 35 feasible fit masks + 93 other fit masks;
    35 feasible audit masks + 13 other audit masks. No observed labels guide this split.
    """
    masks = all_masks(hidden)
    rng = np.random.default_rng(seed)
    feasible = rng.permutation(np.flatnonzero(masks.sum(1) == keep))
    other = rng.permutation(np.flatnonzero((masks.sum(1) != keep) & (masks.sum(1) > 0)))
    cut = max(1, len(feasible)//2)
    fit_feasible = feasible[:min(cut, 64)]
    audit_feasible = feasible[cut:cut+35]
    other_fit_n = min(128-len(fit_feasible), max(1, len(other)-13))
    fitting = np.r_[fit_feasible, other[:other_fit_n]]
    audit = np.r_[audit_feasible, other[other_fit_n:other_fit_n+max(0, 48-len(audit_feasible))]]
    assert not set(fitting) & set(audit)
    return masks, fitting, audit


def surrogate_metrics(true, predicted):
    return {'mae': float(mean_absolute_error(true, predicted)),
            'r2': float(r2_score(true, predicted)) if len(true) > 1 else float('nan'),
            'spearman': float(spearmanr(true, predicted).statistic) if len(true) > 1 else float('nan'),
            'n_masks': len(true)}


def fit_surrogate(initial, data, keep, seed, epochs=200):
    hidden = len(initial['fc1.bias'])
    masks, fit_ids, audit_ids = label_design(hidden, keep, seed)
    labels = {}
    start = time.perf_counter()
    for idx, state in enumerate(np.r_[fit_ids, audit_ids]):
        subnet = rewind_subnetwork(initial, masks[state])
        train(subnet, data, epochs)
        labels[int(state)] = evaluate(subnet, data['val'])['loss']
        if (idx+1) % 40 == 0:
            print(f'  seed {seed}: retrained {idx+1}/{len(fit_ids)+len(audit_ids)} masks', flush=True)
    features = quadratic_features(masks)
    # Hyperparameter selection sees fit masks only; audit masks never tune the model.
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(Ridge(), {'alpha': [0.0001, 0.01, 0.1, 1.0, 10.0]},
                          scoring='neg_mean_squared_error', cv=cv)
    search.fit(features[fit_ids], [labels[int(i)] for i in fit_ids])
    reg = search.best_estimator_
    predictions = reg.predict(features)
    audit_y = np.array([labels[int(i)] for i in audit_ids])
    in_budget = masks[audit_ids].sum(1) == keep
    quality = {'all_audit': surrogate_metrics(audit_y, predictions[audit_ids]),
               'feasible_audit': surrogate_metrics(audit_y[in_budget], predictions[audit_ids][in_budget]),
               'ridge_alpha': float(reg.alpha), 'fit_retrains': len(fit_ids), 'audit_retrains': len(audit_ids),
               'label_seconds': time.perf_counter()-start}
    linear = reg.coef_[:hidden].copy()
    pairwise = np.zeros((hidden, hidden))
    for coefficient, (i, j) in zip(reg.coef_[hidden:], combinations(range(hidden), 2)):
        pairwise[i, j] = coefficient
    direct = reg.intercept_ + masks @ linear + np.einsum('bi,ij,bj->b', masks, pairwise, masks)
    np.testing.assert_allclose(predictions, direct, atol=1e-10)
    label_table = pd.DataFrame([{'state': int(i), 'mask': ''.join(map(str, masks[i])),
                                'split': 'fit' if i in set(fit_ids) else 'audit',
                                'kept': int(masks[i].sum()), 'val_loss': labels[int(i)],
                                'predicted_val_loss': float(predictions[i])} for i in np.r_[fit_ids, audit_ids]])
    return linear, pairwise, float(reg.intercept_), predictions, quality, label_table

# %% Milestone 5: classical comparisons and untouched-test evaluation
def swap_search(start_mask, predicted, masks):
    """Deterministic greedy one-out/one-in search, using only the surrogate."""
    current = start_mask.copy()
    weights = 2**np.arange(len(current))
    evaluations = 0
    while True:
        state = int(current @ weights)
        best_state, best_value = state, predicted[state]
        for i in np.flatnonzero(current):
            for j in np.flatnonzero(1-current):
                candidate = state - int(weights[i]) + int(weights[j])
                evaluations += 1
                if predicted[candidate] < best_value-1e-12:
                    best_state, best_value = candidate, predicted[candidate]
        if best_state == state:
            return current, evaluations
        current = masks[best_state].copy()


def run_experiment(model, data, pruning_percentage=50, seed=11, epochs=200, shots=32):
    """Input an UNTRAINED eight-hidden-unit TinyMLP; its current weights are initialization.

    All samples, hyperparameters and seeds are declared before observing test performance.
    This is an exponential toy demo, not support for arbitrary networks.
    """
    from qaoa_utils import solve_qaoa
    if type(model) is not TinyMLP or model.fc1.out_features != 8:
        raise ValueError('Use the supported TinyMLP architecture with eight hidden neurons')
    if not 0 < pruning_percentage < 100:
        raise ValueError('pruning_percentage must be between 0 and 100')
    hidden = model.fc1.out_features
    keep = int(np.floor(hidden*(1-pruning_percentage/100)+0.5))
    if not 1 <= keep < hidden:
        raise ValueError('The requested percentage must retain 1..hidden-1 neurons after rounding')
    start = time.perf_counter()
    initial = clone_state(model)
    dense = copy.deepcopy(model)
    train(dense, data, epochs)
    mag = magnitude_mask(dense, keep)
    linear, pairwise, constant, predicted, quality, labels = fit_surrogate(initial, data, keep, seed, epochs)
    qaoa_start = time.perf_counter()
    quantum = solve_qaoa(linear, pairwise, constant, keep, seed=seed+1000, shots=shots)
    quantum_seconds = time.perf_counter()-qaoa_start
    masks = all_masks(hidden)
    feasible = np.flatnonzero(masks.sum(1) == keep)
    rng = np.random.default_rng(seed+2000)
    random_ids = rng.choice(feasible, size=shots, replace=True)
    raw_uniform_ids = rng.integers(0, len(masks), size=shots)
    raw_feasible = raw_uniform_ids[masks[raw_uniform_ids].sum(1) == keep]
    best_random = int(random_ids[np.argmin(predicted[random_ids])])
    greedy, greedy_evals = swap_search(masks[random_ids[0]], predicted, masks)
    exact = int(feasible[np.argmin(predicted[feasible])])
    qmask = np.asarray(quantum['selected_mask'], dtype=int)
    selected = {'Magnitude + rewind': mag, 'Random mask': masks[random_ids[0]],
                'Best of 32 random feasible': masks[best_random], 'Classical swap': greedy,
                'Exact QUBO': masks[exact], 'QAOA': qmask}
    # Names reflect actual configured budget.
    selected[f'Best of {shots} random feasible'] = selected.pop('Best of 32 random feasible')
    if len(raw_feasible):
        raw_best = int(raw_feasible[np.argmin(predicted[raw_feasible])])
        selected[f'Best of {shots} uniform raw'] = masks[raw_best]
    rows = []
    for name, mask in [('Dense', np.ones(hidden, int)), *selected.items()]:
        net = copy.deepcopy(dense) if name == 'Dense' else rewind_subnetwork(initial, mask)
        if name != 'Dense':
            train(net, data, epochs)
        state = int(mask @ (2**np.arange(hidden)))
        rows.append({'seed': seed, 'method': name, 'mask': ''.join(map(str, mask)),
                     'kept': int(mask.sum()), 'parameters': sum(p.numel() for p in net.parameters()),
                     'surrogate_loss': float(predicted[state]),
                     **{f'{s}_{k}': v for s in ['val','test'] for k,v in evaluate(net, data[s]).items()}})
    torch.manual_seed(seed+3000)
    fresh = TinyMLP(keep)
    train(fresh, data, epochs)
    rows.append({'seed': seed, 'method': 'Fresh initialization', 'mask': 'new weights', 'kept': keep,
                 'parameters': sum(p.numel() for p in fresh.parameters()), 'surrogate_loss': np.nan,
                 **{f'{s}_{k}': v for s in ['val','test'] for k,v in evaluate(fresh, data[s]).items()}})
    # A distributional comparison avoids attributing best-of-N postselection to QAOA alone.
    feasible_values = predicted[feasible]
    top_cutoff = np.quantile(feasible_values, 0.10)
    probabilities = np.asarray(quantum['probabilities'])
    feasible_mass = probabilities[feasible].sum()
    top_ids = feasible[feasible_values <= top_cutoff+1e-12]
    quality.update({'seed': seed, 'keep': keep, 'actual_pruning_percentage': 100*(1-keep/hidden),
                    'qaoa_seconds': quantum_seconds, 'qaoa_feasible_mass': float(feasible_mass),
                    'qaoa_top_decile_conditional_mass': float(probabilities[top_ids].sum()/feasible_mass),
                    'uniform_top_decile_mass': len(top_ids)/len(feasible),
                    'raw_uniform_feasible_candidates': len(raw_feasible),
                    'greedy_evaluations': greedy_evals, 'total_seconds': time.perf_counter()-start})
    qnet = rewind_subnetwork(initial, qmask)
    train(qnet, data, epochs)
    assert all(torch.equal(initial[k], model.state_dict()[k]) for k in initial)
    return {'table': pd.DataFrame(rows), 'quality': quality, 'labels': labels, 'qaoa': quantum,
            'linear': linear, 'pairwise': pairwise, 'constant': constant, 'initial': initial,
            'qaoa_model': qnet, 'selected_masks': selected}
