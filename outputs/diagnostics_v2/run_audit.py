"""Audit the original three-seed Two Moons experiment without opening test data.

Run from any directory using the same environment as Qismet. Inputs are sibling
results/ checkpoints and immutable fitting/audit assignments. Output is this folder.
"""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import torch

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent
sys.path.insert(0, str(SOURCE))
from experiment_utils import prepare_data, all_masks, rewind_subnetwork, evaluate, TinyMLP

SEEDS = [11, 22, 33]
EPOCHS = 200
ALPHAS = [0.0001, 0.01, 0.1, 1.0, 10.0]
MASKS = all_masks(8)
FEASIBLE = np.flatnonzero(MASKS.sum(axis=1) == 4)
data = prepare_data()
del data['test']  # This follow-up never accesses test examples or test scores.


def state_from_mask(mask):
    return int(np.array(list(map(int, str(mask)))) @ (2**np.arange(8)))


def fit_and_predict(seed, labels):
    fit = labels[labels.split == 'fit']
    search = GridSearchCV(Ridge(), {'alpha': ALPHAS}, scoring='neg_mean_squared_error',
                          cv=KFold(5, shuffle=True, random_state=seed))
    search.fit(MASKS[fit.state], fit.val_loss)
    linear = search.predict(MASKS)
    coefficients = np.load(SOURCE / f'results/qubo_seed_{seed}.npz')
    quadratic = (coefficients['constant'] + MASKS @ coefficients['linear']
                 + np.einsum('bi,ij,bj->b', MASKS, coefficients['pairwise'], MASKS))
    np.testing.assert_allclose(quadratic[labels.state], labels.predicted_val_loss, atol=1e-10)
    return linear, quadratic, float(search.best_params_['alpha'])


def retrain_validation(initial, mask, seed, state):
    model = rewind_subnetwork(initial, mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    x, y = data['train']
    curve = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        val = evaluate(model, data['val'])
        curve.append({'seed': seed, 'state': int(state), 'epoch': epoch,
                      'val_loss': val['loss'], 'val_accuracy': val['accuracy']})
    return curve


def main():
    began = time.perf_counter()
    metrics, comparisons, predictions, selected, curves, spreads = [], [], [], [], [], []
    existing = pd.read_csv(SOURCE / 'results/comparison_all_seeds.csv', usecols=['seed','method','mask'], dtype={'mask': str})
    for seed in SEEDS:
        labels = pd.read_csv(SOURCE / f'results/surrogate_labels_seed_{seed}.csv', dtype={'mask': str})
        linear, quadratic, alpha = fit_and_predict(seed, labels)
        q = json.loads((SOURCE / f'results/qaoa_seed_{seed}.json').read_text())
        initial = torch.load(SOURCE / f'results/original_initialization_seed_{seed}.pt', weights_only=True)
        audit = labels[(labels.split == 'audit') & (labels.kept == 4)]
        audit_ids = audit.state.to_numpy()
        for name, prediction in [('Linear only', linear), ('Quadratic', quadratic)]:
            row = {'seed': seed, 'model': name, 'n_audit': len(audit),
                   'MAE': mean_absolute_error(audit.val_loss, prediction[audit_ids]),
                   'R2': r2_score(audit.val_loss, prediction[audit_ids]),
                   'Spearman_cost_vs_loss': spearmanr(audit.val_loss, prediction[audit_ids]).statistic,
                   'ridge_alpha': alpha if name == 'Linear only' else json.loads((SOURCE/f'results/quality_seed_{seed}.json').read_text())['ridge_alpha']}
            metrics.append(row)
        print('Early surrogate metrics:', metrics[-2:], flush=True)
        pd.DataFrame(metrics).to_csv(OUT / 'surrogate_metrics.csv', index=False)
        original = existing[existing.seed == seed].set_index('method')
        random_state = state_from_mask(original.loc['Random mask','mask'])
        qstate = q['selected_index']
        exact = int(FEASIBLE[np.argmin(quadratic[FEASIBLE])])
        comparisons.append({'seed': seed, 'qaoa_mask': ''.join(map(str,MASKS[qstate])),
                            'random_mask': ''.join(map(str,MASKS[random_state])),
                            'exact_qubo_mask': ''.join(map(str,MASKS[exact])),
                            'qaoa_equals_random': qstate == random_state,
                            'qaoa_random_hamming': int(np.count_nonzero(MASKS[qstate] != MASKS[random_state])),
                            'qaoa_equals_exact': qstate == exact,
                            'qaoa_surrogate_gap': quadratic[qstate]-quadratic[exact],
                            'qaoa_feasible_mass': q['feasible_probability'],
                            'qaoa_top10_given_feasible': q['conditional_top_decile_probability'],
                            'uniform_top10_given_feasible': q['uniform_conditional_top_decile_probability']})
        final_by_state = {}
        for count, state in enumerate(FEASIBLE):
            curve = retrain_validation(initial, MASKS[state], seed, state)
            curves.extend(curve)
            final_by_state[int(state)] = curve[-1]
            previous = labels[labels.state == state].iloc[0]
            np.testing.assert_allclose(curve[-1]['val_loss'], previous.val_loss, atol=1e-7)
            predictions.append({'seed': seed, 'state': int(state), 'mask': ''.join(map(str,MASKS[state])),
                                'split': previous.split, 'val_loss': curve[-1]['val_loss'],
                                'val_accuracy': curve[-1]['val_accuracy'],
                                'linear_prediction': linear[state], 'quadratic_prediction': quadratic[state]})
            if (count+1) % 20 == 0:
                print(f'Seed {seed}: {count+1}/70 exhaustive feasible retrainings', flush=True)
        oracle = min(FEASIBLE, key=lambda s: final_by_state[int(s)]['val_loss'])
        audit_oracle = min(audit_ids, key=lambda s: final_by_state[int(s)]['val_loss'])
        selections = [('QAOA', qstate, 'all70'), ('Random', random_state, 'all70'),
                      ('Exact linear', int(FEASIBLE[np.argmin(linear[FEASIBLE])]), 'all70'),
                      ('Exact quadratic', exact, 'all70'), ('Validation oracle (diagnostic)', oracle, 'all70'),
                      ('Audit-only linear', int(audit_ids[np.argmin(linear[audit_ids])]), 'audit35'),
                      ('Audit-only quadratic', int(audit_ids[np.argmin(quadratic[audit_ids])]), 'audit35'),
                      ('Audit-only oracle (diagnostic)', audit_oracle, 'audit35')]
        for name, state, pool in selections:
            metric = final_by_state[int(state)]
            baseline = final_by_state[int(oracle if pool == 'all70' else audit_oracle)]
            selected.append({'seed': seed, 'method': name, 'state': int(state),
                             'mask': ''.join(map(str,MASKS[state])), 'candidate_pool': pool,
                             'val_loss': metric['val_loss'], 'val_accuracy': metric['val_accuracy'],
                             'val_loss_regret': metric['val_loss']-baseline['val_loss']})
        values = pd.DataFrame(final_by_state.values())
        spreads.append({'seed': seed, 'min_val_accuracy': values.val_accuracy.min(),
                        'max_val_accuracy': values.val_accuracy.max(),
                        'median_val_accuracy': values.val_accuracy.median(),
                        'fraction_within_1pp_best': float((values.val_accuracy >= values.val_accuracy.max()-.01-1e-9).mean()),
                        'min_val_loss': values.val_loss.min(), 'max_val_loss': values.val_loss.max()})
        for row in metrics[-2:]:
            pred = linear if row['model'] == 'Linear only' else quadratic
            acc = [final_by_state[int(s)]['val_accuracy'] for s in audit_ids]
            row['Spearman_cost_vs_accuracy'] = spearmanr(pred[audit_ids], acc).statistic
    frames = {'surrogate_metrics':metrics, 'solver_comparison':comparisons, 'all_feasible_masks':predictions,
              'selection_comparison':selected, 'validation_curves':curves, 'feasible_performance_spread':spreads}
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    make_plot(pd.DataFrame(predictions), pd.DataFrame(selected), pd.DataFrame(curves), pd.DataFrame(metrics))
    write_report(frames, time.perf_counter()-began)


def make_plot(predictions, selected, curves, metrics):
    fig, axs = plt.subplots(2, 2, figsize=(12,9))
    for seed in SEEDS:
        audit = predictions[(predictions.seed == seed) & (predictions.split == 'audit')]
        axs[0,0].scatter(audit.quadratic_prediction, audit.val_loss, s=23, alpha=.7, label=f'Seed {seed}')
    axs[0,0].set(xlabel='Quadratic predicted validation loss', ylabel='Actual validation loss',
                 title='Prediction on held-out four-neuron masks')
    axs[0,0].legend()
    mean = metrics.groupby('model').Spearman_cost_vs_loss.mean()
    axs[0,1].bar(mean.index, mean.values, color=['#94a3b8','#2563eb'])
    for i, name in enumerate(mean.index):
        axs[0,1].scatter(np.full(3,i),metrics[metrics.model==name].Spearman_cost_vs_loss,color='black',s=18)
    axs[0,1].set(ylabel='Spearman correlation (higher is better)', ylim=(-.05,1),title='Linear versus pairwise surrogate')
    for seed in SEEDS:
        vals = predictions[predictions.seed == seed].val_accuracy*100
        axs[1,0].plot(np.sort(vals),label=f'Seed {seed}')
    axs[1,0].set(xlabel='Mask rank by validation accuracy (70 masks)',ylabel='Final validation accuracy (%)',
                 title='How much does mask choice matter?')
    axs[1,0].legend()
    methods = ['QAOA','Random','Exact linear','Exact quadratic']
    for method in methods:
        pairs = selected[selected.method == method][['seed','state']]
        c = curves.merge(pairs,on=['seed','state']).groupby('epoch').val_accuracy.mean()*100
        axs[1,1].plot(c.index,c.values,label=method)
    axs[1,1].set(xlabel='Training epoch (one full-batch update)',ylabel='Mean validation accuracy (%)',
                 title='Rewound retraining curves across three seeds')
    axs[1,1].legend(fontsize=9)
    fig.suptitle('Qismet: diagnose the surrogate before scaling',fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT/'qubo_audit.png',dpi=150,bbox_inches='tight')
    plt.close(fig)


def write_report(frames, seconds):
    metrics=pd.DataFrame(frames['surrogate_metrics'])
    selection=pd.DataFrame(frames['selection_comparison'])
    solver=pd.DataFrame(frames['solver_comparison'])
    def table(df):
        # Avoid a dependency on tabulate.
        cols=list(df.columns)
        return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+ '\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
    report = '''# QUBO diagnostic: original Two Moons experiment

This is an **exploratory follow-up** on the original three initializations and mask audit split.
It reuses existing audit labels; it is not a fresh confirmatory test or a new dataset trial.
Test data was not evaluated or used in this audit; saved test scores were not read.

## Does QAOA solve the surrogate?

'''+table(solver[['seed','qaoa_mask','random_mask','exact_qubo_mask','qaoa_random_hamming','qaoa_equals_exact','qaoa_surrogate_gap']].round(6))+'''

QAOA's selected mask differs from the single random mask in every initialization, and does not
equal the exact surrogate optimum in any of the three. The gap is measured in unpenalized
predicted validation cross-entropy; all compared masks satisfy the same cardinality.

## Do pairwise terms improve prediction?

Both models use exactly the same original 128 fitting labels and five-fold CV with the same
alpha grid. The 35 held-out feasible masks per initialization are excluded from fitting and CV.
The original quadratic coefficients are reused unchanged. Positive cost/loss correlation is
desirable; negative cost/accuracy correlation is desirable.

'''+table(metrics.round(4))+'''

Mean audit metrics:

'''+table(metrics.groupby('model')[['MAE','R2','Spearman_cost_vs_loss','Spearman_cost_vs_accuracy']].mean().round(4).reset_index())+'''

## Does selecting by the surrogate help?

All 70 feasible masks were retrained from saved initializations, with exactly the original
200 full-batch Adam steps and fresh optimizer. Final validation losses matched the saved
labels within 1e-7. Validation accuracy was measured at every epoch. The oracle below is a
retrospective validation-loss reference, not a deployable selector or test-set optimum.
Audit-only selection compares both surrogates on masks whose labels neither saw during fitting.

'''+table(selection.groupby(['candidate_pool','method'])[['val_loss','val_accuracy','val_loss_regret']].mean().round(5).reset_index())+'''

## How much performance variation is available?

'''+table(pd.DataFrame(frames['feasible_performance_spread']).round(5))+f'''

The plot shows held-out cost/loss scatter, the pairwise ablation, the complete feasible-mask
performance spread, and selected-mask retraining curves. These are descriptions of three
initializations, not statistically established advantages. Pairwise features can improve fit
without reliably improving selection; read both tables together. A stronger dataset will not
automatically repair either surrogate error or QAOA optimization error.

Reproducibility: run `python run_audit.py` from this folder (or any working directory), with
the original sibling `results/` and `experiment_utils.py` available. It writes six CSVs and
`qubo_audit.png`. No original files are modified. Audit elapsed time: {seconds:.1f} seconds.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    print(report,flush=True)


if __name__ == '__main__':
    main()
