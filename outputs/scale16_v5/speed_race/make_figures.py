"""Figures for the main README: Qismet + QPU vs brute force, light and dark versions.

    python make_figures.py

Reads ../brute_force/*.csv, qpu_keep_*.json and speed_race.csv; writes figures/*.png.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
for old in FIG.glob('*.png'):
    old.unlink()

THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#ebeae4', axis='#c3c2b7',
                  accent='#2a78d6', accent_soft='#cde2fb', base='#b5b3ab', base_ink='#52514e', hist='#dedcd4'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 accent='#3987e5', accent_soft='#184f95', base='#5c5b56', base_ink='#c3c2b7', hist='#383835'),
}
race = pd.read_csv(HERE / 'speed_race.csv')
race = race[~race.pipeline.str.contains('CPU')]
qpu = {k: json.loads((HERE / f'qpu_keep_{k}.json').read_text()) for k in (8, 4)}
brute = {k: pd.read_csv(ROOT / 'brute_force' / f'seed_1001_keep_{k}.csv') for k in (8, 4)}
N = {8: 12870, 4: 1820}


def row(keep, kind):
    d = race[race.keep == keep]
    return d[d.pipeline.str.startswith('Brute' if kind == 'brute' else 'Qismet')].iloc[0]


def fmt_time(s):
    if s >= 3600:
        return f'{s / 3600:.1f} hours'
    if s >= 120:
        return f'{s / 60:.0f} min'
    return f'{s:.0f} s'


def base_axes(ax, t):
    ax.set_facecolor(t['surface'])
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t['muted'], labelsize=10.5, length=0)
    ax.set_axisbelow(True)


def rounded_bar(ax, x0, y, width, height, color, radius):
    if width <= 0:
        return
    r = min(radius, width / 2, height / 2)
    ax.add_patch(FancyBboxPatch((x0, y - height / 2), width, height, boxstyle=f'round,pad=0,rounding_size={r}',
                                linewidth=0, facecolor=color, zorder=3, mutation_aspect=1))


def save(fig, name, mode):
    fig.savefig(FIG / f'{name}_{mode}.png', dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


def speed_chart(mode):
    """Linear-scale bars: the gap is the story."""
    t = THEMES[mode]
    fig = plt.figure(figsize=(12, 3.5), facecolor=t['surface'])
    for col, keep in enumerate((8, 4)):
        b, q = row(keep, 'brute'), row(keep, 'qpu')
        ax = fig.add_axes([0.03 + col * 0.5, 0.04, 0.44, 0.56])
        base_axes(ax, t)
        total = b.single_core_seconds
        ax.set_xlim(0, total * 1.02)
        ax.set_ylim(-0.5, 1.5)
        rounded_bar(ax, 0, 1, total, 0.7, t['base'], total * 0.012)
        rounded_bar(ax, 0, 0, q.single_core_seconds, 0.7, t['accent'], total * 0.012)
        ax.text(total * 0.015, 1, 'Brute force', color='#ffffff' if mode == 'light' else t['ink'], va='center',
                fontsize=11.5, fontweight='bold', zorder=4)
        ax.text(total * 0.985, 1, fmt_time(total), color='#ffffff' if mode == 'light' else t['ink'], va='center',
                ha='right', fontsize=12, fontweight='bold', zorder=4)
        ax.text(q.single_core_seconds + total * 0.015, 0, f'Qismet + IBM quantum   {fmt_time(q.single_core_seconds)}',
                color=t['accent'], va='center', fontsize=11.5, fontweight='bold')
        ax.set_yticks([])
        ax.set_xticks([])
        speed = total / q.single_core_seconds
        fig.text(0.03 + col * 0.5, 0.86, f'{speed:.0f}×', color=t['accent'], fontsize=40, fontweight='bold', va='center')
        fig.text(0.03 + col * 0.5 + (0.115 if speed >= 10 else 0.085), 0.885, 'faster', color=t['ink'], fontsize=15,
                 fontweight='bold', va='center')
        fig.text(0.03 + col * 0.5 + (0.115 if speed >= 10 else 0.085), 0.83,
                 f'keep {keep} of 16 groups · {N[keep]:,} possible networks', color=t['ink2'], fontsize=11, va='center')
    fig.text(0.03, -0.08, 'Total time to find and retrain a pruned network (one CPU core for training, plus the IBM round trip).',
             color=t['muted'], fontsize=10)
    save(fig, 'speed', mode)


def quality_chart(mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(12, 4.4), facecolor=t['surface'])
    for col, keep in enumerate((8, 4)):
        b, q = row(keep, 'brute'), row(keep, 'qpu')
        loss = brute[keep].val_loss.to_numpy()
        ax = fig.add_axes([0.03 + col * 0.5, 0.12, 0.44, 0.62])
        base_axes(ax, t)
        lo, hi = np.percentile(loss, [0, 99.0])
        bins = np.linspace(lo, hi, 48)
        counts, edges = np.histogram(loss, bins)
        width = edges[1] - edges[0]
        better = edges[:-1] < q.val_loss
        ax.bar(edges[:-1], counts, width=width * 0.86, align='edge', color=np.where(better, t['accent_soft'], t['hist']),
               zorder=2, linewidth=0)
        top = counts.max()
        ax.axvline(q.val_loss, color=t['accent'], linewidth=2.6, zorder=4)
        ax.text(q.val_loss + width * 0.6, top * 1.02, f'Qismet + IBM quantum\ntop {q.top_percent:.1f}%', color=t['accent'],
                fontsize=11, fontweight='bold', va='top')
        ax.set_ylim(0, top * 1.08)
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color(t['axis'])
        ax.set_xlabel('validation loss after retraining  (lower is better)', color=t['ink2'], fontsize=10.5, labelpad=8)
        fig.text(0.03 + col * 0.5, 0.88, f'Keep {keep} of 16 groups', color=t['ink'], fontsize=14, fontweight='bold')
        fig.text(0.03 + col * 0.5, 0.82, f'Rank {int(q['rank']):,} of {N[keep]:,} possible networks', color=t['ink2'], fontsize=11)
    fig.text(0.03, -0.08, 'Every possible pruned network, retrained by brute force. Shaded: networks that beat the quantum pick.',
             color=t['muted'], fontsize=10)
    save(fig, 'quality', mode)


def breakdown_chart(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(12, 2.6), facecolor=t['surface'])
    base_axes(ax, t)
    ys = [1, 0]
    longest = 0
    for y, keep in zip(ys, (8, 4)):
        q = qpu[keep]
        total = row(keep, 'qpu').single_core_seconds
        search = q['local_qaoa']['seconds'] + q['compile_seconds'] + q['submit_to_result_seconds'] + q['select_seconds']
        train = total - search
        longest = max(longest, total)
        rounded_bar(ax, 0, y, train, 0.56, t['base'], 1.2)
        rounded_bar(ax, train + 0.6, y, search, 0.56, t['accent'], 1.2)
        ax.text(1.5, y, f'retrain 193 networks  {train:.0f} s', color='#ffffff' if mode == 'light' else t['ink'], va='center',
                fontsize=10.5, fontweight='bold', zorder=4)
        ax.text(total + 1.5, y, f'quantum search {search:.0f} s', color=t['accent'], va='center', fontsize=10.5, fontweight='bold')
    ax.set_yticks(ys, ['Keep 8', 'Keep 4'], color=t['ink2'], fontsize=11)
    ax.set_xlim(0, longest * 1.3)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis='x', color=t['grid'], linewidth=0.8)
    ax.set_xlabel('seconds', color=t['ink2'], fontsize=10.5)
    fig.text(0.125, 1.0, 'Where the time goes', color=t['ink'], fontsize=14, fontweight='bold')
    fig.text(0.125, -0.12, 'Quantum search = tuning the circuit, compiling it for ibm_fez, and the IBM submit-to-result round trip '
             '(3 s billed QPU time per job).', color=t['muted'], fontsize=10)
    save(fig, 'breakdown', mode)


def scaling_chart(mode):
    t = THEMES[mode]
    per = float(pd.concat([brute[8].training_seconds, brute[4].training_seconds]).median())
    qsearch = 19.0
    groups = np.arange(8, 33, 2)
    brute_s = np.array([math.comb(int(g), int(g) // 2) * per for g in groups])
    qismet_s = np.full(len(groups), 193 * per + qsearch)
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=t['surface'])
    base_axes(ax, t)
    ax.fill_between(groups, qismet_s, brute_s, where=brute_s > qismet_s, color=t['accent'], alpha=0.10, zorder=1,
                    linewidth=0, interpolate=True)
    ax.plot(groups, brute_s, color=t['base'], linewidth=2.6, zorder=3, solid_capstyle='round')
    ax.plot(groups, qismet_s, color=t['accent'], linewidth=2.6, zorder=3, solid_capstyle='round')
    b8, q8 = row(8, 'brute'), row(8, 'qpu')
    for v, c in ((b8.single_core_seconds, t['base']), (q8.single_core_seconds, t['accent'])):
        ax.plot([16], [v], 'o', color=c, markersize=10, markeredgecolor=t['surface'], markeredgewidth=2.5, zorder=5)
    ax.annotate('measured today', (16, b8.single_core_seconds), xytext=(13.2, b8.single_core_seconds * 30),
                color=t['ink2'], fontsize=10, arrowprops=dict(arrowstyle='-', color=t['muted'], lw=0.9))
    ax.set_yscale('log')
    marks = [(60, '1 min'), (3600, '1 hour'), (86400, '1 day'), (86400 * 30, '1 month'), (86400 * 365, '1 year'), (86400 * 3650, '10 years')]
    ax.set_yticks([m[0] for m in marks], [m[1] for m in marks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis='y', color=t['grid'], linewidth=0.8)
    ax.set_ylim(25, brute_s.max() * 4)
    ax.set_xticks(groups)
    ax.set_xlim(groups[0] - 0.4, groups[-1] + 4.8)
    ax.set_xlabel('neuron groups (= qubits), keeping half', color=t['ink2'], fontsize=10.5)
    ax.text(groups[-1] + 0.4, brute_s[-1], f'Brute force\n~{brute_s[-1] / 86400 / 365:.0f} years', color=t['ink'], fontsize=11.5,
            fontweight='bold', va='center')
    ax.text(groups[-1] + 0.4, qismet_s[-1], f'Qismet + IBM quantum\n~{qismet_s[-1] / 60:.0f} min', color=t['accent'], fontsize=11.5,
            fontweight='bold', va='center')
    fig.text(0.125, 0.95, 'The gap explodes as models grow', color=t['ink'], fontsize=14, fontweight='bold')
    fig.text(0.125, -0.02, f'Projection: possible networks × {per:.2f} s per retraining (measured), against 193 retrainings plus '
             f'~{qsearch:.0f} s of quantum search. Larger models may need a larger sample.', color=t['muted'], fontsize=10)
    save(fig, 'scaling', mode)


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif']
    for mode in THEMES:
        speed_chart(mode)
        quality_chart(mode)
        breakdown_chart(mode)
        scaling_chart(mode)
    print('wrote', sorted(p.name for p in FIG.glob('*.png')))


def hw100_chart(mode):
    """Test accuracy of the best pick per seed: full network vs random sampling vs QAOA on 100 qubits."""
    t = THEMES[mode]
    res = pd.read_csv(ROOT.parent / 'hw100_v6' / 'results.csv')
    seeds = sorted(res.seed.unique())
    get = lambda s, m: float(res[(res.seed == s) & (res.method.str.startswith(m))].iloc[0].best_test_accuracy) * 100
    dense = [float(res[res.seed == s].iloc[0].dense_test_accuracy) * 100 for s in seeds]
    rnd = [get(s, 'Random') for s in seeds]
    qa = [get(s, 'QAOA') for s in seeds]
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=t['surface'])
    base_axes(ax, t)
    x = np.arange(len(seeds))
    w = 0.25
    series = [('Full network (400 neurons)', dense, t['hist'] if mode == 'light' else '#383835'),
              ('Random sampling pick', rnd, t['base']), ('Qismet + IBM quantum pick (100 qubits)', qa, t['accent'])]
    for i, (name, vals, color) in enumerate(series):
        xs = x + (i - 1) * (w + 0.03)
        ax.bar(xs, vals, width=w, color=color, zorder=3, linewidth=0)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.05, f'{v:.1f}%', ha='center', va='bottom', fontsize=10.5,
                    color=t['accent'] if i == 2 else t['ink2'], fontweight='bold' if i == 2 else 'normal')
    ax.set_ylim(81.5, 85.2)
    ax.set_xticks(x, [f'Seed {s}' for s in seeds], color=t['ink2'], fontsize=11)
    ax.set_yticks([82, 83, 84, 85], ['82%', '83%', '84%', '85%'])
    ax.grid(axis='y', color=t['grid'], linewidth=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in series]
    leg = ax.legend(handles, [n for n, _, _ in series], loc='upper left', bbox_to_anchor=(0, 1.17), ncol=3, frameon=False,
                    fontsize=10.5, handlelength=1.2)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    fig.text(0.125, 1.04, 'Half the neurons, higher accuracy', color=t['ink'], fontsize=14, fontweight='bold')
    fig.text(0.125, -0.04, 'Test accuracy of each method\'s best pruned network (chosen by validation loss; test data never used to choose). '
             'Axis starts at 81.5%.', color=t['muted'], fontsize=10)
    save(fig, 'hw100', mode)


for _mode in THEMES:
    plt.rcParams['font.family'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif']
    hw100_chart(_mode)
