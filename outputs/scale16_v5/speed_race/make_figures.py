"""Figures for speed_race/README.md, each in a light and a dark version.

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
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  s1='#2a78d6', s2='#eb6834', s3='#1baf7a', neutral='#c3c2b7', hist='#b7d3f6'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 s1='#3987e5', s2='#d95926', s3='#199e70', neutral='#52514e', hist='#1c5cab'),
}
SERIES = [('Brute force', 's1'), ('Qismet + CPU search', 's2'), ('Qismet + QPU (ibm_fez)', 's3')]

race = pd.read_csv(HERE / 'speed_race.csv')
qpu = {k: json.loads((HERE / f'qpu_keep_{k}.json').read_text()) for k in (8, 4)}
brute = {k: pd.read_csv(ROOT / 'brute_force' / f'seed_1001_keep_{k}.csv') for k in (8, 4)}
TITLES = {8: 'Keep 8 of 16 groups · 12,870 networks', 4: 'Keep 4 of 16 groups · 1,820 networks'}


def style(ax, t, xgrid=True, ygrid=False):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['axis'])
    ax.tick_params(colors=t['muted'], labelsize=10, length=0)
    if xgrid:
        ax.grid(axis='x', color=t['grid'], linewidth=0.8)
    if ygrid:
        ax.grid(axis='y', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)


def fmt_time(s):
    return f'{s / 60:.0f} min' if s >= 600 else (f'{s / 60:.1f} min' if s >= 120 else f'{s:.0f} s')


def save(fig, name, mode):
    fig.savefig(FIG / f'{name}_{mode}.png', dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


def time_chart(mode):
    t = THEMES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=t['surface'])
    for ax, keep in zip(axes, (8, 4)):
        d = race[race.keep == keep].reset_index(drop=True)
        y = np.arange(3)[::-1]
        for i, (name, slot) in enumerate(SERIES):
            secs = float(d.single_core_seconds[i])
            ax.barh(y[i], secs, height=0.56, color=t[slot], zorder=2)
            ax.text(secs * 1.12, y[i], fmt_time(secs), va='center', ha='left', color=t['ink'], fontsize=11, fontweight='bold')
        speed = d.single_core_seconds[0] / d.single_core_seconds[1]
        ax.set_xscale('log')
        ax.set_xlim(20, d.single_core_seconds[0] * 6)
        ax.set_yticks(y, [s[0] for s in SERIES], color=t['ink2'], fontsize=10.5)
        ax.set_title(TITLES[keep] + f'\nQismet is {speed:.0f}× faster', loc='left', color=t['ink'], fontsize=12,
                     fontweight='bold', pad=10, linespacing=1.5)
        ticks = {8: [(60, '1 min'), (600, '10 min'), (3600, '1 hour')], 4: [(30, '30 s'), (60, '1 min'), (300, '5 min'), (600, '10 min')]}[keep]
        ax.set_xticks([v for v, _ in ticks], [l for _, l in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        style(ax, t)
    fig.text(0.01, -0.04, 'Total time to choose and retrain a pruned network, on one CPU core (log scale). '
             'QPU bar includes the IBM round trip.', color=t['muted'], fontsize=9.5)
    fig.tight_layout(w_pad=4)
    save(fig, 'time', mode)


def quality_chart(mode):
    t = THEMES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=t['surface'])
    for ax, keep in zip(axes, (8, 4)):
        b = brute[keep]
        d = race[race.keep == keep].reset_index(drop=True)
        loss = b.val_loss.to_numpy()
        lo, hi = np.percentile(loss, [0, 99.5])
        ax.hist(loss, bins=np.linspace(lo, hi, 60), color=t['hist'], edgecolor=t['surface'], linewidth=0.6, zorder=2)
        top = ax.get_ylim()[1]
        offsets = [0.97, 0.79, 0.61]
        for i, (name, slot) in enumerate(SERIES):
            v = float(d.val_loss[i])
            ax.axvline(v, color=t[slot], linewidth=2.2, zorder=3)
            label = 'true best' if i == 0 else f"top {d.top_percent[i]:.1f}%"
            side = 'left'
            ax.text(v + (hi - lo) * (0.012 if side == 'left' else -0.012), top * offsets[i],
                    f'{name.split(" (")[0]}\n{label}', color=t['ink'], fontsize=9.5,
                    ha='left' if side == 'left' else 'right', va='top', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=t['surface'], edgecolor='none', alpha=0.85))
        ax.set_title(TITLES[keep], loc='left', color=t['ink'], fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('validation loss after retraining (lower is better)', color=t['ink2'], fontsize=10)
        ax.set_yticks([])
        style(ax, t, xgrid=False)
    fig.text(0.01, -0.05, 'Histogram: every possible pruned network, retrained by brute force. '
             'Lines: the network each method picked.', color=t['muted'], fontsize=9.5)
    fig.tight_layout(w_pad=4)
    save(fig, 'quality', mode)


def breakdown_chart(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(11, 2.5), facecolor=t['surface'])
    parts = []
    for keep in (8, 4):
        q = qpu[keep]
        d = race[(race.keep == keep)].reset_index(drop=True)
        total = float(d.single_core_seconds[2])
        search = q['local_qaoa']['seconds'] + q['compile_seconds'] + q['submit_to_result_seconds'] + q['select_seconds']
        retrain = total - search
        parts.append((keep, retrain, search, total))
    y = [1, 0]
    for yi, (keep, retrain, search, total) in zip(y, parts):
        ax.barh(yi, retrain, height=0.5, color=t['s1'], zorder=2)
        ax.barh(yi, search, left=retrain + total * 0.004, height=0.5, color=t['s3'], zorder=2)
        ax.text(retrain / 2, yi, f'retraining 193 networks · {retrain:.0f} s', ha='center', va='center',
                color='#ffffff', fontsize=10, fontweight='bold')
        ax.text(total * 1.01, yi, f'+ {search:.0f} s quantum search', ha='left', va='center', color=t['ink'], fontsize=10)
    ax.set_yticks(y, ['Keep 8', 'Keep 4'], color=t['ink2'], fontsize=10.5)
    ax.set_xlim(0, max(p[3] for p in parts) * 1.32)
    ax.set_xlabel('seconds', color=t['ink2'], fontsize=10)
    ax.set_title('Where Qismet + QPU spends its time', loc='left', color=t['ink'], fontsize=12, fontweight='bold', pad=10)
    style(ax, t)
    fig.text(0.01, -0.12, 'Quantum search = circuit tuning on the simulator + compilation + IBM submit-to-result. '
             'The same search on a CPU takes under a millisecond.', color=t['muted'], fontsize=9.5)
    fig.tight_layout()
    save(fig, 'breakdown', mode)


def scaling_chart(mode):
    t = THEMES[mode]
    per = float(pd.concat([brute[8].training_seconds, brute[4].training_seconds]).median())
    groups = np.arange(8, 33, 2)
    brute_s = np.array([math.comb(int(g), int(g) // 2) * per for g in groups])
    qismet_s = np.full(len(groups), 193 * per)
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=t['surface'])
    ax.plot(groups, brute_s, color=t['s1'], linewidth=2, zorder=3)
    ax.plot(groups, qismet_s, color=t['s2'], linewidth=2, zorder=3)
    m8 = race[(race.keep == 8)].reset_index(drop=True)
    ax.plot([16], [m8.single_core_seconds[0]], 'o', color=t['s1'], markersize=9,
            markeredgecolor=t['surface'], markeredgewidth=2, zorder=4)
    ax.plot([16], [m8.single_core_seconds[1]], 'o', color=t['s2'], markersize=9,
            markeredgecolor=t['surface'], markeredgewidth=2, zorder=4)
    ax.annotate('measured today', (16, m8.single_core_seconds[0]), xytext=(17.2, m8.single_core_seconds[0] * 6),
                color=t['ink2'], fontsize=9.5, arrowprops=dict(arrowstyle='-', color=t['muted'], lw=0.8))
    ax.set_yscale('log')
    marks = [(60, '1 min'), (3600, '1 hour'), (86400, '1 day'), (86400 * 365, '1 year'), (86400 * 365 * 1000, '1,000 years')]
    ax.set_yticks([m[0] for m in marks], [m[1] for m in marks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(20, brute_s.max() * 3)
    ax.set_xticks(groups)
    ax.set_xlabel('neuron groups (= qubits), keeping half', color=t['ink2'], fontsize=10)
    ax.text(groups[-1] + 0.3, brute_s[-1], 'Brute force', color=t['ink'], fontsize=10.5, fontweight='bold', va='center')
    ax.text(groups[-1] + 0.3, qismet_s[-1], 'Qismet\n(193 retrainings)', color=t['ink'], fontsize=10.5,
            fontweight='bold', va='center')
    ax.set_xlim(groups[0] - 0.5, groups[-1] + 4.5)
    ax.set_title('Projected time to pick a network as the model grows', loc='left', color=t['ink'],
                 fontsize=12, fontweight='bold', pad=10)
    style(ax, t, xgrid=False, ygrid=True)
    fig.text(0.01, -0.06, f'Projection: number of possible networks × {per:.2f} s per retraining (measured median). '
             'Assumes Qismet keeps the same 192-network sample; a larger model may need more samples to stay accurate.',
             color=t['muted'], fontsize=9.5)
    fig.tight_layout()
    save(fig, 'scaling', mode)


def noise_chart(mode):
    t = THEMES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), facecolor=t['surface'])
    for ax, keep in zip(axes, (8, 4)):
        q = qpu[keep]
        vals = [('Ideal circuit (simulator)', 100.0, t['s1']), ('Real hardware (ibm_fez)', 100 * q['feasible_fraction'], t['s3']),
                ('Random bits', 100 * q['uniform_feasible_fraction'], t['neutral'])]
        y = np.arange(3)[::-1]
        for yi, (name, v, c) in zip(y, vals):
            ax.barh(yi, v, height=0.56, color=c, zorder=2)
            ax.text(v + 1.5, yi, f'{v:.1f}%', va='center', color=t['ink'], fontsize=11, fontweight='bold')
        ax.set_yticks(y, [v[0] for v in vals], color=t['ink2'], fontsize=10.5)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(TITLES[keep].split(' ·')[0], loc='left', color=t['ink'], fontsize=12, fontweight='bold', pad=10)
        style(ax, t)
    fig.text(0.01, -0.06, 'Share of 4,096 shots that were a valid network (the right number of groups kept). '
             f"Circuits compiled to ~{qpu[8]['transpiled_two_qubit_gates']}–{qpu[4]['transpiled_two_qubit_gates']} two-qubit gates.",
             color=t['muted'], fontsize=9.5)
    fig.tight_layout(w_pad=4)
    save(fig, 'noise', mode)


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif']
    for mode in THEMES:
        time_chart(mode)
        quality_chart(mode)
        breakdown_chart(mode)
        scaling_chart(mode)
        noise_chart(mode)
    print('wrote', sorted(p.name for p in FIG.glob('*.png')))
