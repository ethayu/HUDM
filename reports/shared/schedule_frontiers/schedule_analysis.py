"""Per-schedule analysis and plotting, using only this package's local inputs.

Run generate.py to validate inputs and regenerate this figure.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis import SOURCES, SHARED, cell, load_runs, verify, frontier
from frontier_utils import success_at
ALIASES = {"05": "02", "11": "02", "17": "02", "12": "06", "13": "07", "14": "08"}

HERE = Path(__file__).resolve().parent
# L = minimum, F = finest, O = enclosing scheduler's selected level.
RULES = {
    '02': ('F', 'O', 'F→L'), '06': ('F', 'L→F', 'O'),
    '07': ('F', 'L→F', 'F→O'), '08': ('F', 'L→F', 'O→L'),
    '18': ('L→F', 'O', 'O'), '19': ('L→F', 'O', 'F→O'),
    '20': ('L→F', 'O', 'O→L'), '21': ('L→F', 'L→O', 'O'),
    '22': ('L→F', 'L→O', 'F→O'), '23': ('L→F', 'L→O', 'O→L'),
    '24': ('L→F', 'O→F', 'O'), '25': ('L→F', 'O→F', 'F→O'),
    '26': ('L→F', 'O→F', 'O→L'),
}


def analyze():
    result = {}
    for tag, title, folder, filename, minimum in SOURCES:
        source = (SHARED / folder / filename).resolve()
        raw = defaultdict(list)
        for r in load_runs(source):
            key = r['base_name'].split('_', 1)[0]
            if ALIASES.get(key, key) in RULES and not (tag.startswith('ogb_') and r['n_iter'] == 5):
                raw[key].append(r)
        assert set(raw) == set(RULES) | set(ALIASES), (tag, set(raw))
        fixed_source = SHARED / folder / 'data/dense_checkpoint_individual_levels_summary.json'
        fixed = [r for r in load_runs(fixed_source) if '_k192_' in r['base_name']]
        common = set.intersection(set(map(cell, fixed)), *(set(map(cell, rs)) for rs in raw.values()))
        assert common
        fixed = [r for r in fixed if cell(r) in common]
        grouped = defaultdict(list)
        alias_differences = {}
        for key, rows in raw.items():
            rows = [r for r in rows if cell(r) in common]
            assert len(rows) == len(common), (tag, key)
            verify(rows, fixed)
            grouped[ALIASES.get(key, key)].extend(rows)
            if key in ALIASES:
                refs = {cell(r): r for r in raw[ALIASES[key]]}
                alias_differences[key] = max(abs(r['success_rate'] - refs[cell(r)]['success_rate']) for r in rows)
        curves = {key: frontier(rows) for key, rows in sorted(grouped.items())}
        lo = max(fr[0][0] for fr in curves.values())
        hi = max(fr[-1][0] for fr in curves.values())
        assert 0 < lo < hi
        breaks = sorted({lo, hi} | {x for fr in curves.values() for x, _ in fr if lo <= x <= hi})
        values = {k: np.array([success_at(fr, x) for x in breaks]) for k, fr in curves.items()}
        best = np.max(list(values.values()), axis=0)
        winners = [[k for k, v in values.items() if abs(v[i] - best[i]) < 1e-9] for i in range(len(breaks))]
        retained = sorted({k for keys in winners for k in keys})
        omitted = sorted(set(curves) - set(retained))
        # Confirm that filtering preserves the exact envelope, including its endpoint.
        assert np.allclose(np.max([values[k] for k in retained], axis=0), best)
        assert all(np.all(values[k] < best - 1e-9) for k in omitted)
        edges = np.geomspace(lo, hi, 4).tolist()
        bands = []
        for left, right in zip(edges[:-1], edges[1:]):
            knots = sorted({left, right} | {x for x in breaks if left < x < right})
            weights = np.diff(np.log(knots))
            share = {k: 0.0 for k in curves}
            for x, weight in zip(knots[:-1], weights):
                scores = {k: success_at(fr, x) for k, fr in curves.items()}
                peak = max(scores.values())
                for k, value in scores.items():
                    if abs(value - peak) < 1e-9:
                        share[k] += float(weight / np.log(right / left))
            bands.append({'range': [left, right], 'winner_log_fraction': share,
                          'most_frequent_winners': [k for k, v in share.items() if abs(v - max(share.values())) < 1e-9]})
        result[tag] = {'title': title, 'minimum': minimum, 'source': str(source),
                       'fixed_metadata_reference': str(fixed_source), 'cem_cells': sorted(common),
                       'cells_per_schedule_spelling': len(common), 'aliases': ALIASES,
                       'alias_max_success_difference_pp': alias_differences,
                       'frontiers': curves, 'retained': retained, 'omitted': omitted,
                       'budget_range': [lo, hi], 'band_edges': edges, 'bands': bands,
                       'breakpoints': breaks, 'winners': winners, 'envelope': best.tolist(),
                       'peak_winners': winners[-1]}
    return result


def plot(stats, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    palette = ['#2674b5', '#d55e00', '#009e73', '#a36d00', '#7b4ab0',
               '#007f86', '#c34573', '#637c22', '#6256a5', '#b04929',
               '#4d4d4d', '#9b42a8', '#477d92']
    colors = dict(zip(RULES, palette))
    styles = {k: ['-', '--', '-.', ':'][i // 4] for i, k in enumerate(RULES)}
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 9.2))
    for ax, (tag, panel) in zip(axes.flat, stats.items()):
        lo, hi = panel['budget_range']
        for x in panel['band_edges'][1:-1]:
            ax.axvline(x, color='#cccccc', linewidth=.7, linestyle=':')
        for k in panel['retained']:
            fr = panel['frontiers'][k]
            xs, ys = map(list, zip(*fr))
            if xs[-1] < hi:
                xs.append(hi)
                ys.append(ys[-1])
            ax.plot(xs, ys, label=k, color=colors[k], linestyle=styles[k], linewidth=1.25)
            points = [(x, y) for x, y in fr if lo <= x <= hi]
            if points:
                ax.scatter(*zip(*points), s=8, color=colors[k], zorder=3)
        ax.set_xscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(max(0, min(success_at(panel['frontiers'][k], lo) for k in panel['retained']) - 5),
                    min(100, max(panel['envelope']) + 5))
        ax.set_title(panel['title'] + r' — $\Delta=' + tag.split('goal')[1] + '$', loc='left')
        ax.set_ylabel('Success (%)')
        ax.set_xlabel('Dynamics GFLOPs / episode')
        ax.grid(axis='y', color='#eeeeee', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(title='Schedule', ncol=4, fontsize=6, title_fontsize=6, loc='lower right',
                  framealpha=.9, columnspacing=.65, handlelength=1.7, borderpad=.35)
        for left, right, label in zip(panel['band_edges'][:-1], panel['band_edges'][1:], ('Low', 'Mid', 'High')):
            ax.text(np.sqrt(left * right), .97, label, transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=6, color='#666666')
    fig.tight_layout(h_pad=1.5, w_pad=1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'schedule_frontiers.{ext}', dpi=200)
    plt.close(fig)
