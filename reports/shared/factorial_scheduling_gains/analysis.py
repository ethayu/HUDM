"""Snapshot of the matched scheduling analysis; inputs are local to this package."""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
from frontier_utils import pareto_frontier, success_at, cost_to_reach

HERE = Path(__file__).resolve().parent
SHARED = HERE / 'data'
# tag, display name, report, scheduled summary, minimum allowed dimension
SOURCES = json.loads((HERE / 'data/sources.json').read_text())
CUBE = {'000': None, '100': '18', '010': '06', '001': '02',
        '110': '21', '101': '20', '011': '08', '111': '23'}
SCALES = [('episode', '100', 'Episode', r'MPC replans $e$'),
          ('cem', '010', 'CEM', r'Iterations $i$'),
          ('rollout', '001', 'Rollout', r'Prediction steps $h$')]
COLORS = {'episode': '#b36a14', 'cem': '#2166ac', 'rollout': '#8751b0'}
FIXED = '#707070'
ENVELOPE = '#202020'
METADATA = ('checkpoint_epoch', 'manifest_sha256', 'manifest_file_sha256', 'goal_offset', 'seed', 'episodes')


def cell(r):
    return (int(r['pop_size']), int(r['n_iter']), int(r['topk']), float(r['elite_frac']))


def load_runs(path):
    return json.loads(path.read_text())['runs']


def frontier(runs):
    assert runs and all(r['dynamics_flops_total'] > 0 for r in runs)
    return pareto_frontier([(r['dynamics_flops_total'] / r['episodes'] / 1e9,
                             float(r['success_rate'])) for r in runs])


def verify(runs, baseline):
    refs = {cell(r): r for r in baseline}
    for r in runs:
        ref = refs[cell(r)]
        assert Path(r['checkpoint_run_dir']).name == Path(ref['checkpoint_run_dir']).name
        for field in METADATA:
            assert r.get(field) is not None and r[field] == ref[field], (r['name'], field)
        # Candidate action shape verifies a shared planning/action horizon.
        shape = r['candidate_action_values'] / (r['cem_cost_calls'] * r['pop_size'])
        ref_shape = ref['candidate_action_values'] / (ref['cem_cost_calls'] * ref['pop_size'])
        assert abs(shape - ref_shape) < 1e-8, (r['name'], 'candidate action shape')


def shared_grid(frontiers):
    # Evaluate means only where every compared frontier has an affordable point.
    lo = max(f[0][0] for f in frontiers)
    hi = max(f[-1][0] for f in frontiers)
    return np.geomspace(lo, hi, 60)


def analyze():
    out = {}
    for tag, title, folder, scheduled_file, minimum in SOURCES:
        scheduled_path = (SHARED / folder / scheduled_file).resolve()
        fixed_path = SHARED / folder / 'data/dense_checkpoint_individual_levels_summary.json'
        a, f = load_runs(scheduled_path), load_runs(fixed_path)
        keep = lambda r: not tag.startswith('ogb_') or r['n_iter'] != 5
        a, f = [r for r in a if keep(r)], [r for r in f if keep(r)]
        fixed_levels = defaultdict(list)
        for r in f:
            dim = int(re.search(r'_k(\d+)_', r['base_name']).group(1))
            if dim >= minimum:
                fixed_levels[str(dim)].append(r)
        by = defaultdict(list)
        for r in a:
            by[r['base_name'].split('_', 1)[0]].append(r)
        groups = {bits: fixed_levels['192'] if key is None else by[key] for bits, key in CUBE.items()}
        available = ['000'] + [bits for _, bits, _, _ in SCALES if groups[bits]]
        common = set.intersection(*(set(map(cell, groups[b])) for b in available),
                                  *(set(map(cell, rows)) for rows in fixed_levels.values()))
        assert common
        matched = {b: [r for r in groups[b] if cell(r) in common] for b in available}
        assert all(len(rs) == len(common) for rs in matched.values())
        matched_fixed = {dim: [r for r in rows if cell(r) in common] for dim, rows in fixed_levels.items()}
        for rs in list(matched.values()) + list(matched_fixed.values()):
            verify(rs, matched['000'])
        curves = {b: frontier(rs) for b, rs in matched.items()}
        curves['fixed_levels'] = frontier([r for rs in matched_fixed.values() for r in rs])
        grid = shared_grid(list(curves.values()))
        mean = {key: float(np.mean([success_at(fr, c) for c in grid])) for key, fr in curves.items()}
        effects = {}
        for key, bits, _, _ in SCALES:
            if bits not in curves:
                effects[key] = None
                continue
            target = min(curves['000'][-1][1], curves[bits][-1][1])
            effects[key] = {
                'peak': curves[bits][-1][1],
                'peak_change': curves[bits][-1][1] - curves['000'][-1][1],
                'mean_frontier_change': mean[bits] - mean['000'],
                'mean_vs_fixed_levels': mean[bits] - mean['fixed_levels'],
                'matched_success': target,
                'fixed_to_scheduled_cost_ratio': cost_to_reach(curves['000'], target) / cost_to_reach(curves[bits], target),
                'matched_cell_success_change': float(np.mean([r['success_rate'] - next(v['success_rate'] for v in matched['000'] if cell(v) == cell(r)) for r in matched[bits]])),
            }
        factorial = None
        if all(groups[b] for b in CUBE):
            cells8 = set.intersection(*(set(map(cell, groups[b])) for b in CUBE), common)
            cube_rows = {b: [r for r in groups[b] if cell(r) in cells8] for b in CUBE}
            for rows in cube_rows.values():
                assert len(rows) == len(cells8)
                verify(rows, cube_rows['000'])
            cube_fr = {b: frontier(rs) for b, rs in cube_rows.items()}
            grid8 = shared_grid(list(cube_fr.values()))
            areas = {b: float(np.mean([success_at(fr, c) for c in grid8])) for b, fr in cube_fr.items()}
            factorial = {'cells_per_condition': len(cells8), 'budget_range': [float(grid8[0]), float(grid8[-1])], 'effects': {}}
            for index, (key, _, _, _) in enumerate(SCALES):
                pairs = []
                for b in CUBE:
                    if b[index] == '0':
                        on = b[:index] + '1' + b[index + 1:]
                        pairs.append({'off': b, 'on': on,
                                      'mean_frontier_change': areas[on] - areas[b],
                                      'peak_change': cube_fr[on][-1][1] - cube_fr[b][-1][1]})
                factorial['effects'][key] = {
                    'pairs': pairs,
                    'mean_frontier_change': float(np.mean([p['mean_frontier_change'] for p in pairs])),
                    'peak_change': float(np.mean([p['peak_change'] for p in pairs])),
                }
        out[tag] = {
            'task': title, 'goal_offset': int(tag[-2:]), 'd_min': minimum,
            'checkpoint': Path(matched['000'][0]['checkpoint_run_dir']).name,
            'scheduled_source': str(scheduled_path.relative_to(SHARED)),
            'fixed_source': str(fixed_path.relative_to(SHARED)),
            'cells_per_condition': len(common), 'cem_cells': sorted(common),
            'episodes_per_cell': matched['000'][0]['episodes'],
            'fixed_levels': sorted(map(int, matched_fixed)),
            'missing_single_schedulers': [k for k, b, _, _ in SCALES if b not in curves],
            'frontiers': curves, 'effects': effects,
            'budget_range': [float(grid[0]), float(grid[-1])],
            'factorial': factorial,
        }
    return out


def draw_frontier(ax, fr, color, style, xmax, width=1.25, marker='o'):
    x, y = zip(*fr)
    # Connect observations on the log-compute axis for display only;
    # numerical comparisons retain the empirical step frontiers.
    ax.plot(x, y, color=color, ls=style, lw=width)
    ax.plot(x, y, ls='none', marker=marker, ms=2.2, color=color)
    # A larger available budget can reuse the last observed configuration.
    if x[-1] < xmax:
        ax.plot([x[-1], xmax], [y[-1], y[-1]], color=color, ls=style, lw=width, alpha=.55)


@plt.rc_context({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                 'font.size': 7, 'pdf.fonttype': 42})
def frontiers_figure(stats, out):
    # Plot the controlled effect directly, without overlapping success curves.
    # The same budget interval is used for every scheduler within a task/offset.
    fig, axes = plt.subplots(3, len(stats), figsize=(5.9, 3.2), sharey=True,
                             sharex='col')
    positive, negative = '#2166ac', '#b45339'
    fig.subplots_adjust(left=.115, right=.99, bottom=.14, top=.80,
                        wspace=.14, hspace=.22)
    for j, (tag, panel) in enumerate(stats.items()):
        lo, hi = panel['budget_range']
        baseline = panel['frontiers']['000']
        for i, (key, bits, label, progress) in enumerate(SCALES):
            ax = axes[i, j]
            ax.set_xscale('log')
            ax.set_xlim(lo, hi)
            ax.set_ylim(-60, 30)
            ax.set_yticks([-50, 0, 25], ['−50%', '0%', '25%'])
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(labelsize=6.3, length=2, pad=1.5)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_facecolor('#fcfcfb')
            ax.axhline(0, color='#737373', lw=.6, zorder=1)
            for spine in ax.spines.values():
                spine.set_color('#dfdfdc')
                spine.set_linewidth(.5)
            if bits in panel['frontiers']:
                scheduled = panel['frontiers'][bits]
                # Exact empirical breakpoints, with no linear interpolation.
                x = np.array(sorted({lo, hi} | {c for fr in (baseline, scheduled)
                                               for c, _ in fr if lo <= c <= hi}))
                gain = np.array([success_at(scheduled, c) - success_at(baseline, c)
                                 for c in x])
                assert gain.min() >= -60 and gain.max() <= 30, (tag, key, gain)
                ax.fill_between(x, 0, np.maximum(gain, 0), step='post',
                                color=positive, alpha=.24, linewidth=0)
                ax.fill_between(x, 0, np.minimum(gain, 0), step='post',
                                color=negative, alpha=.24, linewidth=0)
                ax.step(x, gain, where='post', color='#343434', lw=.85)
            else:
                ax.text(.5, .5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6.2, color='#777777',
                        bbox={'facecolor': '#fcfcfb', 'edgecolor': 'none', 'pad': 1})
            if i == 0:
                ax.set_title(rf"$\Delta={panel['goal_offset']}$", fontsize=7, pad=4)
            if j == 0:
                ax.set_ylabel(label + ' only', fontsize=7, fontweight='bold', labelpad=3)
    # Group goal offsets beneath a single task heading.
    groups = []
    for j, panel in enumerate(stats.values()):
        if groups and groups[-1][0] == panel['task']:
            groups[-1] = (panel['task'], groups[-1][1], j)
        else:
            groups.append((panel['task'], j, j))
    for label, first, last in groups:
        left = axes[0, first].get_position().x0
        right = axes[0, last].get_position().x1
        fig.text((left + right) / 2, .89, label, ha='center', va='top',
                 fontsize=8, fontweight='bold')
    fig.text(.55, 1.005, r'Success change vs. fixed $d_K$ (percentage points)',
             fontsize=7, ha='center', va='top')
    fig.text(.35, .945, 'Above zero: helps', color=positive, fontsize=7,
             ha='center', va='top')
    fig.text(.72, .945, 'Below zero: hurts', color=negative, fontsize=7,
             ha='center', va='top')
    fig.supxlabel('Dynamics GFLOPs per episode', fontsize=7, y=.025)
    fig.supylabel('Success change (pp)', fontsize=7, x=.005)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'single_scale_frontiers.{ext}', bbox_inches='tight',
                    pad_inches=.03, dpi=250)
    plt.close(fig)


@plt.rc_context({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                 'font.size': 7, 'pdf.fonttype': 42})
def reference_frontiers_figure(stats, out):
    # Match Figure 6's sans-serif typography. One panel per task/offset avoids
    # repeating the two fixed references for each scheduler intervention.
    fig, axes = plt.subplots(2, 4, figsize=(5.9, 3.5))
    columns = {'tworoom': 0, 'pusht': 1, 'ogb': 2, 'reacher': 3}
    markers = {'episode': '^', 'cem': 's', 'rollout': 'D'}
    for tag, panel in stats.items():
        i = {25: 0, 50: 1}[panel['goal_offset']]
        j = columns[tag.split('_')[0]]
        ax = axes[i, j]
        curves = panel['frontiers']
        xmin = min(fr[0][0] for fr in curves.values())
        xmax = max(fr[-1][0] for fr in curves.values())
        ymax = max(fr[-1][1] for fr in curves.values())
        ymin = min(fr[0][1] for fr in curves.values())
        ax.set_xscale('log')
        ax.set_xlim(xmin * .92, xmax * 1.08)
        ax.set_ylim(max(0, np.floor((ymin - 5) / 10) * 10), min(100, np.ceil((ymax + 3) / 10) * 10))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_facecolor('#fcfcfb')
        ax.grid(True, color='#d9d9d6', lw=.45)
        ax.tick_params(labelsize=6.1, length=2, pad=1.7)
        for spine in ax.spines.values():
            spine.set_color('#d9d9d6')
        draw_frontier(ax, curves['fixed_levels'], ENVELOPE, ':', xmax, .9)
        draw_frontier(ax, curves['000'], FIXED, '--', xmax, 1.0)
        for key, bits, label, progress in SCALES:
            if bits in curves:
                draw_frontier(ax, curves[bits], COLORS[key], '-', xmax, 1.15, markers[key])
        ax.set_title(panel['task'] + rf"  $\Delta={panel['goal_offset']}$",
                     fontsize=7.2, fontweight='bold', pad=4)
    # Both offsets have data; place the shared key below the eight panels.
    handles = [Line2D([], [], color=COLORS[key], marker=markers[key], ms=3,
                      lw=1.15, label=label + ' only') for key, _, label, _ in SCALES]
    handles += [Line2D([], [], color=FIXED, ls='--', lw=1, label=r'Fixed $d_K$'),
                Line2D([], [], color=ENVELOPE, ls=':', lw=.9,
                       label='Best fixed level')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .005),
               ncol=5, frameon=False, fontsize=6.2, handlelength=1.8,
               columnspacing=1.0, handletextpad=.5)
    fig.supylabel('Success rate (%)', fontsize=7.5, x=.012)
    fig.supxlabel('Dynamics GFLOPs per episode', fontsize=7.5, y=.095)
    fig.tight_layout(pad=.4, w_pad=.7, h_pad=1.0, rect=(.025, .14, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'single_scale_frontiers_full.{ext}', bbox_inches='tight', pad_inches=.03, dpi=250)
    plt.close(fig)


def factorial_figure(stats, out):
    complete = {tag: panel for tag, panel in stats.items() if panel['factorial'] is not None}
    titles = [p['task'].replace('OGBench-Cube', 'OGB-Cube') + '\n' +
              rf"$\Delta={p['goal_offset']}$" for p in complete.values()]
    fig, axes = plt.subplots(2, 1, figsize=(5.9, 3.3))
    fig.subplots_adjust(left=.125, right=.995, bottom=.23, top=.91, hspace=.77)
    arrays = [np.array([[p['factorial']['effects'][key][metric] for p in complete.values()] for key, _, _, _ in SCALES])
              for metric in ('mean_frontier_change', 'peak_change')]
    limit = float(np.ceil(max(abs(x).max() for x in arrays) / 5) * 5)
    for ax, values, title in zip(axes, arrays, ('(a) Mean frontier change', '(b) Peak-success change')):
        im = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
        for i in range(3):
            for j in range(len(complete)):
                v = values[i, j]
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > limit * .65 else '#252525')
        ax.set_xticks(range(len(complete)), titles, fontsize=6.0)
        ax.set_yticks(range(3), [label for _, _, label, _ in SCALES], fontsize=7)
        ax.tick_params(length=0, pad=3)
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold', pad=6)
        ax.set_xticks(np.arange(.5, len(complete)), minor=True)
        ax.grid(which='minor', axis='x', color='white', lw=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes((.32, .08, .52, .03))
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7, length=2, pad=2)
    cb.outline.set_visible(False)
    cb.set_label('Mean change over four matched pairs (percentage points)', fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'factorial_scheduling_gains.{ext}', bbox_inches='tight', pad_inches=.03, dpi=250)
    plt.close(fig)
