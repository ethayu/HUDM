"""Rebuild seed-level paired estimates and the paper figure from audited episodes.

Run with Python plus matplotlib: python analyze_results.py /path/to/handoff
No model, simulator, or original temporary directory is required.
"""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ENVS = ('ogb_cube', 'reacher', 'pusht', 'tworoom')
LABELS = ('OGB Cube', 'Reacher', 'PushT', 'TwoRoom')
STAGES = {'screen': ((0, 1, 2, 42, 100), 50),
          'final': ((2002, 71623, 82715, 86604, 91943), 100)}
T975 = 2.7764451051977987


def mean_ci(values):
    assert len(values) == 5, 'The source protocol requires five evaluation seeds'
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T975 * sd / math.sqrt(len(values))
    return {'mean': mean, 'sample_sd': sd, 'ci95_low': mean-half, 'ci95_high': mean+half}


def analyze(directory):
    directory = Path(directory)
    with (directory / 'paired_episode_outcomes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        key = (row['stage'], row['environment'], int(row['seed']))
        assert key[0] in STAGES and key[1] in ENVS
        assert key[2] in STAGES[key[0]][0]
        assert row['published_success'] in ('0', '1') and row['retrained_success'] in ('0', '1')
        groups.setdefault(key, []).append(row)
    results, per_seed, skipped = [], [], []
    for stage, (seeds, count) in STAGES.items():
        for env in ENVS:
            if stage == 'final':
                screen = next(r for r in results if r['stage'] == 'screen' and r['environment'] == env)
                if not screen['qualified_for_n100']:
                    assert not any((stage, env, seed) in groups for seed in seeds), 'Unexpected final results for failed screen'
                    skipped.append({'environment': env, 'screen_delta_pp': screen['paired_delta_pp']['mean'],
                                    'reason': 'Mean screen delta below predeclared -5 pp gate'})
                    continue
            seed_results = []
            for seed in seeds:
                episodes = groups[(stage, env, seed)]
                assert len(episodes) == count
                assert sorted(int(r['episode_index']) for r in episodes) == list(range(count))
                published = sum(int(r['published_success']) for r in episodes)
                retrained = sum(int(r['retrained_success']) for r in episodes)
                result = {'stage': stage, 'environment': env, 'seed': seed, 'episodes_per_model': count,
                          'published_successes': published, 'retrained_successes': retrained,
                          'published_success_rate': 100*published/count,
                          'retrained_success_rate': 100*retrained/count,
                          'retrained_minus_published_pp': 100*(retrained-published)/count,
                          'both_succeed': sum(r['published_success'] == r['retrained_success'] == '1' for r in episodes),
                          'published_only': sum(r['published_success'] == '1' and r['retrained_success'] == '0' for r in episodes),
                          'retrained_only': sum(r['retrained_success'] == '1' and r['published_success'] == '0' for r in episodes)}
                seed_results.append(result)
                per_seed.append(result)
            result = {'stage': stage, 'environment': env, 'seeds': list(seeds),
                      'episodes_per_seed_per_model': count, 'paired_results': seed_results,
                      'published': mean_ci([r['published_success_rate'] for r in seed_results]),
                      'retrained': mean_ci([r['retrained_success_rate'] for r in seed_results]),
                      'paired_delta_pp': mean_ci([r['retrained_minus_published_pp'] for r in seed_results])}
            if stage == 'screen':
                result['qualified_for_n100'] = result['paired_delta_pp']['mean'] >= -5.0
            results.append(result)
    summary = {'comparison': 'Published LeWorldModel vs September 22 singleton K=192 separate-optimizer models',
               'confidence_interval': 'Two-sided 95% Student t over five paired evaluation-seed differences (df=4)',
               'interpretation': 'Intervals describe evaluation-manifest variability for these fixed checkpoints, not training-seed uncertainty; the screen gate is a point-estimate selection rule, not a noninferiority test.',
               'results': results, 'skipped_after_screen': skipped}
    (directory/'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (directory/'seed_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    plot(directory, results, skipped)
    return summary


def plot(directory, results, skipped):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True, sharey=True)
    bounds = [v for r in results for v in (r['paired_delta_pp']['ci95_low'], r['paired_delta_pp']['ci95_high'])]
    lower, upper = min(-10, min(bounds)-5), max(10, max(bounds)+5)
    for axis, stage, title in zip(axes, STAGES, ('Screen: 50 episodes × 5 seeds', 'Final: 100 episodes × 5 seeds')):
        axis.axvline(0, color='#555555', lw=1, zorder=0)
        if stage == 'screen':
            axis.axvline(-5, color='#ae3d3d', ls='--', lw=1, label='Screen gate: −5 pp')
        for index, env in enumerate(ENVS):
            matches = [r for r in results if r['stage'] == stage and r['environment'] == env]
            if not matches:
                axis.text(0.03, index, 'Skipped after screen', va='center', color='#666666', transform=axis.get_yaxis_transform())
                continue
            row = matches[0]
            stats = row['paired_delta_pp']
            axis.scatter([r['retrained_minus_published_pp'] for r in row['paired_results']],
                         [index-0.14]*5, s=17, alpha=.55, color='#68727d', zorder=2)
            axis.errorbar(stats['mean'], index, xerr=[[stats['mean']-stats['ci95_low']], [stats['ci95_high']-stats['mean']]],
                          fmt='o', color='#126f87', capsize=4, lw=1.8, markersize=6, zorder=3)
        axis.set_title(title, fontsize=11)
        axis.set_xlim(lower, upper)
        axis.set_xlabel('Retrained − published success (percentage points)')
        axis.set_yticks(range(4), LABELS)
        axis.grid(axis='x', alpha=.15)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(3.5, -.5)
    axes[0].legend(loc='lower left', frameon=False, fontsize=8)
    fig.suptitle('September 22 K=192 models vs published LeWorldModel', fontsize=13)
    fig.text(.5, .01, 'Colored points: means with paired 95% t intervals across 5 evaluation seeds. Gray points: seed differences.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.96))
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(directory/f'paired_success_delta.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    analyze(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent)
