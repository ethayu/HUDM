"""Build a compact, reproducible handoff only after all required runs are audited."""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT/'HUDM'
REPORT = REPO/'reports/research/k192_sepopt_published_parity_20260924'
DEST = REPO/'reports/shared/published_leworldmodel_k192_parity_20260924_rtx5090'
sys.path.insert(0, str(ROOT))
from analyze_results import analyze, ENVS, STAGES


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy(source, relative):
    target = DEST/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    os.chdir(REPO)
    os.environ['GIT_CONFIG_GLOBAL'] = '/dev/null'
    assert json.loads((ROOT/'readiness.json').read_text())['ready'] is True
    assert json.loads((ROOT/'local_run_status.json').read_text())['status'] == 'complete'
    DEST.mkdir(parents=True, exist_ok=True)
    episodes, performance, external = [], [], []
    selected, checkpoints = [], set()
    for stage, (seeds, count) in STAGES.items():
        stage_dir = 'screen_n50' if stage == 'screen' else 'final_n100'
        for env in ENVS:
            if stage == 'final':
                screen = json.loads((REPORT/f'screen_{env}_summary.json').read_text())
                if screen['qualified_for_n100'] is False:
                    continue
                assert screen['qualified_for_n100'] is True
            for seed in seeds:
                seed_dir = REPORT/stage_dir/env/f'seed_{seed}'
                config = REPORT/'generated_configs'/stage_dir/env/f'seed_{seed}.yaml'
                subprocess.run([sys.executable, '-m', 'mwm.benchmark.verify', str(config)], check=True)
                subprocess.run([sys.executable, str(ROOT/'audit_paired.py'), str(config)], check=True)
                audit = json.loads((seed_dir/'paired_audit.json').read_text())
                assert audit['verified'] is True and audit['seed'] == seed
                assert len(audit['paired_episodes']) == count
                for episode in audit['paired_episodes']:
                    episodes.append({'stage':stage, 'environment':env, 'seed':seed,
                                     **{key:int(value) for key,value in episode.items()}})
                summary = json.loads((seed_dir/'summary.json').read_text())
                for row in summary['runs']:
                    checkpoints.add(Path(row['checkpoint_run_dir']).resolve())
                    performance.append({'stage':stage, 'environment':env, 'seed':seed, 'role':row['role'],
                                        **{key:row[key] for key in ('episodes','success_rate','plans','steps','cem_cost_calls','dynamics_flops_total',
                                                                  'plan_time_total_sec','wall_time_sec','manifest_file_sha256')}})
                # Full evaluator artifacts stay at their original locations; compact evidence
                # in this handoff is sufficient to independently reconstruct every statistic.
                compact = {'summary.json','summary.csv','per_env_summary.csv','dependencies.json',
                           'episode_traces.jsonl','metrics.jsonl','resolved_config.yaml','paired_audit.json','run.log'}
                for path in sorted(seed_dir.rglob('*')):
                    if not path.is_file():
                        continue
                    copied = path.name in compact
                    external.append({'path':str(path), 'size_bytes':path.stat().st_size,
                                     'sha256':digest(path), 'included_in_handoff':copied})
                    if copied:
                        copy(path, Path('evidence')/path.relative_to(REPORT))
                copy(config, Path('evidence')/config.relative_to(REPORT))
                manifest = Path(audit['protocol']['eval']['manifest_path']).resolve()
                copy(manifest, Path('evidence')/manifest.relative_to(REPORT))
            selected.append((stage,env))
            copy(REPORT/f'{stage}_{env}_summary.json', Path('source_collector')/f'{stage}_{env}_summary.json')
    for name, rows in (('paired_episode_outcomes.csv',episodes), ('performance.csv',performance)):
        with (DEST/name).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = analyze(DEST)
    # Independently reconstructed estimates must reproduce the source collector.
    for stage, env in selected:
        actual = next(r for r in summary['results'] if r['stage']==stage and r['environment']==env)
        expected = json.loads((REPORT/f'{stage}_{env}_summary.json').read_text())
        for metric in ('published','retrained','paired_delta_pp'):
            for field in ('mean','sample_sd','ci95_low','ci95_high'):
                assert math.isclose(actual[metric][field],expected[metric][field],abs_tol=1e-9)
        if stage == 'screen':
            assert actual['qualified_for_n100'] == expected['qualified_for_n100']
    for name in ('final_summary.json','final_summary.csv'):
        copy(REPORT/name, Path('source_collector')/name)
    for path in sorted((ROOT/'source_snapshot/provenance').glob('*')):
        if path.is_file():
            copy(path, Path('provenance/source')/path.name)
    for name in ('readiness.json','LOCAL_SETUP_STATUS.json','local_dataset_inventory.json','hardware.json',
                 'checkpoint_download_receipt.json','source_checkpoint_comparison.json','source_merge_receipt.json',
                 'package_integrity_comparison.json','local_pip_freeze.txt','pinned_requirements.txt',
                 'cluster_constraints.txt','runtime_smoke_receipt.json','vector_env_smoke_receipt.json',
                 'local_run_status.json','command_history.jsonl','resource_samples.jsonl',
                 'execution_source_hash_check.json','RELAY_STATUS.json','full_run_process.json','full_run.log','resource_policy.json',
                 'full_run_process.before_cube_oom.json','cube_memory_failure_status.json'):
        copy(ROOT/name, Path('provenance/local')/name)
    for path in sorted(ROOT.glob('*receipt*.json')):
        copy(path, Path('provenance/local')/path.name)
    for path in sorted((ROOT/'logs').glob('*.log')):
        copy(path, Path('logs')/path.name)
    for name in ('analyze_results.py','package_results.py','run_local.py','audit_paired.py',
                 'inventory_datasets.py','smoke_runtime.py','smoke_vector_envs.py','run_smoke.py','verify_screen_gates.py'):
        copy(ROOT/name, Path(name) if name=='analyze_results.py' else Path('reproduction')/name)
    for path in sorted(ROOT.glob('cube_allocator_profile*.log')):
        copy(path, Path('logs/diagnostics')/path.name)
    for pattern in ('profile_cube_setup_memory*.py','run_memory_profile_guarded*.py'):
        for path in sorted(ROOT.glob(pattern)):
            copy(path, Path('reproduction/diagnostics')/path.name)
    for folder in ('mwm','configs','scripts'):
        for path in sorted((REPO/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                copy(path, Path('reproduction/HUDM')/path.relative_to(REPO))
    for path in sorted(REPO.glob('*')):
        if path.is_file() and (path.suffix in ('.toml','.txt') or path.name in ('README.md','LICENSE')):
            copy(path, Path('reproduction/HUDM')/path.name)
    for checkpoint in checkpoints:
        metadata = json.loads((checkpoint/'world_metadata.json').read_text())
        for name in ('world_metadata.json',metadata['artifacts']['config']['path'],'resolved_config.yaml','config.yaml'):
            if (checkpoint/name).is_file():
                copy(checkpoint/name, Path('checkpoint_metadata')/checkpoint.name/name)
    (DEST/'external_artifacts.json').write_text(json.dumps(external,indent=2)+'\n')
    readme = '''# RTX 5090 published vs September 22 K=192 comparison

All results use the relayed HUDM source snapshot and the verified published
converted checkpoints and September 22 separate-optimizer checkpoints.
`analysis.json` reports the screen and conditional final runs separately.
`paired_episode_outcomes.csv` contains the paired binary outcomes;
`performance.csv` contains measured planning time, wall time and audited dynamics FLOPs,
alongside plan, step and CEM cost-call counts.

Rebuild all statistics and the PNG/PDF/SVG figure using:

    python analyze_results.py .

The analysis requires matplotlib and the Python standard library. Full execution
uses the pinned environment in `provenance/local/`, the copied experiment code
in `reproduction/`, and the checkpoint/data identities in the provenance records.
The completed source comparison is recorded in `source_input_verification_receipt.json`
under `provenance/local/`, with the actual execution-tree comparison in
`execution_source_hash_check.json`. Preliminary inventory receipts retain their
original capture-time status; `readiness.json` and the final verification receipt
record the completed input checks.
Wall time is the duration of the original per-model evaluator call. Command
history separately records resume and validation executions; verified completed
cells retain their original evaluation totals.
Original evaluator artifacts remain at the paths and SHA-256 hashes recorded in
`external_artifacts.json`. Compact config, manifest, dependency, trace and audit
evidence is included under `evidence/`. Source collector outputs are independently
checked against statistics reconstructed from the episode outcomes.

Protocol: CEM population 300, top-k 30, horizon/receding horizon/action block 5,
10 iterations for Cube/Reacher/TwoRoom and 30 for PushT, rollout budget 50,
50 simultaneous environments, fixed singleton K=192, dynamics-audit FLOPs.
The configured goal offset is 25 with `upstream_lewm_end_exclusive` indexing,
so the actual start-to-goal step difference is 24. The published TwoRoom model
uses history 3 and the retrained model uses history 1. This is a planning
performance comparison of fixed checkpoints; it does not establish that their
training recipes are identical. Local Python is 3.11 while source Python is 3.10;
the core package versions and package integrity checks are recorded.

Screen seeds: 0, 1, 2, 42, 100, at 50 episodes per model per seed.
Only environments with mean retrained-minus-published screen difference >= -5 pp
advance. Final seeds: 2002, 71623, 82715, 86604, 91943, at 100 episodes per model
per seed. Every reported interval is a two-sided Student t interval across five
evaluation seeds (df=4). Intervals describe evaluation-manifest variability for
fixed checkpoints, not training-seed uncertainty. The screen gate uses the point
estimate; passing it is not a statistical noninferiority or equivalence claim.
Conditional final results must be interpreted alongside all four screen results.

Execution recovery: the first Cube screen attempt was killed by host-memory
exhaustion while substantial unrelated shared memory occupied this machine.
No result from that failed attempt was accepted. The already-qualified TwoRoom,
Reacher and PushT final runs proceeded while Cube awaited more available RAM;
all runs retain their original episode counts, 50-environment concurrency,
checkpoints, seeds and planner settings. The failed attempt, retry and actual
execution order remain recorded in the local receipts, logs and command history.
Separate guarded CPU and CUDA setup diagnostics tested unused-memory release;
the CPU test also compared Python allocators with fixed diagnostic seeds.
Model/scaler values and RNG states were preserved. Diagnostic allocator settings
were not applied to the scientific runs. Diagnostic logs and script snapshots
are included alongside their receipts.
'''
    (DEST/'README.md').write_text(readme)
    files = [{'path':str(path.relative_to(DEST)), 'size_bytes':path.stat().st_size, 'sha256':digest(path)}
             for path in sorted(DEST.rglob('*')) if path.is_file() and path.name!='handoff_manifest.json']
    (DEST/'handoff_manifest.json').write_text(json.dumps({'complete':True,
        'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files':files},indent=2)+'\n')
    print(DEST,flush=True)


if __name__ == '__main__':
    main()
