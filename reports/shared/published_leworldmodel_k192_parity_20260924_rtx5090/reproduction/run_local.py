"""Single-GPU runner for the exact relayed screen/conditional-final protocol."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'HUDM'
REPORT = Path('reports/research/k192_sepopt_published_parity_20260924')
ENVIRONMENTS = ('tworoom', 'reacher', 'pusht', 'ogb_cube')
STAGES = {'screen': ('screen_n50', (0, 1, 2, 42, 100), 50),
          'final': ('final_n100', (2002, 71623, 82715, 86604, 91943), 100)}
STATE = ROOT / 'local_run_status.json'
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)

def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(status, **kwargs):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps({'status': status, 'updated_at': timestamp(), 'pid': os.getpid(), **kwargs}, indent=2) + '\n')
    temp.replace(STATE)

def run(command, label, state):
    logfile = LOGS / (label + '.log')
    start = time.monotonic()
    save('running', **state, command=command, log=str(logfile))
    print(timestamp(), label, flush=True)
    with logfile.open('a') as output:
        output.write('\n' + json.dumps({'started_at': timestamp(), 'command': command}) + '\n')
        output.flush()
        child = subprocess.Popen(['/usr/bin/time', '-v', *command], cwd=REPO, env=os.environ, stdout=output, stderr=subprocess.STDOUT)
        samples = 0
        with (ROOT / 'resource_samples.jsonl').open('a') as resources:
            while child.poll() is None:
                try:
                    processes = [psutil.Process(child.pid)]
                    processes += processes[0].children(recursive=True)
                    memory = []
                    for process in processes:
                        try:
                            memory.append({'pid': process.pid, 'rss_bytes': process.memory_info().rss})
                        except psutil.NoSuchProcess:
                            pass
                    sample = {'time': timestamp(), 'label': label, 'processes': memory,
                              'host_available_bytes': psutil.virtual_memory().available}
                    if samples % 5 == 0:
                        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu,power.draw',
                                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                        sample['gpu_memory_used_mib_free_mib_utilization_pct_power_w'] = gpu.stdout.strip()
                    resources.write(json.dumps(sample) + '\n')
                    resources.flush()
                    samples += 1
                except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
                    pass
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        result = subprocess.CompletedProcess(command, child.wait())
    elapsed = time.monotonic() - start
    with (ROOT / 'command_history.jsonl').open('a') as output:
        output.write(json.dumps({'label': label, 'command': command, 'returncode': result.returncode,
                                 'elapsed_seconds': elapsed, 'finished_at': timestamp(), 'log': str(logfile)}) + '\n')
    if result.returncode:
        save('failed', **state, command=command, returncode=result.returncode, log=str(logfile))
        raise RuntimeError(f'{label} failed; inspect {logfile}')

def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--stages', nargs='+', choices=tuple(STAGES))
    parser.add_argument('--environments', nargs='+', choices=ENVIRONMENTS)
    args = parser.parse_args()
    selected_stages = tuple(dict.fromkeys(args.stages or STAGES))
    selected_environments = tuple(dict.fromkeys(args.environments or ENVIRONMENTS))
    full_scope = set(selected_stages) == set(STAGES) and set(selected_environments) == set(ENVIRONMENTS)
    oom_priority = Path('/proc/self/oom_score_adj')
    if oom_priority.exists():
        oom_priority.write_text('1000\n')
    os.chdir(REPO)
    os.environ.update({'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false', 'MPLBACKEND': 'Agg',
                       'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'GIT_CONFIG_GLOBAL': '/dev/null',
                       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                       'MPLCONFIGDIR': str(ROOT / 'cache/matplotlib'), 'HF_HOME': str(ROOT / 'cache/huggingface')})
    lock = (ROOT / 'local_run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.static_only:
        readiness = json.loads((ROOT / 'readiness.json').read_text())
        assert readiness.get('ready') is True, 'Verified setup and GPU smoke tests are required'
    prior = json.loads(STATE.read_text()) if STATE.exists() else {}
    completed = list(dict.fromkeys(prior.get('completed', [])))
    skipped = list(dict.fromkeys(prior.get('skipped_after_screen', [])))
    for stage in selected_stages:
        directory, seeds, episodes = STAGES[stage]
        for environment in selected_environments:
            if stage == 'final' and not args.static_only:
                decision = json.loads((REPO / REPORT / f'screen_{environment}_summary.json').read_text())
                if decision.get('qualified_for_n100') is False:
                    if environment not in skipped:
                        skipped.append(environment)
                    continue
                assert decision.get('qualified_for_n100') is True
            for seed in seeds:
                cfg = OmegaConf.load(REPO / f'configs/research/k192_sepopt_published_parity_20260924/{environment}.yaml')
                cfg.seed = seed
                cfg.output_dir = str(REPORT / directory / environment / f'seed_{seed}')
                cfg.title = f'{cfg.title} (evaluation seed {seed}, n={episodes})'
                cfg.manifest = {'group': f'k192_sepopt_published_parity_{cfg.env_id}_seed{seed}_n{episodes}',
                                'path': str(REPORT / 'manifests' / directory / f'{environment}_seed{seed}.json')}
                for model in cfg.runs:
                    model.eval = {'episodes': episodes, 'num_envs': 50, 'budget': 50}
                config_path = REPORT / 'generated_configs' / directory / environment / f'seed_{seed}.yaml'
                (REPO / config_path).parent.mkdir(parents=True, exist_ok=True)
                (REPO / Path(cfg.manifest.path).parent).mkdir(parents=True, exist_ok=True)
                OmegaConf.save(cfg, REPO / config_path)
                label = f'{stage}_{environment}_seed{seed}'
                state = {'stage': stage, 'environment': environment, 'seed': seed,
                         'completed': completed, 'skipped_after_screen': skipped,
                         'selected_stages': selected_stages, 'selected_environments': selected_environments,
                         'full_scope': full_scope}
                verify = [sys.executable, '-m', 'mwm.benchmark.verify', str(config_path)]
                run([*verify, '--static-only'], label + '_static', state)
                if args.static_only:
                    continue
                run([sys.executable, '-m', 'mwm.benchmark.matrix', str(config_path), '--resume'], label + '_matrix', state)
                run(verify, label + '_verify', state)
                run([sys.executable, str(ROOT / 'audit_paired.py'), str(config_path)], label + '_paired_audit', state)
                if label not in completed:
                    completed.append(label)
            if not args.static_only:
                run([sys.executable, 'scripts/research/collect_k192_sepopt_published_parity_20260924.py',
                     '--stage', stage, '--environment', environment], f'{stage}_{environment}_collect', state)
    if not args.static_only and full_scope:
        run([sys.executable, 'scripts/research/collect_k192_sepopt_published_parity_20260924.py', '--all-final'],
            'final_collect', {'completed': completed, 'skipped_after_screen': skipped})
    status = 'static_checks_passed' if args.static_only else ('complete' if full_scope else 'selected_runs_complete')
    save(status, completed=completed, skipped_after_screen=skipped, selected_stages=selected_stages,
         selected_environments=selected_environments, full_scope=full_scope)

if __name__ == '__main__':
    main()
