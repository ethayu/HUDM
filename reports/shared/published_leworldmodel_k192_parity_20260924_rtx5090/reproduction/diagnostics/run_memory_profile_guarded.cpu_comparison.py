"""Run the CPU Cube allocator diagnostic only after scientific finals finish."""
from pathlib import Path
import argparse
import os
import hashlib
import datetime
import json
import subprocess
import time
import psutil

parser = argparse.ArgumentParser()
parser.add_argument('--allocator', choices=('default', 'malloc'), required=True)
args = parser.parse_args()
variant = args.allocator + '_seeded'
ROOT = Path(__file__).resolve().parent
state = json.loads((ROOT / 'local_run_status.json').read_text())
assert state['status'] == 'selected_runs_complete', 'Wait for the selected scientific finals to finish'
for process in psutil.process_iter(['pid', 'cmdline']):
    process_args = process.info['cmdline'] or []
    if str(ROOT) in ' '.join(process_args) and (str(ROOT / 'run_local.py') in process_args or 'mwm.benchmark.matrix' in process_args):
        raise RuntimeError(f'Scientific process {process.pid} is still live; wait for its exit')
assert psutil.virtual_memory().available >= 6 * 2**30, 'Insufficient RAM for the guarded diagnostic'
command = [str(ROOT / 'venv/bin/python'), '-u', str(ROOT / 'profile_cube_setup_memory.py'), '--label', variant]
child_env = os.environ.copy()
allocator_env = {'PYTHONMALLOC': args.allocator, 'PYTHONHASHSEED': '42'}
child_env.update(allocator_env)
log_path = ROOT / f'cube_allocator_profile_{variant}.log'
start = time.monotonic()
started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
reason = None
peak_rss = 0
minimum_available = psutil.virtual_memory().available
last_print = 0
with log_path.open('a') as output:
    output.write(json.dumps({'started_at': started_at, 'command': command}) + '\n')
    output.flush()
    child = subprocess.Popen(command, cwd=ROOT / 'HUDM', stdout=output, stderr=subprocess.STDOUT, env=child_env)
    print(json.dumps({'profile_pid': child.pid, 'log': str(log_path)}), flush=True)
    while child.poll() is None:
        available = psutil.virtual_memory().available
        try:
            rss = psutil.Process(child.pid).memory_info().rss
        except psutil.NoSuchProcess:
            continue
        peak_rss = max(peak_rss, rss)
        minimum_available = min(minimum_available, available)
        elapsed = time.monotonic() - start
        if available < 3 * 2**30:
            reason = 'Host available memory fell below 3 GiB'
        elif rss > 8 * 2**30:
            reason = 'Diagnostic RSS exceeded 8 GiB'
        elif elapsed > 600:
            reason = 'Diagnostic exceeded its ten-minute time limit'
        if reason:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            break
        if time.monotonic() - last_print >= 30:
            print(json.dumps({'profile_pid': child.pid, 'rss_GiB': round(rss / 2**30, 3),
                              'available_GiB': round(available / 2**30, 3),
                              'elapsed_seconds': round(elapsed, 1)}), flush=True)
            last_print = time.monotonic()
        time.sleep(0.2)
    returncode = child.wait()
receipt = {'diagnostic_only': True, 'started_at': started_at,
           'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'command': command, 'allocator_environment': allocator_env,
           'guard_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'allocator_reference': 'https://docs.python.org/3.11/using/cmdline.html#envvar-PYTHONMALLOC',
           'returncode': returncode, 'guard_stop_reason': reason,
           'peak_diagnostic_rss_bytes': peak_rss, 'minimum_host_available_bytes': minimum_available,
           'elapsed_seconds': time.monotonic() - start, 'log': str(log_path),
           'other_processes_modified': False}
(ROOT / f'cube_allocator_profile_{variant}_guard_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
with log_path.open('rb') as stream:
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - 4500))
    print(stream.read().decode(errors='replace'), flush=True)
raise SystemExit(0 if returncode == 0 and reason is None else 1)
