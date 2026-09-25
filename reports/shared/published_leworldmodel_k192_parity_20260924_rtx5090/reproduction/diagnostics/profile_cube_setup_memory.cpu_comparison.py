"""CPU-only diagnostic of unused memory after exact Cube runtime setup.

Run after the active scientific runner finishes. This does not construct an
environment, execute a rollout, edit source/dependencies, or produce a result cell.
"""
import argparse
import ctypes
import datetime
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--label', choices=('default_seeded', 'malloc_seeded'), required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
run_lock = (ROOT / 'local_run.lock').open('a')
fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.chdir(ROOT / 'HUDM')
sys.path.insert(0, str(ROOT / 'HUDM'))
os.environ.update({'GIT_CONFIG_GLOBAL': '/dev/null', 'OMP_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'MPLBACKEND': 'Agg',
                   'HF_HOME': str(ROOT / 'cache/huggingface'),
                   'MPLCONFIGDIR': str(ROOT / 'cache/matplotlib')})
Path('/proc/self/oom_score_adj').write_text('1000\n')
import numpy as np
import psutil
import pyarrow as pa
import torch
from mwm.eval.runtime import load_eval_runtime

process = psutil.Process()
samples = []

def sample(label):
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem'):
            status[key] = value.strip()
    item = {'phase': label, 'rss_bytes': process.memory_info().rss,
            'host_available_bytes': psutil.virtual_memory().available,
            'arrow_allocated_bytes': pa.total_allocated_bytes(), **status}
    samples.append(item)
    print(json.dumps(item), flush=True)

def fingerprint(runtime):
    h = hashlib.sha256()
    for key, tensor in sorted(runtime.model.state_dict().items()):
        h.update(key.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        h.update(memoryview(data))
    for key, scaler in sorted(runtime.process.items()):
        h.update(key.encode())
        for field in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
            value = np.asarray(getattr(scaler, field))
            h.update(field.encode())
            h.update(str((value.dtype, value.shape)).encode())
            h.update(value.tobytes())
    return h.hexdigest()

def rng_fingerprint():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
                                      torch.get_rng_state().numpy().tobytes()))).hexdigest()

sample('imports_complete')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
config = ROOT / 'HUDM/configs/eval/paper_ogb_cube.yaml'
runtime = load_eval_runtime(str(config), overrides=['device=cpu'])
assert not torch.cuda.is_initialized(), 'The diagnostic must remain CPU-only'
before = fingerprint(runtime)
rng_before = rng_fingerprint()
sample('runtime_ready')
collected = gc.collect()
sample('after_gc')
pa.default_memory_pool().release_unused()
sample('after_arrow_release_unused')
libc = ctypes.CDLL(None)
trim = libc.malloc_trim
trim.argtypes = [ctypes.c_size_t]
trim.restype = ctypes.c_int
trim_result = trim(0)
sample('after_malloc_trim')
after = fingerprint(runtime)
assert before == after, 'Model or scaler values changed'
assert rng_before == rng_fingerprint(), 'RNG state changed'
receipt = {'diagnostic_only': True, 'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'device': 'cpu', 'diagnostic_seed': 42, 'variant': args.label,
           'python_allocator': os.environ.get('PYTHONMALLOC', 'default'),
           'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
           'diagnostic_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'environment_created': False, 'rollouts_executed': False,
           'source_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
           'checkpoint': str(runtime.cfg.checkpoint.run_dir),
           'source_and_dependencies_modified': False, 'arrow_allocator': pa.default_memory_pool().backend_name,
           'gc_objects_collected': collected, 'malloc_trim_result': trim_result,
           'model_and_scaler_values_unchanged': True, 'model_and_scaler_sha256': after,
           'rng_state_unchanged': True, 'rng_sha256': rng_before, 'samples': samples,
           'rss_released_bytes': samples[1]['rss_bytes'] - samples[-1]['rss_bytes'],
           'scope': 'CPU setup memory only; does not prove that the 50-environment GPU experiment fits.'}
(ROOT / f'cube_allocator_profile_{args.label}_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
