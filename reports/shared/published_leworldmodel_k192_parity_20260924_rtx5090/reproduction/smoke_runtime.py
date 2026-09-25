"""Exercise the real GPU checkpoints and all four simulator renderers."""
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'HUDM'
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.update({'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'MPLBACKEND': 'Agg',
                   'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy',
                   'HF_HOME': str(ROOT / 'cache/huggingface'), 'MPLCONFIGDIR': str(ROOT / 'cache/matplotlib'),
                   'GIT_CONFIG_GLOBAL': '/dev/null', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
import numpy as np
import torch
from omegaconf import OmegaConf
from mwm.checkpoint_io import load_world_model_from_checkpoint
from mwm.dependency_refs import dependency_refs
from mwm.swm.envs import make_swm_world, apply_swm_world_runtime_config, validate_swm_world_runtime_config

torch.set_num_threads(4)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.get_device_capability(0) == (12, 0), 'Unexpected GPU architecture'
with torch.inference_mode():
    x = torch.ones((256, 256), device='cuda')
    assert torch.all(x @ x == 256).item()
    y = torch.nn.functional.conv2d(torch.ones((1,3,32,32),device='cuda'), torch.ones((8,3,3,3),device='cuda'))
    assert torch.isfinite(y).all().item()
    del x,y
receipt = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'models': [], 'environments': [], 'dependencies': dependency_refs(REPO)}
for environment in ('tworoom', 'reacher', 'pusht', 'ogb_cube'):
    for name in (f'upstream_lewm_{environment}', f'mwm_paper10_{environment}_k192_sepopt_retrain_20260922'):
        torch.cuda.reset_peak_memory_stats()
        model, metadata, epoch = load_world_model_from_checkpoint(REPO/'checkpoints_mwm'/name, None, torch.device('cuda'))
        assert tuple(model.K) == (192,), (name, model.K)
        history = int(model.history_size)
        model.planning_rollout_semantics = 'upstream_lewm_historical'
        with torch.inference_mode():
            pixels = torch.zeros((1, 2, history, 3, 224, 224), dtype=torch.uint8, device='cuda')
            actions = torch.zeros((1, 2, 5, model.action_dim), device='cuda')
            result = model.rollout_at_level({'pixels': pixels}, actions, 0)
            predicted = result['predicted_emb']
            assert predicted.is_cuda and torch.isfinite(predicted).all().item()
            torch.cuda.synchronize()
            item = {'checkpoint': name, 'history': history, 'prediction_shape': list(predicted.shape),
                    'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated()}
            receipt['models'].append(item)
            print(json.dumps(item), flush=True)
        del model, pixels, actions, result, predicted
        gc.collect()
        torch.cuda.empty_cache()
    cfg = OmegaConf.load(REPO/f'configs/eval/paper_{environment}.yaml')
    world = make_swm_world(cfg.env_id, 1, (224,224), max_episode_steps=100, goal_conditioned=True,
                           env_kwargs=OmegaConf.to_container(cfg.env.kwargs,resolve=True))
    try:
        runtime = OmegaConf.to_container(cfg.env.runtime, resolve=True) if cfg.env.get('runtime') else {}
        provenance = apply_swm_world_runtime_config(world,cfg.env_id,runtime)
        observation, info = world.envs.reset(seed=123)
        # Reacher recompilation can reset its threshold; the full evaluator reapplies it through its restore callback.
        if runtime:
            provenance = apply_swm_world_runtime_config(world,cfg.env_id,runtime)
            validate_swm_world_runtime_config(world,provenance)
        frame = np.asarray(world.envs.envs[0].render())
        assert frame.ndim == 3 and frame.shape[-1] in (3,4), (environment,frame.shape)
        item = {'environment': environment, 'render_shape': list(frame.shape), 'runtime': provenance}
        receipt['environments'].append(item)
        print(json.dumps(item),flush=True)
    finally:
        world.close()
receipt['passed'] = True
(ROOT/'runtime_smoke_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('All eight CUDA model rollouts and four environment renders passed.',flush=True)
