import gc,json,os,resource,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root/'HUDM')
sys.path.insert(0,str(root/'HUDM'))
os.environ.update({'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
import psutil
from omegaconf import OmegaConf
from mwm.swm.envs import make_swm_world
results=[]
process=psutil.Process()
for environment in ('tworoom','reacher','pusht','ogb_cube'):
 cfg=OmegaConf.load(f'configs/eval/paper_{environment}.yaml')
 start=time.monotonic()
 world=make_swm_world(cfg.env_id,50,(224,224),max_episode_steps=100,goal_conditioned=True,env_kwargs=OmegaConf.to_container(cfg.env.kwargs,resolve=True))
 try:
  observations,info=world.envs.reset(seed=123)
  for _ in range(2):
   observations,reward,terminated,truncated,info=world.envs.step(world.envs.action_space.sample())
  result={'environment':environment,'num_envs':50,'elapsed_seconds':time.monotonic()-start,'rss_bytes':process.memory_info().rss,'host_available_bytes':psutil.virtual_memory().available,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  results.append(result)
  print(json.dumps(result),flush=True)
 finally:
  world.close()
 del world,observations,info,reward,terminated,truncated
 gc.collect()
(root/'vector_env_smoke_receipt.json').write_text(json.dumps({'passed':True,'results':results},indent=2)+'\n')
