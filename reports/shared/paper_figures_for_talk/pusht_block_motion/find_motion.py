from pathlib import Path
import json, numpy as np,lance
from omegaconf import OmegaConf
import mwm.benchmark.replay_runtime as rr
from mwm.benchmark.review_media import CapturingFixedActionPolicy,valid_action_prefix
from mwm.swm.envs import make_swm_world,parse_env_kwargs
ROOT=Path(__file__).resolve().parent
class Table:
 def __init__(self,p):
  self.ds=lance.dataset(str(p));self.schema=self.ds.schema;self.version=self.ds.version
 def to_lance(self):return self.ds
 def count_rows(self):return self.ds.count_rows()
rr._open_lance_review_table=Table
class Policy(CapturingFixedActionPolicy):
 def __init__(self,actions):super().__init__(actions);self.poses=[]
 def record_pose(self):
  b=self.env.envs[0].unwrapped.block
  self.poses.append([float(b.position.x),float(b.position.y),float(b.angle)])
 def get_action(self,*a,**kw):self.record_pose();return super().get_action(*a,**kw)

ds=lance.dataset('data/upstream/pusht_expert_train.lance')
candidates=[]
for base in ['release20260728_sepopt_k48to192_pusht_horizon5_all_fidelity_schedules','release20260728_sepopt_k48to192_pusht_goal50_horizon5_all_fidelity_schedules']:
 for p in Path('reports/research',base).glob('*23_mpc*pop100*elite0p1*/eval.json'):
  d=json.loads(p.read_text())
  for r in d['review_rollouts']:
   if not r['success']:continue
   states=np.asarray(ds.take([r['start_row'],r['goal_row']],columns=['state']).column(0).to_pylist()).reshape(2,-1)
   dist=float(np.linalg.norm(states[1,2:4]-states[0,2:4]));angle=float(np.degrees(abs(np.arctan2(np.sin(states[1,4]-states[0,4]),np.cos(states[1,4]-states[0,4])))))
   candidates.append(dict(run=str(p.parent),episode=r['episode_index'],goal_block_distance=dist,goal_angle_degrees=angle,score=dist+angle,rollout=r))
candidates.sort(key=lambda r:r['score'],reverse=True)
# Include the three delivered examples as a direct audit, then high-motion goals.
original=json.loads(Path('reports/shared/paper_figures_for_talk/other_environments/pusht/screening_results.json').read_text())[:3]
selected=[];seen=set()
for src in original:
 match=next(r for r in candidates if r['run']==src['run'] and r['episode']==src['episode'])
 selected.append(match);seen.add((match['run'],match['episode']))
for c in candidates:
 key=(c['run'],c['episode'])
 if key not in seen:selected.append(c);seen.add(key)
 if len(selected)>=33:break
results=[]
for c in selected:
 r=c['rollout'];run=Path(c['run']);print('REPLAY',c['episode'],c['goal_block_distance'],c['goal_angle_degrees'],flush=True)
 rt=rr.load_review_runtime(run/'resolved_config.yaml',start_row=r['start_row'],goal_row=r['goal_row'],load_model=False)
 policy=Policy(valid_action_prefix(r['action_trace']))
 world=make_swm_world(rt.env_id,num_envs=1,image_shape=rt.image_shape,max_episode_steps=int(rt.cfg.env.max_episode_steps),goal_conditioned=True,env_kwargs=parse_env_kwargs(OmegaConf.to_container(rt.cfg.env.kwargs)))
 try:
  world.set_policy(policy)
  result=world.evaluate(dataset=rt.dataset,episodes_idx=[r['dataset_episode']],start_steps=[r['start_step']],eval_budget=len(policy.action_trace),callables=rt.eval_callables,goal_offset=int(rt.cfg.eval.goal_offset))
  policy.record_pose();poses=np.array(policy.poses)
  trans=np.linalg.norm(poses[:,:2]-poses[0,:2],axis=1)
  angles=np.degrees(np.unwrap(poses[:,2])-poses[0,2])
  row={k:v for k,v in c.items() if k not in ['rollout','score']}
  row.update(success=bool(result['episode_successes'][0]),steps=policy._idx,max_translation=float(trans.max()),net_translation=float(trans[-1]),path_length=float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()),max_rotation_degrees=float(np.abs(angles).max()),net_rotation_degrees=float(angles[-1]),block_poses=poses.tolist())
  results.append(row)
  print('RESULT',json.dumps({k:v for k,v in row.items() if k!='block_poses'}),flush=True)
  (ROOT/'motion_results.json').write_text(json.dumps(results,indent=2)+'\n')
 finally:world.close();rt.close()
print('BEST',[(r['episode'],r['steps'],r['max_translation'],r['max_rotation_degrees']) for r in sorted(results,key=lambda r:r['max_translation']+r['max_rotation_degrees'],reverse=True) if r['success']][:8],flush=True)
