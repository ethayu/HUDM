"""Cross-check paired episode evidence beyond the repository's output validator."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'HUDM'
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from omegaconf import OmegaConf
from mwm.benchmark.eval_artifacts import load_eval_artifact, load_planning_diagnostics
from mwm.eval.runtime import effective_goal_offset


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main(config_path):
    config = OmegaConf.load(config_path)
    output = Path(config.output_dir).resolve()
    summary = json.loads((output / 'summary.json').read_text())
    expected_runs = {str(run.role): run for run in config.runs}
    rows = summary['runs']
    require(len(rows) == 2 and {row['role'] for row in rows} == set(expected_runs), 'Expected exactly both checkpoint roles')
    require(set(expected_runs) == {'upstream_lewm_converted', 'sepopt_k192_20260922'}, 'Unexpected role identities')
    identities = None
    protocol = None
    outcomes = {}
    records = []
    for row in rows:
        role = row['role']
        expected = expected_runs[role]
        artifact = Path(row['output_json']).resolve()
        require(artifact.is_relative_to(output), f'Artifact is outside configured output: {artifact}')
        directory = artifact.parent
        cfg = OmegaConf.load(directory / 'resolved_config.yaml')
        payload = load_eval_artifact(artifact, verify='full')
        diagnostics = load_planning_diagnostics(directory, verify='full')
        require(Path(cfg.checkpoint.run_dir).resolve() == Path(expected.checkpoint).resolve(), f'Wrong checkpoint for {role}')
        require(Path(payload['checkpoint_run_dir']).resolve() == Path(expected.checkpoint).resolve(), f'Wrong payload checkpoint for {role}')
        checkpoint = Path(expected.checkpoint)
        metadata = json.loads((checkpoint / 'world_metadata.json').read_text())
        for item in metadata['artifacts'].values():
            require(digest(checkpoint / item['path']) == item['sha256'], f'Checkpoint hash mismatch: {checkpoint}')
        checkpoint_config = json.loads((checkpoint / metadata['artifacts']['config']['path']).read_text())
        require(checkpoint_config['kwargs']['K'] == [192], 'Checkpoint is not singleton K=192')
        require(payload['model_accounting']['K'] == [192], 'Executed model is not singleton K=192')
        count = int(expected.eval.episodes)
        require(int(row['episodes']) == int(payload['episodes']) == int(cfg.eval.episodes) == count, 'Episode count mismatch')
        require(int(cfg.eval.seed) == int(config.seed), 'Evaluation seed mismatch')
        require(str(cfg.env_id) == str(config.env_id) == payload['env_id'], 'Environment mismatch')
        env_runtime = payload.get('env_runtime', {})
        if str(config.env_id) == 'swm/ReacherDMControl-v0':
            require(env_runtime.get('reacher_qpos_threshold') == 0.1, 'Reacher did not execute the required 0.1 threshold')
            require(env_runtime.get('post_reset_validation') == 'passed', 'Reacher threshold was not validated after reset')
            require(all(batch.get('env_runtime') == env_runtime for batch in payload['batches']), 'Reacher runtime differs across batches')
        manifest_value = cfg.eval.get('manifest_path') or cfg.eval.get('write_manifest_path')
        require(bool(manifest_value), 'Resolved config has neither a read nor write manifest path')
        manifest_path = Path(manifest_value).resolve()
        require(manifest_path == Path(config.manifest.path).resolve(), 'Unexpected manifest in resolved config')
        require(manifest_path == Path(payload['manifest']['path']).resolve(), 'Payload manifest path differs from config')
        actual_protocol = {
            'data': OmegaConf.to_container(cfg.data, resolve=True),
            'eval': {key: OmegaConf.to_container(cfg.eval, resolve=True).get(key) for key in
                     ('episodes','num_envs','goal_offset','goal_indexing','seed','budget','sampling','manifest_path')},
            'env': OmegaConf.to_container(cfg.env, resolve=True),
            'executed_env_runtime': env_runtime,
            'restore': OmegaConf.to_container(cfg.restore, resolve=True),
            'planner': OmegaConf.to_container(cfg.planner, resolve=True),
            'device': str(cfg.device),
        }
        # The first checkpoint writes the manifest; later checkpoints read the
        # same immutable file. Compare its identity, not the transport direction.
        actual_protocol['eval']['manifest_path'] = str(manifest_path)
        if protocol is None: protocol = actual_protocol
        require(actual_protocol == protocol, 'Published and retrained evaluation protocols differ')
        for key, value in {'horizon':5, 'receding_horizon':5, 'action_block':5, 'pop_size':300, 'topk':30,
                           'n_iter':30 if str(config.env_id)=='swm/PushT-v1' else 10}.items():
            require(int(cfg.planner[key]) == value, f'Unexpected {key}: {cfg.planner[key]}')
        require(float(cfg.planner.elite_frac) == 0.1 and cfg.planner.flop_accounting == 'dynamics_audit', 'Planner audit protocol mismatch')
        require(int(diagnostics['summary']['flop_audit_error_count']) == 0, 'FLOP audit recorded errors')
        require(int(diagnostics['dynamics_flops_total']) == int(row['dynamics_flops_total']) > 0, 'FLOP accounting mismatch')
        require(digest(manifest_path) == row['manifest_file_sha256'], 'Manifest hash mismatch')
        manifest = json.loads(manifest_path.read_text())
        traces = [json.loads(line) for line in (directory / 'episode_traces.jsonl').read_text().splitlines() if line.strip()]
        pairs = manifest['pairs']
        require(len(traces) == len(pairs) == count, 'Trace or manifest count mismatch')
        effective = effective_goal_offset(int(cfg.eval.goal_offset), str(cfg.eval.goal_indexing))
        pair_ids = []
        successes = []
        for index, (trace, pair) in enumerate(zip(traces, pairs, strict=True)):
            require(int(trace['episode_index']) == index, 'Unexpected episode trace ordering')
            identity = (int(trace['dataset_episode']), int(trace['start_step']), int(trace['goal_step']))
            require(identity == (int(pair['episode']), int(pair['start_step']), int(pair['goal_step'])), 'Trace does not match manifest pair')
            require(identity[2] - identity[1] == effective, 'Actual goal offset mismatch')
            require(type(trace['success']) is bool, 'Non-boolean episode outcome')
            pair_ids.append(identity)
            successes.append(trace['success'])
        batch_pairs = [pair for batch in payload['batches'] for pair in batch['pairs']]
        require(len(batch_pairs) == count, 'Payload batch pair count mismatch')
        for recorded, pair in zip(batch_pairs, pairs, strict=True):
            require(all(int(recorded[key]) == int(pair[key]) for key in ('episode','start_step','goal_step','start_row','goal_row')), 'Payload execution pairs differ from manifest')
        if identities is None: identities = pair_ids
        require(pair_ids == identities, 'Model roles did not evaluate the same ordered start/goal pairs')
        require(payload['swm_results']['episode_successes'] == successes, 'Payload outcomes differ from traces')
        rate = 100.0 * sum(successes) / count
        require(math.isclose(rate, float(row['success_rate']), abs_tol=1e-9), 'Summary success rate differs from actual episode outcomes')
        require(math.isclose(rate, float(payload['swm_results']['success_rate']), abs_tol=1e-9), 'Payload success rate differs from actual outcomes')
        outcomes[role] = successes
        records.append({'role':role,'episodes':count,'successes':sum(successes),'success_rate':rate,
                        'weights_sha256':metadata['artifacts']['weights']['sha256'],
                        'trace_sha256':digest(directory/'episode_traces.jsonl'),'resolved_config_sha256':digest(directory/'resolved_config.yaml'),
                        'manifest_sha256':digest(manifest_path),'dynamics_flops_total':int(row['dynamics_flops_total'])})
    paired = [{'episode_index':index,'dataset_episode':identity[0],'start_step':identity[1],'goal_step':identity[2],
               'published_success':outcomes['upstream_lewm_converted'][index],
               'retrained_success':outcomes['sepopt_k192_20260922'][index]}
              for index, identity in enumerate(identities)]
    result = {'verified':True,'seed':int(config.seed),'env_id':str(config.env_id),'config':str(Path(config_path).resolve()),
              'roles':records,'paired_episodes':paired,'protocol':protocol}
    destination = output/'paired_audit.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified':True,'audit':str(destination),'roles':records}),flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
