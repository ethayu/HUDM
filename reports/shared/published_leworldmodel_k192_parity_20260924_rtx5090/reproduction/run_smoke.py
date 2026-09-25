"""Run an isolated evaluator smoke after byte-complete Lance validation.

This diagnostic does not replace source checksum verification or the full protocol.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import run_local as runner

ROOT=Path(__file__).resolve().parent
ENV_DATA={'ogb_cube':'ogb_cube_single_expert.lance','reacher':'reacher.lance',
          'pusht':'pusht_expert_train.lance','tworoom':'tworoom.lance'}

def main(environment, *, verify_only=False):
    lock=(ROOT/'local_run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import lance
    inventory=json.loads((ROOT/'local_dataset_inventory.json').read_text())
    dataset=next(d for d in inventory['datasets'] if d['name']==ENV_DATA[environment])
    assert dataset['status']=='complete_bytes_pending_source_hash_comparison'
    for record in dataset['files']:
        path=ROOT/'source_snapshot'/record['relative_path']
        stat=path.stat()
        assert record.get('sha256') and stat.st_size==record['expected_bytes']
        assert (stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino)==(record['mtime_ns'],record['ctime_ns'],record['inode'])
    lance.dataset(str(ROOT/'source_snapshot/data/upstream'/ENV_DATA[environment])).validate()
    os.environ.update({'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','MPLBACKEND':'Agg',
                       'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','GIT_CONFIG_GLOBAL':'/dev/null',
                       'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                       'MPLCONFIGDIR':str(ROOT/'cache/matplotlib'),'HF_HOME':str(ROOT/'cache/huggingface')})
    config=runner.REPORT/'generated_configs/local_smoke'/f'{environment}.yaml'
    state={'stage':'isolated_evaluator_smoke','environment':environment,'not_scientific_result':True}
    label=f'local_smoke_{environment}'
    verify=[sys.executable,'-m','mwm.benchmark.verify',str(config)]
    if not verify_only:
        runner.run([*verify,'--static-only'],label+'_static',state)
        runner.run([sys.executable,'-m','mwm.benchmark.matrix',str(config),'--resume'],label+'_matrix',state)
    runner.run(verify,label+'_verify',state)
    runner.run([sys.executable,str(ROOT/'audit_paired.py'),str(config)],label+'_paired_audit',state)
    output=ROOT/'evaluator_smoke_receipt.json'
    receipt=json.loads(output.read_text()) if output.exists() else {}
    receipt[environment]={'passed':True,'finished_at':runner.timestamp(),
                          'config':str(runner.REPO/config),'dataset':dataset,
                          'source_hash_comparison_pending':True,'not_scientific_result':True}
    temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt,indent=2)+'\n')
    temporary.replace(output)
    runner.save('evaluator_smoke_passed',**state)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('environment',choices=ENV_DATA)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    main(args.environment,verify_only=args.verify_only)
