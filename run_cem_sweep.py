"""Run a CEM pop_size sweep across scheduler variants for one env."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

SCHEDULERS = {
    "coarsest": {
        "planner.scheduler.policy": "fixed",
        "planner.scheduler.level": "coarsest",
        "planner.scheduler.rollout_level.mode": "fixed",
        "planner.scheduler.rollout_level.level": "base",
    },
    "coarsest_to_finest": {
        "planner.scheduler.policy": "linear_cem",
        "planner.scheduler.start_level": "coarsest",
        "planner.scheduler.end_level": "finest",
        "planner.scheduler.rollout_level.mode": "fixed",
        "planner.scheduler.rollout_level.level": "base",
    },
    "k96_to_finest": {
        "planner.scheduler.policy": "linear_cem",
        "planner.scheduler.start_level": "3",
        "planner.scheduler.end_level": "finest",
        "planner.scheduler.rollout_level.mode": "fixed",
        "planner.scheduler.rollout_level.level": "base",
    },
    # Same level ramp as k96_to_finest, but pop_size shrinks in lockstep
    # (driven by the shared cem_progress signal) as the level gets finer.
    # The actual start/end pop_size are filled in by run_one() based on
    # --pop-sizes and --pop-shrink-end-frac.
    "k96_to_finest_popshrink": {
        "planner.scheduler.policy": "linear_cem",
        "planner.scheduler.start_level": "3",
        "planner.scheduler.end_level": "finest",
        "planner.scheduler.rollout_level.mode": "fixed",
        "planner.scheduler.rollout_level.level": "base",
    },
    "fixed_finest": {
        "planner.scheduler.policy": "fixed",
        "planner.scheduler.level": "finest",
        "planner.scheduler.rollout_level.mode": "fixed",
        "planner.scheduler.rollout_level.level": "base",
    },
    # Individual fixed levels: K=[6,12,48,96,144,192] → indices 0-5
    "k6":  {"planner.scheduler.policy": "fixed", "planner.scheduler.level": "0",
             "planner.scheduler.rollout_level.mode": "fixed", "planner.scheduler.rollout_level.level": "base"},
    "k12": {"planner.scheduler.policy": "fixed", "planner.scheduler.level": "1",
             "planner.scheduler.rollout_level.mode": "fixed", "planner.scheduler.rollout_level.level": "base"},
    "k48": {"planner.scheduler.policy": "fixed", "planner.scheduler.level": "2",
             "planner.scheduler.rollout_level.mode": "fixed", "planner.scheduler.rollout_level.level": "base"},
    "k96": {"planner.scheduler.policy": "fixed", "planner.scheduler.level": "3",
             "planner.scheduler.rollout_level.mode": "fixed", "planner.scheduler.rollout_level.level": "base"},
    "k144": {"planner.scheduler.policy": "fixed", "planner.scheduler.level": "4",
              "planner.scheduler.rollout_level.mode": "fixed", "planner.scheduler.rollout_level.level": "base"},
    "k192": {"planner.scheduler.policy": "fixed", "planner.scheduler.level": "5",
              "planner.scheduler.rollout_level.mode": "fixed", "planner.scheduler.rollout_level.level": "base"},
}


def run_one(env: str, scheduler_name: str, pop_size: int, n_iter: int, args) -> dict:
    from eval_mwm import main as run_eval

    subdir = getattr(args, "out_subdir", "")
    base = f"rollouts/cem_sweep/{subdir}/{env}" if subdir else f"rollouts/cem_sweep/{env}"
    out_path = Path(f"{base}/{scheduler_name}_pop{pop_size}_niter{n_iter}/eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_cfg = args.pusht_cfg if env == "pusht" else args.tworoom_cfg
    checkpoint = args.pusht_ckpt if env == "pusht" else args.tworoom_ckpt
    data = args.pusht_data if env == "pusht" else args.tworoom_data

    topk = max(1, round(args.elite_frac * pop_size))

    overrides = [
        f"checkpoint.run_dir={checkpoint}",
        f"data.path={data}",
        f"data.split_ratio=1.0",
        f"eval.episodes={args.episodes}",
        f"eval.num_envs=1",
        f"eval.seed=0",
        f"eval.sampling=stable_worldmodel",
        f"eval.output_path={out_path}",
        f"eval.goal_offset={args.goal_offset}",
        f"eval.budget={args.budget}",
        f"planner.pop_size={pop_size}",
        f"planner.n_iter={n_iter}",
        f"planner.topk={topk}",
        f"planner.elite_frac={args.elite_frac}",
        f"planner.batch_size=1",
        f"device=cuda",
    ]
    sched = SCHEDULERS[scheduler_name]
    for k, v in sched.items():
        overrides.append(f"{k}={v}")
    if scheduler_name.endswith("_popshrink"):
        pop_end = max(1, round(pop_size * args.pop_shrink_end_frac))
        overrides.append(f"planner.pop_schedule.start={pop_size}")
        overrides.append(f"planner.pop_schedule.end={pop_end}")

    print(f"\n{'='*60}")
    print(f"  env={env}  scheduler={scheduler_name}  pop_size={pop_size}  n_iter={n_iter}")
    print(f"{'='*60}")
    run_eval(base_cfg, overrides=overrides)

    result = json.loads(out_path.read_text())
    sr = result["swm_results"]["success_rate"]
    print(f"  => success_rate: {sr:.1f}%  (topk={topk}, n_iter={n_iter})")
    return {"env": env, "scheduler": scheduler_name, "pop_size": pop_size, "n_iter": n_iter, "topk": topk, "success_rate": sr}


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    p = argparse.ArgumentParser()
    p.add_argument("--pusht-cfg", default="configs/eval/paper_pusht.yaml")
    p.add_argument("--tworoom-cfg", default="configs/eval/paper_tworoom.yaml")
    p.add_argument("--pusht-ckpt", default="checkpoints_mwm/mwm_dense_pusht")
    p.add_argument("--tworoom-ckpt", default="checkpoints_mwm/mwm_dense_tworoom")
    p.add_argument("--pusht-data", default="data/local/pusht_20ep_224.lance")
    p.add_argument("--tworoom-data", default="data/local/tworoom_20ep_224.lance")
    p.add_argument("--pop-sizes", type=int, nargs="+", default=[3000])
    p.add_argument("--n-iters", type=int, nargs="+", default=[30],
                   help="CEM refinement iterations to sweep")
    p.add_argument("--elite-frac", type=float, default=0.1,
                   help="fraction of pop_size kept as elites; topk = round(elite_frac * pop_size)")
    p.add_argument("--pop-shrink-end-frac", type=float, default=0.1,
                   help="for *_popshrink schedulers: final pop_size as a fraction of the starting pop_size")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--goal-offset", type=int, default=25,
                   help="steps ahead in the demo trajectory the goal state/image is sampled from")
    p.add_argument("--budget", type=int, default=50,
                   help="max env steps the agent gets to reach the goal")
    p.add_argument("--envs", nargs="+", default=["pusht", "tworoom"])
    p.add_argument("--schedulers", nargs="+",
                   default=["coarsest", "coarsest_to_finest", "k96_to_finest", "fixed_finest"])
    p.add_argument("--out-subdir", default="", help="subdirectory under rollouts/cem_sweep/")
    args = p.parse_args()

    rows = []
    for env in args.envs:
        for pop_size in args.pop_sizes:
            for n_iter in args.n_iters:
                for sched in args.schedulers:
                    rows.append(run_one(env, sched, pop_size, n_iter, args))

    print("\n\n=== RESULTS ===")
    print(f"{'env':<10} {'scheduler':<22} {'pop_size':>8} {'n_iter':>6} {'topk':>6} {'success%':>10}")
    print("-" * 72)
    for r in rows:
        print(f"{r['env']:<10} {r['scheduler']:<22} {r['pop_size']:>8} {r['n_iter']:>6} {r['topk']:>6} {r['success_rate']:>9.1f}%")

    out = Path("rollouts/cem_sweep/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
