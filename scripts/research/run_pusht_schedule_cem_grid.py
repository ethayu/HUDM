"""Sweep CEM (pop_size, n_iter) for each of the 26 fidelity schedules in
configs/research/dense_pusht_all_fidelity_schedules.yaml, on PushT.

Mirrors scripts/research/run_cem_sweep.py's approach (call mwm.eval.runner.main
in-process per grid point to avoid per-run Python startup cost) but targets the
current nested planner.scheduler.{mpc,cem,rollout} config schema and the 26
schedules used in today's dense-fidelity benchmark matrix, instead of the old
flat planner.scheduler.policy schema and the 7 ad hoc series names used by the
historical rollouts/pusht_h5_100ep_all_configs.png plot (two of which,
"cascade" and "forward" CEM policies, no longer exist in this codebase).

Output layout matches the historical grid convention so existing plotting code
(plot_tworoom_all_configs.py) can be adapted directly:
  rollouts/cem_sweep/<out_subdir>/<schedule_name>/pop{P}_niter{N}/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mwm.eval.runner import main as run_eval


def load_schedules(schedules_cfg: str) -> list[dict]:
    doc = yaml.safe_load(Path(schedules_cfg).read_text())
    return [{"name": r["name"], "scheduler": r["planner"]["scheduler"]} for r in doc["runs"]]


def write_temp_config(base_cfg_path: str, scheduler: dict) -> str:
    """Load the base eval config and wholesale-replace planner.scheduler.

    OmegaConf.merge is a deep merge — dotlist overrides on individual
    planner.scheduler.<stage>.<key> fields would leave stale sibling keys
    from the base config's default stage blocks (e.g. a leftover `level`
    when switching a stage to mode=linear, which sets start_level/end_level
    instead). mwm.benchmark.matrix avoids exactly this by replacing the
    scheduler node wholesale (see mwm/benchmark/config.py:merged_run_config)
    since the scheduler schema is intentionally closed. Mirror that here.
    """
    cfg = OmegaConf.load(base_cfg_path)
    cfg.planner.scheduler = OmegaConf.create({"enabled": True, **scheduler})
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="mwm_cem_grid_", delete=False)
    tmp.write(OmegaConf.to_yaml(cfg))
    tmp.close()
    return tmp.name


def run_one(
    schedule_name: str,
    scheduler: dict,
    pop_size: int,
    n_iter: int,
    args: argparse.Namespace,
) -> dict:
    out_dir = Path(f"rollouts/cem_sweep/{args.out_subdir}/{schedule_name}/pop{pop_size}_niter{n_iter}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval.json"
    if out_path.exists() and not args.overwrite:
        return {"schedule": schedule_name, "pop_size": pop_size, "n_iter": n_iter, "status": "skipped"}

    topk = max(1, round(args.elite_frac * pop_size))
    overrides = [
        f"checkpoint.run_dir={args.checkpoint}",
        f"eval.episodes={args.episodes}",
        f"eval.num_envs={min(args.episodes, args.max_num_envs)}",
        f"eval.seed={args.seed}",
        f"eval.goal_offset={args.goal_offset}",
        f"eval.output_path={out_path}",
        "eval.manifest_path=null",
        f"planner.pop_size={pop_size}",
        f"planner.n_iter={n_iter}",
        f"planner.topk={topk}",
        f"planner.elite_frac={args.elite_frac}",
        "device=cuda",
    ]

    temp_cfg_path = write_temp_config(args.cfg, scheduler)
    t0 = time.monotonic()
    print(f"[grid] schedule={schedule_name} pop={pop_size} n_iter={n_iter} ...", flush=True)
    try:
        run_eval(temp_cfg_path, overrides=overrides)
    finally:
        Path(temp_cfg_path).unlink(missing_ok=True)
    dt = time.monotonic() - t0

    result = json.loads(out_path.read_text())
    sr = result["swm_results"]["success_rate"]
    bits = result["planning_diagnostics"]["bits_used_total"]
    print(f"[grid]   -> success={sr:.1f}%  bits/ep={bits/args.episodes/1e6:.2f}M  ({dt:.1f}s)", flush=True)
    return {
        "schedule": schedule_name,
        "pop_size": pop_size,
        "n_iter": n_iter,
        "topk": topk,
        "success_rate": sr,
        "bits_used_total": bits,
        "wall_time_sec": dt,
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/eval/paper_pusht.yaml")
    p.add_argument("--schedules-cfg", default="configs/research/dense_pusht_all_fidelity_schedules.yaml")
    p.add_argument("--checkpoint", default="checkpoints_mwm/mwm_paper10_pusht_k96_120_144_168_192_release20260728")
    p.add_argument("--manifest-path", default="rollouts/manifests/pusht_paper_seed42.json")
    p.add_argument("--pop-sizes", type=int, nargs="+", default=[20, 50, 100, 150, 200])
    p.add_argument("--n-iters", type=int, nargs="+", default=[5, 10, 20, 30, 50])
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-num-envs", type=int, default=100)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedules", nargs="+", default=None, help="subset of schedule names to run; default all 26")
    p.add_argument("--out-subdir", default="grid_pusht_paper10_k96to192_100ep")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    schedules = load_schedules(args.schedules_cfg)
    if args.schedules:
        wanted = set(args.schedules)
        schedules = [s for s in schedules if s["name"] in wanted]

    total = len(schedules) * len(args.pop_sizes) * len(args.n_iters)
    print(f"[grid] {len(schedules)} schedules x {len(args.pop_sizes)} pop_sizes x {len(args.n_iters)} n_iters = {total} runs")

    rows = []
    done = 0
    t_start = time.monotonic()
    for sched in schedules:
        for pop_size in args.pop_sizes:
            for n_iter in args.n_iters:
                row = run_one(sched["name"], sched["scheduler"], pop_size, n_iter, args)
                rows.append(row)
                done += 1
                elapsed = time.monotonic() - t_start
                print(f"[grid] progress {done}/{total}  elapsed={elapsed/60:.1f}m", flush=True)

    out = Path(f"rollouts/cem_sweep/{args.out_subdir}/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"[grid] wrote {out}")


if __name__ == "__main__":
    main()
