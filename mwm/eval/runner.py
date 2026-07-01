from __future__ import annotations

from pathlib import Path

from mwm.eval.runtime import DEFAULTS, load_eval_runtime, resolve_device


def main(cfg_path: str, *, overrides: list[str] | None = None) -> None:
    from omegaconf import OmegaConf

    from mwm.dependency_refs import dependency_refs
    from mwm.eval.execution import (
        combine_mwm_diagnostics,
        combine_policy_diagnostics,
        combine_swm_results,
        run_batch,
    )
    from mwm.eval.manifest import pairs_for_eval
    from mwm.eval.policy import model_accounting
    from mwm.eval.review_trace import review_rollouts_for_batches
    from mwm.eval.validation import (
        dataset_path,
        dataset_runtime_metadata,
    )
    from mwm.io import jsonable, write_json

    # load_eval_runtime delegates to load_config(DEFAULTS, cfg_path, overrides).
    runtime = load_eval_runtime(cfg_path, overrides=overrides or [])
    try:
        cfg = runtime.cfg
        pairs, manifest_info = pairs_for_eval(
            dataset=runtime.dataset,
            cfg=cfg,
            env_id=runtime.env_id,
            restore_spec_id=runtime.restore_spec_id,
        )
        all_results = []
        batch_size = max(1, int(cfg.eval.num_envs))
        for batch_index, offset in enumerate(range(0, len(pairs), batch_size)):
            all_results.append(
                run_batch(
                    env_id=runtime.env_id,
                    image_shape=runtime.image_shape,
                    model=runtime.model,
                    metadata=runtime.metadata,
                    dataset=runtime.dataset,
                    pairs=pairs[offset : offset + batch_size],
                    cfg=cfg,
                    device=runtime.device,
                    eval_callables=runtime.eval_callables,
                    batch_index=batch_index,
                    process=runtime.process,
                )
            )
        videos = [video for batch in all_results for video in batch.get("videos", [])]
        checkpoint_ref = str(cfg.checkpoint.run_dir)
        accounting = model_accounting(runtime.model)
        swm_results = combine_swm_results(all_results)

        output = {
            "env_id": runtime.env_id,
            "checkpoint_run_dir": checkpoint_ref,
            "checkpoint_epoch": int(runtime.epoch),
            "dataset": dataset_path(runtime.dataset, cfg),
            "episodes": int(cfg.eval.episodes),
            "goal_offset": int(cfg.eval.goal_offset),
            "eval_budget": int(cfg.eval.budget),
            "restore_spec": runtime.restore_spec_id,
            "swm_results": swm_results,
            "planning_diagnostics": combine_mwm_diagnostics(all_results),
            "policy_diagnostics": combine_policy_diagnostics(all_results),
            "model_accounting": accounting,
            "dataset_metadata": dataset_runtime_metadata(runtime.dataset, cfg),
            "manifest": manifest_info,
            "batches": all_results,
            "review_rollouts": review_rollouts_for_batches(
                batches=all_results,
                successes=list(swm_results.get("episode_successes", [])),
                eval_budget=int(cfg.eval.budget),
                action_block=int(cfg.planner.action_block),
                receding_horizon=int(cfg.planner.receding_horizon),
                k_values=[int(k) for k in accounting.get("K", [])],
            ),
            "videos": videos,
            "dependencies": dependency_refs(Path(__file__).resolve().parents[2]),
        }
        output["schedule"] = jsonable(OmegaConf.to_container(cfg.planner.scheduler, resolve=True))
        output["seed"] = int(cfg.eval.seed)
        if cfg.get("config", None):
            output["config"] = jsonable(OmegaConf.to_container(cfg.config, resolve=True))
        output_path = Path(str(cfg.eval.output_path))
        write_json(output_path, output)
        print(f"Wrote MWM planning results to {output_path}")
    finally:
        runtime.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an MWM checkpoint.")
    parser.add_argument("config", help="Evaluation YAML config")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override, e.g. eval.seed=1")
    args = parser.parse_args()
    main(args.config, overrides=args.set)


__all__ = ["DEFAULTS", "main", "resolve_device"]
