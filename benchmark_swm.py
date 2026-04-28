from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from hudm.swm_artifacts import eval_summary_row, load_json, write_json, write_review_html, write_summary_csv
from plan_swm import main as run_plan_swm


DEFAULTS = {
    "output_dir": "rollouts/swm_benchmark",
    "title": "SWM HUDM Benchmark",
    "runs": [],
}


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return text.strip("_") or "run"


def _merged_run_config(run: Any) -> tuple[str, Any]:
    if "config" not in run:
        raise ValueError("Each benchmark run must define a plan_swm config path under `config`.")
    name = _safe_name(str(run.get("name", Path(str(run.config)).stem)))
    cfg = OmegaConf.load(str(run.config))
    overrides = run.get("overrides", {})
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return name, cfg


def _write_temp_config(cfg: Any) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="hudm_swm_benchmark_", delete=False)
    tmp.write(OmegaConf.to_yaml(cfg))
    tmp.close()
    return tmp.name


def main(cfg_path: str) -> None:
    cfg = OmegaConf.merge(DEFAULTS, OmegaConf.load(cfg_path))
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    runs = list(cfg.runs)
    if not runs:
        raise ValueError("Benchmark config must include at least one run.")

    for idx, run in enumerate(runs):
        name, run_cfg = _merged_run_config(run)
        run_dir = output_dir / f"{idx:03d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_cfg.eval.output_path = str(run_dir / "eval.json")
        if "video_path" not in run_cfg.eval:
            run_cfg.eval.video_path = str(run_dir / "videos")
        if "save_video" not in run_cfg.eval:
            run_cfg.eval.save_video = True
        temp_cfg = _write_temp_config(run_cfg)

        print(f"[benchmark] {idx + 1}/{len(runs)} {name}")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            run_plan_swm(temp_cfg)
        (run_dir / "run.log").write_text(buffer.getvalue(), encoding="utf-8")

        payload = load_json(run_cfg.eval.output_path)
        payload["benchmark_name"] = name
        write_json(run_cfg.eval.output_path, payload)
        rows.append(eval_summary_row(name, run_cfg.eval.output_path, payload))
        outputs.append(payload)

    summary = {
        "title": str(cfg.title),
        "output_dir": str(output_dir),
        "runs": rows,
    }
    write_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", rows)
    write_review_html(output_dir / "review.html", str(cfg.title), rows, outputs)
    print(f"[benchmark] wrote {output_dir / 'summary.json'}")
    print(f"[benchmark] wrote {output_dir / 'review.html'}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python benchmark_swm.py configs/benchmark_swm.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
