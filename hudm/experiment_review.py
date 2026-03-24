from __future__ import annotations

import contextlib
import csv
import hashlib
import html
import itertools
import json
import math
import mimetypes
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

from hudm.experiment_bundle import (
    EXPERIMENT_JSON,
    PAIRED_VS_BASELINE_CSV,
    REVIEWER_SCHEMA_VERSION,
    RUNS_CSV,
    SELECTED_ROLLOUTS_JSON,
    VARIANTS_CSV,
    bundle_paths,
    ensure_dir,
    review_derived_dir,
    review_media_dir,
    trace_dir,
)
from hudm.runtime import format_bits_human, format_flops_human
from planning.cem_core import SharedCEMCore
from scripts import planning_media

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline.offline import get_plotlyjs
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    go = None
    pio = None
    get_plotlyjs = None
    make_subplots = None


REVIEW_MEDIA = (
    "closed_loop_replay",
    "planner_view_replay",
    "predicted_backend_replay",
    "gt_replay",
)

CANONICAL_DOWNLOADS = (
    EXPERIMENT_JSON,
    RUNS_CSV,
    VARIANTS_CSV,
    PAIRED_VS_BASELINE_CSV,
    SELECTED_ROLLOUTS_JSON,
)

FINAL_METRIC_SPECS = (
    ("final_angle_diff", "Final Angle Diff"),
    ("final_eef_diff", "Final EEF Diff"),
    ("final_coverage", "Final Coverage"),
)

COMPUTE_METRIC_SPECS = (
    ("bits_used_total", "Bits Used Total"),
    ("flops_used_total", "FLOPs Used Total"),
    ("plan_time_total_sec", "Plan Time (s)"),
    ("executed_steps", "Executed Steps"),
    ("plans", "Plans"),
)

PAIRED_METRIC_SPECS = (
    ("final_coverage_delta", "Final Coverage Delta"),
    ("bits_used_total_delta", "Bits Used Delta"),
    ("plan_time_total_sec_delta", "Plan Time Delta"),
)

STEPWISE_METRICS = (
    ("pos_diffs", "Position"),
    ("angle_diffs", "Angle"),
    ("eef_diffs", "EEF"),
    ("coverages", "Coverage"),
)

_FIGURE_COUNTER = itertools.count()
_COVERAGE_BINS = dict(start=0.0, end=1.0, size=0.02)
_PARTICLE_MEDIA_RENDER_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExperimentReviewData:
    run_dir: str
    meta: dict[str, Any]
    variant_rows: list[dict[str, Any]]
    run_rows: list[dict[str, Any]]
    paired_rows: list[dict[str, Any]]
    rows_by_key: dict[tuple[str, str], dict[str, Any]]
    rows_by_variant: dict[str, list[dict[str, Any]]]
    variant_by_name: dict[str, dict[str, Any]]

    @property
    def experiment_name(self) -> str:
        return str(self.meta.get("experiment_name", os.path.basename(self.run_dir)))

    @property
    def baseline_variant(self) -> str:
        return str(self.meta.get("baseline_variant", ""))

    @property
    def variant_order(self) -> list[str]:
        order = [str(name) for name in self.meta.get("variant_order", [])]
        if order:
            return order
        return [str(row["variant_name"]) for row in self.variant_rows]


@dataclass
class MediaRenderTask:
    variant_name: str
    rollout_id: str
    media_name: str
    status: str
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    updated_at: float = 0.0


def _require_plotly() -> None:
    if go is None or pio is None or get_plotlyjs is None or make_subplots is None:
        raise RuntimeError(
            "plotly is required for experiment_review. Install it in the hudm environment "
            "(for example: conda run -n hudm python -m pip install plotly)."
        )


def _coerce_value(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower == "nan":
        return float("nan")
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." not in text and "e" not in lower:
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _read_csv_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{key: _coerce_value(value) for key, value in row.items()} for row in reader]


def _safe_join(root: str, rel_path: str) -> str:
    candidate = os.path.abspath(os.path.join(root, rel_path))
    root_abs = os.path.abspath(root)
    if candidate != root_abs and not candidate.startswith(root_abs + os.sep):
        raise ValueError(f"Path escapes experiment root: {rel_path}")
    return candidate


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _safe_float(row.get(key))
        if np.isfinite(value):
            values.append(value)
    return values


def _success_bool(row: dict[str, Any]) -> bool:
    return bool(int(row.get("success", 0)))


def _sort_runs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("rollout_index", 10**9)),
            str(row.get("variant_name", "")),
            str(row.get("rollout_id", "")),
        ),
    )


def _default_reference_variant(data: ExperimentReviewData, *, current_variant: str | None = None) -> str:
    names = [name for name in data.variant_order if data.rows_by_variant.get(name)]
    if len(names) <= 0:
        return ""
    if current_variant is not None:
        for name in names:
            if name != str(current_variant):
                return name
    return names[0]


def _reference_comparison_payload(data: ExperimentReviewData, reference_variant: str) -> dict[str, Any]:
    reference_variant = str(reference_variant)
    reference_rows = {
        str(row.get("rollout_id", "")): row
        for row in data.rows_by_variant.get(reference_variant, [])
    }
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for variant_name in data.variant_order:
        variant_name = str(variant_name)
        if variant_name == reference_variant:
            continue
        rows = data.rows_by_variant.get(variant_name, [])
        if len(rows) <= 0:
            continue
        wins = 0
        losses = 0
        ties = 0
        coverage_better = 0
        coverage_worse = 0
        coverage_ties = 0
        matched = 0
        for row in rows:
            rollout_id = str(row.get("rollout_id", ""))
            reference_row = reference_rows.get(rollout_id)
            if reference_row is None:
                continue
            matched += 1
            success_delta = int(row.get("success", 0)) - int(reference_row.get("success", 0))
            if success_delta > 0:
                wins += 1
            elif success_delta < 0:
                losses += 1
            else:
                ties += 1
            coverage_delta = _safe_float(row.get("final_coverage")) - _safe_float(reference_row.get("final_coverage"))
            if np.isfinite(coverage_delta):
                if coverage_delta > 1e-9:
                    coverage_better += 1
                elif coverage_delta < -1e-9:
                    coverage_worse += 1
                else:
                    coverage_ties += 1
            paired_rows.append(
                {
                    "reference_variant": reference_variant,
                    "variant_name": variant_name,
                    "rollout_id": rollout_id,
                    "success_delta": success_delta,
                    "final_coverage_delta": coverage_delta,
                    "bits_used_total_delta": _safe_float(row.get("bits_used_total")) - _safe_float(reference_row.get("bits_used_total")),
                    "plan_time_total_sec_delta": _safe_float(row.get("plan_time_total_sec")) - _safe_float(reference_row.get("plan_time_total_sec")),
                }
            )
        if matched > 0:
            summary_rows.append(
                {
                    "variant_name": variant_name,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "coverage_better": coverage_better,
                    "coverage_worse": coverage_worse,
                    "coverage_ties": coverage_ties,
                    "matched_rollouts": matched,
                }
            )
    return {
        "reference_variant": reference_variant,
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
    }


def load_experiment_review_data(run_dir: str) -> ExperimentReviewData:
    paths = bundle_paths(run_dir)
    if not os.path.isfile(paths.experiment_json):
        raise FileNotFoundError(f"Experiment bundle metadata not found: {paths.experiment_json}")
    with open(paths.experiment_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    run_rows = _sort_runs(_read_csv_rows(paths.runs_csv))
    if len(run_rows) <= 0:
        raise FileNotFoundError(f"No run rows found under {paths.root}")
    variant_rows = _read_csv_rows(paths.variants_csv)
    paired_rows = _read_csv_rows(paths.paired_vs_baseline_csv)

    variant_order = [str(name) for name in meta.get("variant_order", [])]
    variant_rank = {name: idx for idx, name in enumerate(variant_order)}
    variant_rows = sorted(
        variant_rows,
        key=lambda row: (
            variant_rank.get(str(row.get("variant_name", "")), 10**6),
            str(row.get("variant_name", "")),
        ),
    )

    rows_by_key = {
        (str(row.get("variant_name", "")), str(row.get("rollout_id", ""))): row
        for row in run_rows
    }
    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        rows_by_variant[str(row.get("variant_name", ""))].append(row)
    variant_by_name = {str(row.get("variant_name", "")): row for row in variant_rows}

    return ExperimentReviewData(
        run_dir=paths.root,
        meta=meta,
        variant_rows=variant_rows,
        run_rows=run_rows,
        paired_rows=paired_rows,
        rows_by_key=rows_by_key,
        rows_by_variant=dict(rows_by_variant),
        variant_by_name=variant_by_name,
    )


def resolve_row(data: ExperimentReviewData, variant_name: str, rollout_id: str) -> dict[str, Any]:
    key = (str(variant_name), str(rollout_id))
    row = data.rows_by_key.get(key)
    if row is None:
        raise KeyError(f"Unknown run: {variant_name}/{rollout_id}")
    return row


def _read_log_tail(path: str, *, max_lines: int = 120) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]) if lines else ""


def _load_trace_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _trace_paths_for_variant(data: ExperimentReviewData, variant_name: str) -> list[str]:
    paths = []
    for row in data.rows_by_variant.get(str(variant_name), []):
        npz_path = os.path.join(trace_dir(data.run_dir, str(variant_name), str(row["rollout_id"])), "trace.npz")
        if os.path.isfile(npz_path):
            paths.append(npz_path)
    return paths


def _signature_for_paths(paths: Sequence[str], *, extra: Sequence[str] = ()) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"reviewer:{REVIEWER_SCHEMA_VERSION}".encode("utf-8"))
    for item in extra:
        hasher.update(str(item).encode("utf-8"))
    for path in sorted(set(paths)):
        hasher.update(path.encode("utf-8"))
        if os.path.isfile(path):
            st = os.stat(path)
            hasher.update(str(st.st_mtime_ns).encode("utf-8"))
            hasher.update(str(st.st_size).encode("utf-8"))
        else:
            hasher.update(b"missing")
    return hasher.hexdigest()


def _load_cached_json(path: str, *, signature: str) -> Any | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("signature") != signature:
        return None
    return payload.get("data")


def _write_cached_json(path: str, *, signature: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"signature": signature, "data": data}, f, indent=2)


def _stepwise_summary(arrays: list[np.ndarray], max_steps: int) -> dict[str, list[float]]:
    if len(arrays) <= 0 or max_steps <= 0:
        empty = [float("nan")] * max(0, max_steps)
        return {"median": empty, "q1": empty, "q3": empty}
    padded = []
    for arr in arrays:
        cur = np.asarray(arr, dtype=np.float32)
        if cur.size <= 0:
            cur = np.full((max_steps,), np.nan, dtype=np.float32)
        elif cur.shape[0] < max_steps:
            pad_value = cur[-1]
            cur = np.concatenate([cur, np.full((max_steps - cur.shape[0],), pad_value, dtype=np.float32)], axis=0)
        else:
            cur = cur[:max_steps]
        padded.append(cur)
    mat = np.stack(padded, axis=0)
    return {
        "median": np.nanmedian(mat, axis=0).astype(np.float32).tolist(),
        "q1": np.nanquantile(mat, 0.25, axis=0).astype(np.float32).tolist(),
        "q3": np.nanquantile(mat, 0.75, axis=0).astype(np.float32).tolist(),
    }


def _variant_stepwise_cache_path(data: ExperimentReviewData, variant_name: str) -> str:
    return os.path.join(review_derived_dir(data.run_dir), f"variant_{variant_name}_stepwise.json")


def _overview_stepwise_cache_path(data: ExperimentReviewData) -> str:
    return os.path.join(review_derived_dir(data.run_dir), "overview_stepwise.json")


def compute_variant_stepwise(data: ExperimentReviewData, variant_name: str) -> dict[str, Any]:
    variant_name = str(variant_name)
    trace_paths = _trace_paths_for_variant(data, variant_name)
    signature = _signature_for_paths(
        trace_paths + [bundle_paths(data.run_dir).runs_csv],
        extra=[variant_name, "variant_stepwise_v2"],
    )
    cache_path = _variant_stepwise_cache_path(data, variant_name)
    cached = _load_cached_json(cache_path, signature=signature)
    if cached is not None:
        return cached

    rows = data.rows_by_variant.get(variant_name, [])
    max_steps = max([int(row.get("executed_steps", 0)) for row in rows] + [0])
    success_rows = [row for row in rows if _success_bool(row)]
    failure_rows = [row for row in rows if not _success_bool(row)]
    row_groups = {
        "metrics": rows,
        "success_metrics": success_rows,
        "failure_metrics": failure_rows,
    }
    payload: dict[str, Any] = {"max_steps": int(max_steps), "metrics": {}, "success_metrics": {}, "failure_metrics": {}}
    for key, _label in STEPWISE_METRICS:
        for payload_key, group_rows in row_groups.items():
            arrays = []
            for row in group_rows:
                npz_path = os.path.join(trace_dir(data.run_dir, variant_name, str(row["rollout_id"])), "trace.npz")
                if not os.path.isfile(npz_path):
                    continue
                arrays.append(np.asarray(_load_trace_npz(npz_path).get(key, []), dtype=np.float32))
            payload[payload_key][key] = _stepwise_summary(arrays, max_steps=max_steps)
    _write_cached_json(cache_path, signature=signature, data=payload)
    return payload


def compute_overview_stepwise(data: ExperimentReviewData) -> dict[str, Any]:
    trace_paths = []
    for variant_name in data.variant_order:
        trace_paths.extend(_trace_paths_for_variant(data, variant_name))
    signature = _signature_for_paths(
        trace_paths + [bundle_paths(data.run_dir).runs_csv],
        extra=["overview_stepwise_v2"],
    )
    cache_path = _overview_stepwise_cache_path(data)
    cached = _load_cached_json(cache_path, signature=signature)
    if cached is not None:
        return cached

    payload = {"variants": {}}
    for variant_name in data.variant_order:
        payload["variants"][variant_name] = compute_variant_stepwise(data, variant_name)
    _write_cached_json(cache_path, signature=signature, data=payload)
    return payload


def _wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _beta_pdf_grid(successes: int, failures: int, *, points: int = 256) -> tuple[np.ndarray, np.ndarray, float, float]:
    alpha = float(successes + 1)
    beta = float(failures + 1)
    x = np.linspace(1e-4, 1.0 - 1e-4, points, dtype=np.float64)
    log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    log_pdf = log_norm + (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1.0 - x)
    pdf = np.exp(log_pdf)
    samples = np.random.default_rng(0).beta(alpha, beta, size=20000)
    lo, hi = np.quantile(samples, [0.025, 0.975]).astype(np.float64).tolist()
    return x.astype(np.float32), pdf.astype(np.float32), float(lo), float(hi)


def _figure_html(fig: Any, *, show_barmode_toggle: bool = True) -> str:
    _require_plotly()
    fig.update_layout(
        autosize=True,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend_title_text="",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig_id = f"plotly-figure-{next(_FIGURE_COUNTER)}"
    barmode_toggle_html = ""
    trace_types = {str(getattr(trace, "type", "")) for trace in fig.data}
    current_barmode = str(getattr(fig.layout, "barmode", "") or "")
    if show_barmode_toggle and current_barmode in {"stack", "group", "overlay"} and ("histogram" in trace_types or "bar" in trace_types):
        alt_mode = "overlay" if "histogram" in trace_types else "group"
        option_labels = {
            "stack": "Stacked",
            "overlay": "Overlay",
            "group": "Grouped",
        }
        barmode_toggle_html = (
            "<div class='plot-toolbar'>"
            f"<label for='{fig_id}-barmode'>Display</label>"
            f"<select id='{fig_id}-barmode' data-barmode-target='{fig_id}'>"
            f"<option value='stack'{' selected' if current_barmode == 'stack' else ''}>{option_labels['stack']}</option>"
            f"<option value='{alt_mode}'{' selected' if current_barmode == alt_mode else ''}>{option_labels[alt_mode]}</option>"
            "</select>"
            "</div>"
        )
    plot_html = pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        default_width="100%",
        default_height="100%",
        div_id=fig_id,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    return f"<div class='plot-shell'>{barmode_toggle_html}{plot_html}</div>"


def _plotly_head() -> str:
    _require_plotly()
    return f"<script>{get_plotlyjs()}</script>"


def _format_number(value: Any, *, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(fval):
        return "nan"
    return f"{fval:.{digits}f}"


def _format_compact_number(value: Any, *, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(fval):
        return "nan"
    abs_val = abs(fval)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs_val >= threshold:
            return f"{fval / threshold:.{digits}f}{suffix}"
    return _format_number(fval, digits=digits)


def _format_bits_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(fval):
        return "nan"
    sign = "-" if fval < 0 else ""
    return f"{sign}{format_bits_human(int(round(abs(fval))))}"


def _format_flops_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(fval):
        return "nan"
    sign = "-" if fval < 0 else ""
    return f"{sign}{format_flops_human(int(round(abs(fval))))}"


def _notice_html(message: str | None, *, kind: str = "info") -> str:
    if not message:
        return ""
    return f"<section class='card notice {html.escape(kind)}'>{html.escape(message)}</section>"


def _downloads_html(data: ExperimentReviewData) -> str:
    items = []
    for filename in CANONICAL_DOWNLOADS:
        path = os.path.join(data.run_dir, filename)
        if os.path.isfile(path):
            items.append(f"<li><a href='/files/{quote(filename)}'>{html.escape(filename)}</a></li>")
    return "".join(items)


def _kpi_cards(items: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f"<div class='card kpi'><div class='kpi-label'>{html.escape(label)}</div><div class='kpi-value'>{html.escape(value)}</div></div>"
        for label, value in items
    )


def _html_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str, str]],
    *,
    table_id: str,
) -> str:
    header = "".join(
        f"<th onclick=\"sortTable('{table_id}', {idx})\">{html.escape(label)}</th>"
        for idx, (label, _key, _kind) in enumerate(columns)
    )
    body_parts: list[str] = []
    for row in rows:
        cells = []
        for _label, key, kind in columns:
            raw_value = row.get(key, "")
            sort_value = row.get(f"{key}__sort", raw_value)
            if kind == "number":
                sort_value = _safe_float(sort_value)
                display_value = row.get(f"{key}__display")
                cell_value = str(display_value) if display_value is not None else _format_number(raw_value)
            else:
                cell_value = str(raw_value)
            cells.append(
                f"<td data-sort-value='{html.escape(str(sort_value))}'>{cell_value}</td>"
            )
        body_parts.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table id='{table_id}'><thead><tr>{header}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"


def _table_section(
    title: str,
    *,
    table_id: str,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str, str]],
    filter_id: str | None = None,
    filter_placeholder: str = "Filter...",
) -> str:
    toolbar = ""
    if filter_id is not None:
        toolbar = (
            "<div class='toolbar'>"
            f"<label for='{filter_id}'>Filter</label>"
            f"<input id='{filter_id}' data-filter-target='{table_id}' type='search' placeholder='{html.escape(filter_placeholder)}'>"
            "</div>"
        )
    return (
        "<section class='card'>"
        f"<h2>{html.escape(title)}</h2>"
        f"{toolbar}"
        f"<div class='table-wrap'>{_html_table(rows, columns, table_id=table_id)}</div>"
        "</section>"
    )


def _base_page(title: str, body_html: str, *, auto_refresh_seconds: int | None = None) -> str:
    refresh_meta = f"<meta http-equiv='refresh' content='{int(auto_refresh_seconds)}'>" if auto_refresh_seconds else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  {refresh_meta}
  {_plotly_head()}
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f7fb; color: #18212b; }}
    h1, h2, h3 {{ margin-bottom: 12px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 16px; min-width: 0; }}
    .hero {{ display: flex; align-items: baseline; justify-content: space-between; gap: 20px; flex-wrap: wrap; }}
    .muted {{ color: #52606d; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .kpi-label {{ color: #52606d; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-value {{ font-size: 24px; font-weight: 600; margin-top: 6px; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr)); gap: 16px; align-items: start; }}
    .plot-grid-wide {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .plot-grid > *, .plot-grid-wide > * {{ min-width: 0; }}
    .media-grid > * {{ min-width: 0; }}
    .plot-shell {{ width: 100%; min-width: 0; overflow-x: auto; overflow-y: hidden; }}
    .plot-toolbar {{ display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-bottom: 8px; }}
    .plot-toolbar label {{ color: #52606d; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .plot-shell > div {{ min-width: 0; }}
    .plot-shell .plotly-graph-div,
    .plot-shell .js-plotly-plot,
    .plot-shell .plot-container,
    .plot-shell .svg-container,
    .plot-shell .main-svg {{ width: 100% !important; max-width: 100% !important; }}
    .table-wrap {{ display: block; overflow-x: auto; overflow-y: auto; max-width: 100%; max-height: 60vh; }}
    .toolbar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
    .toolbar.controls-row {{ justify-content: space-between; align-items: end; }}
    .control-group {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; min-width: 0; }}
    input, select {{ padding: 8px 10px; border: 1px solid #ccd5df; border-radius: 8px; min-width: 0; width: min(100%, 320px); max-width: 100%; box-sizing: border-box; }}
    table {{ width: max-content; min-width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7ebf1; text-align: left; vertical-align: top; }}
    th {{ background: #f2f5fa; cursor: pointer; position: sticky; top: 0; }}
    th:hover {{ background: #e8eef7; }}
    a {{ color: #0f62fe; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    video {{ width: 100%; max-height: 420px; background: #111; border-radius: 8px; }}
    .button {{ display: inline-block; padding: 9px 12px; border-radius: 8px; background: #eef2f7; color: #18212b; text-decoration: none; margin-right: 8px; }}
    .button.primary {{ background: #0f62fe; color: white; }}
    .button.disabled {{ pointer-events: none; opacity: 0.65; }}
    .notice.info {{ border-left: 4px solid #0f62fe; }}
    .notice.error {{ border-left: 4px solid #da1e28; }}
    .notice.success {{ border-left: 4px solid #24a148; }}
    .nav-links {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .plot-switcher {{ margin-bottom: 12px; }}
    .hidden {{ display: none; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
    .status-pill {{ display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
    .status-running, .status-pending {{ background: #fff4d6; color: #8a3c00; }}
    .status-succeeded {{ background: #defbe6; color: #198038; }}
    .status-failed {{ background: #ffd7d9; color: #a2191f; }}
    .status-missing {{ background: #eef2f7; color: #52606d; }}
    .media-card video {{ width: 100%; aspect-ratio: 1 / 1; max-height: none; object-fit: contain; background: #111; }}
    .info-chip {{ position: relative; display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; border-radius: 999px; background: #eef2f7; color: #52606d; font-size: 12px; font-weight: 700; cursor: help; margin-left: 6px; vertical-align: middle; }}
    .info-chip::after {{ content: attr(data-tooltip); position: absolute; left: 50%; bottom: calc(100% + 8px); transform: translateX(-50%); width: min(280px, 70vw); padding: 8px 10px; border-radius: 8px; background: #18212b; color: white; font-size: 12px; font-weight: 500; line-height: 1.4; text-transform: none; letter-spacing: 0; white-space: normal; box-shadow: 0 8px 24px rgba(0,0,0,0.18); opacity: 0; pointer-events: none; transition: opacity 120ms ease; z-index: 20; }}
    .info-chip::before {{ content: ""; position: absolute; left: 50%; bottom: calc(100% + 2px); transform: translateX(-50%); border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 6px solid #18212b; opacity: 0; pointer-events: none; transition: opacity 120ms ease; z-index: 21; }}
    .info-chip:hover::after, .info-chip:hover::before, .info-chip:focus-visible::after, .info-chip:focus-visible::before {{ opacity: 1; }}
    @media (min-width: 1600px) {{
      .media-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    }}
    @media (max-width: 960px) {{
      .media-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      body {{ margin: 12px; }}
      .card {{ padding: 14px; }}
      .toolbar {{ align-items: stretch; }}
      input, select {{ width: 100%; }}
      .plot-shell {{ overflow-x: hidden; }}
    }}
  </style>
  <script>
    let mediaPollTimer = null;

    function resizePlotlyFigures() {{
      if (!window.Plotly || !window.Plotly.Plots) return;
      for (const plot of document.querySelectorAll('.js-plotly-plot')) {{
        try {{
          window.Plotly.Plots.resize(plot);
        }} catch (_error) {{
          // Ignore stale or partially initialized plots during page bootstrap.
        }}
      }}
    }}

    window.addEventListener('load', () => {{
      requestAnimationFrame(() => requestAnimationFrame(resizePlotlyFigures));
      setTimeout(resizePlotlyFigures, 150);
    }});
    window.addEventListener('resize', () => requestAnimationFrame(resizePlotlyFigures));

    function sortTable(tableId, columnIdx) {{
      const table = document.getElementById(tableId);
      if (!table) return;
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const current = table.getAttribute('data-sort-col');
      const desc = current === String(columnIdx) && table.getAttribute('data-sort-dir') !== 'desc';
      rows.sort((a, b) => {{
        const av = a.cells[columnIdx].dataset.sortValue || a.cells[columnIdx].textContent;
        const bv = b.cells[columnIdx].dataset.sortValue || b.cells[columnIdx].textContent;
        const an = Number(av);
        const bn = Number(bv);
        let cmp = 0;
        if (!Number.isNaN(an) && !Number.isNaN(bn)) {{
          cmp = an - bn;
        }} else {{
          cmp = String(av).localeCompare(String(bv));
        }}
        return desc ? -cmp : cmp;
      }});
      for (const row of rows) tbody.appendChild(row);
      table.setAttribute('data-sort-col', String(columnIdx));
      table.setAttribute('data-sort-dir', desc ? 'desc' : 'asc');
    }}
    function bindFilters() {{
      for (const input of document.querySelectorAll('input[data-filter-target]')) {{
        input.addEventListener('input', () => {{
          const table = document.getElementById(input.dataset.filterTarget);
          if (!table) return;
          const q = input.value.trim().toLowerCase();
          for (const row of table.tBodies[0].rows) {{
            row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
          }}
        }});
      }}
    }}
    function setPlotBarMode(plotId, mode) {{
      const plot = document.getElementById(plotId);
      if (!plot || !window.Plotly) return;
      window.Plotly.relayout(plot, {{ barmode: mode }});
      requestAnimationFrame(() => requestAnimationFrame(() => {{
        try {{
          window.Plotly.Plots.resize(plot);
        }} catch (_error) {{
          // Ignore stale plots during relayout.
        }}
      }}));
    }}
    function setPlotGroupBarMode(groupId, mode) {{
      for (const node of document.querySelectorAll(`[data-plot-group="${{groupId}}"]`)) {{
        if (node.style.display === 'none') continue;
        const plot = node.querySelector('.js-plotly-plot');
        if (plot && plot.id) {{
          setPlotBarMode(plot.id, mode);
        }}
      }}
    }}
    function bindPlotModeToggles() {{
      for (const select of document.querySelectorAll('select[data-barmode-target]')) {{
        select.addEventListener('change', () => setPlotBarMode(select.dataset.barmodeTarget, select.value));
      }}
      for (const select of document.querySelectorAll('select[data-barmode-group]')) {{
        select.addEventListener('change', () => setPlotGroupBarMode(select.dataset.barmodeGroup, select.value));
        setPlotGroupBarMode(select.dataset.barmodeGroup, select.value);
      }}
    }}
    function showPlotGroup(groupId, value) {{
      for (const node of document.querySelectorAll(`[data-plot-group="${{groupId}}"]`)) {{
        node.style.display = node.dataset.plotValue === value ? '' : 'none';
      }}
      const control = document.querySelector(`select[data-barmode-group="${{groupId}}"]`);
      if (control) {{
        setPlotGroupBarMode(groupId, control.value);
      }}
      requestAnimationFrame(() => requestAnimationFrame(resizePlotlyFigures));
    }}
    function stopMediaPolling() {{
      if (mediaPollTimer) {{
        clearTimeout(mediaPollTimer);
        mediaPollTimer = null;
      }}
    }}
    function replaceMediaSection(html) {{
      const current = document.getElementById('mediaSection');
      if (!current) return;
      const wrapper = document.createElement('div');
      wrapper.innerHTML = html;
      const next = wrapper.firstElementChild;
      if (!next) return;
      current.replaceWith(next);
      bindAsyncRenderControls();
    }}
    async function refreshMediaSection() {{
      const section = document.getElementById('mediaSection');
      if (!section) return;
      const variant = section.dataset.variant || '';
      const rolloutId = section.dataset.rolloutId || '';
      const url = `/run_media?variant=${{encodeURIComponent(variant)}}&rollout_id=${{encodeURIComponent(rolloutId)}}`;
      try {{
        const resp = await fetch(url, {{ headers: {{ 'Accept': 'application/json' }} }});
        if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
        const payload = await resp.json();
        if (payload.media_html) {{
          replaceMediaSection(payload.media_html);
        }}
        if (payload.active_media) {{
          mediaPollTimer = setTimeout(refreshMediaSection, 3000);
        }}
      }} catch (_error) {{
        mediaPollTimer = setTimeout(refreshMediaSection, 3000);
      }}
    }}
    async function handleAsyncRender(link) {{
      const href = link.getAttribute('href');
      if (!href) return;
      const url = new URL(href, window.location.origin);
      url.searchParams.set('format', 'json');
      link.setAttribute('aria-disabled', 'true');
      link.classList.add('disabled');
      stopMediaPolling();
        try {{
          const resp = await fetch(url.toString(), {{ headers: {{ 'Accept': 'application/json' }} }});
          if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
          const payload = await resp.json();
          const notice = document.getElementById('pageNotice');
          if (notice) {{
            notice.innerHTML = payload.notice_html || '';
          }}
        if (payload.media_html) {{
          replaceMediaSection(payload.media_html);
        }}
        if (payload.active_media) {{
          mediaPollTimer = setTimeout(refreshMediaSection, 3000);
        }}
      }} catch (_error) {{
        window.location.href = href;
      }} finally {{
        link.removeAttribute('aria-disabled');
        link.classList.remove('disabled');
      }}
    }}
    function bindAsyncRenderControls() {{
      for (const link of document.querySelectorAll('a[data-async-render]')) {{
        if (link.dataset.asyncBound === 'true') continue;
        link.dataset.asyncBound = 'true';
        link.addEventListener('click', (event) => {{
          event.preventDefault();
          handleAsyncRender(link);
        }});
      }}
    }}
    window.addEventListener('DOMContentLoaded', () => {{
      bindFilters();
      bindPlotModeToggles();
      bindAsyncRenderControls();
      const mediaSection = document.getElementById('mediaSection');
      if (mediaSection && mediaSection.dataset.activeMedia === 'true') {{
        mediaPollTimer = setTimeout(refreshMediaSection, 3000);
      }}
    }});
  </script>
</head>
<body>
{body_html}
</body>
</html>
"""


def _metric_distribution_figure(
    data: ExperimentReviewData,
    *,
    metric_specs: Sequence[tuple[str, str]],
    title: str,
) -> Any:
    _require_plotly()
    fig = make_subplots(rows=1, cols=len(metric_specs), subplot_titles=[label for _, label in metric_specs])
    variant_names = data.variant_order
    for col_idx, (metric_key, label) in enumerate(metric_specs, start=1):
        for variant_name in variant_names:
            values = _finite_values(data.rows_by_variant.get(variant_name, []), metric_key)
            if len(values) <= 0:
                continue
            fig.add_trace(
                go.Box(
                    y=values,
                    name=variant_name,
                    boxmean=True,
                    legendgroup=variant_name,
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_yaxes(title_text=label, row=1, col=col_idx)
    fig.update_layout(title=title)
    return fig


def _success_rate_figure(data: ExperimentReviewData) -> Any:
    _require_plotly()
    variant_names = []
    rates = []
    err_plus = []
    err_minus = []
    for row in data.variant_rows:
        variant_name = str(row["variant_name"])
        n = int(row.get("n_rollouts", 0))
        p = float(row.get("success_rate", 0.0))
        successes = int(round(p * n))
        lo, hi = _wilson_interval(successes, n)
        variant_names.append(variant_name)
        rates.append(p)
        err_plus.append(max(0.0, hi - p))
        err_minus.append(max(0.0, p - lo))
    fig = go.Figure(
        data=[
            go.Bar(
                x=variant_names,
                y=rates,
                text=[f"{rate:.1%}" for rate in rates],
                textposition="outside",
                error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
            )
        ]
    )
    fig.update_layout(
        title="Success Rate by Variant",
        yaxis_title="Success Rate",
        bargap=0.7 if len(variant_names) <= 3 else 0.25,
    )
    return fig


def _compute_vs_outcome_figure(data: ExperimentReviewData) -> Any:
    _require_plotly()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Bits vs Success",
            "Plan Time vs Success",
            "Bits vs Coverage",
            "Plan Time vs Coverage",
        ],
    )
    bits_x = [float(row.get("mean_bits_used_total", float("nan"))) for row in data.variant_rows]
    time_x = [float(row.get("mean_plan_time_total_sec", float("nan"))) for row in data.variant_rows]
    success_y = [float(row.get("success_rate", float("nan"))) for row in data.variant_rows]
    coverage_y = [float(row.get("mean_final_coverage", float("nan"))) for row in data.variant_rows]
    labels = [str(row.get("variant_name", "")) for row in data.variant_rows]
    fig.add_trace(
        go.Scatter(
            x=bits_x,
            y=success_y,
            mode="markers+text",
            text=labels,
            customdata=[[_format_bits_value(value)] for value in bits_x],
            textposition="top center",
            hovertemplate="Variant %{text}<br>Mean Bits Used Total: %{customdata[0]}<br>Success Rate: %{y:.1%}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_x,
            y=success_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertemplate="Variant %{text}<br>Mean Plan Time Total: %{x:.4g}s<br>Success Rate: %{y:.1%}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=bits_x,
            y=coverage_y,
            mode="markers+text",
            text=labels,
            customdata=[[_format_bits_value(value)] for value in bits_x],
            textposition="top center",
            hovertemplate="Variant %{text}<br>Mean Bits Used Total: %{customdata[0]}<br>Mean Coverage: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_x,
            y=coverage_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertemplate="Variant %{text}<br>Mean Plan Time Total: %{x:.4g}s<br>Mean Coverage: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="Mean Bits Used Total", row=1, col=1)
    fig.update_xaxes(title_text="Mean Plan Time Total (s)", row=1, col=2)
    fig.update_xaxes(title_text="Mean Bits Used Total", row=2, col=1)
    fig.update_xaxes(title_text="Mean Plan Time Total (s)", row=2, col=2)
    fig.update_yaxes(title_text="Success Rate", row=1, col=1)
    fig.update_yaxes(title_text="Success Rate", row=1, col=2)
    fig.update_yaxes(title_text="Mean Coverage", row=2, col=1)
    fig.update_yaxes(title_text="Mean Coverage", row=2, col=2)
    fig.update_layout(title="Compute vs Outcome")
    return fig


def _paired_summary_figure(data: ExperimentReviewData, reference_variant: str) -> Any | None:
    _require_plotly()
    payload = _reference_comparison_payload(data, reference_variant)
    rows = payload["summary_rows"]
    if len(rows) <= 0:
        return None
    names = [str(row["variant_name"]) for row in rows]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"Success vs {reference_variant}", f"Coverage vs {reference_variant}"],
    )
    fig.add_trace(
        go.Bar(name="wins", x=names, y=[int(row.get("wins", 0)) for row in rows], hovertemplate="Variant %{x}<br>Wins: %{y}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="losses", x=names, y=[int(row.get("losses", 0)) for row in rows], hovertemplate="Variant %{x}<br>Losses: %{y}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="ties", x=names, y=[int(row.get("ties", 0)) for row in rows], hovertemplate="Variant %{x}<br>Ties: %{y}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="coverage better",
            x=names,
            y=[int(row.get("coverage_better", 0)) for row in rows],
            hovertemplate="Variant %{x}<br>Coverage Better: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="coverage worse",
            x=names,
            y=[int(row.get("coverage_worse", 0)) for row in rows],
            hovertemplate="Variant %{x}<br>Coverage Worse: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="coverage ties",
            x=names,
            y=[int(row.get("coverage_ties", 0)) for row in rows],
            hovertemplate="Variant %{x}<br>Coverage Ties: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title=f"Reference Comparison vs {reference_variant}", barmode="group")
    return fig


def _paired_delta_figure(data: ExperimentReviewData, reference_variant: str) -> Any | None:
    _require_plotly()
    payload = _reference_comparison_payload(data, reference_variant)
    paired_rows = payload["paired_rows"]
    if len(paired_rows) <= 0:
        return None
    fig = make_subplots(rows=1, cols=len(PAIRED_METRIC_SPECS), subplot_titles=[label for _, label in PAIRED_METRIC_SPECS])
    variants = [name for name in data.variant_order if name != str(reference_variant)]
    for col_idx, (metric_key, label) in enumerate(PAIRED_METRIC_SPECS, start=1):
        for variant_name in variants:
            values = _finite_values([row for row in paired_rows if str(row.get("variant_name")) == variant_name], metric_key)
            if len(values) <= 0:
                continue
            fig.add_trace(
                go.Box(y=values, name=variant_name, boxmean=True, legendgroup=variant_name, showlegend=(col_idx == 1)),
                row=1,
                col=col_idx,
            )
        fig.update_yaxes(title_text=label, row=1, col=col_idx)
    fig.update_layout(title=f"Paired Delta Distributions vs {reference_variant}")
    return fig


def _overview_stepwise_figures(data: ExperimentReviewData) -> dict[str, str]:
    _require_plotly()
    payload = compute_overview_stepwise(data)
    out: dict[str, str] = {}
    for metric_key, metric_label in STEPWISE_METRICS:
        fig = go.Figure()
        for variant_name in data.variant_order:
            variant_payload = payload["variants"].get(variant_name, {})
            metric = variant_payload.get("metrics", {}).get(metric_key)
            if metric is None:
                continue
            x = list(range(len(metric["median"])))
            y = metric["median"]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=variant_name))
        fig.update_layout(title=f"Median {metric_label} Over Step", xaxis_title="Step", yaxis_title=metric_label)
        out[metric_key] = _figure_html(fig)
    return out


def _variant_success_figure(rows: Sequence[dict[str, Any]], variant_name: str) -> Any:
    _require_plotly()
    rows = _sort_runs(rows)
    n = len(rows)
    successes = sum(1 for row in rows if _success_bool(row))
    failures = n - successes
    termination_counts = Counter(str(row.get("termination_reason", "unknown")) for row in rows)
    success_rows = [row for row in rows if _success_bool(row)]
    failure_rows = [row for row in rows if not _success_bool(row)]
    success_coverages = _finite_values(success_rows, "final_coverage")
    failure_coverages = _finite_values(failure_rows, "final_coverage")
    x_pdf, y_pdf, ci_lo, ci_hi = _beta_pdf_grid(successes, failures)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Success / Failure",
            "Termination Reasons",
            "Final Coverage Distribution",
            "Success-Rate Posterior",
        ],
    )
    fig.add_trace(
        go.Bar(
            x=["success", "failure"],
            y=[successes, failures],
            marker_color=["#24a148", "#da1e28"],
            hovertemplate="Outcome: %{x}<br>Runs: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(termination_counts.keys()),
            y=list(termination_counts.values()),
            hovertemplate="Termination: %{x}<br>Runs: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    if len(success_coverages) > 0:
        fig.add_trace(
            go.Histogram(
                x=success_coverages,
                name="success",
                marker_color="#24a148",
                xbins=_COVERAGE_BINS,
                hovertemplate="Final Coverage: %{x:.4f}<br>Runs: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    if len(failure_coverages) > 0:
        fig.add_trace(
            go.Histogram(
                x=failure_coverages,
                name="failure",
                marker_color="#da1e28",
                xbins=_COVERAGE_BINS,
                hovertemplate="Final Coverage: %{x:.4f}<br>Runs: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=x_pdf.tolist(),
            y=y_pdf.tolist(),
            mode="lines",
            fill="tozeroy",
            hovertemplate="Success Probability: %{x:.1%}<br>Posterior Density: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_vline(x=ci_lo, line_dash="dash", line_color="#0f62fe", row=2, col=2)
    fig.add_vline(x=ci_hi, line_dash="dash", line_color="#0f62fe", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)
    fig.update_xaxes(title_text="Final Coverage", row=2, col=1)
    fig.update_xaxes(title_text="Success Probability", row=2, col=2)
    fig.update_layout(title=f"{variant_name}: Success Analysis", barmode="stack")
    return fig


def _variant_histogram_figure(
    rows: Sequence[dict[str, Any]],
    *,
    metric_specs: Sequence[tuple[str, str]],
    title: str,
) -> Any:
    _require_plotly()
    cols = 2
    rows_n = int(math.ceil(len(metric_specs) / cols))
    fig = make_subplots(
        rows=rows_n,
        cols=cols,
        subplot_titles=[label for _, label in metric_specs],
        vertical_spacing=0.16 if rows_n > 1 else 0.1,
    )
    success_rows = [row for row in rows if _success_bool(row)]
    fail_rows = [row for row in rows if not _success_bool(row)]
    for idx, (metric_key, label) in enumerate(metric_specs):
        row_idx = idx // cols + 1
        col_idx = idx % cols + 1
        success_vals = _finite_values(success_rows, metric_key)
        fail_vals = _finite_values(fail_rows, metric_key)
        if len(success_vals) > 0:
            fig.add_trace(
                go.Histogram(
                    x=success_vals,
                    name="success",
                    legendgroup="success",
                    marker_color="#24a148",
                    xbins=_COVERAGE_BINS if metric_key == "final_coverage" else None,
                    opacity=0.75,
                    showlegend=(idx == 0),
                ),
                row=row_idx,
                col=col_idx,
            )
        if len(fail_vals) > 0:
            fig.add_trace(
                go.Histogram(
                    x=fail_vals,
                    name="failure",
                    legendgroup="failure",
                    marker_color="#da1e28",
                    xbins=_COVERAGE_BINS if metric_key == "final_coverage" else None,
                    opacity=0.6,
                    showlegend=(idx == 0),
                ),
                row=row_idx,
                col=col_idx,
            )
        fig.update_xaxes(title_text=label, row=row_idx, col=col_idx)
    fig.update_layout(
        title=title,
        barmode="stack",
        height=280 * rows_n + 160,
        margin=dict(t=96, b=72, l=60, r=32),
    )
    return fig


def _variant_relationship_figure(rows: Sequence[dict[str, Any]], variant_name: str) -> Any:
    _require_plotly()
    grouped_rows = [
        ("success", [row for row in rows if _success_bool(row)], "#24a148"),
        ("failure", [row for row in rows if not _success_bool(row)], "#da1e28"),
    ]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Bits vs Coverage",
            "Plan Time vs Coverage",
            "Bits vs Plan Time",
        ],
    )
    for outcome_name, outcome_rows, color in grouped_rows:
        labels = [str(row.get("rollout_id", "")) for row in outcome_rows]
        if outcome_rows:
            fig.add_trace(
                go.Scatter(
                    x=[_safe_float(row.get("bits_used_total")) for row in outcome_rows],
                    y=[_safe_float(row.get("final_coverage")) for row in outcome_rows],
                    mode="markers",
                    name=outcome_name,
                    legendgroup=outcome_name,
                    marker=dict(color=color),
                    text=labels,
                    customdata=[[_format_bits_value(row.get("bits_used_total"))] for row in outcome_rows],
                    hovertemplate="Rollout %{text}<br>Bits Used Total: %{customdata[0]}<br>Final Coverage: %{y:.4f}<extra></extra>",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[_safe_float(row.get("plan_time_total_sec")) for row in outcome_rows],
                    y=[_safe_float(row.get("final_coverage")) for row in outcome_rows],
                    mode="markers",
                    name=outcome_name,
                    legendgroup=outcome_name,
                    marker=dict(color=color),
                    text=labels,
                    hovertemplate="Rollout %{text}<br>Plan Time Total: %{x:.4f}s<br>Final Coverage: %{y:.4f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=[_safe_float(row.get("bits_used_total")) for row in outcome_rows],
                    y=[_safe_float(row.get("plan_time_total_sec")) for row in outcome_rows],
                    mode="markers",
                    name=outcome_name,
                    legendgroup=outcome_name,
                    marker=dict(color=color),
                    text=labels,
                    customdata=[[_format_bits_value(row.get("bits_used_total"))] for row in outcome_rows],
                    hovertemplate="Rollout %{text}<br>Bits Used Total: %{customdata[0]}<br>Plan Time Total: %{y:.4f}s<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
    fig.update_xaxes(title_text="Bits Used Total", row=1, col=1)
    fig.update_yaxes(title_text="Final Coverage", row=1, col=1)
    fig.update_xaxes(title_text="Plan Time Total (s)", row=1, col=2)
    fig.update_yaxes(title_text="Final Coverage", row=1, col=2)
    fig.update_xaxes(title_text="Bits Used Total", row=1, col=3)
    fig.update_yaxes(title_text="Plan Time Total (s)", row=1, col=3)
    fig.update_layout(title=f"{variant_name}: Relationship Plots")
    return fig


def _variant_stepwise_figure(data: ExperimentReviewData, variant_name: str) -> Any:
    _require_plotly()
    payload = compute_variant_stepwise(data, variant_name)
    fig = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label in STEPWISE_METRICS])
    groups = [
        ("success_metrics", "success", "#24a148", "rgba(36,161,72,0.15)"),
        ("failure_metrics", "failure", "#da1e28", "rgba(218,30,40,0.12)"),
    ]
    for idx, (metric_key, label) in enumerate(STEPWISE_METRICS):
        row_idx = idx // 2 + 1
        col_idx = idx % 2 + 1
        for payload_key, legend_name, line_color, fill_color in groups:
            metric = payload.get(payload_key, {}).get(metric_key, {})
            median = metric.get("median", [])
            q1 = metric.get("q1", [])
            q3 = metric.get("q3", [])
            if len(median) <= 0 or all(not np.isfinite(value) for value in median):
                continue
            x = list(range(len(median)))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=q3,
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=legend_name,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=q1,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    legendgroup=legend_name,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=median,
                    mode="lines",
                    line=dict(color=line_color, width=3),
                    name=legend_name,
                    legendgroup=legend_name,
                    hovertemplate=f"Step %{{x}}<br>{label}: %{{y:.4f}}<extra></extra>",
                    showlegend=(idx == 0),
                ),
                row=row_idx,
                col=col_idx,
            )
        fig.update_xaxes(title_text="Step", row=row_idx, col=col_idx)
        fig.update_yaxes(title_text=label, row=row_idx, col=col_idx)
    fig.update_layout(title=f"{variant_name}: Stepwise Trace Summary")
    return fig


def _variant_paired_summary_figure(data: ExperimentReviewData, variant_name: str, reference_variant: str) -> Any | None:
    _require_plotly()
    variant_name = str(variant_name)
    reference_variant = str(reference_variant)
    if variant_name == reference_variant:
        return None
    payload = _reference_comparison_payload(data, reference_variant)
    summary_row = next((row for row in payload["summary_rows"] if str(row.get("variant_name")) == variant_name), None)
    if summary_row is None:
        return None
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"Success vs {reference_variant}", f"Coverage vs {reference_variant}"],
    )
    fig.add_trace(
        go.Bar(
            x=["wins", "losses", "ties"],
            y=[
                int(summary_row.get("wins", 0)),
                int(summary_row.get("losses", 0)),
                int(summary_row.get("ties", 0)),
            ],
            hovertemplate="Outcome: %{x}<br>Count: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["better", "worse", "ties"],
            y=[
                int(summary_row.get("coverage_better", 0)),
                int(summary_row.get("coverage_worse", 0)),
                int(summary_row.get("coverage_ties", 0)),
            ],
            hovertemplate="Coverage: %{x}<br>Count: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(title=f"{variant_name}: Reference Comparison vs {reference_variant}")
    return fig


def _variant_paired_delta_figure(data: ExperimentReviewData, variant_name: str, reference_variant: str) -> Any | None:
    _require_plotly()
    variant_name = str(variant_name)
    reference_variant = str(reference_variant)
    if variant_name == reference_variant:
        return None
    payload = _reference_comparison_payload(data, reference_variant)
    paired_rows = [row for row in payload["paired_rows"] if str(row.get("variant_name", "")) == variant_name]
    if len(paired_rows) <= 0:
        return None
    fig = make_subplots(rows=1, cols=len(PAIRED_METRIC_SPECS), subplot_titles=[label for _, label in PAIRED_METRIC_SPECS])
    for col_idx, (metric_key, label) in enumerate(PAIRED_METRIC_SPECS, start=1):
        values = _finite_values(paired_rows, metric_key)
        if len(values) > 0:
            fig.add_trace(go.Box(y=values, boxmean=True, name=label, showlegend=False), row=1, col=col_idx)
        fig.update_yaxes(title_text=label, row=1, col=col_idx)
    fig.update_layout(title=f"{variant_name}: Paired Delta Distributions vs {reference_variant}")
    return fig


def _single_run_trace_figure(
    trace_arrays: dict[str, np.ndarray],
    variant_name: str,
    rollout_id: str,
    *,
    trace_meta: dict[str, Any] | None = None,
) -> Any:
    _require_plotly()
    fig = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label in STEPWISE_METRICS])
    max_steps = 0
    replans = trace_meta.get("replans", []) if trace_meta is not None else []
    for idx, (metric_key, label) in enumerate(STEPWISE_METRICS):
        row_idx = idx // 2 + 1
        col_idx = idx % 2 + 1
        values = np.asarray(trace_arrays.get(metric_key, []), dtype=np.float32)
        steps = int(values.shape[0])
        max_steps = max(max_steps, steps)
        x_values = list(range(1, steps + 1))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=values.tolist(),
                mode="lines+markers" if steps > 1 else "markers",
                line=dict(color="#0f62fe", width=3),
                marker=dict(size=8, color="#0f62fe"),
                hovertemplate=f"Step %{{x}}<br>{label}: %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Step", row=row_idx, col=col_idx)
        fig.update_yaxes(title_text=label, row=row_idx, col=col_idx)
        if steps <= 1:
            fig.update_xaxes(range=[0.85, 1.15], dtick=1, row=row_idx, col=col_idx)
        else:
            fig.update_xaxes(dtick=1, row=row_idx, col=col_idx)
        marker_x: list[int] = []
        marker_y: list[float] = []
        marker_meta: list[list[Any]] = []
        for replan in replans:
            step_start = int(replan.get("step_start", -1))
            if step_start < 0 or steps <= 0 or step_start >= steps:
                continue
            x_step = step_start + 1
            marker_x.append(x_step)
            marker_y.append(float(values[min(step_start, steps - 1)]))
            marker_meta.append(
                [
                    int(replan.get("replan_idx", -1)),
                    _replan_display_level(trace_meta, replan),
                    _format_bits_value(replan.get("bits_used_estimate")),
                    _format_number(replan.get("plan_time_sec")),
                ]
            )
            fig.add_vline(
                x=x_step,
                row=row_idx,
                col=col_idx,
                line_dash="dot",
                line_color="#8a3ffc",
                opacity=0.45,
            )
        if len(marker_x) > 0:
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers",
                    marker=dict(size=11, symbol="diamond", color="#8a3ffc", line=dict(width=1, color="#ffffff")),
                    customdata=marker_meta,
                    name="Replan",
                    showlegend=(idx == 0),
                    legendgroup="replan",
                    hovertemplate=(
                        "Replan %{customdata[0]}<br>"
                        "Step %{x}<br>"
                        f"{label}: %{{y:.4f}}<br>"
                        "Base level %{customdata[1]}<br>"
                        "Bits %{customdata[2]}<br>"
                        "Plan time %{customdata[3]} s<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )
    if max_steps <= 1:
        fig.add_annotation(
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            text="Only 1 recorded step is available in trace.npz for this rollout.",
            showarrow=False,
            font=dict(color="#525252", size=12),
        )
    fig.update_layout(
        title=f"{variant_name} / {rollout_id}: Single-Run Trace Curves ({max_steps} recorded step{'s' if max_steps != 1 else ''})",
        height=760,
        margin=dict(t=96, b=56, l=60, r=32),
    )
    return fig


def _replan_rows(trace_meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for replan in trace_meta.get("replans", []):
        bits_value = int(replan.get("bits_used_estimate", 0))
        action_horizon = replan.get("action_horizon", None)
        if action_horizon is None:
            action_horizon = len(replan.get("action_seq", []))
        if int(action_horizon) <= 0:
            action_horizon = len(replan.get("rollout_level_indices", []))
        display_level = _replan_display_level(trace_meta, replan)
        rows.append(
            {
                "replan_idx": int(replan.get("replan_idx", -1)),
                "step_start": int(replan.get("step_start", -1)),
                "mpc_progress": _format_number(replan.get("mpc_progress")),
                "base_level_idx": display_level,
                "bits_used_estimate": bits_value,
                "bits_used_estimate__display": _format_bits_value(bits_value),
                "bits_used_estimate__sort": bits_value,
                "plan_time_sec": _format_number(replan.get("plan_time_sec")),
                "action_horizon": int(action_horizon),
            }
        )
    return rows


def _replan_display_level(trace_meta: dict[str, Any], replan: dict[str, Any]) -> int:
    if "start_level_idx" in replan:
        return int(replan.get("start_level_idx", -1))

    plan_cfg = trace_meta.get("plan_config", {})
    fidelity_cfg = plan_cfg.get("fidelity", {}) if isinstance(plan_cfg, dict) else {}
    cem_cfg = plan_cfg.get("cem", {}) if isinstance(plan_cfg, dict) else {}
    mpc_cfg = plan_cfg.get("mpc", {}) if isinstance(plan_cfg, dict) else {}
    num_levels = fidelity_cfg.get("num_levels", None)
    if num_levels is None:
        return int(replan.get("base_level_idx", -1))

    horizon = replan.get("action_horizon", None)
    if horizon is None:
        horizon = len(replan.get("action_seq", [])) or int(mpc_cfg.get("horizon", 1))
    action_seq = replan.get("action_seq", [])
    action_dim = len(action_seq[0]) if len(action_seq) > 0 else 2

    try:
        core = SharedCEMCore(
            horizon=max(1, int(horizon)),
            action_dim=max(1, int(action_dim)),
            pop_size=max(1, int(cem_cfg.get("pop_size", 1))),
            elite_frac=float(cem_cfg.get("elite_frac", 0.5)),
            n_iter=max(1, int(cem_cfg.get("n_iter", 1))),
            init_std=max(1e-6, float(cem_cfg.get("init_std", 1.0))),
            action_low=cem_cfg.get("action_low", None),
            action_high=cem_cfg.get("action_high", None),
            fidelity_cfg=fidelity_cfg,
            num_levels=int(num_levels),
            rollout_modes={"fixed", "linear", "uncertainty_downshift"},
        )
        return int(core.base_level_index(float(replan.get("mpc_progress", 0.0)), 0.0))
    except Exception:
        return int(replan.get("base_level_idx", -1))


def _media_path_for_name(media_dir: str, media_name: str) -> str | None:
    for filename in planning_media.MEDIA_ALIASES.get(media_name, []):
        path = os.path.join(media_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def _trace_backend_for_run(experiment_root: str, *, variant_name: str, rollout_id: str) -> str:
    trace_json_path = os.path.join(trace_dir(experiment_root, variant_name, rollout_id), "trace.json")
    if not os.path.isfile(trace_json_path):
        return ""
    with open(trace_json_path, "r", encoding="utf-8") as f:
        trace_meta = json.load(f)
    plan_cfg = trace_meta.get("plan_config", {})
    backend = plan_cfg.get("backend") or trace_meta.get("backend") or ""
    return str(backend).strip().lower()


def render_media_for_run(
    experiment_root: str,
    *,
    variant_name: str,
    rollout_id: str,
    media: list[str],
) -> tuple[list[str], list[str]]:
    outputs: list[str] = []
    errors: list[str] = []
    media_dir = review_media_dir(experiment_root, variant_name, rollout_id)
    ensure_dir(media_dir)
    for media_name in media:
        try:
            outputs.extend(
                planning_media.render_media(
                    experiment_root,
                    schedule=variant_name,
                    rollout_id=rollout_id,
                    media=[media_name],
                    output_dir=media_dir,
                )
            )
        except Exception as exc:
            errors.append(f"{media_name}: {type(exc).__name__}: {exc}")
    return outputs, errors


def _status_badge_html(status: str) -> str:
    if status == "succeeded":
        return ""
    labels = {
        "pending": "Queued",
        "running": "Rendering",
        "failed": "Failed",
        "missing": "Not Generated",
    }
    safe_status = status if status in labels else "missing"
    return f"<span class='status-pill status-{safe_status}'>{labels[safe_status]}</span>"


def _media_backend_label(trace_meta: dict[str, Any] | None) -> str:
    if trace_meta is None:
        return "planner backend"
    plan_cfg = trace_meta.get("plan_config", {})
    backend = str(plan_cfg.get("backend") or trace_meta.get("backend") or "").strip()
    if backend:
        if backend == "wm":
            world_model = plan_cfg.get("world_model", {})
            run_dir = str(world_model.get("run_dir") or "").strip()
            if run_dir:
                return f"wm ({os.path.basename(run_dir)})"
        return backend
    return "planner backend"


def _media_description(media_name: str, *, trace_meta: dict[str, Any] | None = None) -> str:
    backend_label = _media_backend_label(trace_meta)
    descriptions = {
        "closed_loop_replay": (
            "s_{t+1} = f(s_t, a_t), where f is the GT environment transition. "
            "States are rendered in the GT environment. "
            "Here T is bounded by plan.budget.max_env_steps, and is smaller if the rollout terminates early."
        ),
        "planner_view_replay": (
            "s_{t+1} = f(s_t, a_t), where f is the GT environment transition. "
            f"States are rendered in the planner backend ({backend_label}). "
            "Here T is bounded by plan.budget.max_env_steps, and is smaller if the rollout terminates early."
        ),
        "predicted_backend_replay": (
            f"s_{{t+1}} = f(s_t, a_t), where f is the planner backend ({backend_label}). "
            f"States are rendered in the planner backend ({backend_label}). "
            "Here T is plan.planner.horizon * num_replans."
        ),
        "gt_replay": (
            "s_{t+1} = f(s_t, a_t), where f is the GT environment transition. "
            "States are rendered in the GT environment. "
            "Here T is plan.task.init_goal.dataset.trajectory_len."
        ),
    }
    return descriptions.get(media_name, "Review media artifact.")


def _media_title_html(media_name: str, *, trace_meta: dict[str, Any] | None = None) -> str:
    description = html.escape(_media_description(media_name, trace_meta=trace_meta))
    return (
        f"{html.escape(media_name)}"
        f" <span class='info-chip' tabindex='0' data-tooltip='{description}' aria-label='{description}'>i</span>"
    )


def _media_card_title_html(media_name: str, status: str, *, trace_meta: dict[str, Any] | None = None) -> str:
    badge = _status_badge_html(status)
    suffix = f" {badge}" if badge else ""
    return f"<h3>{_media_title_html(media_name, trace_meta=trace_meta)}{suffix}</h3>"


def _media_panel_html(
    data: ExperimentReviewData,
    row: dict[str, Any],
    *,
    media_tasks: dict[str, MediaRenderTask] | None = None,
    trace_meta: dict[str, Any] | None = None,
) -> str:
    media_dir = review_media_dir(data.run_dir, str(row["variant_name"]), str(row["rollout_id"]))
    pieces: list[str] = []
    variant = str(row.get("variant_name", ""))
    rollout_id = str(row.get("rollout_id", ""))
    for media_name in REVIEW_MEDIA:
        task = media_tasks.get(media_name) if media_tasks is not None else None
        media_path = _media_path_for_name(media_dir, media_name)
        if media_path is None:
            href = (
                f"/render?variant={quote(variant)}&rollout_id={quote(rollout_id)}"
                f"&media={quote(media_name)}"
            )
            status = task.status if task is not None else "missing"
            detail = ""
            if status == "failed":
                detail = f"<p class='muted'>Last render failed: {html.escape(task.error or 'unknown error')}</p>"
            pieces.append(
                "<section class='card media-card'>"
                f"{_media_card_title_html(media_name, status, trace_meta=trace_meta)}"
                f"{detail}"
                f"<a class='button' data-async-render='true' href='{href}'>Render</a>"
                "</section>"
            )
            continue
        rel_path = os.path.relpath(media_path, data.run_dir)
        pieces.append(
            "<section class='card media-card'>"
            f"{_media_card_title_html(media_name, 'succeeded', trace_meta=trace_meta)}"
            f"<video controls preload='metadata' src='/files/{quote(rel_path)}'></video>"
            "</section>"
        )
    return "".join(pieces)


def _media_section_html(
    data: ExperimentReviewData,
    row: dict[str, Any],
    *,
    trace_meta: dict[str, Any] | None = None,
    media_tasks: dict[str, MediaRenderTask] | None = None,
) -> str:
    variant = str(row.get("variant_name", ""))
    rollout_id = str(row.get("rollout_id", ""))
    render_all_href = (
        f"/render?variant={quote(variant)}&rollout_id={quote(rollout_id)}"
        + "".join(f"&media={quote(media_name)}" for media_name in REVIEW_MEDIA)
    )
    active_media = any(task.status in {"pending", "running"} for task in (media_tasks or {}).values())
    return (
        f"<section class='card' id='mediaSection' data-variant='{html.escape(variant)}' "
        f"data-rollout-id='{html.escape(rollout_id)}' data-active-media='{'true' if active_media else 'false'}'>"
        "<h2>Media</h2>"
        f"<p><a class='button primary' data-async-render='true' href='{render_all_href}'>Render all review media</a></p>"
        "<div class='media-grid'>"
        f"{_media_panel_html(data, row, media_tasks=media_tasks, trace_meta=trace_meta)}"
        "</div>"
        "</section>"
    )


def _representative_run_rows(rows: Sequence[dict[str, Any]], *, success: bool, limit: int = 8) -> list[dict[str, Any]]:
    filtered = [row for row in rows if _success_bool(row) == success]
    if success:
        ranked = sorted(
            filtered,
            key=lambda row: (
                _safe_float(row.get("final_pos_diff")),
                -_safe_float(row.get("final_coverage")),
                _safe_float(row.get("bits_used_total")),
            ),
        )
    else:
        ranked = sorted(
            filtered,
            key=lambda row: (
                _safe_float(row.get("final_coverage")),
                -_safe_float(row.get("final_pos_diff")),
                -_safe_float(row.get("bits_used_total")),
            ),
        )
    return ranked[:limit]


def build_summary_page(data: ExperimentReviewData, *, notice: str | None = None) -> str:
    kpi_items: list[tuple[str, str]] = [
        ("Experiment", data.experiment_name),
        ("Variants", str(len(data.variant_rows))),
        ("Runs", str(len(data.run_rows))),
    ]
    if len(data.variant_rows) == 1:
        only_variant = data.variant_rows[0]
        kpi_items.extend(
            [
                ("Mean Coverage", _format_number(only_variant.get("mean_final_coverage"))),
                ("Mean Bits", _format_bits_value(only_variant.get("mean_bits_used_total"))),
                ("Mean FLOPs", _format_flops_value(only_variant.get("mean_flops_used_total"))),
                ("Mean Plan Time", _format_number(only_variant.get("mean_plan_time_total_sec"))),
            ]
        )
    kpi_html = _kpi_cards(kpi_items)

    variant_rows = []
    for row in data.variant_rows:
        variant_name = str(row.get("variant_name", ""))
        variant_rows.append(
            {
                "variant_name": f"<a href='/variant?name={quote(variant_name)}'>{html.escape(variant_name)}</a>",
                "n_rollouts": int(row.get("n_rollouts", 0)),
                "success_rate": row.get("success_rate"),
                "mean_final_coverage": row.get("mean_final_coverage"),
                "mean_bits_used_total": row.get("mean_bits_used_total"),
                "mean_bits_used_total__display": _format_bits_value(row.get("mean_bits_used_total")),
                "mean_bits_used_total__sort": row.get("mean_bits_used_total"),
                "mean_flops_used_total": row.get("mean_flops_used_total"),
                "mean_flops_used_total__display": _format_flops_value(row.get("mean_flops_used_total")),
                "mean_flops_used_total__sort": row.get("mean_flops_used_total"),
                "mean_plan_time_total_sec": row.get("mean_plan_time_total_sec"),
            }
        )
    run_rows = []
    for row in data.run_rows:
        variant = str(row.get("variant_name", ""))
        rollout_id = str(row.get("rollout_id", ""))
        run_rows.append(
            {
                "variant_name": f"<a href='/variant?name={quote(variant)}'>{html.escape(variant)}</a>",
                "rollout_id": f"<a href='/run?variant={quote(variant)}&rollout_id={quote(rollout_id)}'>{html.escape(rollout_id)}</a>",
                "success": int(row.get("success", 0)),
                "termination_reason": html.escape(str(row.get("termination_reason", ""))),
                "final_pos_diff": _safe_float(row.get("final_pos_diff")),
                "final_coverage": _safe_float(row.get("final_coverage")),
                "bits_used_total": _safe_float(row.get("bits_used_total")),
                "bits_used_total__display": _format_bits_value(row.get("bits_used_total")),
                "bits_used_total__sort": _safe_float(row.get("bits_used_total")),
                "plan_time_total_sec": _safe_float(row.get("plan_time_total_sec")),
            }
        )

    if len(data.variant_rows) == 1:
        variant_name = data.variant_order[0]
        variant_rows_only = data.rows_by_variant.get(variant_name, [])
        success_analysis_fig = _figure_html(_variant_success_figure(variant_rows_only, variant_name))
        stepwise_fig = _figure_html(_variant_stepwise_figure(data, variant_name))
        best_success_rows = _representative_run_rows(variant_rows_only, success=True)
        hardest_failure_rows = _representative_run_rows(variant_rows_only, success=False)
        best_success_table_rows = [
            {
                "rollout_id": f"<a href='/run?variant={quote(variant_name)}&rollout_id={quote(str(row.get('rollout_id', '')))}'>{html.escape(str(row.get('rollout_id', '')))}</a>",
                "final_coverage": _safe_float(row.get("final_coverage")),
                "bits_used_total": _safe_float(row.get("bits_used_total")),
                "bits_used_total__display": _format_bits_value(row.get("bits_used_total")),
                "bits_used_total__sort": _safe_float(row.get("bits_used_total")),
                "plan_time_total_sec": _safe_float(row.get("plan_time_total_sec")),
            }
            for row in best_success_rows
        ]
        hardest_failure_table_rows = [
            {
                "rollout_id": f"<a href='/run?variant={quote(variant_name)}&rollout_id={quote(str(row.get('rollout_id', '')))}'>{html.escape(str(row.get('rollout_id', '')))}</a>",
                "termination_reason": html.escape(str(row.get("termination_reason", ""))),
                "final_coverage": _safe_float(row.get("final_coverage")),
                "bits_used_total": _safe_float(row.get("bits_used_total")),
                "bits_used_total__display": _format_bits_value(row.get("bits_used_total")),
                "bits_used_total__sort": _safe_float(row.get("bits_used_total")),
            }
            for row in hardest_failure_rows
        ]
        body = f"""
  {_notice_html(notice)}
  <section class='card'>
    <div class='hero'>
      <div>
        <h1>{html.escape(data.experiment_name)}</h1>
        <p class='muted'>Interactive review for experiment bundle <code>{html.escape(os.path.relpath(data.run_dir, os.getcwd()))}</code></p>
      </div>
      <div class='nav-links'>
        <a class='button' href='/'>Overview</a>
        <a class='button primary' href='/variant?name={quote(variant_name)}'>Open Variant Page</a>
      </div>
    </div>
  </section>
  <section class='kpi-grid'>{kpi_html}</section>
  <div class='plot-grid'>
    <section class='card'>{success_analysis_fig}</section>
    <section class='card'>{stepwise_fig}</section>
  </div>
  {_table_section(
        "Best Successful Runs",
        table_id="bestSuccessRunsTable",
        rows=best_success_table_rows,
        columns=[
            ("Rollout", "rollout_id", "text"),
            ("Final Coverage", "final_coverage", "number"),
            ("Bits", "bits_used_total", "number"),
            ("Plan Time", "plan_time_total_sec", "number"),
        ],
    ) if best_success_table_rows else ""}
  {_table_section(
        "Representative Failures",
        table_id="representativeFailuresTable",
        rows=hardest_failure_table_rows,
        columns=[
            ("Rollout", "rollout_id", "text"),
            ("Termination", "termination_reason", "text"),
            ("Final Coverage", "final_coverage", "number"),
            ("Bits", "bits_used_total", "number"),
        ],
    ) if hardest_failure_table_rows else ""}
  {_table_section(
        "Runs",
        table_id="overviewRunsTable",
        rows=run_rows,
        columns=[
            ("Variant", "variant_name", "text"),
            ("Rollout", "rollout_id", "text"),
            ("Success", "success", "number"),
            ("Termination", "termination_reason", "text"),
            ("Final Coverage", "final_coverage", "number"),
            ("Bits", "bits_used_total", "number"),
            ("Plan Time", "plan_time_total_sec", "number"),
        ],
        filter_id="overviewRunsFilter",
        filter_placeholder="variant, rollout, termination...",
    )}
  <section class='card'>
    <h2>Downloads</h2>
    <ul>{_downloads_html(data)}</ul>
  </section>
"""
        return _base_page(f"{data.experiment_name} review", body)

    final_metrics_fig = _figure_html(_metric_distribution_figure(data, metric_specs=FINAL_METRIC_SPECS, title="Cross-Variant Final Metric Distributions"))
    compute_metrics_fig = _figure_html(_metric_distribution_figure(data, metric_specs=COMPUTE_METRIC_SPECS, title="Cross-Variant Compute Distributions"))
    compute_vs_outcome_fig = _figure_html(_compute_vs_outcome_figure(data))
    overview_reference_default = _default_reference_variant(data)
    overview_reference_selector = (
        "<div class='toolbar plot-switcher controls-row'>"
        "<div class='control-group'>"
        "<h2>Reference Comparisons</h2>"
        "</div>"
        "<div class='control-group'>"
        "<label for='overview-reference-select'>Reference Variant</label>"
        "<select id='overview-reference-select' onchange=\"showPlotGroup('overview-reference-summary', this.value); showPlotGroup('overview-reference-delta', this.value)\">"
        + "".join(
            f"<option value='{html.escape(reference_name)}'{' selected' if reference_name == overview_reference_default else ''}>{html.escape(reference_name)}</option>"
            for reference_name in data.variant_order
        )
        + "</select>"
        "</div>"
        "</div>"
    )
    paired_summary_blocks = "".join(
        f"<div data-plot-group='overview-reference-summary' data-plot-value='{html.escape(reference_name)}' style='display:{'' if reference_name == overview_reference_default else 'none'}'>"
        + (
            _figure_html(fig)
            if (fig := _paired_summary_figure(data, reference_name)) is not None
            else f"<p class='muted'>No matched rollouts are available to compare against <code>{html.escape(reference_name)}</code>.</p>"
        )
        + "</div>"
        for reference_name in data.variant_order
    )
    paired_delta_blocks = "".join(
        f"<div data-plot-group='overview-reference-delta' data-plot-value='{html.escape(reference_name)}' style='display:{'' if reference_name == overview_reference_default else 'none'}'>"
        + (
            _figure_html(fig)
            if (fig := _paired_delta_figure(data, reference_name)) is not None
            else f"<p class='muted'>No paired delta data are available for <code>{html.escape(reference_name)}</code>.</p>"
        )
        + "</div>"
        for reference_name in data.variant_order
    )
    stepwise_figs = _overview_stepwise_figures(data)
    overview_success_selector = (
        "<div class='toolbar plot-switcher controls-row'>"
        "<div class='control-group'>"
        "<label for='overview-success-select'>Variant</label>"
        "<select id='overview-success-select' onchange=\"showPlotGroup('overview-success-analysis', this.value)\">"
        + "".join(f"<option value='{html.escape(variant_name)}'>{html.escape(variant_name)}</option>" for variant_name in data.variant_order)
        + "</select>"
        "</div>"
        "<div class='control-group'>"
        "<label for='overview-success-display'>Display</label>"
        "<select id='overview-success-display' data-barmode-group='overview-success-analysis'>"
        "<option value='stack' selected>Stacked</option>"
        "<option value='overlay'>Overlay</option>"
        "</select>"
        "</div>"
        "</div>"
    )
    overview_success_blocks = "".join(
        f"<div data-plot-group='overview-success-analysis' data-plot-value='{html.escape(variant_name)}' style='display:{'' if idx == 0 else 'none'}'>{_figure_html(_variant_success_figure(data.rows_by_variant.get(variant_name, []), variant_name), show_barmode_toggle=False)}</div>"
        for idx, variant_name in enumerate(data.variant_order)
    )

    stepwise_selector = (
        "<div class='toolbar plot-switcher'>"
        "<label for='overview-stepwise-select'>Stepwise Metric</label>"
        "<select id='overview-stepwise-select' onchange=\"showPlotGroup('overview-stepwise', this.value)\">"
        + "".join(f"<option value='{key}'>{html.escape(label)}</option>" for key, label in STEPWISE_METRICS)
        + "</select></div>"
    )
    stepwise_blocks = "".join(
        f"<div class='card' data-plot-group='overview-stepwise' data-plot-value='{key}' style='display:{'' if idx == 0 else 'none'}'>{plot_html}</div>"
        for idx, (key, _label) in enumerate(STEPWISE_METRICS)
        for plot_html in [stepwise_figs[key]]
    )

    body = f"""
  {_notice_html(notice)}
  <section class='card'>
    <div class='hero'>
      <div>
        <h1>{html.escape(data.experiment_name)}</h1>
        <p class='muted'>Interactive review for experiment bundle <code>{html.escape(os.path.relpath(data.run_dir, os.getcwd()))}</code></p>
      </div>
      <div class='nav-links'>
        <a class='button' href='/'>Overview</a>
      </div>
    </div>
  </section>
  <section class='kpi-grid'>{kpi_html}</section>
  {_table_section(
        "Variant Summary",
        table_id="variantSummaryTable",
        rows=variant_rows,
        columns=[
            ("Variant", "variant_name", "text"),
            ("Rollouts", "n_rollouts", "number"),
            ("Success Rate", "success_rate", "number"),
            ("Mean Coverage", "mean_final_coverage", "number"),
            ("Mean Bits", "mean_bits_used_total", "number"),
            ("Mean FLOPs", "mean_flops_used_total", "number"),
            ("Mean Plan Time", "mean_plan_time_total_sec", "number"),
        ],
    )}
  <div class='plot-grid'>
    <section class='card'>
      <h2>Variant Success Analysis</h2>
      {overview_success_selector}
      {overview_success_blocks}
    </section>
    <section class='card'>{compute_vs_outcome_fig}</section>
  </div>
  <div class='plot-grid-wide'>
    <section class='card'>{final_metrics_fig}</section>
    <section class='card'>{compute_metrics_fig}</section>
    <section class='card'>
      {overview_reference_selector}
      {paired_summary_blocks}
      {paired_delta_blocks}
    </section>
  </div>
  <section class='card'>
    <h2>Cross-Variant Stepwise Trace Summary</h2>
    {stepwise_selector}
    {stepwise_blocks}
  </section>
  {_table_section(
        "Runs",
        table_id="overviewRunsTable",
        rows=run_rows,
        columns=[
            ("Variant", "variant_name", "text"),
            ("Rollout", "rollout_id", "text"),
            ("Success", "success", "number"),
            ("Termination", "termination_reason", "text"),
            ("Final Coverage", "final_coverage", "number"),
            ("Bits", "bits_used_total", "number"),
            ("Plan Time", "plan_time_total_sec", "number"),
        ],
        filter_id="overviewRunsFilter",
        filter_placeholder="variant, rollout, termination...",
    )}
  <section class='card'>
    <h2>Downloads</h2>
    <ul>{_downloads_html(data)}</ul>
  </section>
"""
    return _base_page(f"{data.experiment_name} review", body)


def build_variant_page(data: ExperimentReviewData, variant_name: str, *, notice: str | None = None) -> str:
    variant_name = str(variant_name)
    rows = data.rows_by_variant.get(variant_name, [])
    if len(rows) <= 0:
        raise KeyError(f"Unknown variant: {variant_name}")
    variant_row = data.variant_by_name[variant_name]
    success_fig = _figure_html(_variant_success_figure(rows, variant_name))
    final_metrics_fig = _figure_html(_variant_histogram_figure(rows, metric_specs=FINAL_METRIC_SPECS, title=f"{variant_name}: Final Metric Distributions"))
    compute_metrics_fig = _figure_html(_variant_histogram_figure(rows, metric_specs=COMPUTE_METRIC_SPECS, title=f"{variant_name}: Compute and Effort Distributions"))
    relationship_fig = _figure_html(_variant_relationship_figure(rows, variant_name))
    stepwise_fig = _figure_html(_variant_stepwise_figure(data, variant_name))
    reference_section_html = ""
    if len(data.variant_order) > 1:
        reference_default = _default_reference_variant(data, current_variant=variant_name)
        reference_selector = (
            "<div class='toolbar plot-switcher controls-row'>"
            "<div class='control-group'>"
            "<h2>Reference Comparison</h2>"
            "</div>"
            "<div class='control-group'>"
            "<label for='variant-reference-select'>Reference Variant</label>"
            "<select id='variant-reference-select' onchange=\"showPlotGroup('variant-reference-comparison', this.value)\">"
            + "".join(
                f"<option value='{html.escape(reference_name)}'{' selected' if reference_name == reference_default else ''}>{html.escape(reference_name)}</option>"
                for reference_name in data.variant_order
            )
            + "</select>"
            "</div>"
            "</div>"
        )
        reference_blocks_list: list[str] = []
        for reference_name in data.variant_order:
            summary_fig = _variant_paired_summary_figure(data, variant_name, reference_name)
            delta_fig = _variant_paired_delta_figure(data, variant_name, reference_name)
            if summary_fig is not None or delta_fig is not None:
                inner_html = (
                    (_figure_html(summary_fig) if summary_fig is not None else "")
                    + (_figure_html(delta_fig) if delta_fig is not None else "")
                )
            elif reference_name == variant_name:
                inner_html = f"<p class='muted'>Choose a reference variant other than <code>{html.escape(variant_name)}</code> to see paired comparisons.</p>"
            else:
                inner_html = f"<p class='muted'>No matched rollouts are available to compare <code>{html.escape(variant_name)}</code> against <code>{html.escape(reference_name)}</code>.</p>"
            reference_blocks_list.append(
                f"<div data-plot-group='variant-reference-comparison' data-plot-value='{html.escape(reference_name)}' style='display:{'' if reference_name == reference_default else 'none'}'>{inner_html}</div>"
            )
        reference_section_html = f"<section class='card'>{reference_selector}{''.join(reference_blocks_list)}</section>"

    kpi_html = _kpi_cards(
        [
            ("Variant", variant_name),
            ("Rollouts", str(int(variant_row.get("n_rollouts", 0)))),
            ("Success Rate", _format_number(variant_row.get("success_rate"))),
            ("Mean Coverage", _format_number(variant_row.get("mean_final_coverage"))),
            ("Mean Bits", _format_bits_value(variant_row.get("mean_bits_used_total"))),
            ("Mean FLOPs", _format_flops_value(variant_row.get("mean_flops_used_total"))),
            ("Mean Plan Time", _format_number(variant_row.get("mean_plan_time_total_sec"))),
        ]
    )

    run_rows = []
    for row in _sort_runs(rows):
        rollout_id = str(row.get("rollout_id", ""))
        run_rows.append(
            {
                "rollout_id": f"<a href='/run?variant={quote(variant_name)}&rollout_id={quote(rollout_id)}'>{html.escape(rollout_id)}</a>",
                "success": int(row.get("success", 0)),
                "termination_reason": html.escape(str(row.get("termination_reason", ""))),
                "final_pos_diff": _safe_float(row.get("final_pos_diff")),
                "final_coverage": _safe_float(row.get("final_coverage")),
                "bits_used_total": _safe_float(row.get("bits_used_total")),
                "bits_used_total__display": _format_bits_value(row.get("bits_used_total")),
                "bits_used_total__sort": _safe_float(row.get("bits_used_total")),
                "plan_time_total_sec": _safe_float(row.get("plan_time_total_sec")),
            }
        )

    body = f"""
  {_notice_html(notice)}
  <section class='card'>
    <div class='hero'>
      <div>
        <p><a href='/'>&larr; Back to experiment overview</a></p>
        <h1>{html.escape(variant_name)}</h1>
        <p class='muted'>Variant detail page with success analysis, distributions, stepwise traces, and reference comparisons.</p>
      </div>
      <div class='nav-links'>
        <a class='button' href='/'>Overview</a>
      </div>
    </div>
  </section>
  <section class='kpi-grid'>{kpi_html}</section>
  <div class='plot-grid'>
    <section class='card'>{success_fig}</section>
    <section class='card'>{relationship_fig}</section>
  </div>
  <div class='plot-grid-wide'>
    <section class='card'>{final_metrics_fig}</section>
    <section class='card'>{compute_metrics_fig}</section>
    <section class='card'>{stepwise_fig}</section>
    {reference_section_html}
  </div>
  {_table_section(
        f"{variant_name} Runs",
        table_id="variantRunsTable",
        rows=run_rows,
        columns=[
            ("Rollout", "rollout_id", "text"),
            ("Success", "success", "number"),
            ("Termination", "termination_reason", "text"),
            ("Final Coverage", "final_coverage", "number"),
            ("Bits", "bits_used_total", "number"),
            ("Plan Time", "plan_time_total_sec", "number"),
        ],
        filter_id="variantRunsFilter",
        filter_placeholder="rollout, termination...",
    )}
"""
    return _base_page(f"{variant_name} review", body)


def build_run_page(
    data: ExperimentReviewData,
    row: dict[str, Any],
    *,
    notice: str | None = None,
    errors: list[str] | None = None,
    media_tasks: dict[str, MediaRenderTask] | None = None,
) -> str:
    variant = str(row.get("variant_name", ""))
    rollout_id = str(row.get("rollout_id", ""))
    raw_trace_dir = trace_dir(data.run_dir, variant, rollout_id)
    trace_json_path = os.path.join(raw_trace_dir, "trace.json")
    trace_npz_path = os.path.join(raw_trace_dir, "trace.npz")
    metadata_path = os.path.join(raw_trace_dir, "metadata.json")
    run_log_path = os.path.join(raw_trace_dir, "run.log")
    if not os.path.isfile(trace_json_path) or not os.path.isfile(trace_npz_path):
        raise FileNotFoundError(f"Trace files missing for {variant}/{rollout_id}")

    with open(trace_json_path, "r", encoding="utf-8") as f:
        trace_meta = json.load(f)
    trace_arrays = _load_trace_npz(trace_npz_path)
    trace_fig = _figure_html(_single_run_trace_figure(trace_arrays, variant, rollout_id, trace_meta=trace_meta))
    log_tail = _read_log_tail(run_log_path)
    error_html = _notice_html(" | ".join(errors), kind="error") if errors else ""
    active_media_tasks = [task for task in (media_tasks or {}).values() if task.status in {"pending", "running"}]

    kpi_html = _kpi_cards(
        [
            ("Success", str(int(row.get("success", 0)))),
            ("Termination", str(row.get("termination_reason", ""))),
            ("Executed Steps", _format_number(row.get("executed_steps"), digits=0)),
            ("Final Pos", _format_number(row.get("final_pos_diff"))),
            ("Final Angle", _format_number(row.get("final_angle_diff"))),
            ("Final Coverage", _format_number(row.get("final_coverage"))),
            ("Bits", _format_bits_value(row.get("bits_used_total"))),
            ("Plan Time", _format_number(row.get("plan_time_total_sec"))),
        ]
    )

    replan_rows = _replan_rows(trace_meta)
    replan_section = ""
    if len(replan_rows) > 0:
        replan_section = _table_section(
            "Replan Diagnostics",
            table_id="replanTable",
            rows=replan_rows,
            columns=[
                ("Replan", "replan_idx", "number"),
                ("Step Start", "step_start", "number"),
                ("MPC Progress", "mpc_progress", "number"),
                ("Base Level", "base_level_idx", "number"),
                ("Bits", "bits_used_estimate", "number"),
                ("Plan Time", "plan_time_sec", "number"),
                ("Action Horizon", "action_horizon", "number"),
            ],
        )

    raw_links = []
    for path in (trace_json_path, trace_npz_path, metadata_path, run_log_path):
        if os.path.isfile(path):
            rel = os.path.relpath(path, data.run_dir)
            raw_links.append(f"<li><a href='/files/{quote(rel)}'>{html.escape(os.path.basename(path))}</a></li>")
    log_html = ""
    if log_tail is not None:
        log_html = f"<details class='card'><summary>Run Log Tail</summary><pre>{html.escape(log_tail)}</pre></details>"

    body = f"""
  <div id='pageNotice'>{_notice_html(notice)}</div>
  {error_html}
  <section class='card'>
    <div class='hero'>
      <div>
        <p><a href='/variant?name={quote(variant)}'>&larr; Back to {html.escape(variant)}</a></p>
        <h1>{html.escape(variant)} / {html.escape(rollout_id)}</h1>
        <p class='muted'>Raw trace dir: <code>{html.escape(os.path.relpath(raw_trace_dir, data.run_dir))}</code></p>
      </div>
      <div class='nav-links'>
        <a class='button' href='/'>Overview</a>
        <a class='button' href='/variant?name={quote(variant)}'>Variant</a>
      </div>
    </div>
  </section>
  <section class='kpi-grid'>{kpi_html}</section>
  <section class='card'>
    <h2>Single-Run Trace Curves</h2>
    {trace_fig}
  </section>
  {replan_section}
  {_media_section_html(data, row, media_tasks=media_tasks, trace_meta=trace_meta)}
  <section class='card'>
    <h2>Raw Artifacts</h2>
    <ul>{''.join(raw_links)}</ul>
  </section>
  {log_html}
"""
    return _base_page(
        f"{variant} / {rollout_id}",
        body,
        auto_refresh_seconds=3 if len(active_media_tasks) > 0 else None,
    )


def build_run_media_section(
    data: ExperimentReviewData,
    row: dict[str, Any],
    *,
    media_tasks: dict[str, MediaRenderTask] | None = None,
) -> tuple[str, bool]:
    variant = str(row.get("variant_name", ""))
    rollout_id = str(row.get("rollout_id", ""))
    raw_trace_dir = trace_dir(data.run_dir, variant, rollout_id)
    trace_json_path = os.path.join(raw_trace_dir, "trace.json")
    if not os.path.isfile(trace_json_path):
        raise FileNotFoundError(f"Trace JSON missing for {variant}/{rollout_id}")
    with open(trace_json_path, "r", encoding="utf-8") as f:
        trace_meta = json.load(f)
    active_media = any(task.status in {"pending", "running"} for task in (media_tasks or {}).values())
    return _media_section_html(data, row, media_tasks=media_tasks, trace_meta=trace_meta), active_media


class ExperimentReviewApp:
    def __init__(self, run_dir: str):
        self.data = load_experiment_review_data(run_dir)
        self._media_tasks: dict[tuple[str, str, str], MediaRenderTask] = {}
        self._media_tasks_lock = threading.Lock()

    def summary_page(self, *, notice: str | None = None) -> str:
        return build_summary_page(self.data, notice=notice)

    def variant_page(self, *, variant_name: str, notice: str | None = None) -> str:
        return build_variant_page(self.data, variant_name, notice=notice)

    def run_page(
        self,
        *,
        variant_name: str,
        rollout_id: str,
        notice: str | None = None,
        errors: list[str] | None = None,
    ) -> str:
        row = resolve_row(self.data, variant_name, rollout_id)
        return build_run_page(
            self.data,
            row,
            notice=notice,
            errors=errors,
            media_tasks=self.media_tasks_for_run(variant_name, rollout_id),
        )

    def media_tasks_for_run(self, variant_name: str, rollout_id: str) -> dict[str, MediaRenderTask]:
        variant_name = str(variant_name)
        rollout_id = str(rollout_id)
        with self._media_tasks_lock:
            return {
                media_name: MediaRenderTask(
                    variant_name=task.variant_name,
                    rollout_id=task.rollout_id,
                    media_name=task.media_name,
                    status=task.status,
                    outputs=list(task.outputs),
                    error=task.error,
                    updated_at=task.updated_at,
                )
                for (_variant, _rollout, media_name), task in self._media_tasks.items()
                if _variant == variant_name and _rollout == rollout_id
            }

    def queue_media_render(self, *, variant_name: str, rollout_id: str, media: Sequence[str]) -> tuple[int, int]:
        queued = 0
        skipped = 0
        media_dir = review_media_dir(self.data.run_dir, str(variant_name), str(rollout_id))
        for media_name in media:
            if _media_path_for_name(media_dir, str(media_name)) is not None:
                skipped += 1
                continue
            key = (str(variant_name), str(rollout_id), str(media_name))
            with self._media_tasks_lock:
                existing = self._media_tasks.get(key)
                if existing is not None and existing.status in {"pending", "running"}:
                    skipped += 1
                    continue
                self._media_tasks[key] = MediaRenderTask(
                    variant_name=str(variant_name),
                    rollout_id=str(rollout_id),
                    media_name=str(media_name),
                    status="pending",
                    updated_at=time.time(),
                )
            worker = threading.Thread(
                target=self._run_media_render_task,
                kwargs={
                    "variant_name": str(variant_name),
                    "rollout_id": str(rollout_id),
                    "media_name": str(media_name),
                },
                daemon=True,
            )
            worker.start()
            queued += 1
        return queued, skipped

    def _run_media_render_task(self, *, variant_name: str, rollout_id: str, media_name: str) -> None:
        key = (variant_name, rollout_id, media_name)
        with self._media_tasks_lock:
            task = self._media_tasks[key]
            task.status = "running"
            task.updated_at = time.time()
        backend = _trace_backend_for_run(
            self.data.run_dir,
            variant_name=variant_name,
            rollout_id=rollout_id,
        )
        render_context = _PARTICLE_MEDIA_RENDER_LOCK if backend == "particle_sim" else contextlib.nullcontext()
        with render_context:
            outputs, errors = render_media_for_run(
                self.data.run_dir,
                variant_name=variant_name,
                rollout_id=rollout_id,
                media=[media_name],
            )
        with self._media_tasks_lock:
            task = self._media_tasks[key]
            task.outputs = outputs
            task.error = " | ".join(errors) if errors else None
            task.status = "failed" if errors else "succeeded"
            task.updated_at = time.time()


def make_review_handler(app: ExperimentReviewApp):
    class ReviewHandler(BaseHTTPRequestHandler):
        review_app = app

        @staticmethod
        def _is_client_disconnect(exc: BaseException) -> bool:
            return isinstance(exc, (BrokenPipeError, ConnectionResetError))

        def _send_error_quietly(self, status: HTTPStatus, message: str) -> None:
            try:
                self.send_error(status, message)
            except Exception as exc:  # pragma: no cover - defensive server path
                if not self._is_client_disconnect(exc):
                    raise

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"", "/"}:
                    self._send_html(self.review_app.summary_page())
                    return
                if parsed.path == "/variant":
                    params = parse_qs(parsed.query)
                    variant = params.get("name", [""])[0]
                    self._send_html(self.review_app.variant_page(variant_name=variant))
                    return
                if parsed.path == "/run":
                    params = parse_qs(parsed.query)
                    variant = params.get("variant", [""])[0]
                    rollout_id = params.get("rollout_id", [""])[0]
                    self._send_html(self.review_app.run_page(variant_name=variant, rollout_id=rollout_id))
                    return
                if parsed.path == "/run_media":
                    params = parse_qs(parsed.query)
                    variant = params.get("variant", [""])[0]
                    rollout_id = params.get("rollout_id", [""])[0]
                    row = resolve_row(self.review_app.data, variant, rollout_id)
                    media_tasks = self.review_app.media_tasks_for_run(variant, rollout_id)
                    media_html, active_media = build_run_media_section(
                        self.review_app.data,
                        row,
                        media_tasks=media_tasks,
                    )
                    self._send_json({"media_html": media_html, "active_media": active_media})
                    return
                if parsed.path == "/render":
                    params = parse_qs(parsed.query)
                    variant = params.get("variant", [""])[0]
                    rollout_id = params.get("rollout_id", [""])[0]
                    media = [item for item in params.get("media", []) if item]
                    response_format = params.get("format", ["html"])[0]
                    if len(media) <= 0:
                        media = list(REVIEW_MEDIA)
                    queued, skipped = self.review_app.queue_media_render(
                        variant_name=variant,
                        rollout_id=rollout_id,
                        media=media,
                    )
                    notice = None
                    if queued > 0:
                        notice = f"Queued {queued} media artifact(s)."
                    if response_format == "json":
                        row = resolve_row(self.review_app.data, variant, rollout_id)
                        media_tasks = self.review_app.media_tasks_for_run(variant, rollout_id)
                        media_html, active_media = build_run_media_section(
                            self.review_app.data,
                            row,
                            media_tasks=media_tasks,
                        )
                        self._send_json(
                            {
                                "notice_html": "",
                                "media_html": media_html,
                                "active_media": active_media,
                                "queued": queued,
                                "skipped": skipped,
                            }
                        )
                        return
                    self._send_html(
                        self.review_app.run_page(
                            variant_name=variant,
                            rollout_id=rollout_id,
                            notice=notice,
                        )
                    )
                    return
                if parsed.path.startswith("/files/"):
                    rel_path = unquote(parsed.path[len("/files/") :])
                    self._serve_file(rel_path)
                    return
                self._send_error_quietly(HTTPStatus.NOT_FOUND, "Not found")
            except (BrokenPipeError, ConnectionResetError):
                return
            except KeyError as exc:
                self._send_error_quietly(HTTPStatus.NOT_FOUND, str(exc))
            except ValueError as exc:
                self._send_error_quietly(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # pragma: no cover - defensive server path
                self._send_error_quietly(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send_html(self, text: str, *, status: int = HTTPStatus.OK) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_file(self, rel_path: str) -> None:
            path = _safe_join(self.review_app.data.run_dir, rel_path)
            if not os.path.isfile(path):
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            ctype, _ = mimetypes.guess_type(path)
            ctype = ctype or "application/octet-stream"
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ReviewHandler


def create_review_server(run_dir: str, *, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    app = ExperimentReviewApp(run_dir)
    handler_cls = make_review_handler(app)
    return ThreadingHTTPServer((host, port), handler_cls)


def serve_review(run_dir: str, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = create_review_server(run_dir, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", ""} else str(bound_host)
    print(f"[experiment-review] serving {os.path.abspath(run_dir)}")
    print(f"[experiment-review] open http://{display_host}:{bound_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
