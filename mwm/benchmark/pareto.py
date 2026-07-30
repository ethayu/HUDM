from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def pareto_frontier(rows: Iterable[dict[str, Any]], *, cost_key: str = "dynamics_flops_total") -> list[dict[str, Any]]:
    """Return nondominated points for minimum cost and maximum success."""

    points = sorted(
        rows,
        key=lambda row: (float(row.get(cost_key, 0)), -float(row.get("success_rate", 0))),
    )
    frontier: list[dict[str, Any]] = []
    best_success = float("-inf")
    for row in points:
        success = float(row.get("success_rate", 0))
        if success > best_success:
            frontier.append(row)
            best_success = success
    return frontier


def write_pareto_html(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    cost_key: str = "dynamics_flops_total",
) -> str:
    import plotly.graph_objects as go

    rows = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = go.Figure()
    schedules = sorted(
        {
            str(row.get("schedule") or row.get("strategy") or row.get("role", "run"))
            for row in rows
        }
    )
    for schedule in schedules:
        group = [
            row
            for row in rows
            if str(row.get("schedule") or row.get("strategy") or row.get("role", "run")) == schedule
        ]
        customdata = [
            [
                str(row.get("name", "")),
                int(row.get("pop_size", 0)),
                float(row.get("elite_frac", 0)),
                int(row.get("topk", 0)),
                int(row.get("n_iter", 0)),
                int(row.get("candidate_action_values", 0)),
                int(row.get("latent_work_total", 0)),
                float(row.get("wall_time_sec", 0)),
            ]
            for row in group
        ]
        figure.add_trace(
            go.Scatter(
                x=[float(row.get(cost_key, 0)) for row in group],
                y=[float(row.get("success_rate", 0)) for row in group],
                mode="markers",
                name=schedule,
                customdata=customdata,
                marker={"size": 9, "opacity": 0.8},
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>cell=%{customdata[0]}<br>"
                    "audited dynamics FLOPs=%{x:,.0f}<br>success=%{y:.1%}<br>"
                    "population=%{customdata[1]}<br>elite frac=%{customdata[2]:.4g}<br>"
                    "effective topk=%{customdata[3]}<br>CEM iterations=%{customdata[4]}<br>"
                    "candidate action values=%{customdata[5]:,}<br>latent work=%{customdata[6]:,}<br>"
                    "wall time=%{customdata[7]:.2f}s<extra></extra>"
                ),
            )
        )

    frontier = pareto_frontier(rows, cost_key=cost_key)
    if frontier:
        figure.add_trace(
            go.Scatter(
                x=[float(row.get(cost_key, 0)) for row in frontier],
                y=[float(row.get("success_rate", 0)) for row in frontier],
                mode="lines+markers",
                name="Global Pareto frontier",
                line={"color": "black", "width": 3},
                marker={"color": "black", "size": 7},
                hovertemplate="frontier<br>audited dynamics FLOPs=%{x:,.0f}<br>success=%{y:.1%}<extra></extra>",
            )
        )

    figure.update_layout(
        title="Success vs audited dynamics compute",
        xaxis_title="Audited dynamics FLOPs (lower is better)",
        yaxis_title="Success rate (higher is better)",
        yaxis={"tickformat": ".0%", "range": [0, 1.02]},
        hovermode="closest",
        template="plotly_white",
        legend={"groupclick": "toggleitem"},
        margin={"l": 80, "r": 30, "t": 70, "b": 70},
    )
    figure.write_html(str(out), include_plotlyjs=True, full_html=True, config={"responsive": True})
    return str(out)


__all__ = ["pareto_frontier", "write_pareto_html"]
