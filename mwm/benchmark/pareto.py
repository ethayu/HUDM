from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


_HIGH_SEPARATION_COLORS = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#00a6a6",
    "#0a0af0",
    "#6a441d",
    "#f01df0",
    "#f00a7d",
    "#b6a30a",
    "#f090a3",
    "#0a6af0",
    "#7d0a30",
    "#1d5730",
    "#570aa3",
    "#a3a36a",
    "#0ac90a",
    "#dd9057",
    "#b6a3f0",
    "#57446a",
    "#c90aa3",
    "#901d0a",
    "#f0576a",
    "#303090",
    "#576a0a",
    "#f06ab6",
    "#c96af0",
    "#0a44f0",
    "#7db60a",
    "#a30af0",
    "#0a6a7d",
)


def _schedule_label(row: dict[str, Any]) -> str:
    return str(row.get("schedule") or row.get("strategy") or row.get("role", "run"))


def _strategy_color_map(schedules: Iterable[str]) -> dict[str, str]:
    labels = sorted({str(value) for value in schedules})
    if len(labels) > len(_HIGH_SEPARATION_COLORS):
        raise ValueError(
            f"Strategy color key supports at most {len(_HIGH_SEPARATION_COLORS)} strategies, got {len(labels)}."
        )
    return {label: _HIGH_SEPARATION_COLORS[index] for index, label in enumerate(labels)}


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
    provisional: bool = False,
) -> str:
    import plotly.graph_objects as go

    rows = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = go.Figure()
    frontier = pareto_frontier(rows, cost_key=cost_key)
    frontier_objects = {id(row) for row in frontier}
    dominated = [row for row in rows if id(row) not in frontier_objects]
    schedules = sorted(
        {
            _schedule_label(row)
            for row in rows
        }
    )
    strategy_colors = _strategy_color_map(schedules)
    percent_scale = max((float(row.get("success_rate", 0)) for row in rows), default=0.0) > 1.5
    success_hover = "%{y:.1f}%" if percent_scale else "%{y:.1%}"

    def _customdata(group: list[dict[str, Any]]) -> list[list[Any]]:
        return [
            [
                str(row.get("name", "")),
                _schedule_label(row),
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

    hovertemplate = (
        "<b>%{customdata[1]}</b><br>cell=%{customdata[0]}<br>"
        "audited dynamics FLOPs=%{x:,.0f}<br>success=" + success_hover + "<br>"
        "population=%{customdata[2]}<br>elite frac=%{customdata[3]:.4g}<br>"
        "effective topk=%{customdata[4]}<br>CEM iterations=%{customdata[5]}<br>"
        "candidate action values=%{customdata[6]:,}<br>latent work=%{customdata[7]:,}<br>"
        "wall time=%{customdata[8]:.2f}s<extra></extra>"
    )

    figure.add_trace(
        go.Scattergl(
            x=[float(row.get(cost_key, 0)) for row in dominated],
            y=[float(row.get("success_rate", 0)) for row in dominated],
            mode="markers",
            name="Dominated cells",
            customdata=_customdata(dominated),
            marker={"size": 5, "opacity": 0.16, "color": "#64748b"},
            hovertemplate=hovertemplate,
        )
    )

    for schedule in schedules:
        group = [row for row in rows if _schedule_label(row) == schedule]
        schedule_frontier = pareto_frontier(group, cost_key=cost_key)
        figure.add_trace(
            go.Scattergl(
                x=[float(row.get(cost_key, 0)) for row in schedule_frontier],
                y=[float(row.get("success_rate", 0)) for row in schedule_frontier],
                mode="lines+markers",
                name=schedule,
                customdata=_customdata(schedule_frontier),
                line={"color": strategy_colors[schedule], "width": 2.0},
                marker={"color": strategy_colors[schedule], "size": 4},
                opacity=0.78,
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )

    for schedule in schedules:
        group = [row for row in rows if _schedule_label(row) == schedule]
        figure.add_trace(
            go.Scattergl(
                x=[float(row.get(cost_key, 0)) for row in group],
                y=[float(row.get("success_rate", 0)) for row in group],
                mode="markers",
                name=schedule,
                customdata=_customdata(group),
                marker={
                    "size": 8,
                    "opacity": 0.78,
                    "color": strategy_colors[schedule],
                    "line": {"color": "#0f172a", "width": 0.6},
                },
                hovertemplate=hovertemplate,
                visible=False,
                showlegend=False,
            )
        )

    if frontier:
        figure.add_trace(
            go.Scatter(
                x=[float(row.get(cost_key, 0)) for row in frontier],
                y=[float(row.get("success_rate", 0)) for row in frontier],
                mode="lines+markers",
                name=(
                    "Provisional global Pareto frontier (reference)"
                    if provisional
                    else "Global Pareto frontier (reference)"
                ),
                customdata=_customdata(frontier),
                line={"color": "#64748b", "width": 1.4, "dash": "dot"},
                marker={"color": "#64748b", "symbol": "circle-open", "size": 7},
                opacity=0.62,
                hovertemplate=hovertemplate,
            )
        )

    all_visible = [True] + [True] * len(schedules) + [False] * len(schedules) + ([True] if frontier else [])
    buttons = [
        {
            "label": "All schedules",
            "method": "update",
            "args": [{"visible": all_visible}],
        }
    ]
    for schedule_index, schedule in enumerate(schedules):
        visible = [True] + [True] * len(schedules)
        visible.extend(index == schedule_index for index in range(len(schedules)))
        if frontier:
            visible.append(True)
        buttons.append(
            {
                "label": schedule,
                "method": "update",
                "args": [{"visible": visible}],
            }
        )

    positive_costs = [float(row.get(cost_key, 0)) for row in rows if float(row.get(cost_key, 0)) > 0]
    use_log_cost = bool(positive_costs and max(positive_costs) / min(positive_costs) >= 20)

    figure.update_layout(
        title=(
            "Provisional success vs audited dynamics compute (completed cells only)"
            if provisional
            else "Success vs audited dynamics compute"
        ),
        xaxis_title="Audited dynamics FLOPs (lower is better)",
        yaxis_title="Success rate (higher is better)",
        xaxis={"type": "log" if use_log_cost else "linear", "showgrid": True, "gridcolor": "#e2e8f0"},
        yaxis={
            "ticksuffix": "%" if percent_scale else "",
            "tickformat": None if percent_scale else ".0%",
            "range": [0, 102] if percent_scale else [0, 1.02],
            "showgrid": True,
            "gridcolor": "#e2e8f0",
        },
        hovermode="closest",
        template="plotly_white",
        legend={"orientation": "h", "x": 0, "y": 1.08},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        annotations=[
            {
                "text": "Highlight schedule",
                "showarrow": False,
                "x": 1,
                "xanchor": "right",
                "xref": "paper",
                "y": 1.13,
                "yref": "paper",
                "font": {"size": 11, "color": "#475569"},
            }
        ],
        height=680,
        margin={"l": 84, "r": 30, "t": 115, "b": 74},
    )
    figure.write_html(str(out), include_plotlyjs=True, full_html=True, config={"responsive": True})
    return str(out)


__all__ = ["pareto_frontier", "write_pareto_html"]
