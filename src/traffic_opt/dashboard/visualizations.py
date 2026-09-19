"""Plotly figures built only from available experiment data."""

import plotly.graph_objects as go

from traffic_opt.experiments.results import ComparisonRow, ScenarioExperimentResult
from .data import result_to_event_rows, result_to_snapshot_rows


def metric_time_series(result: ScenarioExperimentResult, metric: str, title: str, y_title: str) -> go.Figure:
    rows = result_to_snapshot_rows(result)
    figure = go.Figure()
    if rows:
        figure.add_trace(
            go.Scatter(
                x=[row["time_seconds"] for row in rows],
                y=[row[metric] for row in rows],
                mode="lines+markers",
                name=title,
            )
        )
    figure.update_layout(title=title, xaxis_title="Simulation time (s)", yaxis_title=y_title)
    return figure


def event_timeline(result: ScenarioExperimentResult) -> go.Figure:
    rows = result_to_event_rows(result)
    figure = go.Figure()
    for row in rows:
        figure.add_trace(
            go.Bar(
                x=[row["duration_seconds"]],
                y=[f"{row['event_type']} · {row['affected_roads']}"],
                base=[row["start_time_seconds"]],
                orientation="h",
                name=row["event_type"],
                hovertemplate=(
                    "%{y}<br>Start: %{base}s<br>Duration: %{x}s<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Scheduled traffic events",
        xaxis_title="Simulation time (s)",
        barmode="overlay",
        showlegend=False,
    )
    return figure


def comparison_chart(rows: tuple[ComparisonRow, ...], metric: str, title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[f"{row.controller_name} · {row.scenario}" for row in rows],
            y=[getattr(row, metric) for row in rows],
            name=y_title,
        )
    )
    figure.update_layout(title=title, xaxis_title="Experiment", yaxis_title=y_title)
    return figure


def topology_chart(topology, affected_roads: set[str] | None = None) -> go.Figure:
    figure = go.Figure()
    positions = {node: (index, 0) for index, node in enumerate(sorted(topology.nodes))}
    for start, end, attributes in topology.edges(data=True):
        road_id = attributes.get("road_id", f"{start}->{end}")
        color = "#e45756" if affected_roads and road_id in affected_roads else "#6688aa"
        figure.add_trace(
            go.Scatter(
                x=[positions[start][0], positions[end][0]],
                y=[0, 0],
                mode="lines",
                line={"color": color, "width": 3},
                hoverinfo="text",
                text=[road_id, road_id],
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in sorted(positions)],
            y=[0 for _ in positions],
            mode="markers+text",
            text=list(sorted(positions)),
            textposition="top center",
            marker={"size": 14, "color": "#13b5a4"},
            name="Intersections",
        )
    )
    figure.update_layout(title="Network topology", showlegend=False, xaxis_visible=False, yaxis_visible=False)
    return figure