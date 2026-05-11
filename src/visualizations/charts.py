from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, template="plotly_dark", xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=40, b=10), height=420)
    return fig


def style_figure(fig: go.Figure, dark: bool = True) -> go.Figure:
    template = "plotly_dark" if dark else "plotly_white"
    fig.update_layout(template=template, margin=dict(l=10, r=10, t=40, b=10), height=420, legend_title_text="")
    return fig


def latency_over_time(data: pd.DataFrame, dark: bool = True) -> go.Figure:
    if data.empty:
        return _empty_figure("Latency Over Time", "No data available for the selected filters.")
    fig = px.line(data, x="timestamp", y="latency_ms", color="tower_id", title="Latency Over Time", labels={"latency_ms": "Latency (ms)"})
    return style_figure(fig, dark)


def throughput_trends(data: pd.DataFrame, dark: bool = True) -> go.Figure:
    if data.empty:
        return _empty_figure("Throughput Trends", "No data available for the selected filters.")
    melted = data.melt(id_vars=["timestamp", "tower_id"], value_vars=["download_speed_mbps", "upload_speed_mbps"], var_name="metric", value_name="speed_mbps")
    fig = px.line(melted, x="timestamp", y="speed_mbps", color="metric", title="Throughput Trends")
    return style_figure(fig, dark)


def signal_strength_trends(data: pd.DataFrame, dark: bool = True) -> go.Figure:
    if data.empty:
        return _empty_figure("Signal Strength Trends", "No data available for the selected filters.")
    fig = px.line(data, x="timestamp", y="signal_strength_dbm", color="tower_id", title="Signal Strength Trends")
    return style_figure(fig, dark)


def tower_comparison(data: pd.DataFrame, dark: bool = True) -> go.Figure:
    if data.empty:
        return _empty_figure("Tower Comparison", "No tower comparison data for the selected filters.")
    summary = data.groupby(["tower_id", "location"], as_index=False).agg(avg_latency_ms=("latency_ms", "mean"), avg_download_speed_mbps=("download_speed_mbps", "mean"), avg_qos_score=("qos_score", "mean"))
    fig = px.scatter(summary, x="avg_latency_ms", y="avg_download_speed_mbps", size="avg_qos_score", color="tower_id", hover_data=["location"], title="Tower Comparison")
    return style_figure(fig, dark)


def congestion_heatmap(data: pd.DataFrame, dark: bool = True) -> go.Figure:
    if data.empty:
        return _empty_figure("Congestion Heatmap", "No congestion data for the selected filters.")
    heatmap_frame = data.copy()
    heatmap_frame["hour"] = heatmap_frame["timestamp"].dt.hour
    heatmap_frame["day"] = heatmap_frame["timestamp"].dt.day_name()
    pivot = heatmap_frame.pivot_table(index="day", columns="hour", values="congestion_level", aggfunc="mean")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(day_order)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Turbo", title="Congestion Heatmap")
    return style_figure(fig, dark)


def packet_loss_trends(data: pd.DataFrame, dark: bool = True) -> go.Figure:
    if data.empty:
        return _empty_figure("Packet Loss Trends", "No data available for the selected filters.")
    fig = px.line(data, x="timestamp", y="packet_loss_percent", color="tower_id", title="Packet Loss Trends")
    return style_figure(fig, dark)


def anomaly_scatter(data: pd.DataFrame, anomaly_column: str, dark: bool = True, title: str = "Anomaly View") -> go.Figure:
    if data.empty:
        return _empty_figure(title, "No data available for anomaly analysis.")
    color_column = anomaly_column if anomaly_column in data.columns else None
    fig = px.scatter(data, x="timestamp", y="latency_ms", color=color_column, hover_data=["tower_id", "packet_loss_percent", "signal_strength_dbm", "congestion_level"], title=title)
    return style_figure(fig, dark)