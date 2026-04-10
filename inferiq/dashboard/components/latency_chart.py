"""Latency distribution and percentile visualization components."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from src.benchmark.metrics import BenchmarkMetrics


def plot_latency_violin(
    metrics_list: list[BenchmarkMetrics],
    metric_type: str = "total_time",
) -> go.Figure:
    """Create violin plot of latency distributions.
    
    Args:
        metrics_list: List of benchmark metrics
        metric_type: 'total_time' or 'ttft'
        
    Returns:
        Plotly figure
    """
    data = []
    for m in metrics_list:
        latency_data = m.raw_results if m.raw_results else []
        
        for result in latency_data:
            latency = (
                result.total_time_ms if metric_type == "total_time"
                else result.ttft_ms
            )
            data.append({
                "Backend": m.backend.value,
                "Model": m.model_name,
                "Latency (ms)": latency,
                "Prompt Length": m.prompt_length,
                "Batch Size": m.batch_size,
            })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = px.violin(
        df,
        x="Backend",
        y="Latency (ms)",
        color="Backend",
        box=True,
        points="all",
        hover_data=["Model", "Prompt Length", "Batch Size"],
        title=f"{metric_type.replace('_', ' ').title()} Distribution by Backend",
    )
    
    fig.update_layout(
        xaxis_title="Backend",
        yaxis_title=f"{metric_type.replace('_', ' ').title()} (ms)",
        showlegend=False,
    )
    
    return fig


def plot_latency_percentiles(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Create grouped bar chart of latency percentiles.
    
    Args:
        metrics_list: List of benchmark metrics
        
    Returns:
        Plotly figure
    """
    data = []
    for m in metrics_list:
        data.append({
            "Backend": m.backend.value,
            "Model": m.model_name,
            "Metric": "p50",
            "TTFT (ms)": m.ttft.p50_ms,
            "Total Time (ms)": m.total_time.p50_ms,
        })
        data.append({
            "Backend": m.backend.value,
            "Model": m.model_name,
            "Metric": "p95",
            "TTFT (ms)": m.ttft.p95_ms,
            "Total Time (ms)": m.total_time.p95_ms,
        })
        data.append({
            "Backend": m.backend.value,
            "Model": m.model_name,
            "Metric": "p99",
            "TTFT (ms)": m.ttft.p99_ms,
            "Total Time (ms)": m.total_time.p99_ms,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("TTFT Percentiles", "Total Time Percentiles"),
    )
    
    for metric in ["p50", "p95", "p99"]:
        metric_df = df[df["Metric"] == metric]
        
        fig.add_trace(
            go.Bar(
                name=metric,
                x=metric_df["Backend"],
                y=metric_df["TTFT (ms)"],
                legendgroup=metric,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        
        fig.add_trace(
            go.Bar(
                name=metric,
                x=metric_df["Backend"],
                y=metric_df["Total Time (ms)"],
                legendgroup=metric,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    
    fig.update_layout(
        barmode="group",
        height=500,
        title_text="Latency Percentiles by Backend",
    )
    
    return fig


def plot_latency_heatmap(
    metrics_list: list[BenchmarkMetrics],
    metric_type: str = "total_time",
    percentile: str = "p99",
) -> go.Figure:
    """Create heatmap of latency by prompt length and batch size.
    
    Args:
        metrics_list: List of benchmark metrics
        metric_type: 'total_time' or 'ttft'
        percentile: 'p50', 'p95', or 'p99'
        
    Returns:
        Plotly figure
    """
    data = []
    for m in metrics_list:
        latency = (
            getattr(m.total_time, f"{percentile}_ms")
            if metric_type == "total_time"
            else getattr(m.ttft, f"{percentile}_ms")
        )
        data.append({
            "Backend": m.backend.value,
            "Prompt Length": m.prompt_length,
            "Batch Size": m.batch_size,
            "Latency (ms)": latency,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    # Create separate heatmaps per backend
    backends = df["Backend"].unique()
    fig = make_subplots(
        rows=len(backends),
        cols=1,
        subplot_titles=[f"{b} Backend" for b in backends],
        vertical_spacing=0.1,
    )
    
    for i, backend in enumerate(backends):
        backend_df = df[df["Backend"] == backend]
        
        pivot = backend_df.pivot(
            index="Prompt Length",
            columns="Batch Size",
            values="Latency (ms)",
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn_r",
                text=pivot.values,
                texttemplate="%{text:.0f}",
                name=backend,
                colorbar=dict(title="Latency (ms)"),
            ),
            row=i + 1,
            col=1,
        )
    
    fig.update_layout(
        height=300 * len(backends),
        title_text=f"{percentile.upper()} {metric_type.replace('_', ' ').title()} Heatmap",
    )
    
    return fig


def render_latency_section(metrics_list: list[BenchmarkMetrics]) -> None:
    """Render latency visualization section in Streamlit."""
    st.header("Latency Analysis")
    
    if not metrics_list:
        st.warning("No benchmark data available for latency analysis.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        metric_type = st.selectbox(
            "Metric Type",
            ["total_time", "ttft"],
            format_func=lambda x: "Total Generation Time" if x == "total_time" else "Time to First Token",
        )
    
    with col2:
        view_type = st.radio(
            "View",
            ["Distribution", "Percentiles", "Heatmap"],
            horizontal=True,
        )
    
    # Render selected view
    if view_type == "Distribution":
        fig = plot_latency_violin(metrics_list, metric_type)
        st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Percentiles":
        fig = plot_latency_percentiles(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Heatmap":
        percentile = st.selectbox("Percentile", ["p50", "p95", "p99"])
        fig = plot_latency_heatmap(metrics_list, metric_type, percentile)
        st.plotly_chart(fig, use_container_width=True)
