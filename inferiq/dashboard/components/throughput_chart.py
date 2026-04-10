"""Throughput visualization components."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from src.benchmark.metrics import BenchmarkMetrics


def plot_throughput_bar(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Create bar chart of throughput by backend and batch size.
    
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
            "Batch Size": m.batch_size,
            "Tokens/sec": m.tokens_per_second,
            "Prompt Length": m.prompt_length,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x="Backend",
        y="Tokens/sec",
        color="Batch Size",
        barmode="group",
        hover_data=["Model", "Prompt Length"],
        title="Throughput (tokens/sec) by Backend and Batch Size",
    )
    
    fig.update_layout(
        xaxis_title="Backend",
        yaxis_title="Tokens per Second",
        legend_title="Batch Size",
    )
    
    return fig


def plot_throughput_line(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Create line chart of throughput vs prompt length.
    
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
            "Prompt Length": m.prompt_length,
            "Tokens/sec": m.tokens_per_second,
            "Batch Size": m.batch_size,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = px.line(
        df,
        x="Prompt Length",
        y="Tokens/sec",
        color="Backend",
        line_dash="Batch Size",
        markers=True,
        title="Throughput vs Prompt Length",
    )
    
    fig.update_layout(
        xaxis_title="Prompt Length (tokens)",
        yaxis_title="Tokens per Second",
    )
    
    return fig


def plot_throughput_heatmap(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Create heatmap of throughput by prompt length and batch size.
    
    Args:
        metrics_list: List of benchmark metrics
        
    Returns:
        Plotly figure
    """
    data = []
    for m in metrics_list:
        data.append({
            "Backend": m.backend.value,
            "Prompt Length": m.prompt_length,
            "Batch Size": m.batch_size,
            "Tokens/sec": m.tokens_per_second,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    backends = df["Backend"].unique()
    fig = make_subplots(
        rows=len(backends),
        cols=1,
        subplot_titles=[f"{b} Throughput" for b in backends],
        vertical_spacing=0.1,
    )
    
    for i, backend in enumerate(backends):
        backend_df = df[df["Backend"] == backend]
        
        pivot = backend_df.pivot(
            index="Prompt Length",
            columns="Batch Size",
            values="Tokens/sec",
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="Viridis",
                text=pivot.values,
                texttemplate="%{text:.0f}",
                name=backend,
                colorbar=dict(title="Tokens/sec"),
            ),
            row=i + 1,
            col=1,
        )
    
    fig.update_layout(
        height=300 * len(backends),
        title_text="Throughput Heatmap",
    )
    
    return fig


def plot_efficiency_scatter(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Create scatter plot of throughput vs latency (efficiency frontier).
    
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
            "Tokens/sec": m.tokens_per_second,
            "Latency (ms)": m.total_time.p50_ms,
            "Batch Size": m.batch_size,
            "Prompt Length": m.prompt_length,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = px.scatter(
        df,
        x="Latency (ms)",
        y="Tokens/sec",
        color="Backend",
        size="Batch Size",
        hover_data=["Model", "Prompt Length"],
        title="Efficiency Frontier: Throughput vs Latency",
    )
    
    fig.update_layout(
        xaxis_title="Latency p50 (ms)",
        yaxis_title="Tokens per Second",
    )
    
    return fig


def render_throughput_section(metrics_list: list[BenchmarkMetrics]) -> None:
    """Render throughput visualization section in Streamlit."""
    st.header("Throughput Analysis")
    
    if not metrics_list:
        st.warning("No benchmark data available for throughput analysis.")
        return
    
    # View selector
    view_type = st.radio(
        "View",
        ["Bar Chart", "Line Chart", "Heatmap", "Efficiency Frontier"],
        horizontal=True,
    )
    
    if view_type == "Bar Chart":
        fig = plot_throughput_bar(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Line Chart":
        fig = plot_throughput_line(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Heatmap":
        fig = plot_throughput_heatmap(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Efficiency Frontier":
        fig = plot_efficiency_scatter(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
