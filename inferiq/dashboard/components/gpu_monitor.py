"""GPU utilization and memory monitoring components."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from src.benchmark.metrics import BenchmarkMetrics


def plot_gpu_memory_timeline(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Plot GPU memory usage over benchmark runs.
    
    Args:
        metrics_list: List of benchmark metrics
        
    Returns:
        Plotly figure
    """
    data = []
    for m in metrics_list:
        for i, result in enumerate(m.raw_results):
            if result.gpu_stats:
                data.append({
                    "Run": i + 1,
                    "Backend": m.backend.value,
                    "Model": m.model_name,
                    "GPU Memory Used (MB)": result.gpu_stats.get("used_memory_mb", 0),
                    "GPU Utilization (%)": result.gpu_stats.get("utilization_percent", 0),
                    "Prompt Length": m.prompt_length,
                    "Batch Size": m.batch_size,
                })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No GPU data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("GPU Memory Usage", "GPU Utilization"),
        vertical_spacing=0.15,
    )
    
    for backend in df["Backend"].unique():
        backend_df = df[df["Backend"] == backend]
        
        fig.add_trace(
            go.Scatter(
                x=backend_df["Run"],
                y=backend_df["GPU Memory Used (MB)"],
                mode="lines+markers",
                name=f"{backend} - Memory",
                legendgroup=backend,
            ),
            row=1,
            col=1,
        )
        
        fig.add_trace(
            go.Scatter(
                x=backend_df["Run"],
                y=backend_df["GPU Utilization (%)"],
                mode="lines+markers",
                name=f"{backend} - Util",
                legendgroup=backend,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    
    fig.update_layout(
        height=600,
        title_text="GPU Metrics Over Benchmark Runs",
    )
    
    fig.update_xaxes(title_text="Run Number", row=2, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=1)
    fig.update_yaxes(title_text="Utilization (%)", row=2, col=1)
    
    return fig


def plot_gpu_memory_by_config(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Create bar chart of peak GPU memory by configuration.
    
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
            "Batch Size": m.batch_size,
            "Peak Memory (MB)": m.gpu.peak_memory_mb,
            "Avg Memory (MB)": m.gpu.avg_memory_mb,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No GPU data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x="Backend",
        y=["Peak Memory (MB)", "Avg Memory (MB)"],
        color="Batch Size",
        barmode="group",
        facet_col="Prompt Length",
        title="GPU Memory Usage by Configuration",
    )
    
    fig.update_layout(
        yaxis_title="Memory (MB)",
        height=500,
    )
    
    return fig


def plot_memory_efficiency(
    metrics_list: list[BenchmarkMetrics],
) -> go.Figure:
    """Plot memory efficiency (tokens per GB) by backend.
    
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
            "Memory Efficiency (tokens/GB)": m.gpu.memory_efficiency,
            "Throughput (tokens/sec)": m.tokens_per_second,
            "Prompt Length": m.prompt_length,
            "Batch Size": m.batch_size,
        })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No GPU data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    fig = px.scatter(
        df,
        x="Backend",
        y="Memory Efficiency (tokens/GB)",
        color="Batch Size",
        size="Throughput (tokens/sec)",
        hover_data=["Model", "Prompt Length"],
        title="Memory Efficiency by Backend",
    )
    
    fig.update_layout(
        yaxis_title="Tokens per GB of GPU Memory",
        height=500,
    )
    
    return fig


def render_gpu_section(metrics_list: list[BenchmarkMetrics]) -> None:
    """Render GPU monitoring section in Streamlit."""
    st.header("GPU Monitoring")
    
    if not metrics_list:
        st.warning("No benchmark data available for GPU analysis.")
        return
    
    # Check if any metrics have GPU data
    has_gpu_data = any(
        m.gpu.peak_memory_mb > 0 or m.gpu.avg_utilization > 0
        for m in metrics_list
    )
    
    if not has_gpu_data:
        st.info("No GPU metrics collected during benchmark runs.")
        return
    
    # View selector
    view_type = st.radio(
        "View",
        ["Timeline", "Memory by Config", "Efficiency"],
        horizontal=True,
    )
    
    if view_type == "Timeline":
        fig = plot_gpu_memory_timeline(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Memory by Config":
        fig = plot_gpu_memory_by_config(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
    
    elif view_type == "Efficiency":
        fig = plot_memory_efficiency(metrics_list)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    st.subheader("GPU Memory Summary")
    col1, col2, col3 = st.columns(3)
    
    peak_memories = [m.gpu.peak_memory_mb for m in metrics_list if m.gpu.peak_memory_mb > 0]
    if peak_memories:
        with col1:
            st.metric("Peak Memory (MB)", f"{max(peak_memories):.0f}")
        with col2:
            st.metric("Avg Peak Memory (MB)", f"{sum(peak_memories)/len(peak_memories):.0f}")
        with col3:
            avg_utils = [m.gpu.avg_utilization for m in metrics_list if m.gpu.avg_utilization > 0]
            if avg_utils:
                st.metric("Avg Utilization (%)", f"{sum(avg_utils)/len(avg_utils):.1f}")
