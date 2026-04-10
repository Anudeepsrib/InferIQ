"""Side-by-side backend comparison table component."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.benchmark.metrics import BenchmarkMetrics


def create_comparison_dataframe(
    metrics_list: list[BenchmarkMetrics],
) -> pd.DataFrame:
    """Create DataFrame for comparison table.
    
    Args:
        metrics_list: List of benchmark metrics
        
    Returns:
        Comparison DataFrame
    """
    data = []
    for m in metrics_list:
        data.append({
            "Model": m.model_name,
            "Backend": m.backend.value,
            "Prompt Length": m.prompt_length,
            "Batch Size": m.batch_size,
            "TTFT p50 (ms)": f"{m.ttft.p50_ms:.2f}",
            "TTFT p99 (ms)": f"{m.ttft.p99_ms:.2f}",
            "Total p50 (ms)": f"{m.total_time.p50_ms:.2f}",
            "Total p99 (ms)": f"{m.total_time.p99_ms:.2f}",
            "Tokens/sec": f"{m.tokens_per_second:.2f}",
            "Peak Memory (MB)": f"{m.gpu.peak_memory_mb:.0f}",
            "Memory Eff. (tok/GB)": f"{m.gpu.memory_efficiency:.2f}",
            "Cost/1K tokens": f"${m.cost.cost_per_1k_tokens:.4f}",
        })
    
    return pd.DataFrame(data)


def create_summary_comparison(
    metrics_list: list[BenchmarkMetrics],
) -> pd.DataFrame:
    """Create summary comparison by backend.
    
    Args:
        metrics_list: List of benchmark metrics
        
    Returns:
        Summary DataFrame
    """
    # Group by backend
    backend_metrics: dict[str, list[BenchmarkMetrics]] = {}
    for m in metrics_list:
        key = f"{m.backend.value}"
        if key not in backend_metrics:
            backend_metrics[key] = []
        backend_metrics[key].append(m)
    
    data = []
    for backend, metrics in backend_metrics.items():
        avg_throughput = sum(m.tokens_per_second for m in metrics) / len(metrics)
        avg_latency_p50 = sum(m.total_time.p50_ms for m in metrics) / len(metrics)
        avg_latency_p99 = sum(m.total_time.p99_ms for m in metrics) / len(metrics)
        avg_ttft_p50 = sum(m.ttft.p50_ms for m in metrics) / len(metrics)
        avg_memory = sum(m.gpu.peak_memory_mb for m in metrics) / len(metrics)
        avg_cost = sum(m.cost.cost_per_1k_tokens for m in metrics) / len(metrics)
        
        data.append({
            "Backend": backend,
            "Runs": len(metrics),
            "Avg Tokens/sec": f"{avg_throughput:.2f}",
            "Avg TTFT p50 (ms)": f"{avg_ttft_p50:.2f}",
            "Avg Latency p50 (ms)": f"{avg_latency_p50:.2f}",
            "Avg Latency p99 (ms)": f"{avg_latency_p99:.2f}",
            "Avg Peak Memory (MB)": f"{avg_memory:.0f}",
            "Avg Cost/1K tokens": f"${avg_cost:.4f}",
        })
    
    df = pd.DataFrame(data)
    
    # Sort by throughput descending
    df["_sort"] = df["Avg Tokens/sec"].astype(float)
    df = df.sort_values("_sort", ascending=False).drop("_sort", axis=1)
    
    return df


def highlight_best_values(df: pd.DataFrame) -> pd.DataFrame:
    """Highlight best values in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Styled DataFrame
    """
    # Define which columns are better when lower vs higher
    higher_better = ["Avg Tokens/sec"]
    lower_better = [
        "Avg TTFT p50 (ms)",
        "Avg Latency p50 (ms)",
        "Avg Latency p99 (ms)",
        "Avg Peak Memory (MB)",
        "Avg Cost/1K tokens",
    ]
    
    def style_column(col: pd.Series) -> list[str]:
        """Style a single column."""
        col_name = col.name
        
        # Try to convert to numeric
        try:
            numeric_col = col.str.replace("$", "").str.replace(",", "").astype(float)
        except (ValueError, AttributeError):
            return [""] * len(col)
        
        if col_name in higher_better:
            max_val = numeric_col.max()
            return ["background-color: #90EE90" if v == max_val else "" for v in numeric_col]
        elif col_name in lower_better:
            min_val = numeric_col.min()
            return ["background-color: #90EE90" if v == min_val else "" for v in numeric_col]
        
        return [""] * len(col)
    
    return df.style.apply(style_column)


def render_comparison_section(metrics_list: list[BenchmarkMetrics]) -> None:
    """Render comparison section in Streamlit."""
    st.header("Backend Comparison")
    
    if not metrics_list:
        st.warning("No benchmark data available for comparison.")
        return
    
    # View selector
    view_type = st.radio(
        "View",
        ["Summary by Backend", "Detailed View", "Filtered View"],
        horizontal=True,
    )
    
    if view_type == "Summary by Backend":
        df = create_summary_comparison(metrics_list)
        st.dataframe(
            highlight_best_values(df),
            use_container_width=True,
            hide_index=True,
        )
        
        # Add explanation
        st.caption("Green highlighting indicates best values (higher throughput is better, lower latency/cost is better).")
    
    elif view_type == "Detailed View":
        df = create_comparison_dataframe(metrics_list)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    elif view_type == "Filtered View":
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backends = ["All"] + list(set(m.backend.value for m in metrics_list))
            selected_backend = st.selectbox("Backend", backends)
        
        with col2:
            prompt_lengths = ["All"] + sorted(set(m.prompt_length for m in metrics_list))
            selected_prompt = st.selectbox("Prompt Length", prompt_lengths)
        
        with col3:
            batch_sizes = ["All"] + sorted(set(m.batch_size for m in metrics_list))
            selected_batch = st.selectbox("Batch Size", batch_sizes)
        
        # Apply filters
        filtered = metrics_list
        if selected_backend != "All":
            filtered = [m for m in filtered if m.backend.value == selected_backend]
        if selected_prompt != "All":
            filtered = [m for m in filtered if m.prompt_length == int(selected_prompt)]
        if selected_batch != "All":
            filtered = [m for m in filtered if m.batch_size == int(selected_batch)]
        
        df = create_comparison_dataframe(filtered)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.caption(f"Showing {len(filtered)} results")
