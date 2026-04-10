"""Streamlit dashboard for InferIQ benchmark visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.components.latency_chart import render_latency_section
from dashboard.components.throughput_chart import render_throughput_section
from dashboard.components.gpu_monitor import render_gpu_section
from dashboard.components.comparison_table import render_comparison_section
from src.benchmark.metrics import BenchmarkMetrics
from src.gateway.schemas import ModelBackend
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="InferIQ Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_results_from_directory(results_dir: Path) -> list[BenchmarkMetrics]:
    """Load all benchmark results from a directory.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        List of BenchmarkMetrics
    """
    metrics_list = []
    
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return metrics_list
    
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Convert raw results
            raw_results = []
            if "raw_results" in data:
                from src.gateway.schemas import GenerateResult
                raw_results = [GenerateResult(**r) for r in data["raw_results"]]
            
            # Reconstruct metrics
            metrics = BenchmarkMetrics(
                model_name=data["model_name"],
                backend=ModelBackend(data["backend"]),
                prompt_length=data["prompt_length"],
                batch_size=data["batch_size"],
                max_tokens=data.get("max_tokens", 128),
                num_runs=data["num_runs"],
                raw_results=raw_results,
            )
            
            # Load computed metrics
            if "latency" in data:
                from src.benchmark.metrics import LatencyMetrics
                metrics.ttft = LatencyMetrics(
                    p50_ms=data["latency"]["ttft_p50_ms"],
                    p95_ms=data["latency"]["ttft_p95_ms"],
                    p99_ms=data["latency"]["ttft_p99_ms"],
                )
                metrics.total_time = LatencyMetrics(
                    p50_ms=data["latency"]["total_time_p50_ms"],
                    p95_ms=data["latency"]["total_time_p95_ms"],
                    p99_ms=data["latency"]["total_time_p99_ms"],
                )
            
            if "throughput" in data:
                metrics.tokens_per_second = data["throughput"]["tokens_per_second"]
                metrics.tokens_per_second_per_gpu = data["throughput"].get("tokens_per_second_per_gpu")
            
            if "gpu" in data:
                from src.benchmark.metrics import GPUMetrics
                metrics.gpu = GPUMetrics(
                    peak_memory_mb=data["gpu"]["peak_memory_mb"],
                    avg_memory_mb=data["gpu"]["avg_memory_mb"],
                    avg_utilization=data["gpu"]["avg_utilization"],
                    peak_utilization=data["gpu"]["peak_utilization"],
                    memory_efficiency=data["gpu"]["memory_efficiency"],
                )
            
            if "cost" in data:
                from src.benchmark.metrics import CostMetrics
                metrics.cost = CostMetrics(
                    gpu_hour_rate=data["cost"]["gpu_hour_rate"],
                    total_gpu_hours=data["cost"]["total_gpu_hours"],
                    total_cost_usd=data["cost"]["total_cost_usd"],
                    cost_per_1k_tokens=data["cost"]["cost_per_1k_tokens"],
                )
            
            metrics_list.append(metrics)
            logger.info(f"Loaded metrics from {json_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
    
    return metrics_list


def render_sidebar() -> tuple[list[BenchmarkMetrics], Path | None]:
    """Render sidebar and return loaded metrics."""
    st.sidebar.title("⚡ InferIQ Dashboard")
    st.sidebar.markdown("GPU-optimized LLM inference benchmarking")
    
    st.sidebar.divider()
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select data source",
        ["Default Results Directory", "Upload Files"],
    )
    
    metrics_list: list[BenchmarkMetrics] = []
    uploaded_file: Path | None = None
    
    if data_source == "Default Results Directory":
        results_dir = Path("results")
        metrics_list = load_results_from_directory(results_dir)
        st.sidebar.info(f"Loaded {len(metrics_list)} benchmark results from {results_dir}")
    
    else:
        uploaded_files = st.sidebar.file_uploader(
            "Upload benchmark result JSON files",
            type=["json"],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    data = json.load(uploaded_file)
                    # Parse and create BenchmarkMetrics (simplified)
                    metrics_list.append(BenchmarkMetrics(**data))
                except Exception as e:
                    st.sidebar.error(f"Failed to load {uploaded_file.name}: {e}")
            
            st.sidebar.info(f"Loaded {len(metrics_list)} benchmark results")
    
    st.sidebar.divider()
    
    # Filters
    st.sidebar.subheader("Filters")
    
    if metrics_list:
        backends = ["All"] + sorted(list(set(m.backend.value for m in metrics_list)))
        selected_backend = st.sidebar.selectbox("Backend", backends)
        
        models = ["All"] + sorted(list(set(m.model_name for m in metrics_list)))
        selected_model = st.sidebar.selectbox("Model", models)
        
        # Apply filters
        filtered_metrics = metrics_list
        if selected_backend != "All":
            filtered_metrics = [m for m in filtered_metrics if m.backend.value == selected_backend]
        if selected_model != "All":
            filtered_metrics = [m for m in filtered_metrics if m.model_name == selected_model]
        
        metrics_list = filtered_metrics
    
    st.sidebar.divider()
    
    # Navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Select page",
        ["Overview", "Latency Analysis", "Throughput Analysis", "GPU Monitoring", "Backend Comparison"],
    )
    
    return metrics_list, page


def render_overview(metrics_list: list[BenchmarkMetrics]) -> None:
    """Render overview page with key metrics."""
    st.title("⚡ InferIQ Dashboard")
    st.markdown("### GPU-optimized LLM inference benchmarking")
    
    if not metrics_list:
        st.warning("No benchmark data available. Run benchmarks or upload result files.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Benchmark Runs", len(metrics_list))
    
    with col2:
        backends = set(m.backend.value for m in metrics_list)
        st.metric("Backends Tested", len(backends))
    
    with col3:
        models = set(m.model_name for m in metrics_list)
        st.metric("Models Tested", len(models))
    
    with col4:
        total_tokens = sum(m.total_completion_tokens for m in metrics_list)
        st.metric("Total Tokens Generated", f"{total_tokens:,}")
    
    st.divider()
    
    # Performance summary
    st.subheader("Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Best Throughput**")
        best = max(metrics_list, key=lambda x: x.tokens_per_second)
        st.markdown(f"**{best.tokens_per_second:.2f}** tokens/sec")
        st.markdown(f"{best.backend.value} @ {best.prompt_length} tokens, batch {best.batch_size}")
    
    with col2:
        st.markdown("**Best Latency (p50)**")
        best = min(metrics_list, key=lambda x: x.total_time.p50_ms)
        st.markdown(f"**{best.total_time.p50_ms:.2f}** ms")
        st.markdown(f"{best.backend.value} @ {best.prompt_length} tokens, batch {best.batch_size}")
    
    with col3:
        st.markdown("**Most Cost Effective**")
        best = min(metrics_list, key=lambda x: x.cost.cost_per_1k_tokens or float('inf'))
        st.markdown(f"**${best.cost.cost_per_1k_tokens:.4f}** / 1K tokens")
        st.markdown(f"{best.backend.value} @ {best.prompt_length} tokens, batch {best.batch_size}")
    
    st.divider()
    
    # Backend distribution
    st.subheader("Test Coverage")
    
    backend_counts: dict[str, int] = {}
    for m in metrics_list:
        backend_counts[m.backend.value] = backend_counts.get(m.backend.value, 0) + 1
    
    cols = st.columns(len(backend_counts))
    for i, (backend, count) in enumerate(backend_counts.items()):
        with cols[i]:
            st.markdown(f"**{backend}**")
            st.markdown(f"{count} runs")


def main() -> None:
    """Main entry point for Streamlit dashboard."""
    metrics_list, page = render_sidebar()
    
    if page == "Overview":
        render_overview(metrics_list)
    
    elif page == "Latency Analysis":
        render_latency_section(metrics_list)
    
    elif page == "Throughput Analysis":
        render_throughput_section(metrics_list)
    
    elif page == "GPU Monitoring":
        render_gpu_section(metrics_list)
    
    elif page == "Backend Comparison":
        render_comparison_section(metrics_list)
    
    # Footer
    st.divider()
    st.caption("InferIQ - Production-grade GPU-optimized LLM serving benchmark")


if __name__ == "__main__":
    main()
