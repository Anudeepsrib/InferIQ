"""Benchmark metrics computation: latency percentiles, throughput, cost analysis."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.gateway.schemas import GenerateResult, ModelBackend
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyMetrics:
    """Latency statistics for a benchmark run."""
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0


@dataclass
class GPUMetrics:
    """GPU utilization and memory metrics."""
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    avg_utilization: float = 0.0
    peak_utilization: float = 0.0
    memory_efficiency: float = 0.0  # tokens per GB


@dataclass
class CostMetrics:
    """Cost analysis metrics."""
    gpu_hour_rate: float = 2.0  # USD per GPU hour
    total_gpu_hours: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_1k_tokens: float = 0.0
    cost_per_1k_prompt_tokens: float = 0.0
    cost_per_1k_completion_tokens: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Complete metrics for a benchmark run."""
    # Configuration
    model_name: str
    backend: ModelBackend
    prompt_length: int
    batch_size: int
    max_tokens: int
    num_runs: int
    
    # Latency metrics
    ttft: LatencyMetrics = field(default_factory=LatencyMetrics)
    total_time: LatencyMetrics = field(default_factory=LatencyMetrics)
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    tokens_per_second_per_gpu: float = 0.0
    prompts_per_second: float = 0.0
    
    # Token metrics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    
    # GPU metrics
    gpu: GPUMetrics = field(default_factory=GPUMetrics)
    
    # Cost metrics
    cost: CostMetrics = field(default_factory=CostMetrics)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    config: dict[str, Any] = field(default_factory=dict)
    raw_results: list[GenerateResult] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "backend": self.backend.value,
            "prompt_length": self.prompt_length,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
            "num_runs": self.num_runs,
            "timestamp": self.timestamp.isoformat(),
            "latency": {
                "ttft_p50_ms": self.ttft.p50_ms,
                "ttft_p95_ms": self.ttft.p95_ms,
                "ttft_p99_ms": self.ttft.p99_ms,
                "total_time_p50_ms": self.total_time.p50_ms,
                "total_time_p95_ms": self.total_time.p95_ms,
                "total_time_p99_ms": self.total_time.p99_ms,
            },
            "throughput": {
                "tokens_per_second": self.tokens_per_second,
                "tokens_per_second_per_gpu": self.tokens_per_second_per_gpu,
                "prompts_per_second": self.prompts_per_second,
            },
            "tokens": {
                "total_prompt": self.total_prompt_tokens,
                "total_completion": self.total_completion_tokens,
                "total": self.total_tokens,
                "avg_prompt": self.avg_prompt_tokens,
                "avg_completion": self.avg_completion_tokens,
            },
            "gpu": {
                "peak_memory_mb": self.gpu.peak_memory_mb,
                "avg_memory_mb": self.gpu.avg_memory_mb,
                "avg_utilization": self.gpu.avg_utilization,
                "peak_utilization": self.gpu.peak_utilization,
                "memory_efficiency": self.gpu.memory_efficiency,
            },
            "cost": {
                "gpu_hour_rate": self.cost.gpu_hour_rate,
                "total_gpu_hours": self.cost.total_gpu_hours,
                "total_cost_usd": self.cost.total_cost_usd,
                "cost_per_1k_tokens": self.cost.cost_per_1k_tokens,
            },
        }
    
    def save_json(self, path: Path | str) -> None:
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info("Metrics saved", path=str(path))


class MetricsComputer:
    """Compute benchmark metrics from raw results."""
    
    def __init__(self, gpu_hour_rate: float = 2.0) -> None:
        """Initialize metrics computer.
        
        Args:
            gpu_hour_rate: Cost per GPU hour in USD
        """
        self.gpu_hour_rate = gpu_hour_rate
    
    @staticmethod
    def compute_percentiles(values: list[float]) -> LatencyMetrics:
        """Compute latency percentiles from values."""
        if not values:
            return LatencyMetrics()
        
        arr = np.array(values)
        return LatencyMetrics(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
        )
    
    def compute_metrics(
        self,
        model_name: str,
        backend: ModelBackend,
        prompt_length: int,
        batch_size: int,
        max_tokens: int,
        results: list[GenerateResult],
        gpu_stats_list: list[dict[str, Any]] | None = None,
    ) -> BenchmarkMetrics:
        """Compute all metrics from generation results.
        
        Args:
            model_name: Model identifier
            backend: Backend type
            prompt_length: Target prompt length in tokens
            batch_size: Number of prompts in batch
            max_tokens: Maximum tokens to generate
            results: List of generation results
            gpu_stats_list: Optional list of GPU statistics during runs
            
        Returns:
            BenchmarkMetrics with all computed statistics
        """
        if not results:
            raise ValueError("Cannot compute metrics from empty results")
        
        # Extract latency values
        ttft_values = [r.ttft_ms for r in results]
        total_time_values = [r.total_time_ms for r in results]
        
        # Compute latency metrics
        ttft_metrics = self.compute_percentiles(ttft_values)
        total_metrics = self.compute_percentiles(total_time_values)
        
        # Token metrics
        total_prompt_tokens = sum(r.prompt_tokens for r in results)
        total_completion_tokens = sum(r.completion_tokens for r in results)
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        avg_completion_tokens = total_completion_tokens / len(results)
        
        # Throughput calculations
        # Total time is sum of all generation times (sequential processing)
        total_time_sec = sum(total_time_values) / 1000
        
        if total_time_sec > 0:
            tokens_per_second = total_completion_tokens / total_time_sec
            prompts_per_second = len(results) * batch_size / total_time_sec
        else:
            tokens_per_second = 0.0
            prompts_per_second = 0.0
        
        # GPU metrics
        gpu_metrics = self._compute_gpu_metrics(results, gpu_stats_list)
        
        # Cost metrics
        cost_metrics = self._compute_cost_metrics(
            total_time_sec, total_tokens, self.gpu_hour_rate
        )
        
        metrics = BenchmarkMetrics(
            model_name=model_name,
            backend=backend,
            prompt_length=prompt_length,
            batch_size=batch_size,
            max_tokens=max_tokens,
            num_runs=len(results),
            ttft=ttft_metrics,
            total_time=total_metrics,
            tokens_per_second=tokens_per_second,
            prompts_per_second=prompts_per_second,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            avg_completion_tokens=avg_completion_tokens,
            gpu=gpu_metrics,
            cost=cost_metrics,
            raw_results=results,
        )
        
        logger.info(
            "Metrics computed",
            model=model_name,
            backend=backend.value,
            throughput=f"{tokens_per_second:.2f} tok/s",
            p99_latency=f"{total_metrics.p99_ms:.2f}ms",
        )
        
        return metrics
    
    def _compute_gpu_metrics(
        self,
        results: list[GenerateResult],
        gpu_stats_list: list[dict[str, Any]] | None = None,
    ) -> GPUMetrics:
        """Compute GPU utilization metrics."""
        # Extract GPU stats from results
        memory_values = []
        utilization_values = []
        
        for result in results:
            if result.gpu_stats:
                mem = result.gpu_stats.get("used_memory_mb", 0)
                util = result.gpu_stats.get("utilization_percent", 0)
                if mem > 0:
                    memory_values.append(mem)
                if util > 0:
                    utilization_values.append(util)
        
        if gpu_stats_list:
            for stats in gpu_stats_list:
                mem = stats.get("used_memory_mb", 0)
                util = stats.get("utilization_percent", 0)
                if mem > 0:
                    memory_values.append(mem)
                if util > 0:
                    utilization_values.append(util)
        
        if not memory_values:
            return GPUMetrics()
        
        peak_memory = max(memory_values)
        avg_memory = statistics.mean(memory_values)
        
        avg_util = statistics.mean(utilization_values) if utilization_values else 0.0
        peak_util = max(utilization_values) if utilization_values else 0.0
        
        # Memory efficiency: tokens per GB
        total_completion_tokens = sum(r.completion_tokens for r in results)
        memory_efficiency = (total_completion_tokens / (peak_memory / 1024)) if peak_memory > 0 else 0.0
        
        return GPUMetrics(
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            avg_utilization=avg_util,
            peak_utilization=peak_util,
            memory_efficiency=memory_efficiency,
        )
    
    def _compute_cost_metrics(
        self,
        total_time_sec: float,
        total_tokens: int,
        gpu_hour_rate: float,
    ) -> CostMetrics:
        """Compute cost estimation metrics."""
        # Convert seconds to GPU hours
        total_gpu_hours = total_time_sec / 3600
        total_cost = total_gpu_hours * gpu_hour_rate
        
        # Cost per 1K tokens
        cost_per_1k = (total_cost / total_tokens) * 1000 if total_tokens > 0 else 0.0
        
        return CostMetrics(
            gpu_hour_rate=gpu_hour_rate,
            total_gpu_hours=total_gpu_hours,
            total_cost_usd=total_cost,
            cost_per_1k_tokens=cost_per_1k,
        )


def format_metrics_table(metrics: BenchmarkMetrics) -> str:
    """Format metrics as a readable table string."""
    lines = [
        f"Benchmark Results: {metrics.model_name} ({metrics.backend.value})",
        f"Configuration: {metrics.prompt_length} tokens × batch {metrics.batch_size}",
        "",
        "Latency (ms):",
        f"  TTFT  p50: {metrics.ttft.p50_ms:8.2f}  p95: {metrics.ttft.p95_ms:8.2f}  p99: {metrics.ttft.p99_ms:8.2f}",
        f"  Total p50: {metrics.total_time.p50_ms:8.2f}  p95: {metrics.total_time.p95_ms:8.2f}  p99: {metrics.total_time.p99_ms:8.2f}",
        "",
        "Throughput:",
        f"  Tokens/sec:    {metrics.tokens_per_second:10.2f}",
        f"  Prompts/sec:   {metrics.prompts_per_second:10.2f}",
        "",
        "GPU Memory:",
        f"  Peak: {metrics.gpu.peak_memory_mb:8.1f} MB  Avg: {metrics.gpu.avg_memory_mb:8.1f} MB",
        f"  Efficiency: {metrics.gpu.memory_efficiency:8.2f} tokens/GB",
        "",
        "Cost:",
        f"  Per 1K tokens: ${metrics.cost.cost_per_1k_tokens:.4f}",
        f"  Total:         ${metrics.cost.total_cost_usd:.4f}",
    ]
    return "\n".join(lines)
