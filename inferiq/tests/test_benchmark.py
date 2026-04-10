"""Tests for benchmark runner, workload generator, and metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from src.benchmark.metrics import (
    BenchmarkMetrics,
    LatencyMetrics,
    GPUMetrics,
    CostMetrics,
    MetricsComputer,
    format_metrics_table,
)
from src.benchmark.workloads import WorkloadGenerator
from src.benchmark.runner import BenchmarkRunner
from src.gateway.schemas import GenerateResult, ModelBackend


# Fixtures
@pytest.fixture
def sample_generate_results():
    """Create sample generation results."""
    return [
        GenerateResult(
            text="Output 1",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            ttft_ms=45.0,
            total_time_ms=90.0,
            tokens_per_second=555.56,
            finish_reason="stop",
        ),
        GenerateResult(
            text="Output 2",
            prompt_tokens=100,
            completion_tokens=52,
            total_tokens=152,
            ttft_ms=48.0,
            total_time_ms=95.0,
            tokens_per_second=547.37,
            finish_reason="stop",
        ),
        GenerateResult(
            text="Output 3",
            prompt_tokens=100,
            completion_tokens=48,
            total_tokens=148,
            ttft_ms=42.0,
            total_time_ms=85.0,
            tokens_per_second=564.71,
            finish_reason="length",
        ),
    ]


@pytest.fixture
def sample_benchmark_metrics(sample_generate_results):
    """Create sample benchmark metrics."""
    return BenchmarkMetrics(
        model_name="test-model",
        backend=ModelBackend.VLLM,
        prompt_length=128,
        batch_size=1,
        max_tokens=128,
        num_runs=3,
        ttft=LatencyMetrics(p50_ms=45.0, p95_ms=48.0, p99_ms=48.0),
        total_time=LatencyMetrics(p50_ms=90.0, p95_ms=95.0, p99_ms=95.0),
        tokens_per_second=555.88,
        total_prompt_tokens=300,
        total_completion_tokens=150,
        total_tokens=450,
        gpu=GPUMetrics(
            peak_memory_mb=20480.0,
            avg_memory_mb=18432.0,
            avg_utilization=85.5,
        ),
        cost=CostMetrics(
            gpu_hour_rate=2.5,
            total_gpu_hours=0.0001,
            total_cost_usd=0.00025,
            cost_per_1k_tokens=0.00056,
        ),
        raw_results=sample_generate_results,
    )


# Workload generator tests
class TestWorkloadGenerator:
    """Test workload generation."""
    
    def test_workload_generator_initialization(self):
        """Test workload generator initialization."""
        gen = WorkloadGenerator()
        assert gen.tokenizer_name is None
        assert gen._tokenizer is None
    
    def test_token_estimation(self):
        """Test token estimation."""
        gen = WorkloadGenerator()
        text = "This is a test sentence with eight words"
        estimated = gen.estimate_tokens(text)
        # Should be roughly words * 1.3
        assert estimated > 0
        assert estimated > len(text.split())
    
    def test_generate_prompt_fixed_length(self):
        """Test prompt generation with fixed length."""
        gen = WorkloadGenerator()
        prompt = gen.generate_prompt(128, distribution="fixed")
        
        assert len(prompt) > 0
        assert isinstance(prompt, str)
    
    def test_generate_batch(self):
        """Test batch generation."""
        gen = WorkloadGenerator()
        batch = gen.generate_batch(128, batch_size=4)
        
        assert len(batch) == 4
        for prompt in batch:
            assert len(prompt) > 0
    
    def test_generate_variable_batch(self):
        """Test variable-length batch generation."""
        gen = WorkloadGenerator()
        batch = gen.generate_variable_batch((64, 256), batch_size=5)
        
        assert len(batch) == 5
    
    def test_generate_dataset(self):
        """Test full dataset generation."""
        gen = WorkloadGenerator()
        dataset = gen.generate_dataset(
            prompt_lengths=[128, 256],
            batch_sizes=[1, 4],
            samples_per_config=2,
        )
        
        assert len(dataset) == 4  # 2 lengths * 2 batch sizes
        assert (128, 1) in dataset
        assert (256, 4) in dataset


# Metrics computation tests
class TestMetricsComputer:
    """Test metrics computation."""
    
    def test_percentile_computation(self, sample_generate_results):
        """Test percentile computation."""
        computer = MetricsComputer()
        
        latencies = [r.total_time_ms for r in sample_generate_results]
        metrics = computer.compute_percentiles(latencies)
        
        assert metrics.p50_ms > 0
        assert metrics.p95_ms >= metrics.p50_ms
        assert metrics.p99_ms >= metrics.p95_ms
        assert metrics.min_ms <= metrics.max_ms
    
    def test_compute_metrics(self, sample_generate_results):
        """Test full metrics computation."""
        computer = MetricsComputer(gpu_hour_rate=2.5)
        
        metrics = computer.compute_metrics(
            model_name="test-model",
            backend=ModelBackend.VLLM,
            prompt_length=128,
            batch_size=1,
            max_tokens=128,
            results=sample_generate_results,
        )
        
        assert metrics.model_name == "test-model"
        assert metrics.backend == ModelBackend.VLLM
        assert metrics.num_runs == 3
        assert metrics.tokens_per_second > 0
        assert metrics.cost.cost_per_1k_tokens > 0
    
    def test_empty_results_error(self):
        """Test error on empty results."""
        computer = MetricsComputer()
        
        with pytest.raises(ValueError, match="empty results"):
            computer.compute_metrics(
                model_name="test",
                backend=ModelBackend.VLLM,
                prompt_length=128,
                batch_size=1,
                max_tokens=128,
                results=[],
            )


# BenchmarkMetrics tests
class TestBenchmarkMetrics:
    """Test BenchmarkMetrics dataclass."""
    
    def test_to_dict(self, sample_benchmark_metrics):
        """Test conversion to dictionary."""
        data = sample_benchmark_metrics.to_dict()
        
        assert data["model_name"] == "test-model"
        assert data["backend"] == "vllm"
        assert "latency" in data
        assert "throughput" in data
        assert "gpu" in data
        assert "cost" in data
    
    def test_save_json(self, sample_benchmark_metrics):
        """Test saving to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_metrics.json"
            sample_benchmark_metrics.save_json(path)
            
            assert path.exists()
            
            # Verify content
            with open(path) as f:
                data = json.load(f)
            assert data["model_name"] == "test-model"


# Formatting tests
class TestFormatting:
    """Test output formatting."""
    
    def test_format_metrics_table(self, sample_benchmark_metrics):
        """Test metrics table formatting."""
        table = format_metrics_table(sample_benchmark_metrics)
        
        assert "test-model" in table
        assert "vLLM" in table
        assert "tokens/sec" in table
        assert "ms" in table


# LatencyMetrics tests
class TestLatencyMetrics:
    """Test LatencyMetrics dataclass."""
    
    def test_latency_metrics_initialization(self):
        """Test latency metrics initialization."""
        metrics = LatencyMetrics(
            p50_ms=100.0,
            p95_ms=150.0,
            p99_ms=200.0,
            min_ms=50.0,
            max_ms=250.0,
            mean_ms=120.0,
            std_ms=30.0,
        )
        
        assert metrics.p50_ms == 100.0
        assert metrics.p95_ms == 150.0


# GPUMetrics tests
class TestGPUMetrics:
    """Test GPUMetrics dataclass."""
    
    def test_gpu_metrics_initialization(self):
        """Test GPU metrics initialization."""
        metrics = GPUMetrics(
            peak_memory_mb=16384.0,
            avg_memory_mb=8192.0,
            avg_utilization=75.0,
            memory_efficiency=500.0,
        )
        
        assert metrics.peak_memory_mb == 16384.0
        assert metrics.avg_utilization == 75.0


# CostMetrics tests
class TestCostMetrics:
    """Test CostMetrics dataclass."""
    
    def test_cost_metrics_initialization(self):
        """Test cost metrics initialization."""
        metrics = CostMetrics(
            gpu_hour_rate=3.0,
            total_gpu_hours=1.5,
            total_cost_usd=4.5,
            cost_per_1k_tokens=0.001,
        )
        
        assert metrics.gpu_hour_rate == 3.0
        assert metrics.total_cost_usd == 4.5
