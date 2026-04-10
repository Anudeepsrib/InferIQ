"""pytest configuration and shared fixtures."""

import pytest
from pathlib import Path
import tempfile
import json


@pytest.fixture
def temp_results_dir():
    """Provide a temporary directory for benchmark results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model_config():
    """Provide a sample model configuration."""
    return {
        "name": "test-model-vllm",
        "display_name": "Test Model (vLLM)",
        "model_id": "test/model-id",
        "backend": "vllm",
        "parameters": {
            "size": "7B",
            "architecture": "transformer",
            "context_length": 4096,
        },
        "config": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
        },
        "tags": ["test", "vllm"],
    }


@pytest.fixture
def sample_benchmark_config():
    """Provide a sample benchmark configuration."""
    return {
        "benchmark": {
            "name": "test_benchmark",
            "prompt_lengths": [128, 256],
            "batch_sizes": [1, 4],
            "max_tokens": [128],
            "warmup_runs": 1,
            "measured_runs": 2,
            "output_dir": "results",
            "resume": False,
        }
    }


@pytest.fixture
def mock_gpu_stats():
    """Provide mock GPU statistics."""
    return {
        "device_id": 0,
        "name": "NVIDIA A100-PCIE-40GB",
        "total_memory_mb": 40960,
        "used_memory_mb": 20480,
        "free_memory_mb": 20480,
        "utilization_percent": 75.5,
        "temperature_c": 65,
        "power_draw_w": 250.5,
        "power_limit_w": 400.0,
    }
