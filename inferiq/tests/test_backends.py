"""Tests for backend implementations."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.backends.base import Backend, BackendStats, ModelLoadError, GenerationError
from src.backends.vllm_backend import VLLMBackend
from src.backends.nim_backend import NIMBackend
from src.backends.nemo_backend import NeMoBackend, _check_nemo_available
from src.gateway.schemas import (
    GenerateParams,
    GenerateResult,
    GPUStats,
    ModelConfig,
    ModelBackend,
)


# Fixtures
@pytest.fixture
def mock_model_config():
    """Create mock model config."""
    return ModelConfig(
        name="test-model",
        display_name="Test Model",
        model_id="test/model",
        backend=ModelBackend.VLLM,
        parameters={"size": "7B"},
        config={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
        tags=["test"],
    )


@pytest.fixture
def mock_nim_config():
    """Create mock NIM config."""
    return ModelConfig(
        name="test-nim-model",
        display_name="Test NIM Model",
        model_id="test/nim-model",
        backend=ModelBackend.NIM,
        parameters={"size": "8B"},
        config={"base_url": "http://localhost:8000", "timeout": 60.0},
        tags=["test", "nim"],
    )


@pytest.fixture
def mock_nemo_config():
    """Create mock NeMo config."""
    return ModelConfig(
        name="test-nemo-model",
        display_name="Test NeMo Model",
        model_id="test/nemo-model",
        backend=ModelBackend.NEMO,
        parameters={"size": "8B"},
        config={"tensor_model_parallel_size": 1, "precision": "bf16"},
        tags=["test", "nemo"],
    )


# Backend base class tests
class TestBackendBase:
    """Test base backend class."""
    
    def test_backend_stats_initialization(self):
        """Test BackendStats initialization."""
        stats = BackendStats()
        assert stats.total_requests == 0
        assert stats.failed_requests == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.error_rate == 0.0
    
    def test_backend_stats_record_request(self):
        """Test recording requests."""
        stats = BackendStats()
        stats.record_request(100.0, 50, success=True)
        
        assert stats.total_requests == 1
        assert stats.failed_requests == 0
        assert stats.avg_latency_ms == 100.0
        assert stats.total_tokens_generated == 50
        
        # Record failure
        stats.record_request(0, 0, success=False)
        assert stats.total_requests == 2
        assert stats.failed_requests == 1
        assert stats.error_rate == 0.5
    
    def test_backend_stats_memory_update(self):
        """Test memory tracking."""
        stats = BackendStats()
        stats.update_memory(1024.0)
        assert stats.peak_memory_mb == 1024.0
        
        stats.update_memory(2048.0)
        assert stats.peak_memory_mb == 2048.0
        
        stats.update_memory(512.0)
        assert stats.peak_memory_mb == 2048.0  # Should not decrease


# Backend exceptions tests
class TestBackendExceptions:
    """Test backend exceptions."""
    
    def test_model_load_error(self):
        """Test ModelLoadError creation."""
        error = ModelLoadError("Failed to load", "test-model")
        assert str(error) == "Failed to load"
        assert error.backend_name == "test-model"
        assert error.original_error is None
    
    def test_model_load_error_with_original(self):
        """Test ModelLoadError with original error."""
        original = ValueError("Original error")
        error = ModelLoadError("Failed to load", "test-model", original)
        assert error.original_error == original
    
    def test_generation_error(self):
        """Test GenerationError creation."""
        error = GenerationError("Generation failed", "test-model")
        assert str(error) == "Generation failed"
        assert error.backend_name == "test-model"


# vLLM backend tests
class TestVLLMBackend:
    """Test vLLM backend implementation."""
    
    @pytest.mark.asyncio
    async def test_vllm_backend_initialization(self, mock_model_config):
        """Test vLLM backend initialization."""
        backend = VLLMBackend(mock_model_config)
        assert backend.model_config == mock_model_config
        assert backend.name == "test-model"
        assert not backend.loaded
        assert backend.backend_type == "vllm"
    
    @pytest.mark.asyncio
    async def test_vllm_generate_without_load(self, mock_model_config):
        """Test generation without loading model."""
        backend = VLLMBackend(mock_model_config)
        
        with pytest.raises(GenerationError):
            await backend.generate("Hello", GenerateParams(max_tokens=10))
    
    @pytest.mark.asyncio
    async def test_vllm_batch_generate_without_load(self, mock_model_config):
        """Test batch generation without loading model."""
        backend = VLLMBackend(mock_model_config)
        
        with pytest.raises(GenerationError):
            await backend.generate_batch(["Hello", "World"], GenerateParams(max_tokens=10))


# NIM backend tests
class TestNIMBackend:
    """Test NVIDIA NIM backend implementation."""
    
    @pytest.mark.asyncio
    async def test_nim_backend_initialization(self, mock_nim_config):
        """Test NIM backend initialization."""
        backend = NIMBackend(mock_nim_config)
        assert backend.model_config == mock_nim_config
        assert backend.base_url == "http://localhost:8000"
        assert backend.timeout == 60.0
    
    @pytest.mark.asyncio
    async def test_nim_generate_without_load(self, mock_nim_config):
        """Test generation without initialization."""
        backend = NIMBackend(mock_nim_config)
        
        with pytest.raises(GenerationError):
            await backend.generate("Hello", GenerateParams(max_tokens=10))


# NeMo backend tests
class TestNeMoBackend:
    """Test NeMo backend implementation."""
    
    def test_nemo_availability_check(self):
        """Test NeMo availability check function."""
        available, error = _check_nemo_available()
        # Result depends on whether NeMo is installed
        # Just verify function runs without error
        assert isinstance(available, bool)
    
    @pytest.mark.asyncio
    async def test_nemo_backend_initialization(self, mock_nemo_config):
        """Test NeMo backend initialization."""
        backend = NeMoBackend(mock_nemo_config)
        assert backend.model_config == mock_nemo_config
        assert backend.tensor_parallel_size == 1
        assert backend.precision == "bf16"
    
    def test_nemo_available_property(self, mock_nemo_config):
        """Test NeMo available property."""
        backend = NeMoBackend(mock_nemo_config)
        # Property should return whether NeMo is available
        assert isinstance(backend.available, bool)
    
    @pytest.mark.asyncio
    async def test_nemo_generate_without_nemo(self, mock_nemo_config):
        """Test generation when NeMo not available."""
        backend = NeMoBackend(mock_nemo_config)
        
        if not backend.available:
            with pytest.raises(GenerationError):
                await backend.generate("Hello", GenerateParams(max_tokens=10))


# Backend interface compliance tests
class TestBackendInterface:
    """Test that all backends implement the interface correctly."""
    
    def test_all_backends_inherit_from_base(self):
        """Test that all backends inherit from Backend."""
        assert issubclass(VLLMBackend, Backend)
        assert issubclass(NIMBackend, Backend)
        assert issubclass(NeMoBackend, Backend)
    
    def test_backend_abstract_methods(self, mock_model_config):
        """Test that abstract methods are implemented."""
        backend = VLLMBackend(mock_model_config)
        
        # Check required methods exist
        assert hasattr(backend, 'load_model')
        assert hasattr(backend, 'generate')
        assert hasattr(backend, 'generate_batch')
        assert hasattr(backend, 'get_gpu_stats')
        assert hasattr(backend, 'health_check')
        assert hasattr(backend, 'shutdown')
        
        # Check they are coroutines
        import inspect
        assert inspect.iscoroutinefunction(backend.load_model)
        assert inspect.iscoroutinefunction(backend.generate)
        assert inspect.iscoroutinefunction(backend.generate_batch)
        assert inspect.iscoroutinefunction(backend.health_check)
        assert inspect.iscoroutinefunction(backend.shutdown)


# GPU stats tests
class TestGPUStats:
    """Test GPU statistics functionality."""
    
    def test_gpu_stats_creation(self):
        """Test GPUStats creation."""
        stats = GPUStats(
            device_id=0,
            name="NVIDIA A100",
            total_memory_mb=40960.0,
            used_memory_mb=20480.0,
            free_memory_mb=20480.0,
            utilization_percent=75.5,
            temperature_c=65.0,
            power_draw_w=250.0,
            power_limit_w=400.0,
        )
        
        assert stats.device_id == 0
        assert stats.name == "NVIDIA A100"
        assert stats.utilization_percent == 75.5
