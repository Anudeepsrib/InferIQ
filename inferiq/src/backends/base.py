"""Abstract Backend class for LLM inference engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.gateway.schemas import (
    GenerateParams,
    GenerateResult,
    GPUStats,
    ModelConfig,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BackendStats:
    """Runtime statistics for a backend."""
    total_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    avg_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    last_request_time: Optional[datetime] = None
    rolling_latencies: list[float] = field(default_factory=list)
    
    def record_request(self, latency_ms: float, tokens: int, success: bool = True) -> None:
        """Record a request completion."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        
        self.total_tokens_generated += tokens
        self.last_request_time = datetime.now(timezone.utc)
        
        # Update rolling average (keep last 100)
        self.rolling_latencies.append(latency_ms)
        if len(self.rolling_latencies) > 100:
            self.rolling_latencies.pop(0)
        
        self.avg_latency_ms = sum(self.rolling_latencies) / len(self.rolling_latencies)
    
    def update_memory(self, memory_mb: float) -> None:
        """Update peak memory tracking."""
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


class Backend(ABC):
    """Abstract base class for LLM inference backends.
    
    All inference backends (vLLM, NIM, NeMo) must implement this interface
    to participate in the InferIQ benchmark and gateway.
    """
    
    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize backend with model configuration.
        
        Args:
            model_config: Model configuration including backend-specific settings
        """
        self.model_config = model_config
        self.stats = BackendStats()
        self._loaded: bool = False
        self._gpu_stats: Optional[GPUStats] = None
        
        logger.info(
            "Backend initialized",
            backend=self.__class__.__name__,
            model=model_config.name,
        )
    
    @property
    def name(self) -> str:
        """Return backend identifier."""
        return self.model_config.name
    
    @property
    def loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._loaded
    
    @property
    def backend_type(self) -> str:
        """Return backend type identifier."""
        return self.model_config.backend.value
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the model into memory.
        
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        params: GenerateParams,
    ) -> GenerateResult:
        """Generate completion for a single prompt.
        
        Args:
            prompt: Input prompt text
            params: Generation parameters
            
        Returns:
            GenerateResult with generated text and timing metrics
            
        Raises:
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_batch(
        self,
        prompts: list[str],
        params: GenerateParams,
    ) -> list[GenerateResult]:
        """Generate completions for multiple prompts.
        
        Args:
            prompts: List of input prompts
            params: Generation parameters (applied to all prompts)
            
        Returns:
            List of GenerateResult, one per prompt
            
        Raises:
            RuntimeError: If batch generation fails
        """
        pass
    
    @abstractmethod
    def get_gpu_stats(self) -> Optional[GPUStats]:
        """Get current GPU statistics.
        
        Returns:
            GPUStats with memory and utilization, or None if unavailable
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy and responsive.
        
        Returns:
            True if backend is operational
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources and shutdown backend."""
        pass
    
    async def __aenter__(self) -> Backend:
        """Async context manager entry."""
        if not self._loaded:
            await self.load_model()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()
    
    def _update_stats(self, result: GenerateResult, success: bool = True) -> None:
        """Update backend statistics from generation result."""
        self.stats.record_request(
            latency_ms=result.total_time_ms,
            tokens=result.completion_tokens,
            success=success,
        )
        if result.gpu_stats:
            memory_mb = result.gpu_stats.get("used_memory_mb", 0)
            self.stats.update_memory(memory_mb)


class BackendError(Exception):
    """Base exception for backend errors."""
    
    def __init__(self, message: str, backend_name: str, original_error: Exception | None = None):
        super().__init__(message)
        self.backend_name = backend_name
        self.original_error = original_error


class ModelLoadError(BackendError):
    """Raised when model loading fails."""
    pass


class GenerationError(BackendError):
    """Raised when text generation fails."""
    pass


class HealthCheckError(BackendError):
    """Raised when health check fails."""
    pass
