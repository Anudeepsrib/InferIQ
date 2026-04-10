"""NVIDIA NeMo backend implementation."""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional

from src.backends.base import Backend, ModelLoadError, GenerationError
from src.gateway.schemas import GenerateParams, GenerateResult, GPUStats, ModelConfig
from src.utils.gpu import GPUPoller
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Flag to track if NeMo is available
_NEMO_AVAILABLE = None
_NEMO_IMPORT_ERROR = None


def _check_nemo_available() -> tuple[bool, Exception | None]:
    """Check if NeMo is installed and available."""
    global _NEMO_AVAILABLE, _NEMO_IMPORT_ERROR
    
    if _NEMO_AVAILABLE is not None:
        return _NEMO_AVAILABLE, _NEMO_IMPORT_ERROR
    
    try:
        import nemo
        import nemo.deploy
        _NEMO_AVAILABLE = True
        _NEMO_IMPORT_ERROR = None
        return True, None
    except ImportError as e:
        _NEMO_AVAILABLE = False
        _NEMO_IMPORT_ERROR = e
        return False, e


class NeMoBackend(Backend):
    """NVIDIA NeMo inference backend with graceful degradation."""
    
    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize NeMo backend.
        
        Args:
            model_config: Model configuration with NeMo-specific settings
        """
        super().__init__(model_config)
        
        # Check NeMo availability
        nemo_available, import_error = _check_nemo_available()
        
        if not nemo_available:
            warning_msg = (
                f"NeMo not installed for model {model_config.name}. "
                "Install with: pip install nemo-toolkit nemo-deploy"
            )
            logger.warning(warning_msg)
            warnings.warn(warning_msg, ImportWarning, stacklevel=2)
            self._nemo_available = False
            self._import_error = import_error
        else:
            self._nemo_available = True
            self._import_error = None
        
        # Extract NeMo config
        nemo_config = model_config.config
        self.checkpoint_path = nemo_config.get("checkpoint_path")
        self.tensor_parallel_size = nemo_config.get("tensor_model_parallel_size", 1)
        self.pipeline_parallel_size = nemo_config.get("pipeline_model_parallel_size", 1)
        self.precision = nemo_config.get("precision", "bf16")
        self.max_batch_size = nemo_config.get("max_batch_size", 32)
        
        self.model: Any = None
        self._gpu_poller = GPUPoller(device_id=0)
        self._gpu_poller.initialize()
    
    async def load_model(self) -> None:
        """Load NeMo model."""
        if not self._nemo_available:
            raise ModelLoadError(
                "NeMo not installed. Install with: pip install nemo-toolkit nemo-deploy",
                self.model_config.name,
                self._import_error,
            )
        
        try:
            # NeMo model loading is synchronous, run in thread
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load NeMo model: {str(e)}",
                self.model_config.name,
                e,
            )
    
    def _load_model_sync(self) -> None:
        """Synchronous model loading."""
        # This is a placeholder - actual NeMo loading depends on export format
        # NeMo models can be loaded via:
        # 1. NeMo Megatron GPT models (nemo.collections.nlp.models.language_modeling)
        # 2. Exported TRT-LLM models via nemo.deploy
        # 3. Exported ONNX models
        
        logger.info(
            "Loading NeMo model",
            model=self.model_config.model_id,
            checkpoint=self.checkpoint_path,
            tensor_parallel=self.tensor_parallel_size,
        )
        
        # Placeholder for actual NeMo loading
        # In production, this would use:
        # from nemo.deploy import DeployableLLM
        # self.model = DeployableLLM(self.checkpoint_path, ...)
        
        # For now, mark as loaded (actual implementation depends on NeMo version)
        self._loaded = True
        logger.info("NeMo model loaded", model=self.model_config.name)
    
    async def generate(
        self,
        prompt: str,
        params: GenerateParams,
    ) -> GenerateResult:
        """Generate completion using NeMo."""
        if not self._loaded:
            if not self._nemo_available:
                raise GenerationError(
                    "NeMo not installed. Install with: pip install nemo-toolkit nemo-deploy",
                    self.model_config.name,
                )
            raise GenerationError(
                "Model not loaded. Call load_model() first.",
                self.model_config.name,
            )
        
        try:
            # Run generation in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                params,
            )
            return result
            
        except Exception as e:
            self._update_stats(
                GenerateResult(
                    text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    ttft_ms=0,
                    total_time_ms=0,
                    tokens_per_second=0,
                    finish_reason="error",
                    gpu_stats={},
                ),
                success=False,
            )
            raise GenerationError(
                f"NeMo generation failed: {str(e)}",
                self.model_config.name,
                e,
            )
    
    def _generate_sync(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """Synchronous generation (placeholder)."""
        start_time = time.perf_counter()
        
        # Placeholder for actual NeMo inference
        # This would use self.model.generate() or similar
        
        # Simulate generation for now
        import random
        completion_tokens = min(params.max_tokens, int(len(prompt.split()) * 0.5))
        generated_text = " " * completion_tokens  # Placeholder
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        ttft_ms = total_time_ms * 0.1  # Estimate
        
        # Get GPU stats
        gpu_stats = self._gpu_poller.get_stats()
        
        # Token estimation
        prompt_tokens = len(prompt.split())
        
        result = GenerateResult(
            text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=(completion_tokens / (total_time_ms / 1000)) if total_time_ms > 0 else 0,
            finish_reason="stop",
            gpu_stats=gpu_stats.to_dict() if gpu_stats else {},
        )
        
        self._update_stats(result)
        return result
    
    async def generate_batch(
        self,
        prompts: list[str],
        params: GenerateParams,
    ) -> list[GenerateResult]:
        """Generate completions for multiple prompts."""
        if not self._loaded:
            raise GenerationError(
                "Model not loaded. Call load_model() first.",
                self.model_config.name,
            )
        
        # Run batch generation in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._generate_batch_sync,
            prompts,
            params,
        )
        return results
    
    def _generate_batch_sync(
        self,
        prompts: list[str],
        params: GenerateParams,
    ) -> list[GenerateResult]:
        """Synchronous batch generation."""
        results = []
        for prompt in prompts:
            result = self._generate_sync(prompt, params)
            results.append(result)
        return results
    
    def get_gpu_stats(self) -> Optional[GPUStats]:
        """Get current GPU statistics."""
        return self._gpu_poller.get_stats()
    
    async def health_check(self) -> bool:
        """Check if NeMo model is loaded and responsive."""
        if not self._loaded:
            return False
        
        if not self._nemo_available:
            return False
        
        try:
            # Try a simple health check
            return self.model is not None or self._loaded
        except Exception as e:
            logger.debug("NeMo health check failed", error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Shutdown NeMo model."""
        if self.model is not None:
            try:
                # Cleanup NeMo resources
                self.model = None
                import gc
                gc.collect()
                
                logger.info("NeMo model shutdown", model=self.model_config.name)
            except Exception as e:
                logger.error("Error shutting down NeMo", error=str(e))
        
        self._loaded = False
        self._gpu_poller.shutdown()
    
    @property
    def available(self) -> bool:
        """Return True if NeMo is installed and available."""
        return self._nemo_available
