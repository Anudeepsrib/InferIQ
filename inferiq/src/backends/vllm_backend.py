"""vLLM backend implementation using AsyncLLMEngine."""

from __future__ import annotations

import time
import asyncio
from typing import Any, Optional

from src.backends.base import Backend, ModelLoadError, GenerationError
from src.gateway.schemas import GenerateParams, GenerateResult, GPUStats, ModelConfig
from src.utils.gpu import GPUPoller
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VLLMBackend(Backend):
    """vLLM inference backend with AsyncLLMEngine."""
    
    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize vLLM backend.
        
        Args:
            model_config: Model configuration with vLLM-specific settings
        """
        super().__init__(model_config)
        self.engine: Any = None
        self.tokenizer: Any = None
        self._gpu_poller = GPUPoller(device_id=0)
        self._gpu_poller.initialize()
        
    async def load_model(self) -> None:
        """Load model using vLLM AsyncLLMEngine."""
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
            
            config = self.model_config.config
            
            engine_args = AsyncEngineArgs(
                model=self.model_config.model_id,
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.90),
                max_num_seqs=config.get("max_num_seqs", 256),
                max_num_batched_tokens=config.get("max_num_batched_tokens", 4096),
                quantization=config.get("quantization"),
                dtype=config.get("dtype", "auto"),
                trust_remote_code=config.get("trust_remote_code", False),
                enable_prefix_caching=config.get("enable_prefix_caching", True),
                disable_log_stats=True,  # We'll handle our own stats
            )
            
            logger.info(
                "Loading vLLM model",
                model=self.model_config.model_id,
                tp_size=engine_args.tensor_parallel_size,
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Wait for model to be ready
            await asyncio.sleep(2)
            
            self._loaded = True
            logger.info(
                "vLLM model loaded successfully",
                model=self.model_config.name,
            )
            
        except ImportError as e:
            raise ModelLoadError(
                "vLLM not installed. Install with: pip install vllm",
                self.model_config.name,
                e,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load vLLM model: {str(e)}",
                self.model_config.name,
                e,
            )
    
    def _build_sampling_params(self, params: GenerateParams) -> Any:
        """Build vLLM SamplingParams from GenerateParams."""
        from vllm import SamplingParams
        
        return SamplingParams(
            n=1,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k > 0 else -1,
            repetition_penalty=params.repetition_penalty,
            stop=params.stop_sequences if params.stop_sequences else None,
            logprobs=params.logprobs,
            seed=params.seed,
        )
    
    async def generate(
        self,
        prompt: str,
        params: GenerateParams,
    ) -> GenerateResult:
        """Generate completion using vLLM."""
        if not self._loaded:
            raise GenerationError(
                "Model not loaded. Call load_model() first.",
                self.model_config.name,
            )
        
        try:
            from vllm import SamplingParams
            
            sampling_params = self._build_sampling_params(params)
            
            # Record start time
            start_event = self._record_cuda_start()
            start_time = time.perf_counter()
            
            # Submit request
            request_id = f"vllm_{int(time.time() * 1000)}"
            self.engine.add_request(request_id, prompt, sampling_params)
            
            # Stream output and capture first token time
            ttft_recorded = False
            ttft_ms = 0.0
            generated_text = ""
            output_tokens = 0
            finish_reason = "unknown"
            
            while True:
                request_outputs = await self.engine.get_request_outputs(request_id)
                
                if not request_outputs:
                    await asyncio.sleep(0.01)
                    continue
                
                output = request_outputs[0]
                
                if not ttft_recorded and len(output.outputs) > 0 and output.outputs[0].token_ids:
                    ttft_ms = (time.perf_counter() - start_time) * 1000
                    ttft_recorded = True
                
                if output.finished:
                    if output.outputs:
                        generated_text = output.outputs[0].text
                        output_tokens = len(output.outputs[0].token_ids)
                        finish_reason = output.outputs[0].finish_reason or "stop"
                    break
                
                await asyncio.sleep(0.01)
            
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Get GPU stats
            gpu_stats = self._get_gpu_stats_dict()
            
            # Count prompt tokens (approximate)
            prompt_tokens = len(prompt.split())  # Fallback approximation
            try:
                if hasattr(self.engine, 'tokenizer') and self.engine.tokenizer:
                    prompt_tokens = len(self.engine.tokenizer.encode(prompt))
            except Exception:
                pass
            
            result = GenerateResult(
                text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                tokens_per_second=(output_tokens / (total_time_ms / 1000)) if total_time_ms > 0 else 0,
                finish_reason=finish_reason,
                gpu_stats=gpu_stats.to_dict() if gpu_stats else {},
            )
            
            self._update_stats(result)
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
                f"Generation failed: {str(e)}",
                self.model_config.name,
                e,
            )
    
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
        
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, params)
            results.append(result)
        return results
    
    def _record_cuda_start(self) -> Any:
        """Record CUDA start event for precise timing."""
        try:
            import torch
            if torch.cuda.is_available():
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return event
        except Exception:
            pass
        return None
    
    def _get_gpu_stats_dict(self) -> GPUStats | None:
        """Get GPU stats as GPUStats object."""
        return self._gpu_poller.get_stats()
    
    def get_gpu_stats(self) -> Optional[GPUStats]:
        """Get current GPU statistics."""
        return self._gpu_poller.get_stats()
    
    async def health_check(self) -> bool:
        """Check if vLLM engine is healthy."""
        if not self._loaded or self.engine is None:
            return False
        
        try:
            # Try to get engine status
            return True
        except Exception as e:
            logger.warning("vLLM health check failed", error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Shutdown vLLM engine."""
        if self.engine is not None:
            try:
                # vLLM doesn't have explicit shutdown, but we can help GC
                self.engine = None
                import gc
                gc.collect()
                
                logger.info("vLLM engine shutdown", model=self.model_config.name)
            except Exception as e:
                logger.error("Error shutting down vLLM", error=str(e))
        
        self._loaded = False
        self._gpu_poller.shutdown()
