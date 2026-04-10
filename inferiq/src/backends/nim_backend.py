"""NVIDIA NIM backend implementation using HTTP client."""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.backends.base import Backend, ModelLoadError, GenerationError
from src.gateway.schemas import GenerateParams, GenerateResult, GPUStats, ModelConfig
from src.utils.gpu import GPUPoller
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NIMBackend(Backend):
    """NVIDIA NIM inference backend via HTTP API."""
    
    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize NIM backend.
        
        Args:
            model_config: Model configuration with NIM-specific settings
        """
        super().__init__(model_config)
        
        # Extract NIM config
        nim_config = model_config.config
        self.base_url = nim_config.get("base_url", "http://localhost:8000")
        self.api_key = nim_config.get("api_key")
        self.timeout = nim_config.get("timeout", 120.0)
        self.max_retries = nim_config.get("max_retries", 3)
        self.retry_delay = nim_config.get("retry_delay", 1.0)
        self.health_check_interval = nim_config.get("health_check_interval", 30)
        
        self.client: Optional[httpx.AsyncClient] = None
        self._gpu_poller = GPUPoller(device_id=0)
        self._gpu_poller.initialize()
        
    async def load_model(self) -> None:
        """Initialize HTTP client and verify NIM availability."""
        try:
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=50,
                ),
            )
            
            # Verify NIM is accessible
            logger.info(
                "Connecting to NIM",
                base_url=self.base_url,
                model=self.model_config.model_id,
            )
            
            # Try health check
            if not await self.health_check():
                raise ModelLoadError(
                    f"NIM not responding at {self.base_url}",
                    self.model_config.name,
                )
            
            self._loaded = True
            logger.info(
                "NIM backend initialized",
                model=self.model_config.name,
                base_url=self.base_url,
            )
            
        except httpx.HTTPError as e:
            raise ModelLoadError(
                f"Failed to connect to NIM: {str(e)}",
                self.model_config.name,
                e,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize NIM backend: {str(e)}",
                self.model_config.name,
                e,
            )
    
    def _build_request_payload(
        self,
        prompt: str,
        params: GenerateParams,
        is_chat: bool = False,
    ) -> dict[str, Any]:
        """Build NIM API request payload."""
        payload: dict[str, Any] = {
            "model": self.model_config.model_id,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        
        if is_chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        
        if params.stop_sequences:
            payload["stop"] = params.stop_sequences
        
        if params.seed is not None:
            payload["seed"] = params.seed
        
        return payload
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _make_request(self, payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
        """Make HTTP request with retry logic."""
        if self.client is None:
            raise GenerationError("Client not initialized", self.model_config.name)
        
        response = await self.client.post(
            endpoint,
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    async def generate(
        self,
        prompt: str,
        params: GenerateParams,
    ) -> GenerateResult:
        """Generate completion via NIM API."""
        if not self._loaded or self.client is None:
            raise GenerationError(
                "NIM not initialized. Call load_model() first.",
                self.model_config.name,
            )
        
        try:
            payload = self._build_request_payload(prompt, params, is_chat=False)
            
            # Record timing
            start_time = time.perf_counter()
            
            # Make request
            result = await self._make_request(payload, "/v1/completions")
            
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Parse response (OpenAI-compatible format)
            choice = result["choices"][0]
            generated_text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "stop")
            
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            # NIM doesn't provide TTFT separately, estimate based on response time
            # For streaming, this would be actual TTFT
            ttft_ms = total_time_ms * 0.1  # Conservative estimate
            
            # Get GPU stats
            gpu_stats = self._gpu_poller.get_stats()
            
            gen_result = GenerateResult(
                text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                tokens_per_second=(completion_tokens / (total_time_ms / 1000)) if total_time_ms > 0 else 0,
                finish_reason=finish_reason,
                gpu_stats=gpu_stats.to_dict() if gpu_stats else {},
            )
            
            self._update_stats(gen_result)
            return gen_result
            
        except httpx.HTTPStatusError as e:
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
                f"NIM HTTP error: {e.response.status_code} - {e.response.text}",
                self.model_config.name,
                e,
            )
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
                f"NIM generation failed: {str(e)}",
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
                "NIM not initialized. Call load_model() first.",
                self.model_config.name,
            )
        
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, params)
            results.append(result)
        return results
    
    def get_gpu_stats(self) -> Optional[GPUStats]:
        """Get current GPU statistics."""
        return self._gpu_poller.get_stats()
    
    async def health_check(self) -> bool:
        """Check if NIM is healthy."""
        if self.client is None:
            return False
        
        try:
            response = await self.client.get("/v1/health")
            if response.status_code == 200:
                return True
            
            # Try models endpoint as fallback
            response = await self.client.get("/v1/models")
            return response.status_code == 200
            
        except Exception as e:
            logger.debug("NIM health check failed", error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Close HTTP client and cleanup."""
        if self.client is not None:
            try:
                await self.client.aclose()
                logger.info("NIM client closed", model=self.model_config.name)
            except Exception as e:
                logger.error("Error closing NIM client", error=str(e))
        
        self.client = None
        self._loaded = False
        self._gpu_poller.shutdown()
