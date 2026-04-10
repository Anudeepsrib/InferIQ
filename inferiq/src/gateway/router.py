"""Dynamic model routing logic with load balancing strategies."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from src.backends.base import Backend
from src.gateway.schemas import GenerateParams, GenerateResult, ModelConfig, RoutingStrategy
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BackendInstance:
    """A backend instance with routing metadata."""
    backend: Backend
    model_config: ModelConfig
    request_count: int = 0
    total_latency_ms: float = 0.0
    current_load: int = 0
    errors: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.request_count == 0:
            return 0.0
        return self.errors / self.request_count
    
    def record_request(self, latency_ms: float, success: bool = True) -> None:
        """Record request metrics."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.current_load -= 1
        if not success:
            self.errors += 1
    
    def increment_load(self) -> None:
        """Increment current load counter."""
        self.current_load += 1


class ModelRouter:
    """Dynamic model router with multiple routing strategies."""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.LEAST_LATENCY) -> None:
        """Initialize router.
        
        Args:
            strategy: Routing strategy for selecting backends
        """
        self.strategy = strategy
        self.backends: dict[str, list[BackendInstance]] = {}
        self._round_robin_index: dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    def register_backend(self, backend: Backend, model_config: ModelConfig) -> None:
        """Register a backend for a model."""
        model_name = model_config.name
        
        if model_name not in self.backends:
            self.backends[model_name] = []
            self._round_robin_index[model_name] = 0
        
        instance = BackendInstance(backend=backend, model_config=model_config)
        self.backends[model_name].append(instance)
        
        logger.info(
            "Backend registered",
            model=model_name,
            backend=backend.__class__.__name__,
            instance_count=len(self.backends[model_name]),
        )
    
    def unregister_backend(self, backend: Backend) -> None:
        """Unregister a backend."""
        for model_name, instances in self.backends.items():
            self.backends[model_name] = [
                inst for inst in instances if inst.backend != backend
            ]
    
    async def get_backend(self, model_name: str) -> BackendInstance | None:
        """Get backend for model using routing strategy."""
        if model_name not in self.backends or not self.backends[model_name]:
            return None
        
        instances = self.backends[model_name]
        
        # Filter out unhealthy backends
        healthy_instances = [inst for inst in instances if await inst.backend.health_check()]
        
        if not healthy_instances:
            logger.warning("No healthy backends available", model=model_name)
            # Fall back to any backend and hope for the best
            healthy_instances = instances
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(model_name, healthy_instances)
        
        elif self.strategy == RoutingStrategy.LEAST_LATENCY:
            return self._least_latency_select(healthy_instances)
        
        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_select(healthy_instances)
        
        else:
            # Default to round robin
            return self._round_robin_select(model_name, healthy_instances)
    
    def _round_robin_select(
        self,
        model_name: str,
        instances: list[BackendInstance],
    ) -> BackendInstance:
        """Select backend using round-robin."""
        index = self._round_robin_index[model_name]
        instance = instances[index % len(instances)]
        self._round_robin_index[model_name] = (index + 1) % len(instances)
        return instance
    
    def _least_latency_select(self, instances: list[BackendInstance]) -> BackendInstance:
        """Select backend with lowest average latency."""
        return min(instances, key=lambda x: x.avg_latency_ms)
    
    def _least_loaded_select(self, instances: list[BackendInstance]) -> BackendInstance:
        """Select backend with lowest current load."""
        return min(instances, key=lambda x: x.current_load)
    
    async def route_generate(
        self,
        model_name: str,
        prompt: str,
        params: GenerateParams,
    ) -> tuple[GenerateResult, BackendInstance]:
        """Route generation request and return result with backend info."""
        instance = await self.get_backend(model_name)
        
        if instance is None:
            raise ValueError(f"No backend available for model: {model_name}")
        
        # Track load
        instance.increment_load()
        
        try:
            result = await instance.backend.generate(prompt, params)
            instance.record_request(result.total_time_ms, success=True)
            return result, instance
        except Exception as e:
            instance.record_request(0, success=False)
            raise e
    
    async def route_generate_batch(
        self,
        model_name: str,
        prompts: list[str],
        params: GenerateParams,
    ) -> tuple[list[GenerateResult], BackendInstance]:
        """Route batch generation request."""
        instance = await self.get_backend(model_name)
        
        if instance is None:
            raise ValueError(f"No backend available for model: {model_name}")
        
        instance.increment_load()
        
        try:
            results = await instance.backend.generate_batch(prompts, params)
            total_latency = sum(r.total_time_ms for r in results)
            instance.record_request(total_latency / len(results), success=True)
            return results, instance
        except Exception as e:
            instance.record_request(0, success=False)
            raise e
    
    def get_model_list(self) -> list[ModelConfig]:
        """Get list of available models."""
        configs = []
        for instances in self.backends.values():
            if instances:
                configs.append(instances[0].model_config)
        return configs
    
    def get_backend_stats(self) -> dict[str, list[dict[str, Any]]]:
        """Get statistics for all backends."""
        stats = {}
        for model_name, instances in self.backends.items():
            stats[model_name] = [
                {
                    "backend_type": inst.backend.__class__.__name__,
                    "request_count": inst.request_count,
                    "avg_latency_ms": inst.avg_latency_ms,
                    "current_load": inst.current_load,
                    "error_rate": inst.error_rate,
                }
                for inst in instances
            ]
        return stats
