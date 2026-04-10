"""Health and readiness probes with Prometheus metrics."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.backends.base import Backend
from src.gateway.schemas import HealthStatus, ReadyStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "inferiq_requests_total",
    "Total requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "inferiq_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

GPU_UTILIZATION = Histogram(
    "inferiq_gpu_utilization_percent",
    "GPU utilization percentage",
    ["device_id"],
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

GPU_MEMORY = Histogram(
    "inferiq_gpu_memory_used_mb",
    "GPU memory used in MB",
    ["device_id"],
    buckets=[1024, 2048, 4096, 8192, 16384, 32768, 65536]
)


class HealthManager:
    """Manages health checks and backend monitoring."""
    
    def __init__(self) -> None:
        """Initialize health manager."""
        self.backends: dict[str, Backend] = {}
        self.startup_time = datetime.utcnow()
        self._health_status: dict[str, bool] = {}
    
    def register_backend(self, name: str, backend: Backend) -> None:
        """Register a backend for health monitoring."""
        self.backends[name] = backend
        logger.info("Backend registered for health monitoring", name=name)
    
    async def check_health(self) -> HealthStatus:
        """Perform health check on all backends.
        
        Returns:
            HealthStatus with overall and per-backend health
        """
        backend_status: dict[str, str] = {}
        healthy_count = 0
        
        for name, backend in self.backends.items():
            try:
                is_healthy = await backend.health_check()
                backend_status[name] = "healthy" if is_healthy else "unhealthy"
                if is_healthy:
                    healthy_count += 1
            except Exception as e:
                backend_status[name] = "error"
                logger.warning("Health check failed", backend=name, error=str(e))
        
        # Determine overall status
        if healthy_count == len(self.backends):
            status = "healthy"
        elif healthy_count > 0:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthStatus(
            status=status,  # type: ignore
            timestamp=datetime.utcnow(),
            backends=backend_status,
        )
    
    async def check_readiness(self) -> ReadyStatus:
        """Check if all backends are loaded and ready.
        
        Returns:
            ReadyStatus with loaded and failed backends
        """
        loaded = []
        failed = []
        
        for name, backend in self.backends.items():
            try:
                is_healthy = await backend.health_check()
                if is_healthy and backend.loaded:
                    loaded.append(name)
                else:
                    failed.append(name)
            except Exception as e:
                failed.append(name)
                logger.warning("Readiness check failed", backend=name, error=str(e))
        
        return ReadyStatus(
            ready=len(failed) == 0,
            timestamp=datetime.utcnow(),
            loaded_backends=loaded,
            failed_backends=failed,
        )
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        """Record request metrics."""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint
        ).observe(latency_seconds)
    
    def record_gpu_stats(self, device_id: str, utilization: float, memory_mb: float) -> None:
        """Record GPU statistics."""
        GPU_UTILIZATION.labels(device_id=device_id).observe(utilization)
        GPU_MEMORY.labels(device_id=device_id).observe(memory_mb)


# Global health manager instance
_health_manager: HealthManager | None = None


def get_health_manager() -> HealthManager:
    """Get or create health manager instance."""
    global _health_manager
    if _health_manager is None:
        _health_manager = HealthManager()
    return _health_manager


# Create router for health endpoints
health_router = APIRouter(prefix="", tags=["health"])


@health_router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Liveness probe - returns overall health status.
    
    Returns HTTP 200 if the service is running, even if degraded.
    Returns HTTP 503 if completely unhealthy.
    """
    manager = get_health_manager()
    status = await manager.check_health()
    
    if status.status == "unhealthy":
        raise HTTPException(status_code=503, detail=status.model_dump())
    
    return status


@health_router.get("/ready", response_model=ReadyStatus)
async def readiness_check() -> ReadyStatus:
    """Readiness probe - returns whether backends are loaded.
    
    Returns HTTP 200 if all backends are ready.
    Returns HTTP 503 if any backend is not ready.
    """
    manager = get_health_manager()
    status = await manager.check_readiness()
    
    if not status.ready:
        raise HTTPException(status_code=503, detail=status.model_dump())
    
    return status


@health_router.get("/metrics")
async def metrics() -> Response:
    """Prometheus-compatible metrics endpoint."""
    from fastapi import Response
    
    metrics_data = generate_latest()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST,
    )
