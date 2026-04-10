"""Request/response logging middleware with structured JSON and latency headers."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging with structured JSON."""
    
    def __init__(
        self,
        app: FastAPI,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ) -> None:
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.perf_counter()
        
        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else None,
        }
        
        if self.log_request_body:
            try:
                body = await request.body()
                log_data["body_size"] = len(body)
            except Exception:
                pass
        
        logger.info("Request started", **log_data)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate latency
            process_time = (time.perf_counter() - start_time) * 1000
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
            
            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                process_time_ms=f"{process_time:.2f}",
                response_size=response.headers.get("content-length"),
            )
            
            return response
            
        except Exception as e:
            process_time = (time.perf_counter() - start_time) * 1000
            
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                process_time_ms=f"{process_time:.2f}",
            )
            raise


class LatencyHeaderMiddleware(BaseHTTPMiddleware):
    """Add latency headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add inference timing headers."""
        response = await call_next(request)
        
        # Add inference-specific headers if available in state
        if hasattr(request.state, "inference_time_ms"):
            response.headers["X-Inference-Time-Ms"] = f"{request.state.inference_time_ms:.2f}"
        
        if hasattr(request.state, "ttft_ms"):
            response.headers["X-TTFT-Ms"] = f"{request.state.ttft_ms:.2f}"
        
        if hasattr(request.state, "tokens_generated"):
            response.headers["X-Tokens-Generated"] = str(request.state.tokens_generated)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-model rate limiting."""
    
    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = 1000,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}
        self._window_seconds = 60.0
    
    def _is_rate_limited(self, key: str) -> bool:
        """Check if key is rate limited."""
        now = time.time()
        
        # Clean old entries
        if key in self._requests:
            self._requests[key] = [
                t for t in self._requests[key]
                if now - t < self._window_seconds
            ]
        
        # Check limit
        if key not in self._requests:
            self._requests[key] = []
        
        if len(self._requests[key]) >= self.requests_per_minute:
            return True
        
        self._requests[key].append(now)
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        # Create rate limit key from client + model (if in request body)
        client = request.client.host if request.client else "unknown"
        key = f"{client}"
        
        if self._is_rate_limited(key):
            from fastapi import HTTPException
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        return await call_next(request)


def setup_middleware(app: FastAPI, rate_limit: int = 1000) -> None:
    """Configure all middleware for the application."""
    # Order matters - first added is first executed
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(LatencyHeaderMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)
