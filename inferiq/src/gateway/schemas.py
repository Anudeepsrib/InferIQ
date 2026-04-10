"""Pydantic request/response schemas for the inference gateway."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class ModelBackend(str, Enum):
    """Supported inference backends."""
    VLLM = "vllm"
    NIM = "nim"
    NEMO = "nemo"


class RoutingStrategy(str, Enum):
    """Request routing strategies for multi-backend deployments."""
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    LEAST_LOADED = "least_loaded"


class ModelInfo(BaseModel):
    """Information about a registered model."""
    id: str = Field(..., description="Unique model identifier")
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    owned_by: str = Field(default="inferiq")
    backend: ModelBackend = Field(..., description="Inference backend type")
    display_name: str = Field(..., description="Human-readable model name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    tags: list[str] = Field(default_factory=list, description="Model tags")


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = Field(..., description="Model ID to use")
    prompt: str | list[str] = Field(..., description="Prompt text(s)")
    suffix: Optional[str] = Field(None, description="Suffix for infill")
    max_tokens: Optional[int] = Field(16, ge=1, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Stream response tokens")
    stop: Optional[str | list[str]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logprobs: Optional[int] = Field(None, ge=0, le=5, description="Return top logprobs")
    echo: Optional[bool] = Field(False)
    best_of: Optional[int] = Field(None, ge=0)
    logit_bias: Optional[dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)
    
    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v: str | list[str] | None) -> list[str] | None:
        """Normalize stop sequences to list."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v


class CompletionChoice(BaseModel):
    """Single completion choice."""
    text: str = Field(..., description="Generated text")
    index: int = Field(..., description="Choice index")
    logprobs: Optional[dict[str, Any]] = Field(None)
    finish_reason: Optional[str] = Field(None, description="Reason for stopping")


class CompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str = Field(..., description="Unique completion ID")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str = Field(..., description="Model ID used")
    choices: list[CompletionChoice] = Field(...)
    usage: CompletionUsage = Field(...)
    system_fingerprint: Optional[str] = Field(None)


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: Literal["system", "user", "assistant", "tool"] = Field(...)
    content: str = Field(...)
    name: Optional[str] = Field(None)
    tool_calls: Optional[list[dict[str, Any]]] = Field(None)
    tool_call_id: Optional[str] = Field(None)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model ID to use")
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(16, ge=1)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = Field(False)
    stop: Optional[str | list[str]] = Field(None)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)
    response_format: Optional[dict[str, Any]] = Field(None)
    seed: Optional[int] = Field(None)
    tools: Optional[list[dict[str, Any]]] = Field(None)
    tool_choice: Optional[str | dict[str, Any]] = Field(None)
    
    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v: str | list[str] | None) -> list[str] | None:
        """Normalize stop sequences to list."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v


class ChatCompletionChoice(BaseModel):
    """Single chat completion choice."""
    index: int = Field(...)
    message: ChatMessage = Field(...)
    finish_reason: Optional[str] = Field(None)
    logprobs: Optional[dict[str, Any]] = Field(None)


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(...)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str = Field(...)
    choices: list[ChatCompletionChoice] = Field(...)
    usage: CompletionUsage = Field(...)
    system_fingerprint: Optional[str] = Field(None)


class GenerateParams(BaseModel):
    """Internal generation parameters."""
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(-1, ge=-1)
    repetition_penalty: float = Field(1.0, ge=1.0)
    stop_sequences: list[str] = Field(default_factory=list)
    seed: Optional[int] = Field(None)
    logprobs: Optional[int] = Field(None)
    echo: bool = Field(False)


class GenerateResult(BaseModel):
    """Result of a single generation."""
    text: str = Field(..., description="Generated text")
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    ttft_ms: float = Field(..., description="Time to first token in milliseconds")
    total_time_ms: float = Field(..., description="Total generation time in milliseconds")
    tokens_per_second: float = Field(..., description="Throughput in tokens/second")
    finish_reason: str = Field(..., description="Reason for stopping")
    gpu_stats: dict[str, Any] = Field(default_factory=dict, description="GPU statistics during generation")
    logprobs: Optional[list[dict[str, Any]]] = Field(None)


class ModelListResponse(BaseModel):
    """List of available models response."""
    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(...)


class HealthStatus(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy", "degraded"] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="0.1.0")
    backends: dict[str, str] = Field(default_factory=dict, description="Per-backend health status")


class ReadyStatus(BaseModel):
    """Readiness probe response."""
    ready: bool = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    loaded_backends: list[str] = Field(default_factory=list)
    failed_backends: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: dict[str, Any] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None)


class BenchmarkResult(BaseModel):
    """Single benchmark run result."""
    model_name: str = Field(...)
    backend: ModelBackend = Field(...)
    prompt_length: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    max_tokens: int = Field(..., ge=1)
    
    # Latency metrics (milliseconds)
    ttft_p50_ms: float = Field(..., description="TTFT p50")
    ttft_p95_ms: float = Field(..., description="TTFT p95")
    ttft_p99_ms: float = Field(..., description="TTFT p99")
    total_time_p50_ms: float = Field(..., description="Total time p50")
    total_time_p95_ms: float = Field(..., description="Total time p95")
    total_time_p99_ms: float = Field(..., description="Total time p99")
    
    # Throughput metrics
    tokens_per_second: float = Field(...)
    tokens_per_second_per_gpu: Optional[float] = Field(None)
    
    # GPU metrics
    peak_gpu_memory_mb: float = Field(...)
    avg_gpu_utilization: float = Field(...)
    
    # Cost metrics
    cost_per_1k_tokens: Optional[float] = Field(None, description="USD per 1K tokens")
    
    # Run metadata
    num_runs: int = Field(..., ge=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: dict[str, Any] = Field(default_factory=dict)
    raw_results: list[GenerateResult] = Field(default_factory=list)


class GPUStats(BaseModel):
    """GPU statistics snapshot."""
    device_id: int = Field(..., ge=0)
    name: str = Field(...)
    total_memory_mb: float = Field(...)
    used_memory_mb: float = Field(...)
    free_memory_mb: float = Field(...)
    utilization_percent: float = Field(..., ge=0.0, le=100.0)
    temperature_c: Optional[float] = Field(None)
    power_draw_w: Optional[float] = Field(None)
    power_limit_w: Optional[float] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelConfig(BaseModel):
    """Model configuration from registry."""
    name: str = Field(...)
    display_name: str = Field(...)
    model_id: str = Field(...)
    backend: ModelBackend = Field(...)
    parameters: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
