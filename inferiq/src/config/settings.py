"""Configuration loader using Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.gateway.schemas import ModelBackend, ModelConfig


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: Optional[str] = Field(default=None)


class TimeoutConfig(BaseModel):
    """Timeout settings."""
    model_load: int = Field(default=300)
    inference: int = Field(default=120)
    health_check: int = Field(default=30)


class ProfilingConfig(BaseModel):
    """CUDA profiling configuration."""
    enabled: bool = Field(default=True)
    record_cuda_kernels: bool = Field(default=True)
    export_chrome_trace: bool = Field(default=True)
    export_nsys_format: bool = Field(default=True)
    max_profiler_entries: int = Field(default=1_000_000)


class BenchmarkConfig(BaseModel):
    """Benchmark runner configuration."""
    name: str = Field(default="default_benchmark")
    description: str = Field(default="")
    prompt_lengths: list[int] = Field(default_factory=lambda: [128, 256, 512, 1024])
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 4, 8, 16, 32])
    max_tokens: list[int] = Field(default_factory=lambda: [128, 256])
    warmup_runs: int = Field(default=3, ge=0)
    measured_runs: int = Field(default=10, ge=1)
    output_dir: str = Field(default="results")
    result_filename_template: str = Field(
        default="{model}_{backend}_{prompt_len}_{batch_size}_{timestamp}.json"
    )
    resume: bool = Field(default=True)
    resume_state_file: str = Field(default=".benchmark_state.json")
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    gpu_poll_interval: float = Field(default=0.1, ge=0.01)
    gpu_pricing: dict[str, float] = Field(default_factory=dict)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("prompt_lengths", "batch_sizes", "max_tokens")
    @classmethod
    def validate_positive_ints(cls, v: list[int]) -> list[int]:
        """Ensure all values are positive integers."""
        if not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("All values must be positive integers")
        return v


class VLLMBackendConfig(BaseModel):
    """vLLM-specific configuration."""
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.90, ge=0.0, le=1.0)
    max_num_seqs: int = Field(default=256, ge=1)
    max_num_batched_tokens: int = Field(default=4096, ge=1)
    quantization: Optional[str] = Field(default=None)
    dtype: str = Field(default="auto")
    trust_remote_code: bool = Field(default=False)


class NIMBackendConfig(BaseModel):
    """NVIDIA NIM-specific configuration."""
    base_url: str = Field(default="http://localhost:8000")
    api_key: Optional[str] = Field(default=None)
    timeout: float = Field(default=120.0, ge=0.0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    health_check_interval: int = Field(default=30, ge=1)


class NeMoBackendConfig(BaseModel):
    """NeMo-specific configuration."""
    checkpoint_path: Optional[str] = Field(default=None)
    tensor_model_parallel_size: int = Field(default=1, ge=1)
    pipeline_model_parallel_size: int = Field(default=1, ge=1)
    precision: str = Field(default="bf16")
    max_batch_size: int = Field(default=32, ge=1)


class BackendsConfig(BaseModel):
    """All backend configurations."""
    vllm: VLLMBackendConfig = Field(default_factory=VLLMBackendConfig)
    nim: NIMBackendConfig = Field(default_factory=NIMBackendConfig)
    nemo: NeMoBackendConfig = Field(default_factory=NeMoBackendConfig)


class GatewayConfig(BaseModel):
    """Gateway server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")
    routing_strategy: str = Field(default="least_latency")
    rate_limit_requests_per_minute: int = Field(default=1000, ge=1)
    request_timeout: float = Field(default=120.0, ge=0.0)


class Settings(BaseSettings):
    """Application settings loaded from environment and config files."""
    
    model_config = SettingsConfigDict(
        env_prefix="INFERIQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Paths
    config_dir: Path = Field(default=Path("configs"))
    results_dir: Path = Field(default=Path("results"))
    
    # Config files
    benchmark_config_file: str = Field(default="default.yaml")
    models_config_file: str = Field(default="models.yaml")
    
    # Sub-configs (loaded from YAML)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    backends: BackendsConfig = Field(default_factory=BackendsConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    
    # Model registry
    models: list[ModelConfig] = Field(default_factory=list)
    default_models: list[str] = Field(default_factory=list)
    
    # Runtime settings
    cuda_visible_devices: Optional[str] = Field(default=None)
    torch_compile: bool = Field(default=False)
    
    @field_validator("config_dir", "results_dir", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Ensure path is a Path object."""
        return Path(v) if isinstance(v, str) else v
    
    def load_config_files(self) -> None:
        """Load configuration from YAML files."""
        # Load benchmark config
        benchmark_path = self.config_dir / self.benchmark_config_file
        if benchmark_path.exists():
            with open(benchmark_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "benchmark" in data:
                    self.benchmark = BenchmarkConfig(**data["benchmark"])
                if data and "backends" in data:
                    self.backends = BackendsConfig(**data["backends"])
        
        # Load models config
        models_path = self.config_dir / self.models_config_file
        if models_path.exists():
            with open(models_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "registry" in data:
                    self.models = [ModelConfig(**m) for m in data["registry"]]
                if data and "defaults" in data:
                    defaults = data["defaults"]
                    if "benchmark_models" in defaults:
                        self.default_models = defaults["benchmark_models"]
    
    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        for model in self.models:
            if model.name == name:
                return model
        return None
    
    def get_models_by_backend(self, backend: ModelBackend) -> list[ModelConfig]:
        """Get all models for a specific backend."""
        return [m for m in self.models if m.backend == backend]
    
    def get_backend_config(self, backend: ModelBackend) -> BaseModel:
        """Get configuration for a specific backend."""
        config_map = {
            ModelBackend.VLLM: self.backends.vllm,
            ModelBackend.NIM: self.backends.nim,
            ModelBackend.NEMO: self.backends.nemo,
        }
        return config_map.get(backend, self.backends.vllm)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.load_config_files()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from config files."""
    global _settings
    _settings = Settings()
    _settings.load_config_files()
    return _settings
