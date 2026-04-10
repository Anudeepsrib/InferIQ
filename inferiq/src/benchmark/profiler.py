"""CUDA kernel profiler using torch.profiler with Chrome/NSight trace export."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KernelInfo:
    """Information about a single CUDA kernel execution."""
    name: str
    duration_us: float
    occupancy: Optional[float] = None
    grid_size: Optional[tuple[int, int, int]] = None
    block_size: Optional[tuple[int, int, int]] = None


@dataclass
class ProfileResult:
    """Result from a profiling run."""
    kernel_events: list[KernelInfo] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    total_cuda_time_ms: float = 0.0
    trace_path: Optional[Path] = None
    
    def top_kernels(self, k: int = 10) -> list[KernelInfo]:
        """Get top k most expensive kernels by duration."""
        return sorted(
            self.kernel_events,
            key=lambda x: x.duration_us,
            reverse=True,
        )[:k]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cuda_time_ms": self.total_cuda_time_ms,
            "num_kernels": len(self.kernel_events),
            "top_kernels": [
                {"name": k.name, "duration_us": k.duration_us, "occupancy": k.occupancy}
                for k in self.top_kernels(10)
            ],
            "trace_path": str(self.trace_path) if self.trace_path else None,
        }


class CUDAProfiler:
    """CUDA kernel profiler for inference benchmarks."""
    
    def __init__(
        self,
        enabled: bool = True,
        record_cuda_kernels: bool = True,
        export_chrome_trace: bool = True,
        export_nsys_format: bool = True,
        max_profiler_entries: int = 1_000_000,
        output_dir: str | Path = "results",
    ) -> None:
        """Initialize CUDA profiler.
        
        Args:
            enabled: Whether profiling is enabled
            record_cuda_kernels: Record individual CUDA kernel execution
            export_chrome_trace: Export Chrome-compatible JSON trace
            export_nsys_format: Export Nsight-compatible format
            max_profiler_entries: Maximum profiler entries to prevent OOM
            output_dir: Directory for trace outputs
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.record_cuda_kernels = record_cuda_kernels
        self.export_chrome_trace = export_chrome_trace
        self.export_nsys_format = export_nsys_format
        self.max_profiler_entries = max_profiler_entries
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._prof: Optional[profile] = None
        self._cuda_start_event: Optional[torch.cuda.Event] = None
        self._cuda_end_event: Optional[torch.cuda.Event] = None
        
        if not self.enabled:
            logger.warning("CUDA profiling disabled (CUDA not available or disabled)")
        else:
            logger.info(
                "CUDA profiler initialized",
                enabled=enabled,
                record_kernels=record_cuda_kernels,
            )
    
    def start(self) -> None:
        """Start CUDA profiling."""
        if not self.enabled:
            return
        
        try:
            activities = [ProfilerActivity.CPU]
            if self.record_cuda_kernels:
                activities.append(ProfilerActivity.CUDA)
            
            self._prof = profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                profile_memory=True,
            )
            self._prof.start()
            
            # Record CUDA events for precise timing
            self._cuda_start_event = torch.cuda.Event(enable_timing=True)
            self._cuda_end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_start_event.record()
            
            logger.debug("CUDA profiling started")
            
        except Exception as e:
            logger.error("Failed to start CUDA profiler", error=str(e))
            self.enabled = False
    
    def stop(self, trace_name: str | None = None) -> ProfileResult:
        """Stop profiling and return results.
        
        Args:
            trace_name: Name for the trace file (timestamp used if None)
            
        Returns:
            ProfileResult with kernel execution data
        """
        result = ProfileResult()
        
        if not self.enabled or self._prof is None:
            return result
        
        try:
            # Record end event
            if self._cuda_end_event:
                self._cuda_end_event.record()
                torch.cuda.synchronize()
                cuda_time_ms = self._cuda_start_event.elapsed_time(self._cuda_end_event)
                result.total_cuda_time_ms = cuda_time_ms
            
            self._prof.stop()
            
            # Process events
            kernel_events = []
            memory_events = []
            
            for event in self._prof.events():
                if event.cuda_time_total > 0:
                    kernel_events.append(KernelInfo(
                        name=event.name,
                        duration_us=event.cuda_time_total,
                    ))
                
                # Memory events
                if event.device_memory_usage > 0:
                    memory_events.append({
                        "name": event.name,
                        "device_memory_usage": event.device_memory_usage,
                        "cpu_memory_usage": event.cpu_memory_usage,
                    })
            
            result.kernel_events = kernel_events
            result.memory_events = memory_events
            
            # Export traces
            if trace_name is None:
                trace_name = f"trace_{int(time.time() * 1000)}"
            
            trace_path = self.output_dir / trace_name
            
            if self.export_chrome_trace:
                chrome_path = trace_path.with_suffix(".json")
                self._export_chrome_trace(chrome_path)
                result.trace_path = chrome_path
            
            if self.export_nsys_format:
                nsys_path = trace_path.with_suffix(".nsys-rep")
                # Note: Actual NSight export requires nsys CLI
                # We export metadata that can be converted
                self._export_nsys_metadata(nsys_path)
            
            logger.info(
                "CUDA profiling complete",
                num_kernels=len(kernel_events),
                cuda_time_ms=cuda_time_ms,
                trace_path=str(result.trace_path) if result.trace_path else None,
            )
            
        except Exception as e:
            logger.error("Error during profiling stop", error=str(e))
        
        finally:
            self._prof = None
            self._cuda_start_event = None
            self._cuda_end_event = None
        
        return result
    
    def _export_chrome_trace(self, output_path: Path) -> None:
        """Export profiler data to Chrome-compatible JSON trace format."""
        if self._prof is None:
            return
        
        try:
            self._prof.export_chrome_trace(str(output_path))
            logger.debug("Chrome trace exported", path=str(output_path))
        except Exception as e:
            logger.error("Failed to export Chrome trace", error=str(e))
    
    def _export_nsys_metadata(self, output_path: Path) -> None:
        """Export metadata compatible with NSight Systems."""
        if self._prof is None:
            return
        
        try:
            # Create metadata file that can be used with nsys CLI
            metadata = {
                "version": "1.0",
                "tool": "InferIQ CUDA Profiler",
                "timestamp": time.time(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "events": [
                    {
                        "name": e.name,
                        "cuda_time_us": e.cuda_time_total,
                        "cpu_time_us": e.cpu_time_total,
                    }
                    for e in self._prof.events()
                    if e.cuda_time_total > 0
                ],
            }
            
            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug("NSys metadata exported", path=str(output_path))
            
        except Exception as e:
            logger.error("Failed to export NSys metadata", error=str(e))
    
    def __enter__(self) -> CUDAProfiler:
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
    
    async def wrap_inference(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        trace_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ProfileResult]:
        """Wrap an inference function with profiling.
        
        Args:
            func: Async function to profile
            *args: Arguments for func
            trace_name: Name for the trace file
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (function result, ProfileResult)
        """
        self.start()
        try:
            result = await func(*args, **kwargs)
            profile_result = self.stop(trace_name=trace_name)
            return result, profile_result
        except Exception as e:
            self.stop(trace_name=trace_name)
            raise e


def profile_inference(
    enabled: bool = True,
    output_dir: str | Path = "results",
) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, tuple[Any, ProfileResult]]]]:
    """Decorator for profiling inference functions.
    
    Usage:
        @profile_inference(enabled=True)
        async def my_inference(prompt: str) -> str:
            ...
    """
    def decorator(
        func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, tuple[Any, ProfileResult]]]:
        async def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, ProfileResult]:
            profiler = CUDAProfiler(enabled=enabled, output_dir=output_dir)
            return await profiler.wrap_inference(
                func, *args, trace_name=func.__name__, **kwargs
            )
        return wrapper
    return decorator
