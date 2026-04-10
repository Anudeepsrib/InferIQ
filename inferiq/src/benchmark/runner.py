"""Benchmark runner orchestrating model sweeps with warmup and measurement."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.backends.base import Backend
from src.backends.vllm_backend import VLLMBackend
from src.backends.nim_backend import NIMBackend
from src.backends.nemo_backend import NeMoBackend
from src.benchmark.metrics import BenchmarkMetrics, MetricsComputer, format_metrics_table
from src.benchmark.profiler import CUDAProfiler, ProfileResult
from src.benchmark.workloads import WorkloadGenerator
from src.config.settings import BenchmarkConfig, ModelConfig, Settings, get_settings
from src.gateway.schemas import GenerateParams, GenerateResult, GPUStats, ModelBackend
from src.utils.gpu import GPUMonitor
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


class BenchmarkRunner:
    """Orchestrates benchmark sweeps across models and configurations."""
    
    BACKEND_MAP: dict[ModelBackend, type[Backend]] = {
        ModelBackend.VLLM: VLLMBackend,
        ModelBackend.NIM: NIMBackend,
        ModelBackend.NEMO: NeMoBackend,
    }
    
    def __init__(self, config: BenchmarkConfig | None = None, settings: Settings | None = None) -> None:
        """Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration (loads from file if None)
            settings: Application settings (creates new if None)
        """
        self.settings = settings or get_settings()
        self.config = config or self.settings.benchmark
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.workload_gen = WorkloadGenerator()
        self.metrics_computer = MetricsComputer(
            gpu_hour_rate=self.config.gpu_pricing.get("default", 2.0)
        )
        self.gpu_monitor = GPUMonitor()
        
        # State file for resumption
        self.state_file = self.output_dir / self.config.resume_state_file
        self.completed_runs: set[str] = set()
        
        if self.config.resume and self.state_file.exists():
            self._load_state()
    
    def _load_state(self) -> None:
        """Load completed runs from state file."""
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                self.completed_runs = set(state.get("completed", []))
            logger.info("Loaded resume state", completed=len(self.completed_runs))
        except Exception as e:
            logger.warning("Failed to load state file", error=str(e))
    
    def _save_state(self) -> None:
        """Save completed runs to state file."""
        state = {
            "completed": list(self.completed_runs),
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)
    
    def _get_run_id(
        self,
        model: ModelConfig,
        prompt_length: int,
        batch_size: int,
    ) -> str:
        """Generate unique identifier for a benchmark run."""
        return f"{model.name}_{model.backend.value}_{prompt_length}_{batch_size}"
    
    def _is_completed(self, run_id: str) -> bool:
        """Check if a run is already completed."""
        return run_id in self.completed_runs
    
    def _mark_completed(self, run_id: str) -> None:
        """Mark a run as completed."""
        self.completed_runs.add(run_id)
        self._save_state()
    
    def _get_backend(self, model_config: ModelConfig) -> Backend:
        """Create backend instance for model."""
        backend_class = self.BACKEND_MAP.get(model_config.backend)
        if backend_class is None:
            raise ValueError(f"Unknown backend: {model_config.backend}")
        
        return backend_class(model_config)
    
    async def _run_single_benchmark(
        self,
        backend: Backend,
        prompt: str,
        max_tokens: int,
        gpu_stats_during_run: list[GPUStats],
    ) -> GenerateResult:
        """Execute a single benchmark run."""
        params = GenerateParams(max_tokens=max_tokens)
        
        # Run with profiling if enabled
        if self.config.profiling.enabled:
            profiler = CUDAProfiler(
                enabled=True,
                record_cuda_kernels=self.config.profiling.record_cuda_kernels,
                export_chrome_trace=self.config.profiling.export_chrome_trace,
                export_nsys_format=self.config.profiling.export_nsys_format,
                max_profiler_entries=self.config.profiling.max_profiler_entries,
                output_dir=self.output_dir,
            )
            
            with profiler:
                result = await backend.generate(prompt, params)
            
            # GPU polling during profiling
            gpu_stat = self.gpu_monitor.get_all_stats()
            if gpu_stat:
                gpu_stats_during_run.extend(gpu_stat)
        else:
            result = await backend.generate(prompt, params)
            
            # GPU polling
            gpu_stat = self.gpu_monitor.get_all_stats()
            if gpu_stat:
                gpu_stats_during_run.extend(gpu_stat)
        
        return result
    
    async def _run_benchmark_config(
        self,
        model: ModelConfig,
        prompt_length: int,
        batch_size: int,
        max_tokens: int,
    ) -> BenchmarkMetrics | None:
        """Run benchmark for a single configuration."""
        run_id = self._get_run_id(model, prompt_length, batch_size)
        
        if self._is_completed(run_id):
            logger.info("Skipping completed run", run_id=run_id)
            return None
        
        logger.info(
            "Running benchmark",
            model=model.name,
            prompt_length=prompt_length,
            batch_size=batch_size,
        )
        
        # Initialize backend
        backend = self._get_backend(model)
        
        try:
            await backend.load_model()
            
            # Warmup runs
            logger.info(f"Running {self.config.warmup_runs} warmup iterations")
            warmup_prompt = self.workload_gen.generate_prompt(prompt_length)
            for i in range(self.config.warmup_runs):
                await backend.generate(warmup_prompt, GenerateParams(max_tokens=max_tokens))
            
            # Measured runs
            logger.info(f"Running {self.config.measured_runs} measured iterations")
            results: list[GenerateResult] = []
            gpu_stats_during_run: list[GPUStats] = []
            
            for i in range(self.config.measured_runs):
                # Generate batch of prompts
                prompts = self.workload_gen.generate_batch(prompt_length, batch_size)
                
                for prompt in prompts:
                    result = await self._run_single_benchmark(
                        backend, prompt, max_tokens, gpu_stats_during_run
                    )
                    results.append(result)
                
                logger.debug(f"Completed run {i + 1}/{self.config.measured_runs}")
            
            # Compute metrics
            gpu_stats_dicts = [s.to_dict() for s in gpu_stats_during_run]
            metrics = self.metrics_computer.compute_metrics(
                model_name=model.name,
                backend=model.backend,
                prompt_length=prompt_length,
                batch_size=batch_size,
                max_tokens=max_tokens,
                results=results,
                gpu_stats_list=gpu_stats_dicts,
            )
            
            # Save results
            await self._save_results(metrics, model, prompt_length, batch_size)
            
            # Mark as completed
            self._mark_completed(run_id)
            
            return metrics
            
        except Exception as e:
            logger.error(
                "Benchmark run failed",
                run_id=run_id,
                error=str(e),
                exc_info=True,
            )
            return None
        
        finally:
            await backend.shutdown()
    
    async def _save_results(
        self,
        metrics: BenchmarkMetrics,
        model: ModelConfig,
        prompt_length: int,
        batch_size: int,
    ) -> None:
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = self.config.result_filename_template.format(
            model=model.name,
            backend=model.backend.value,
            prompt_len=prompt_length,
            batch_size=batch_size,
            timestamp=timestamp,
        )
        
        output_path = self.output_dir / filename
        metrics.save_json(output_path)
        
        # Also print to console
        console.print(format_metrics_table(metrics))
    
    async def run(
        self,
        models: list[ModelConfig] | None = None,
        prompt_lengths: list[int] | None = None,
        batch_sizes: list[int] | None = None,
        max_tokens: int | None = None,
    ) -> list[BenchmarkMetrics]:
        """Run full benchmark sweep.
        
        Args:
            models: List of models to benchmark (uses defaults if None)
            prompt_lengths: Prompt lengths to test
            batch_sizes: Batch sizes to test
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of benchmark metrics for all completed runs
        """
        # Use defaults if not specified
        if models is None:
            models = [
                m for m in self.settings.models
                if m.name in self.settings.default_models
            ]
        
        prompt_lengths = prompt_lengths or self.config.prompt_lengths
        batch_sizes = batch_sizes or self.config.batch_sizes
        max_tokens = max_tokens or self.config.max_tokens[0]
        
        all_metrics: list[BenchmarkMetrics] = []
        total_runs = len(models) * len(prompt_lengths) * len(batch_sizes)
        
        console.print(f"[bold blue]Starting benchmark sweep: {total_runs} configurations[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total_runs)
            
            for model in models:
                for prompt_length in prompt_lengths:
                    for batch_size in batch_sizes:
                        progress.update(
                            task,
                            description=f"{model.name} @ {prompt_length} tokens, batch={batch_size}"
                        )
                        
                        metrics = await self._run_benchmark_config(
                            model, prompt_length, batch_size, max_tokens
                        )
                        
                        if metrics:
                            all_metrics.append(metrics)
                        
                        progress.advance(task)
        
        console.print(f"[bold green]Benchmark complete: {len(all_metrics)}/{total_runs} runs successful[/bold green]")
        
        return all_metrics


async def run_benchmark(
    config_path: str | None = None,
    models: list[str] | None = None,
) -> list[BenchmarkMetrics]:
    """Entry point for running benchmarks.
    
    Args:
        config_path: Path to benchmark config YAML
        models: Specific model names to benchmark
        
    Returns:
        List of benchmark metrics
    """
    settings = get_settings()
    
    if config_path:
        settings.benchmark_config_file = config_path
        settings.load_config_files()
    
    runner = BenchmarkRunner(config=settings.benchmark)
    
    # Filter models if specified
    model_configs = None
    if models:
        model_configs = [m for m in settings.models if m.name in models]
    
    return await runner.run(models=model_configs)
