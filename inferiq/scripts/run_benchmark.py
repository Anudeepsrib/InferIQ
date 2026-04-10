"""CLI entrypoint for running benchmarks."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.benchmark.runner import run_benchmark
from src.config.settings import get_settings

console = Console()
app = typer.Typer(help="InferIQ Benchmark Runner")


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to benchmark configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    models: list[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Specific model(s) to benchmark (can specify multiple)",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results (overrides config)",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume from previous run if available",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Run InferIQ benchmark suite.
    
    Examples:
        # Run with default config
        python scripts/run_benchmark.py
        
        # Run with custom config
        python scripts/run_benchmark.py --config configs/custom.yaml
        
        # Run specific models
        python scripts/run_benchmark.py --model mistral-7b-instruct-vllm --model llama-3-8b-nim
        
        # Resume from previous run
        python scripts/run_benchmark.py --resume
    """
    # Display header
    header = Text()
    header.append("⚡ ", style="bold yellow")
    header.append("InferIQ Benchmark Runner\n", style="bold blue")
    header.append("GPU-optimized LLM inference benchmarking", style="dim")
    
    console.print(Panel(header, border_style="blue"))
    
    # Load settings
    settings = get_settings()
    
    # Override config if specified
    if config != Path("configs/default.yaml"):
        settings.benchmark_config_file = str(config)
        settings.load_config_files()
    
    # Override output directory if specified
    if output_dir:
        settings.benchmark.output_dir = str(output_dir)
    
    # Override resume setting
    settings.benchmark.resume = resume
    
    # Set logging level
    if verbose:
        settings.benchmark.logging.level = "DEBUG"
    
    # Display configuration
    console.print(f"[dim]Config:[/dim] {config}")
    console.print(f"[dim]Output:[/dim] {settings.benchmark.output_dir}")
    console.print(f"[dim]Resume:[/dim] {resume}")
    
    if models:
        console.print(f"[dim]Models:[/dim] {', '.join(models)}")
    else:
        console.print(f"[dim]Models:[/dim] {', '.join(settings.default_models)}")
    
    console.print()
    
    # Run benchmark
    try:
        results = asyncio.run(run_benchmark(
            config_path=str(config) if config else None,
            models=models if models else None,
        ))
        
        # Summary
        console.print()
        console.print(Panel(
            f"[bold green]Benchmark Complete![/bold green]\n"
            f"Completed {len(results)} configurations successfully",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"\n[bold red]Benchmark failed:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
