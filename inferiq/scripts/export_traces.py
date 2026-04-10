"""Convert torch profiler traces to Nsight-compatible format."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def load_chrome_trace(trace_path: Path) -> dict[str, Any]:
    """Load Chrome trace JSON file.
    
    Args:
        trace_path: Path to Chrome trace JSON file
        
    Returns:
        Trace data dictionary
    """
    with open(trace_path, 'r') as f:
        return json.load(f)


def convert_to_nsys_format(
    chrome_trace: dict[str, Any],
    output_path: Path,
) -> None:
    """Convert Chrome trace to NSight Systems compatible format.
    
    Args:
        chrome_trace: Chrome trace data
        output_path: Output path for NSys metadata
    """
    events = chrome_trace.get("traceEvents", [])
    
    # Extract CUDA kernel events
    cuda_kernels = []
    for event in events:
        if event.get("ph") == "X" and "cuda" in event.get("cat", "").lower():
            cuda_kernels.append({
                "name": event.get("name"),
                "start_us": event.get("ts"),
                "duration_us": event.get("dur"),
                "device": event.get("args", {}).get("device"),
                "stream": event.get("args", {}).get("stream"),
            })
    
    # Create NSys-compatible metadata
    nsys_metadata = {
        "version": "1.0",
        "tool": "InferIQ Trace Converter",
        "source": "torch.profiler",
        "converted_at": str(Path.cwd()),
        "num_cuda_kernels": len(cuda_kernels),
        "cuda_kernels": cuda_kernels[:1000],  # Limit for file size
        "trace_events_summary": {
            "total_events": len(events),
            "cuda_events": len([e for e in events if "cuda" in e.get("cat", "").lower()]),
            "cpu_events": len([e for e in events if "cpu" in e.get("cat", "").lower()]),
        },
    }
    
    # Write metadata JSON
    with open(output_path, 'w') as f:
        json.dump(nsys_metadata, f, indent=2)
    
    console.print(f"[green]NSys metadata exported:[/green] {output_path}")
    
    # Also generate a summary report
    report_path = output_path.with_suffix(".report.txt")
    with open(report_path, 'w') as f:
        f.write("InferIQ CUDA Trace Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total CUDA kernels: {len(cuda_kernels)}\n")
        f.write(f"Total trace events: {len(events)}\n\n")
        
        # Top kernels by duration
        sorted_kernels = sorted(cuda_kernels, key=lambda x: x.get("duration_us", 0), reverse=True)
        f.write("Top 20 CUDA kernels by duration:\n")
        f.write("-" * 50 + "\n")
        for i, kernel in enumerate(sorted_kernels[:20], 1):
            duration_ms = kernel.get("duration_us", 0) / 1000.0
            f.write(f"{i}. {kernel.get('name', 'Unknown')[:60]:60s} {duration_ms:8.2f} ms\n")
    
    console.print(f"[green]Report exported:[/green] {report_path}")


def main():
    """Main entry point for trace export utility."""
    parser = argparse.ArgumentParser(
        description="Convert torch profiler traces to Nsight-compatible format"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input Chrome trace JSON file or directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all trace files in directory",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.batch and args.input.is_dir():
        # Process all trace files in directory
        trace_files = list(args.input.glob("*.json"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Converting traces...", total=len(trace_files))
            
            for trace_file in trace_files:
                progress.update(task, description=f"Processing {trace_file.name}...")
                
                try:
                    chrome_trace = load_chrome_trace(trace_file)
                    output_path = output_dir / trace_file.with_suffix(".nsys.json").name
                    convert_to_nsys_format(chrome_trace, output_path)
                except Exception as e:
                    console.print(f"[red]Error processing {trace_file}:[/red] {e}")
                
                progress.advance(task)
    
    else:
        # Process single file
        if not args.input.exists():
            console.print(f"[red]Error:[/red] File not found: {args.input}")
            return 1
        
        try:
            chrome_trace = load_chrome_trace(args.input)
            output_path = output_dir / args.input.with_suffix(".nsys.json").name
            convert_to_nsys_format(chrome_trace, output_path)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
    
    console.print("\n[bold green]Conversion complete![/bold green]")
    return 0


if __name__ == "__main__":
    exit(main())
