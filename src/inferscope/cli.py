"""
InferScope CLI - Command-line interface for profiling inference workloads
"""

import sys
import os
import subprocess
import json
import argparse
import traceback
from pathlib import Path
from typing import Optional

from .profiler import Profiler
from .analyzer import BottleneckAnalyzer
from .reporter import ReportGenerator


def run_command(script: str, args: list, report_path: str = "report.md", 
                log_level: str = "info", trace_size_mb: int = 100) -> int:
    """
    Execute script with InferScope tracing.
    
    Usage:
        inferscope run my_inference.py --arg1 value1
    
    Args:
        script: Path to Python script
        args: Arguments to pass to script
        report_path: Output report file path
        log_level: Logging verbosity (debug, info, warn)
        trace_size_mb: Ring buffer size in MB
        
    Returns:
        Exit code from script execution
    """
    
    # Verify script exists
    if not os.path.exists(script):
        print(f"Error: Script '{script}' not found", file=sys.stderr)
        return 1
    
    # Create profiler
    trace_buffer_size = trace_size_mb * 1024 * 1024
    
    # We'll inject profiling by executing the script with INFERSCOPE env vars set
    # In a real implementation, we'd use sys.settrace or similar
    
    env = os.environ.copy()
    env["INFERSCOPE_ENABLED"] = "1"
    env["INFERSCOPE_LOG_LEVEL"] = log_level
    env["INFERSCOPE_TRACE_SIZE_MB"] = str(trace_size_mb)
    env["INFERSCOPE_REPORT_PATH"] = report_path
    
    # Execute script
    cmd = [sys.executable, script] + args
    
    try:
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())
        return result.returncode
    except Exception as e:
        print(f"Error executing script: {e}", file=sys.stderr)
        return 1


def analyze_command(trace_file: str, output_path: str = "report.md", 
                   rules_file: Optional[str] = None, output_format: str = "markdown") -> int:
    """
    Analyze existing trace file.
    
    Usage:
        inferscope analyze trace.json --output report.html --format html
    
    Args:
        trace_file: Path to trace JSON file
        output_path: Output report file path
        rules_file: Optional custom bottleneck rules YAML
        output_format: Output format (markdown, html, json)
        
    Returns:
        Exit code (0 for success)
    """
    
    # Verify trace file exists
    if not os.path.exists(trace_file):
        print(f"Error: Trace file '{trace_file}' not found", file=sys.stderr)
        return 1
    
    # Load trace
    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        if isinstance(trace_data, dict) and "events" in trace_data:
            timeline = trace_data["events"]
        elif isinstance(trace_data, dict) and "timeline" in trace_data:
            timeline = trace_data["timeline"]
        elif isinstance(trace_data, list):
            timeline = trace_data
        else:
            timeline = trace_data
        
    except Exception as e:
        print(f"Error loading trace file: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    
    # Analyze
    try:
        analyzer = BottleneckAnalyzer(timeline)
        analysis = analyzer.analyze()
        
        # Generate report
        reporter = ReportGenerator(analysis, timeline)
        reporter.save(output_path, format=output_format)
        
        print(f"Report generated: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error analyzing trace: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


def config_command(action: str = "show") -> int:
    """
    Manage InferScope configuration.
    
    Usage:
        inferscope config show
    
    Args:
        action: Configuration action (show, get, set)
        
    Returns:
        Exit code (0 for success)
    """
    
    if action == "show":
        config = {
            "version": "0.1-alpha",
            "default_trace_size_mb": 100,
            "default_report_format": "markdown",
            "supported_frameworks": ["pytorch"],
            "gpu_support": "nvidia_cupti",
            "platform": sys.platform,
        }
        
        print("InferScope Configuration:")
        print("=" * 50)
        for key, value in config.items():
            print(f"  {key}: {value}")
        return 0
    
    else:
        print(f"Unknown config action: {action}")
        return 1


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        prog="inferscope",
        description="InferScope - AI Inference Bottleneck Analysis Tool",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 'run' command
    run_parser = subparsers.add_parser("run", help="Execute script with profiling")
    run_parser.add_argument("script", help="Python script to profile")
    run_parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments for script")
    run_parser.add_argument("--report", default="report.md", help="Output report path")
    run_parser.add_argument("--log-level", default="info", choices=["debug", "info", "warn"])
    run_parser.add_argument("--trace-size-mb", type=int, default=100, help="Ring buffer size (MB)")
    
    # 'analyze' command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze trace file")
    analyze_parser.add_argument("trace_file", help="Trace JSON file to analyze")
    analyze_parser.add_argument("--output", default="report.md", help="Output report path")
    analyze_parser.add_argument("--format", default="markdown", choices=["markdown", "html", "json"])
    analyze_parser.add_argument("--rules-file", help="Custom bottleneck rules YAML")
    
    # 'config' command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("action", nargs="?", default="show", help="Config action")
    
    args = parser.parse_args()
    
    if args.command == "run":
        return run_command(
            args.script,
            args.script_args,
            report_path=args.report,
            log_level=args.log_level,
            trace_size_mb=args.trace_size_mb,
        )
    
    elif args.command == "analyze":
        return analyze_command(
            args.trace_file,
            output_path=args.output,
            rules_file=args.rules_file,
            output_format=args.format,
        )
    
    elif args.command == "config":
        return config_command(args.action)
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
