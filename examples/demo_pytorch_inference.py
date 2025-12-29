#!/usr/bin/env python3
"""
Real inference demo using PyTorch (if available), instrumented with InferScope.
- Uses a simple MLP on random data
- Records CPU scopes via API
- Injects H2D/D2H and GPU kernel events to produce clear durations
- Saves trace to demo_real_trace.json
"""

import os
import sys
import time
import json

# Ensure src/ is importable when running from workspace
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(WORKSPACE_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# InferScope API
from inferscope import Profiler
from inferscope.api import scope, mark_event, set_global_profiler

# Minimal trace buffer implementation
class SimpleTraceBuffer:
    def __init__(self, capacity_mb: int = 100):
        self.capacity_mb = capacity_mb
        self.events = []
        self._full = False
    def enqueue(self, event):
        if self._full:
            return False
        self.events.append(event.copy())
        return True
    def read_all(self):
        return [e.copy() for e in self.events]
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"events": self.events}, f, indent=2)

"""Optional PyTorch import (CUDA path used if available)."""
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def run_inference_batches(num_batches: int = 3, batch_size: int = 32):
    buf = SimpleTraceBuffer(capacity_mb=100)
    profiler = Profiler(buf)
    set_global_profiler(profiler)

    # Start collectors
    profiler.start()

    # Prepare model (GPU if available and single device; else CPU)
    device = torch.device("cpu") if not (TORCH_AVAILABLE and torch.cuda.is_available()) else torch.device("cuda:0")
    single_gpu = False
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            single_gpu = (torch.cuda.device_count() == 1)
        except Exception:
            single_gpu = False
    if TORCH_AVAILABLE:
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        model.eval()
        try:
            model.to(device)
        except Exception:
            device = torch.device("cpu")
    else:
        model = None

    for b in range(num_batches):
        mark_event("batch_start", {"batch": b})

        # Data loading / preprocessing (CPU)
        with scope("data_loading"):
            time.sleep(0.01)
        with scope("preprocessing"):
            time.sleep(0.01)

        # H2D transfer: measure if on CUDA single-GPU; else inject
        if TORCH_AVAILABLE and device.type == "cuda" and single_gpu:
            x_cpu = torch.randn(batch_size, 128)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            x = x_cpu.to(device)
            end.record()
            torch.cuda.synchronize()
            h2d_ms = start.elapsed_time(end)
            duration_us = int(h2d_ms * 1000)
            now_us = int(time.monotonic_ns() // 1000)
            profiler.trace_buffer.enqueue({
                'type': 'h2d_copy',
                'name': 'h2d_batch',
                'timestamp_start_us': now_us - duration_us,
                'timestamp_end_us': now_us,
                'duration_us': duration_us,
                'thread_id': 1,
                'origin': 'cpu',
                'bytes': x_cpu.numel() * 4,
            })
        else:
            profiler.gpu._inject_copy_event("h2d_batch", bytes_transferred=4_096_000, direction="h2d")

        # Inference
        with scope("inference"):
            if TORCH_AVAILABLE and device.type == "cuda" and single_gpu:
                with torch.no_grad():
                    # Measure GPU compute with CUDA events
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    y = model(x)
                    end.record()
                    torch.cuda.synchronize()
                    compute_ms = start.elapsed_time(end)
                    profiler.gpu._inject_kernel_event("forward_pass", duration_us=int(compute_ms * 1000))
            else:
                # CPU forward and injected GPU kernels for demo
                if TORCH_AVAILABLE:
                    x = torch.randn(batch_size, 128)
                    with torch.no_grad():
                        _ = model(x)
                profiler.gpu._inject_kernel_event("forward_pass", duration_us=300_000)
                profiler.gpu._inject_kernel_event("attention", duration_us=120_000)
                profiler.gpu._inject_kernel_event("projection", duration_us=90_000)
                time.sleep(0.51)

        # D2H transfer: measure if on CUDA single-GPU; else inject
        if TORCH_AVAILABLE and device.type == "cuda" and single_gpu:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y_cpu = y.to("cpu")
            end.record()
            torch.cuda.synchronize()
            d2h_ms = start.elapsed_time(end)
            duration_us = int(d2h_ms * 1000)
            now_us = int(time.monotonic_ns() // 1000)
            profiler.trace_buffer.enqueue({
                'type': 'd2h_copy',
                'name': 'd2h_results',
                'timestamp_start_us': now_us - duration_us,
                'timestamp_end_us': now_us,
                'duration_us': duration_us,
                'thread_id': 1,
                'origin': 'cpu',
                'bytes': y_cpu.numel() * 4,
            })
        else:
            profiler.gpu._inject_copy_event("d2h_results", bytes_transferred=1_024_000, direction="d2h")

        # Postprocess and save (CPU)
        with scope("postprocessing"):
            time.sleep(0.01)
        with scope("save_results"):
            time.sleep(0.02)

        mark_event("batch_complete", {"batch": b})

    profiler.stop()

    # Save trace to file
    trace_path = os.path.join(WORKSPACE_ROOT, "demo_real_trace.json")
    buf.save(trace_path)
    print(f"[INFO] Trace saved: {trace_path}")

    # Return path for convenience
    return trace_path


if __name__ == "__main__":
    print("=" * 70)
    print("InferScope Demo: Real Inference Script")
    print("=" * 70)
    print(f"[INFO] PyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"[INFO] CUDA available: {torch.cuda.is_available()} | device_count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    p = run_inference_batches(num_batches=3, batch_size=32)
    print("\nNext:")
    print(f"  - Analyze (MD): python scripts/inferscope analyze {os.path.basename(p)} --output real_report.md")
    print(f"  - Analyze (HTML): python scripts/inferscope analyze {os.path.basename(p)} --output real_report.html --format html")
