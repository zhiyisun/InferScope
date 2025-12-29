#!/usr/bin/env python3
"""
Run a simple end-to-end profiling demo using Profiler.
Collects CPU events from a small workload and injects a synthetic GPU event,
then prints a unified, synchronized timeline.
"""

import os
import sys
import time
from typing import Dict, Any, List

# Ensure project root is on sys.path to import src.inferscope
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inferscope import Profiler  # type: ignore


class SimpleTraceBuffer:
    """Minimal trace buffer for demo (in-memory)."""
    def __init__(self, capacity_mb: int = 100):
        self.capacity_mb = capacity_mb
        self._events: List[Dict[str, Any]] = []
        self._is_full = False

    def enqueue(self, event: Dict[str, Any]) -> bool:
        if self._is_full:
            return False
        self._events.append(event.copy())
        return True

    def read_all(self) -> List[Dict[str, Any]]:
        return [e.copy() for e in self._events]


def small_workload() -> int:
    """A tiny CPU workload to generate call/return events."""
    def inner(x: int) -> int:
        time.sleep(0.001)  # minimal delay
        return x * x
    return sum(inner(i) for i in range(5))


def main() -> None:
    buf = SimpleTraceBuffer()
    profiler = Profiler(buf)

    print("Starting profiler...")
    profiler.start()

    # CPU activity
    result = small_workload()

    # Inject synthetic GPU events (mock/stub)
    profiler.gpu._inject_kernel_event(name="demo_kernel", duration_us=200, stream_id=0)
    profiler.gpu._inject_copy_event(name="input_copy", bytes_transferred=256 * 1024, direction='h2d')

    profiler.stop()
    stats = profiler.get_stats()

    print("CPU stats:", stats.cpu)
    print("GPU stats:", stats.gpu)

    timeline = profiler.get_unified_timeline()
    print("\nUnified timeline (ts_us type name thread/stream):")
    for e in timeline:
        ts = e.get('global_ts_us')
        etype = e.get('type')
        name = e.get('name')
        tid = e.get('thread_id')
        sid = e.get('stream_id')
        print(f"{ts} {etype} {name} tid={tid} stream={sid}")

    print("\nDemo complete. Workload result:", result)


if __name__ == "__main__":
    main()
