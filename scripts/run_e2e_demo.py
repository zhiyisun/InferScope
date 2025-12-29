#!/usr/bin/env python3
"""
End-to-end demo: Profiler → Analyzer → Reporter

Demonstrates the complete InferScope pipeline:
1. Collect CPU and GPU events
2. Merge into unified timeline
3. Analyze bottlenecks
4. Generate report
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inferscope import Profiler, BottleneckAnalyzer, ReportGenerator  # type: ignore


class SimpleTraceBuffer:
    """Minimal trace buffer for demo."""
    def __init__(self):
        self._events = []
    
    def enqueue(self, event):
        self._events.append(event.copy())
        return True
    
    def read_all(self):
        return [e.copy() for e in self._events]


def cpu_heavy_workload():
    """Simulate CPU-intensive preprocessing."""
    result = 0
    for i in range(1000):
        result += i ** 2
    time.sleep(0.002)  # 2ms
    return result


def main():
    print("=" * 60)
    print("InferScope End-to-End Demo")
    print("=" * 60)
    print()
    
    # Step 1: Collect events
    print("Step 1: Profiling workload...")
    buf = SimpleTraceBuffer()
    profiler = Profiler(buf)
    
    profiler.start()
    
    # Run workload (CPU-heavy)
    result = cpu_heavy_workload()
    
    # Inject synthetic GPU events (less time than CPU)
    profiler.gpu._inject_kernel_event(name="gemm_kernel", duration_us=50, stream_id=0)
    profiler.gpu._inject_copy_event(name="input_h2d", bytes_transferred=1024, direction='h2d')
    
    profiler.stop()
    
    stats = profiler.get_stats()
    print(f"  CPU events: {stats.cpu['total_events_captured']}")
    print(f"  GPU events: {stats.gpu['event_count']}")
    print()
    
    # Step 2: Get unified timeline
    print("Step 2: Merging timelines...")
    unified_timeline = profiler.get_unified_timeline()
    print(f"  Unified events: {len(unified_timeline)}")
    print()
    
    # Step 3: Analyze bottlenecks
    print("Step 3: Analyzing bottlenecks...")
    analyzer = BottleneckAnalyzer(unified_timeline)
    analysis = analyzer.analyze()
    
    bottleneck = analysis['bottleneck']
    print(f"  Bottleneck type: {bottleneck['type']}")
    print(f"  Confidence: {bottleneck['confidence'] * 100:.0f}%")
    print(f"  Primary cause: {bottleneck['primary_cause']}")
    print()
    
    # Step 4: Generate report
    print("Step 4: Generating report...")
    reporter = ReportGenerator(analysis, unified_timeline)
    
    # Save to file
    output_path = "demo_report.md"
    reporter.save(output_path, format='markdown')
    print(f"  Report saved to: {output_path}")
    print()
    
    # Display report preview
    print("=" * 60)
    print("Report Preview (first 20 lines):")
    print("=" * 60)
    markdown = reporter.to_markdown()
    lines = markdown.split('\n')[:20]
    for line in lines:
        print(line)
    print("...")
    print()
    
    print("=" * 60)
    print("Demo complete!")
    print(f"  Workload result: {result}")
    print(f"  Full report: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
