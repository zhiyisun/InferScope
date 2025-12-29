#!/usr/bin/env python3
"""
Demo: Generate a mock inference trace with simulated bottlenecks.
This creates a realistic trace that shows typical inference bottlenecks.
"""

import json
import time
import random
from datetime import datetime

def create_mock_trace():
    """Create a mock trace file simulating an inference workload with durations."""
    events = []
    event_id = 0

    # Start time in microseconds
    start_time = int(time.time() * 1e6)
    current_time = start_time
    
    # Simulate 3 batches of inference
    for batch_num in range(3):
        print(f"  Generating batch {batch_num + 1} events...")
        
        # Batch start event
        current_time += 100
        events.append({
            "id": event_id,
            "type": "instant_event",
            "name": "batch_start",
            "timestamp_us": current_time,
            "thread_id": 1,
            "origin": "cpu",
            "metadata": {"batch_num": batch_num}
        })
        event_id += 1
        
        # CPU preprocessing (10ms)
        start = current_time
        end = start + 10_000
        events.append({
            "id": event_id,
            "type": "cpu_preprocessing",
            "name": "data_loading",
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "duration_us": 10_000,
            "thread_id": 1,
            "origin": "cpu"
        })
        event_id += 1
        current_time = end
        
        # CPU preprocessing (10ms)
        start = current_time
        end = start + 10_000
        events.append({
            "id": event_id,
            "type": "cpu_preprocessing",
            "name": "preprocessing",
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "duration_us": 10_000,
            "thread_id": 1,
            "origin": "cpu"
        })
        event_id += 1
        current_time = end
        
        # H2D Transfer (20ms)
        start = current_time
        end = start + 20_000
        events.append({
            "id": event_id,
            "type": "h2d_copy",
            "name": "data_transfer_h2d",
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "duration_us": 20_000,
            "thread_id": 1,
            "origin": "cpu"
        })
        event_id += 1
        current_time = end
        
        # INFERENCE - MAJOR BOTTLENECK (300ms on GPU)
        start = current_time
        end = start + 300_000
        events.append({
            "id": event_id,
            "type": "gpu_compute",
            "name": "inference",
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "duration_us": 300_000,
            "thread_id": 1,
            "origin": "gpu"
        })
        event_id += 1
        current_time = end
        
        # D2H Transfer (15ms)
        start = current_time
        end = start + 15_000
        events.append({
            "id": event_id,
            "type": "d2h_copy",
            "name": "data_transfer_d2h",
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "duration_us": 15_000,
            "thread_id": 1,
            "origin": "cpu"
        })
        event_id += 1
        current_time = end
        
        # Postprocessing (10ms)
        start = current_time
        end = start + 10_000
        events.append({
            "id": event_id,
            "type": "cpu_preprocessing",
            "name": "postprocessing",
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "duration_us": 10_000,
            "thread_id": 1,
            "origin": "cpu"
        })
        event_id += 1
        current_time = end
        
        # Storage (30ms)
        start = current_time
        end = start + 30_000
        events.append({
            "id": event_id,
            "type": "cpu_preprocessing",
            "name": "result_storage",
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "duration_us": 30_000,
            "thread_id": 1,
            "origin": "cpu"
        })
        event_id += 1
        current_time = end
        
        # Batch complete
        events.append({
            "id": event_id,
            "type": "instant_event",
            "name": "batch_complete",
            "timestamp_us": current_time,
            "thread_id": 1,
            "origin": "cpu",
            "metadata": {"batch_num": batch_num}
        })
        event_id += 1
        
        current_time += 100
    
    total_duration_s = (current_time - start_time) / 1e6
    return {
        "metadata": {
            "start_time_us": start_time,
            "end_time_us": current_time,
            "duration_us": current_time - start_time,
            "thread_count": 1,
            "application": "inference_demo",
            "timestamp": datetime.now().isoformat()
        },
        "events": events
    }


if __name__ == "__main__":
    print("=" * 70)
    print("InferScope Demo: Creating Mock Inference Trace")
    print("=" * 70)
    print("\n[INFO] Generating realistic inference trace with bottlenecks...\n")
    
    trace = create_mock_trace()
    
    total_duration = trace["metadata"]["duration_us"] / 1e6
    total_events = len(trace["events"])
    
    print(f"\n[INFO] Trace generation complete!")
    print(f"[INFO] Total events: {total_events}")
    print(f"[INFO] Total duration: {total_duration:.3f} seconds")
    print(f"[INFO] Event breakdown:")
    print(f"       - scope_enter: {sum(1 for e in trace['events'] if e['type'] == 'scope_enter')}")
    print(f"       - scope_exit: {sum(1 for e in trace['events'] if e['type'] == 'scope_exit')}")
    print(f"       - instant_event: {sum(1 for e in trace['events'] if e['type'] == 'instant_event')}")
    
    # Save trace
    trace_file = "demo_trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace, f, indent=2)
    
    print(f"\n[INFO] Trace saved to: {trace_file}")
    print("\n" + "=" * 70)
    print("Next steps - Analyze the trace using the CLI:")
    print("=" * 70)
    print(f"\n1. Generate markdown report:")
    print(f"   $ python scripts/inferscope analyze {trace_file} --output demo_report.md")
    print(f"\n2. Generate HTML report:")
    print(f"   $ python scripts/inferscope analyze {trace_file} --output demo_report.html --format html")
    print(f"\n3. Show configuration:")
    print(f"   $ python scripts/inferscope config show")
    print("=" * 70)
