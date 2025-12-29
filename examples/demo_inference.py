#!/usr/bin/env python3
"""
Demo inference application for InferScope bottleneck analysis.
Simulates a typical ML inference pipeline with CPU and GPU operations.
"""

import sys
import time
import json
sys.path.insert(0, 'src')

from inferscope import Profiler, scope, mark_event, set_global_profiler

# Create a mock trace buffer
class MockTraceBuffer:
    def __init__(self):
        self.events = []
    
    def add_event(self, event):
        self.events.append(event)
    
    def save_to_file(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({"events": self.events}, f, indent=2)

# Initialize profiler
trace_buffer = MockTraceBuffer()
profiler = Profiler(trace_buffer)
set_global_profiler(profiler)


def simulate_data_loading(batch_size=32):
    """Simulate data loading from disk."""
    with scope("data_loading"):
        time.sleep(0.05)  # 50ms - simulate I/O
        mark_event("batch_loaded", metadata={"batch_size": batch_size})
        return [{"id": i, "data": [j for j in range(10)]} for i in range(batch_size)]


def preprocess_batch(batch):
    """Preprocess input batch."""
    with scope("preprocessing"):
        # Simulate CPU-bound preprocessing
        for item in batch:
            # Simulate complex preprocessing
            _ = [x * 2 for x in item["data"]]
        
        time.sleep(0.03)  # 30ms - CPU preprocessing
        mark_event("batch_preprocessed", metadata={"items": len(batch)})
        return batch


def transfer_to_gpu(batch):
    """Simulate data transfer to GPU."""
    with scope("data_transfer_h2d"):
        # Simulate H2D transfer (even without actual GPU)
        time.sleep(0.02)  # 20ms - simulated transfer
        mark_event("data_on_gpu", metadata={"batch_size": len(batch)})
        return batch


def model_forward_pass(batch):
    """Simulate model forward pass."""
    with scope("inference"):
        # Simulate GPU computation
        time.sleep(0.15)  # 150ms - GPU computation (this will be the bottleneck)
        
        # Simulate memory operations
        with scope("attention_layers"):
            time.sleep(0.08)  # 80ms
        
        with scope("output_projection"):
            time.sleep(0.07)  # 70ms
        
        mark_event("inference_complete", metadata={"batch_size": len(batch)})
        return [{"output": i * 2} for i in range(len(batch))]


def transfer_from_gpu(results):
    """Simulate data transfer from GPU."""
    with scope("data_transfer_d2h"):
        # Simulate D2H transfer
        time.sleep(0.015)  # 15ms - simulated transfer
        mark_event("data_on_cpu", metadata={"results": len(results)})
        return results


def postprocess_results(results):
    """Postprocess model outputs."""
    with scope("postprocessing"):
        # Simulate postprocessing
        time.sleep(0.02)  # 20ms
        mark_event("results_postprocessed", metadata={"count": len(results)})
        return results


def save_results(results):
    """Save results to storage."""
    with scope("result_storage"):
        # Simulate storage I/O
        time.sleep(0.04)  # 40ms
        mark_event("results_saved", metadata={"count": len(results)})


def inference_pipeline(num_batches=3):
    """Run complete inference pipeline."""
    with scope("inference_pipeline"):
        profiler.start()
        
        for batch_num in range(num_batches):
            mark_event("batch_start", metadata={"batch_num": batch_num})
            
            # Data loading
            batch = simulate_data_loading(batch_size=32)
            
            # Preprocessing
            batch = preprocess_batch(batch)
            
            # GPU transfer
            batch = transfer_to_gpu(batch)
            
            # Model inference
            results = model_forward_pass(batch)
            
            # GPU transfer back
            results = transfer_from_gpu(results)
            
            # Postprocessing
            results = postprocess_results(results)
            
            # Storage
            save_results(results)
            
            mark_event("batch_complete", metadata={"batch_num": batch_num})
        
        profiler.stop()
        
        # Get unified timeline
        timeline = profiler.get_unified_timeline()
        
        return timeline


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("InferScope Demo: Inference Bottleneck Analysis")
    print("=" * 70)
    print("\n[INFO] Starting inference pipeline...")
    print("[INFO] Running 3 batches of 32 samples each...\n")
    
    # Run inference pipeline
    timeline = inference_pipeline(num_batches=3)
    
    print(f"\n[INFO] Inference complete!")
    print(f"[INFO] Total events collected: {len(timeline)}")
    
    # Calculate total duration
    if timeline:
        start_time = min(e.get('timestamp_us', 0) for e in timeline)
        end_time = max(e.get('timestamp_us', 0) for e in timeline)
        duration_us = end_time - start_time
        print(f"[INFO] Timeline duration: {duration_us / 1e6:.3f} seconds")
    
    # Generate trace file
    trace_file = "demo_trace.json"
    trace_buffer.save_to_file(trace_file)
    print(f"[INFO] Trace saved to: {trace_file}")
    print("\n" + "=" * 70)
    print("Next steps:")
    print(f"  1. Analyze the trace: inferscope analyze {trace_file} --output report.md")
    print(f"  2. View HTML report:  inferscope analyze {trace_file} --output report.html --format html")
    print("=" * 70)
