# Integration Strategy

This document outlines how the InferScope components integrate at runtime and during development.

## Component Integration Map

```
User Code (PyTorch)
     │
     ├─→ [API Layer] scope(), mark_event()
     │      │
     │      ├─→ [CPU Collector] sys.settrace hooks
     │      │      │
     │      │      └─→ [Trace Buffer] (ring buffer, per-thread)
     │      │
     │      ├─→ [GPU Collector] CUPTI callbacks
     │      │      │
     │      │      └─→ [Trace Buffer] (GPU events)
     │      │
     │      └─→ [Memory Collector] /proc/stat
     │             │
     │             └─→ [Trace Buffer] (memory events)
     │
     └─→ [Merger] Clock sync + event ordering
            │
            └─→ [Analyzer] Bottleneck rules → diagnosis
                   │
                   └─→ [Reporter] Markdown/HTML/JSON
                          │
                          └─→ Output: report.md
```

## Integration Points (API Boundaries)

### 1. Collectors → Trace Buffer
**Interface:** TraceBuffer.enqueue(event: Dict) → bool

**Contract:**
- Event = {"type": str, "name": str, "timestamp_start_us": int, ...}
- Enqueue returns success/overflow indicator
- Lock-free: each thread has own buffer

**Example:**
```python
# CPU Collector
cpu_events_buffer.enqueue({
    "type": "cpu_call",
    "name": "tokenize",
    "timestamp_start_us": 1000000,
    "timestamp_end_us": 1031200
})

# GPU Collector
gpu_events_buffer.enqueue({
    "type": "gpu_kernel",
    "name": "attention_forward",
    "timestamp_start_us": 1031200,
    "timestamp_end_us": 1051500
})
```

### 2. Trace Buffer → Merger
**Interface:** TraceBuffer.read_all() → List[Dict]

**Contract:**
- Returns all events in insertion order
- Timestamps may be in different clock domains (CPU vs GPU)
- No guarantee of global ordering

**Example:**
```python
all_events = trace_buffer.read_all()
# Result: unsorted mix of CPU, GPU, memory events
```

### 3. Merger → Analyzer
**Interface:** TimelineMerger.get_unified_timeline() → List[Dict]

**Contract:**
- Returns events sorted by synchronized timestamp
- All timestamps in unified global clock domain
- Causality preserved (parent/child scopes)

**Example:**
```python
unified_timeline = merger.get_unified_timeline()
# Result: fully ordered events, ready for analysis
```

### 4. Analyzer → Reporter
**Interface:** BottleneckAnalyzer.analyze() → Dict

**Contract:**
- Returns bottleneck classification + suggestions
- Each suggestion backed by evidence
- Confidence score [0.0, 1.0]

**Example:**
```python
analysis = analyzer.analyze()
# Result: {
#   "bottleneck_type": "cpu_bound",
#   "confidence": 0.92,
#   "suggestions": [
#     {"action": "Increase batch size", "priority": "high", "impact": 12}
#   ]
# }
```

## Data Flow During Runtime

### Collection Phase (0-100 ms, example timing)
```
t=0ms:   User calls model.forward(inputs)
         → scope("inference").__enter__()
         → CPU Collector hooks sys.settrace
         → GPU Collector registers CUPTI callbacks

t=30ms:  CPU event: enter tokenize()
         → enqueue to trace_buffer[thread_0]

t=35ms:  GPU event: cuLaunchKernel(attention_forward)
         → enqueue to trace_buffer[gpu]

t=60ms:  CPU event: exit tokenize()
         → enqueue to trace_buffer[thread_0]

t=85ms:  GPU event: kernel complete
         → enqueue to trace_buffer[gpu]

t=100ms: model.forward() returns
         → scope("inference").__exit__()
         → CPU Collector uninstalls hooks
         → GPU Collector flushes CUPTI buffer
         → Return to user code
```

### Merge Phase (<1ms, after inference)
```
Merger input:
  - CPU events: [enter tokenize @ 30ms, exit tokenize @ 60ms]
  - GPU events: [kernel launch @ 35ms, kernel end @ 85ms]

Merger:
  1. Calibrate CPU/GPU clock (CUDA event sync)
  2. Apply offset to CPU timestamps
  3. Sort all events by timestamp
  4. Build scope hierarchy

Merger output:
  - Sorted: [cpu enter @ 30ms, gpu launch @ 35ms, cpu exit @ 60ms, gpu end @ 85ms]
  - Scopes: scope("inference") spans all events
```

### Analysis Phase (<10ms, after merge)
```
Analyzer input: Unified timeline

Analyzer rules:
  1. GPU idle = (100ms - 50ms) / 100ms = 50% idle → CPU-bound
  2. H2D copy = 18ms / 100ms = 18% → Memory overhead
  3. CPU time = 30ms / 100ms = 30% → Dominant

Analyzer output:
  - Bottleneck: CPU-bound
  - Suggestions:
    1. "Increase batch size" (amortize H2D copy)
    2. "Move tokenization off critical path" (async)
```

### Reporting Phase (<100ms, after analysis)
```
Reporter input: Analyzer results + timeline

Reporter:
  1. Render Markdown template
  2. Insert values: CPU time, GPU time, suggestions
  3. Write to report.md

Reporter output: report.md (file on disk)
```

## Integration Testing Strategy

### Test 1: Collector Integration
**Validates:** Collectors → Trace Buffer

```python
def test_collectors_write_to_buffer():
    buffer = TraceBuffer(size_mb=100)
    cpu_collector = CpuCollector(buffer)
    gpu_collector = GpuCollector(buffer)
    
    cpu_collector.start()
    gpu_collector.start()
    
    # Synthetic workload
    _synthetic_cpu_work()
    _synthetic_gpu_kernel()
    
    cpu_collector.stop()
    gpu_collector.stop()
    
    events = buffer.read_all()
    assert len(events) > 10  # Some events captured
    assert any(e["type"] == "cpu_call" for e in events)
    assert any(e["type"] == "gpu_kernel" for e in events)
```

### Test 2: Merger Integration
**Validates:** Trace Buffer → Merger → Analyzer

```python
def test_merger_analyzer_integration():
    # Load example trace
    trace_json = load_json("fixtures/example_trace.json")
    cpu_events = [e for e in trace_json["events"] if e["type"].startswith("cpu")]
    gpu_events = [e for e in trace_json["events"] if e["type"].startswith("gpu")]
    
    merger = TimelineMerger(cpu_events, gpu_events)
    merger.synchronize_clocks()
    unified = merger.get_unified_timeline()
    
    analyzer = BottleneckAnalyzer(unified)
    analysis = analyzer.analyze()
    
    assert analysis["bottleneck_type"] in ["cpu_bound", "gpu_bound", "memory_bound"]
    assert analysis["confidence"] > 0.5
    assert len(analysis["suggestions"]) >= 1
```

### Test 3: End-to-End Integration
**Validates:** User API → Collectors → Merger → Analyzer → Reporter

```python
def test_end_to_end_with_real_pytorch():
    import torch
    
    with scope("inference"):
        model = torch.nn.Linear(10, 10).cuda()
        input_batch = torch.randn(32, 10).cuda()
        output = model(input_batch)
    
    # Report should be generated automatically
    report_path = get_report_path()
    assert os.path.exists(report_path)
    
    report_text = read_file(report_path)
    assert "End-to-end latency" in report_text
    assert "CPU" in report_text or "GPU" in report_text
    assert "Suggestion" in report_text
```

## Dependency Management During Integration

### Critical Dependencies
1. **Trace Buffer** – No dependencies (utility)
2. **Collectors** – Depend on Trace Buffer only
3. **Merger** – Depends on Collectors + Trace Buffer
4. **Analyzer** – Depends on Merger output
5. **Reporter** – Depends on Analyzer output
6. **CLI** – Depends on all (orchestrates)

### Optional Dependencies
- PyTorch (only for example workloads)
- TensorFlow (future; parallel to PyTorch)
- CUDA (GPU collection; CPU-only mode if unavailable)

### Dependency Isolation
```python
# Good: CLI doesn't import analyzer directly; uses via merger
from inferscope.merger import TimelineMerger
timeline = TimelineMerger(...).get_unified_timeline()
# Analyzer is separate concern

# Bad: Tight coupling
from inferscope.merger import TimelineMerger, BottleneckAnalyzer
# Harder to mock, test separately
```

## API Version Compatibility

### Breaking Changes Policy
- Major version bump (v1.0 → v2.0) for breaking API changes
- Deprecation period (minimum 2 releases) before removal
- Always provide migration guide

### Example: Hypothetical v1.0 → v2.0
```python
# v1.0 API (deprecated in v1.2)
with scope("inference"):
    output = model.forward(inputs)

# v2.0 API (new in v2.0)
with profile("inference", tags={"model": "gpt2"}):
    output = model.forward(inputs)

# Compatibility shim in v1.2-v2.0
scope = profile  # Alias; deprecated warning
```

## Integration Checklist (Pre-Release)

- [ ] All collectors functional
- [ ] Merger achieves <1% clock error
- [ ] Analyzer passes all rule tests
- [ ] Reporter generates valid Markdown/HTML
- [ ] CLI runs without errors
- [ ] End-to-end test passes (PyTorch model)
- [ ] No segfaults on shutdown
- [ ] Overhead <5% measured on real model
- [ ] Documentation complete and accurate
- [ ] All tests pass (unit + integration + system)
