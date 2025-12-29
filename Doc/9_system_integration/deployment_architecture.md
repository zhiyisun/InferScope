# Deployment Architecture

## Runtime Topology

InferScope operates as a **library-injected profiler**, not a separate service.

```
User Application (PyTorch/TensorFlow)
│
├─ Load inferscope package (import)
│  │
│  ├─ CPU Collector (sys.settrace hooks)
│  ├─ GPU Collector (CUPTI callbacks)
│  └─ Trace Buffer (in-process ring buffer)
│
├─ Execute inference (model.forward(), model.generate(), etc.)
│  │
│  ├─ CPU events → Trace Buffer
│  ├─ GPU events → Trace Buffer (async)
│  └─ Memory events → Trace Buffer
│
└─ Post-execution
   │
   ├─ Retrieve events from Trace Buffer
   ├─ Timeline Merger (CPU/GPU synchronization)
   ├─ Analyzer Engine (bottleneck detection)
   ├─ Report Generator (Markdown/HTML/JSON)
   └─ Write report to disk
```

## Deployment Models

### 1. CLI Mode (Primary)
```bash
inferscope run python my_inference.py --model llama-7b --batch-size 32
# Generates: report.md (in working directory)
```

**Process Flow:**
1. CLI spawns child process: `python my_inference.py`
2. Sets `INFERSCOPE_ENABLED=1` environment variable
3. Child process imports inferscope; collection starts
4. Child exits; parent merges traces and generates report
5. Report written to disk (default: `report.md` in cwd)

### 2. Library API Mode (Secondary)
```python
from inferscope import scope

with scope("inference"):
    output = model.forward(inputs)

# Traces available after scope exits
# User can call inferscope.save_report("report.md")
```

**Process Flow:**
1. User imports inferscope in their code
2. Wraps inference in `scope()` context
3. Collection occurs automatically
4. User calls API to save or retrieve traces

### 3. Post-Hoc Analysis Mode (Tertiary)
```bash
inferscope analyze trace.json --output report.md
```

**Process Flow:**
1. Load previously saved trace file
2. Run analyzer engine (no collection)
3. Generate report
4. Write to disk

## Resource Allocation

### Memory

| Component | Peak Memory | Note |
|-----------|------------|------|
| Trace Buffer | ~100 MB | Configurable (INFERSCOPE_TRACE_SIZE_MB) |
| CPU Collector | ~10 MB | Per-thread overhead |
| GPU Collector | ~20 MB | CUPTI activity buffer |
| Merger/Analyzer | ~50 MB | Temporary during post-processing |
| **Total** | **~180 MB** | Typical for 100ms trace |

**Scaling:**
- Longer traces → larger buffer → linear memory growth
- More threads → more per-thread buffers → linear growth

### CPU Overhead

| Operation | Overhead | Note |
|-----------|----------|------|
| sys.settrace hook | ~2-3% | Per function call |
| perf sampling | ~0.5-1% | Background sampling |
| Lock-free enqueue | <1% | Per event |
| **Total** | **<5%** | Target for MVP |

**Scaling:**
- High-frequency functions → higher overhead
- Consider sampling for very hot paths

### GPU Overhead

| Operation | Overhead | Note |
|-----------|----------|------|
| CUPTI callbacks | ~0.5-1% | Per kernel launch |
| Activity buffer | ~0.1-0.2% | Async collection |
| **Total** | **<2%** | Minimal impact |

## Single-Node Constraint

**Enforced at runtime:**
```python
# inferscope/__init__.py
def _validate_single_gpu():
    gpu_count = torch.cuda.device_count()  # or equivalent
    if gpu_count > 1:
        raise RuntimeError(
            "InferScope supports single-GPU only. "
            f"Detected {gpu_count} GPUs. "
            "Set CUDA_VISIBLE_DEVICES=0 to use one GPU."
        )
```

**User mitigation:**
```bash
CUDA_VISIBLE_DEVICES=0 inferscope run python infer.py
```

## Configuration at Deployment

### Environment Variables
```bash
# Enable/disable (default: enabled if inferscope imported)
export INFERSCOPE_ENABLED=1

# Trace buffer size (MB, default: 100)
export INFERSCOPE_TRACE_SIZE_MB=200

# Logging verbosity (debug, info, warn, error; default: info)
export INFERSCOPE_LOG_LEVEL=debug

# Clock sync error margin (microseconds; default: 1000)
export INFERSCOPE_SYNC_ERROR_MARGIN_US=1000

# Output format (markdown, html, json; default: markdown)
export INFERSCOPE_REPORT_FORMAT=markdown

# Report output path (default: ./report.md)
export INFERSCOPE_REPORT_PATH=/tmp/inference_report.md
```

### Config File (.inferscope.yaml)
```yaml
enabled: true
log_level: info
trace_size_mb: 100
sync_error_margin_us: 1000
report_format: markdown
report_path: ./report.md
cuda_version: 12.0
cpu_sampling_frequency_hz: 1000
```

## Isolation & Security

**No inter-process communication:**
- Traces stored in process memory only
- No network calls
- No data upload

**Workload isolation:**
- Each process has its own trace buffer
- No cross-process state
- Safe to run concurrently (different processes)

**Privilege requirements:**
- User-level execution (no root)
- May require debugger permissions for perf (on some systems)

## Failure Modes & Recovery

| Scenario | Default Behavior | User Recovery |
|----------|-----------------|----------------|
| CUDA unavailable | CPU-only mode (warn) | Ignore; analysis continues |
| CUPTI load fails | Skip GPU events (warn) | Set INFERSCOPE_LOG_LEVEL=debug for details |
| Trace buffer overflow | Wrap oldest events (warn) | Increase INFERSCOPE_TRACE_SIZE_MB |
| Clock sync error >5% | Use conservative margin | Check GPU clock stability; try --gpu-reset |
| Report generation fails | Exception + error message | Check disk space; verify REPORT_PATH writable |
| Multi-GPU detected | Abort with error | Set CUDA_VISIBLE_DEVICES=0 |

## Monitoring & Observability

### Logging Output
```
[2025-12-27 10:30:45.123] INFO     Trace collection started (buffer_size_mb=100)
[2025-12-27 10:30:45.456] DEBUG    CPU collector initialized (threads=8)
[2025-12-27 10:30:45.789] DEBUG    GPU collector initialized (device=0)
[2025-12-27 10:31:12.456] INFO     Trace collection completed (events=5432, duration_ms=87.4)
[2025-12-27 10:31:12.789] INFO     Timeline merger: CPU/GPU clock sync (error_us=42)
[2025-12-27 10:31:13.123] INFO     Bottleneck analysis: CPU-bound (confidence=0.92)
[2025-12-27 10:31:13.456] INFO     Report generated: /tmp/report.md
```

### Metrics to Track
- Collection time (ms)
- Events collected (count)
- Clock sync error (μs)
- Report generation time (ms)
- Profiling overhead (%)

## Scaling Considerations

**For longer traces (>1 second):**
1. Increase INFERSCOPE_TRACE_SIZE_MB
2. Consider disk spilling (future feature)
3. Monitor memory usage

**For high-concurrency scenarios (many threads):**
1. Per-thread buffers scale linearly
2. Profiling overhead increases
3. Consider sampling-based collection (future)

**For batch processing multiple inferences:**
- Reset trace buffer between runs
- Or use separate processes (preferred)
