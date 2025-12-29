# Technology Selection Rationale

## Language & Runtime

### Selected: Python (CLI/API) + C++ (Core Collectors)

**Rationale:**
- **Python**: User-friendly CLI, easy integration with ML frameworks (PyTorch, TensorFlow)
- **C++**: High-performance trace collection with minimal overhead (<5% goal)
- **Hybrid**: C++ extensions via pybind11 for critical path

**Alternatives Considered:**
- Pure Python: Slow for trace collection, overhead would exceed 5% target
- Pure Rust: Overkill for MVP, longer development cycle
- Go: No direct CUDA/CUPTI support

## Trace Collection Technology

### CPU Profiling: sys.settrace() + perf

**Selected:**
- `sys.settrace()`: Python function-level visibility
- Linux `perf`: System-level CPU sampling
- `/proc/self/stat`: Process-wide CPU time

**Rationale:**
- No kernel modifications required
- Works with all Python interpreters
- Proven low overhead

**Limitations:**
- GIL introduces instrumentation bias (acceptable for non-GIL-bound code)
- Python 2 unsupported (acceptable; Python 3.8+ only)

### GPU Profiling: NVIDIA CUPTI

**Selected:**
- CUPTI Callback API: Lightweight kernel launch interception
- CUPTI Activity API: Post-collection GPU event buffer

**Rationale:**
- Official NVIDIA profiling API; well-documented
- Supports all CUDA kernels (including custom kernels)
- Hardware-level precision

**Limitations:**
- NVIDIA-only (ROCm support deferred)
- Requires CUDA 11.8+

## Timeline Synchronization

### Selected: CUDA Event + CPU rdtsc Calibration

**Algorithm:**
1. Record CPU rdtsc at known wall-clock time
2. Record CUDA event on GPU stream
3. Query CUDA event's CPU-side timestamp
4. Calculate rdtsc → GPU timestamp mapping
5. Use mapping to convert all CPU timestamps to GPU clock domain

**Rationale:**
- Achieves <1% sync error with minimal overhead
- No kernel module required
- Portable across NVIDIA GPUs

**Alternatives:**
- PTP (Precision Time Protocol): Overkill for single-node
- GPU firmware timestamps: Firmware-dependent accuracy

## Data Format

### Selected: JSON + Optional Binary (CEF - Common Event Format-inspired)

**Rationale:**
- JSON: Human-readable, simple schema evolution
- Binary: Optional for large-scale traces (>1GB)

**Schema:**
```json
{
  "event_type": "gpu_kernel",
  "name": "attention_forward",
  "start_us": 1234567,
  "end_us": 1234589,
  "device_id": 0,
  "grid_dim": [32, 1, 1],
  "block_dim": [256, 1, 1]
}
```

## Report Format

### Selected: Markdown + Optional HTML

**Rationale:**
- Markdown: Version-controllable, Git-friendly
- HTML: Optional for prettier presentation
- Both: Machine-readable JSON also exported

**Template Example:**
```markdown
# InferScope Report

End-to-end latency: 87.4 ms

## Timeline Breakdown
- CPU preprocessing: 31.2 ms (35.7%)
- H2D copy: 18.4 ms (21.0%)
- GPU compute: 24.1 ms (27.6%)
- Framework overhead: 13.7 ms (15.7%)
```

## Build & Packaging

### Selected: pip + Poetry/setuptools

**Rationale:**
- Standard Python packaging
- Easy distribution via PyPI
- Automatic CUDA detection

**C++ Build:**
- CMake 3.18+ for CUDA compilation
- Pre-built wheels for common CUDA versions (11.8, 12.0+)

## Testing Framework

### Selected: pytest (Python) + Google Test (C++)

**Rationale:**
- pytest: Mature, widely-used in ML community
- Google Test: C++ standard; good fixture support
- CI/CD ready

## Configuration

### Selected: YAML + Environment Variables

**Rationale:**
- YAML: Human-readable, hierarchical
- Env vars: Container/cloud-friendly
- Precedence: CLI args > env vars > config file > defaults
