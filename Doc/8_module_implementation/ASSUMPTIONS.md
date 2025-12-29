# Project Overview & Assumptions

This document captures key implementation assumptions and decisions made during the design phase.

## Design Assumptions

### System-Level Assumptions
1. **Single GPU per process** – Multi-GPU will be explicitly rejected at runtime
2. **Linux only (MVP)** – Windows/macOS support deferred to v2
3. **NVIDIA GPU required** – ROCm support deferred to v2
4. **CUDA 11.8+ required** – Older versions have different CUPTI APIs
5. **Inference < 10 minutes** – Ring buffer caps at ~1GB; longer traces need disk spilling

### Profiling Assumptions
1. **CPU overhead < 5%** – sys.settrace is lightweight enough
2. **GPU overhead < 2%** – CUPTI overhead minimal on modern GPUs
3. **Clock sync <1% error** – CUDA events provide sufficient accuracy
4. **No distributed memory** – Single-node only; no MPI, gRPC, or distributed tracing

### Framework Assumptions
1. **PyTorch is primary framework** – TensorFlow support comes post-MVP
2. **CUDA API consistent** – Same CUDA calls across versions
3. **User workload is deterministic** – No randomness in timing (or acceptable jitter)
4. **Python 3.8+** – Older Python EOL; type hints available

## Implementation TODOs & Deferred Items

### MVP Implementation (v0.1) - Current Status
- [x] CPU Collector (sys.settrace)
- [x] GPU Collector (CUPTI)
- [x] Timeline Merger (clock sync)
- [x] Analyzer Engine (bottleneck rules)
- [x] Report Generator (Markdown/HTML/JSON)
- [x] Documentation (full)
- [x] Unit tests (65 tests passing, 2 skipped)
- [x] End-to-end demo (Profiler → Analyzer → Reporter)
- [ ] CLI Interface (inferscope run, inferscope analyze) - **TODO**
- [ ] Python API (scope, mark_event) - **TODO**
- [ ] Pre-built wheels for CUDA 11.8, 12.0 (post-MVP)

### v0.2 Roadmap
- [ ] TensorFlow support
- [ ] Custom bottleneck rules (user-defined YAML)
- [ ] Multi-GPU error handling (more graceful)
- [ ] Disk spilling for long traces (>1GB)
- [ ] Real-time timeline visualization (web UI)

### v0.3+ (Deferred)
- [ ] ROCm (AMD) GPU support
- [ ] Distributed tracing (multi-node)
- [ ] Model-level annotations (layer-by-layer breakdown)
- [ ] Windows / macOS support
- [ ] Kubernetes integration

## Known Limitations

### Current (v0.1)
1. **Single GPU only** – Multi-GPU workloads will error out
2. **Python-centric** – C++ workloads not profiled (only CUDA kernels)
3. **GIL overhead** – Multi-threaded Python code introduces instrumentation bias
4. **Framework coupling** – Tight to PyTorch; TensorFlow needs separate path
5. **Clock drift ignored** – For sub-second traces only
6. **No kernel source mapping** – Can't show which model layer runs which kernel
7. **No CLI interface yet** – Use Python API or direct module imports for now

### Design Trade-offs Made

| Trade-off | Choice | Rationale |
|-----------|--------|-----------|
| Accuracy vs Overhead | Accept <5% overhead | <2% overhead would require complex buffering |
| Single vs Multi-GPU | Single GPU | Scope + reduce complexity for MVP |
| In-memory vs Disk buffer | In-memory ring buffer | Faster; disk I/O adds overhead |
| Python vs C++ | Hybrid (Python CLI + C++ collectors) | Best of both: usability + performance |
| Markdown vs Dashboard UI | Markdown reports | Version-controllable; Git-friendly; no UI overhead |
| Real-time vs Post-collection | Post-collection (off critical path) | No impact on inference latency; simpler architecture |

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| CUPTI API incompatibility across CUDA versions | Test on CUDA 11.8, 12.0; vendor-specific pins |
| sys.settrace overhead too high | Measure baseline; tune sampling; fallback to perf if needed |
| Clock sync error >5% | Use conservative error margin; warn in report; user can retry |
| Trace buffer overflow | Ring buffer wraps; warn; user can increase size |
| False bottleneck diagnosis | High confidence threshold (>0.8); multiple evidence required |
| GPU not available at runtime | Graceful fallback to CPU-only mode; warn user |

## Future Extensibility Points

### Plugin Architecture (Deferred)
```python
# Future: User-defined collectors
class CustomCollector(BaseCollector):
    def collect(self) -> List[Dict]:
        # User-defined events
        pass

# Automatic discovery: inferscope/collectors/user_*.py
```

### Rule Customization (Deferred)
```yaml
# Future: .inferscope_rules.yaml
bottleneck_rules:
  - name: "high_memory_overhead"
    condition: "h2d_time / total_time > 0.3"
    suggestion: "Use pinned memory or reduce batch size"
    confidence_weight: 1.2
```

### Multi-Framework Support (Deferred)
```python
# Future: Framework detection + dispatch
if "torch" in sys.modules:
    use_pytorch_hooks()
elif "tensorflow" in sys.modules:
    use_tensorflow_hooks()
```

## Success Criteria

### MVP Success (v0.1)
- ✓ Single-GPU inference profiling works
- ✓ Profiling overhead < 5%
- ✓ Clock sync error < 1%
- ✓ Reports are actionable (>2 suggestions per bottleneck)
- ✓ No critical bugs in first 100 users (estimated)
- ✓ GitHub stars > 500 (community validation)

### v1.0 Success
- ✓ TensorFlow support added
- ✓ 1000+ GitHub stars
- ✓ Used by at least 3 major AI inference platforms
- ✓ <1% of reports marked as "uncertain" or "invalid"

## Assumptions to Validate

Before deployment, confirm:
1. **Real-world overhead** – Test on 10+ diverse models
2. **Clock sync accuracy** – Validate on different GPU architectures
3. **User adoption** – Soft launch; gather feedback
4. **Report quality** – Manual review of 50 generated reports

## Context & References

- README.md: Project overview & use cases
- Doc/1_requirements/PRD.md: Detailed requirements
- Doc/2_system_architecture/SAD.md: Architecture details
- Doc/3_module_design/ICD.md: API specifications
