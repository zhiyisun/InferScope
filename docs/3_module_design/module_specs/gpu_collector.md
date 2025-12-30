# GPU Collector Module Specification

## Overview
The GPU Collector captures NVIDIA CUDA kernel execution, memory transfers (H2D/D2H), and GPU timeline events using the NVIDIA CUPTI profiling API.

## Responsibilities
- Intercept CUDA kernel launches via CUPTI Callback API
- Record GPU kernel execution via CUPTI Activity API
- Capture Host-to-Device and Device-to-Host memory transfers
- Provide accurate GPU timestamps with minimal profiling overhead

## Public APIs

### GpuCollector Class

```python
class GpuCollector:
    def __init__(self, trace_buffer: TraceBuffer):
        """Initialize GPU collector; load CUPTI library."""
    
    def start(self) -> None:
        """Enable CUPTI callbacks and activity collection."""
    
    def stop(self) -> None:
        """Disable CUPTI collection; finalize GPU buffer."""
    
    def get_gpu_events(self) -> List[Dict]:
        """Return collected GPU events with synchronized timestamps."""
```

## Input/Output Contracts

**Input:**
- CUDA API calls from user code
- CUDA runtime stream state

**Output:**
- Stream of GPU events to shared trace buffer
- Event types: `gpu_kernel`, `h2d_copy`, `d2h_copy`, `gpu_memory`
- Synchronized timestamps (see timeline merger)

## Error Handling

| Error | Behavior |
|-------|----------|
| CUPTI initialization fails | Graceful degradation: warn, skip GPU collection |
| CUDA not available | Warn; CPU-only mode |
| Activity buffer overflow | Warn; oldest events dropped |
| Clock sync failure | Use conservative margin; annotate in report |

## Concurrency Model

- **Main thread only**: CUPTI callbacks must run on CUDA-context-owning thread
- **Async buffer**: GPU activity buffer filled asynchronously by GPU device
- **Flush on stop**: Explicit flush to ensure all pending GPU events captured

## State Machine

```
[Uninitialized] --init--> [Idle] --start--> [Collecting] --stop--> [Finalized]
```

## CUPTI Integration Details

**Callback API:**
- Hook `cuLaunchKernel` for kernel launch annotations

**Activity API:**
- Global buffer for GPU kernel execution, memory copies
- Periodic flushing to retrieve completed events

**Clock Synchronization:**
- Use CUDA events to establish CPU ↔ GPU timestamp mapping
- Applies offset to all GPU timestamps during finalization

## Testing Notes

- Requires NVIDIA GPU and CUDA toolkit
- Mock CUPTI for unit tests
- Integration tests: use simple CUDA kernels to verify event capture

## Known Limitations

- NVIDIA-only (ROCm support deferred to v2)
- Cannot profile kernels without CUPTI (e.g., closed-source kernels)
- Profiling overhead ~1-2% on typical GPU workloads
