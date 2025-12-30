# CPU Collector Module Specification

## Overview
The CPU Collector captures Python function calls, system calls, and CPU execution time, providing insight into CPU-side bottlenecks during inference.

## Responsibilities
- Hook into Python interpreter to capture function enter/exit events
- Sample CPU execution using Linux perf or similar
- Track NUMA-aware memory allocations and migrations
- Maintain per-thread trace buffers with minimal overhead

## Public APIs

### CpuCollector Class

```python
class CpuCollector:
    def __init__(self, trace_buffer: TraceBuffer):
        """Initialize CPU collector with reference to shared trace buffer."""
    
    def start(self) -> None:
        """Enable profiling hooks; begin collecting events."""
    
    def stop(self) -> None:
        """Disable profiling hooks; finalize collection."""
    
    def get_thread_local_buffer(self) -> List[Dict]:
        """Return accumulated events for current thread (non-blocking)."""
```

## Input/Output Contracts

**Input:**
- User code execution (Python interpreter)
- NUMA memory system state

**Output:**
- Stream of CPU events to shared trace buffer
- Event types: `cpu_call`, `cpu_syscall`, `memory_alloc`, `memory_free`

## Error Handling

| Error | Behavior |
|-------|----------|
| sys.settrace already hooked | Chain hook; call previous handler |
| perf unavailable | Warn; skip perf sampling (settrace only) |
| NUMA API unavailable | Skip NUMA tracking; continue |
| Thread local storage full | Warn; discard oldest events |

## Concurrency Model

- **Per-thread buffers**: Each thread maintains its own lock-free circular buffer
- **Merge at end**: Main thread collects all per-thread buffers during finalization
- **No synchronization**: Collectors and main thread don't contend on locks

## State Machine

```
[Uninitialized] --init--> [Idle] --start--> [Collecting] --stop--> [Finalized]
```

## Testing Notes

- Mock sys.settrace for unit tests
- Use synthetic workloads for integration tests
- Measure overhead: target <5% CPU overhead from instrumentation

## Known Limitations

- GIL introduces bias for multi-threaded Python code
- C-extension calls (PyTorch, NumPy) not directly instrumented (frame events only)
- Memory tracking is approximation (sampled)
