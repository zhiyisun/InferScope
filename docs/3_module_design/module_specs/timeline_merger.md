# Timeline Merger Module Specification

## Overview
The Timeline Merger synchronizes CPU and GPU clocks, orders all events by global wall-clock time, and produces a unified timeline for analysis.

## Responsibilities
- Establish CPU ↔ GPU clock relationship via calibration
- Apply clock offset to all collected events
- Sort events by global timestamp
- Handle concurrent events and maintain causality

## Public APIs

### TimelineMerger Class

```python
class TimelineMerger:
    def __init__(self, cpu_events: List[Dict], gpu_events: List[Dict]):
        """Initialize merger with separate CPU and GPU event streams."""
    
    def synchronize_clocks(self) -> Dict[str, float]:
        """
        Calibrate CPU ↔ GPU timestamp mapping.
        Returns: {"slope": float, "intercept": float, "error_us": float}
        """
    
    def get_unified_timeline(self) -> List[Dict]:
        """Return all events sorted by global timestamp."""
    
    def get_sync_metadata(self) -> Dict:
        """Return clock sync calibration details for report."""
```

## Input/Output Contracts

**Input:**
- CPU events (timestamps in `rdtsc` or wall-clock format)
- GPU events (timestamps in GPU clock format)

**Output:**
- Unified event list sorted by timestamp
- All events with synchronized timestamps
- Sync metadata (slope, intercept, error margin)

## Clock Synchronization Algorithm

**Calibration Procedure:**

1. Record CPU timestamp T_cpu_1 = rdtsc()
2. Issue CUDA event E on active GPU stream
3. Record CPU timestamp T_cpu_2 = rdtsc()
4. Query CUDA event with cuEventQuery(); get GPU-side timestamp T_gpu
5. Calculate:
   - CPU midpoint: T_cpu_mid = (T_cpu_1 + T_cpu_2) / 2
   - Slope: m = (T_gpu - T_cpu_mid) / (T_cpu_2 - T_cpu_1)
   - Intercept: b = T_gpu - m * T_cpu_mid
6. Apply to all CPU events: T_synchronized = m * T_cpu + b

**Accuracy:** <1% error typical (±100 μs for 100ms trace)

## Error Handling

| Error | Behavior |
|-------|----------|
| CUDA event query fails | Use conservative error margin (1000 μs); warn |
| Empty CPU events | CPU-only timeline (no GPU events) |
| Empty GPU events | GPU-only timeline (no CPU events) |
| Clock skew too large (>5%) | Abort; report error; suggest GPU reset |

## Concurrency Model

- **Single-threaded**: Merge only occurs after collection finalized
- **No collectors running**: Safe to sort and reorder events

## State Machine

```
[Uninitialized] --init--> [Ready] --sync--> [Synchronized] --merge--> [Finalized]
```

## Invariants

1. **Non-overlapping GPU kernels**: On same device, no kernels overlap in time
2. **Total time consistency**: Sum of timeline components ≈ end-to-end latency ± 1%
3. **Causality preservation**: Parent scope spans all child events

## Testing Notes

- Unit tests: mock CUDA event queries with known offsets
- Integration tests: real GPU with controlled kernel timing
- Validation: verify timestamp ordering and causality

## Known Limitations

- Assumes single GPU with synchronized clock
- Clock drift ignored (ok for sub-second traces)
- Concurrent multi-GPU events not supported (MVP single GPU)
