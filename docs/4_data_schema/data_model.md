# Data Model Specification

## Overview
This document defines the data structures, relationships, and constraints for InferScope traces and analysis results.

## 1. Trace Event Model

### 1.1 Core Event Structure

```json
{
  "id": "unique-event-id",
  "type": "cpu_call|gpu_kernel|h2d_copy|d2h_copy|memory_event|instant",
  "name": "function_or_kernel_name",
  "timestamp_start_us": 1234567890,
  "timestamp_end_us": 1234567920,
  "wall_clock_start": "2025-12-27T10:30:45.123456Z",
  "wall_clock_end": "2025-12-27T10:30:45.123486Z",
  "duration_us": 30,
  "metadata": {
    // Type-specific fields below
  }
}
```

### 1.2 Event Types & Schemas

#### CPU Call Event
```json
{
  "type": "cpu_call",
  "name": "tokenize",
  "duration_us": 31200,
  "metadata": {
    "thread_id": 12345,
    "cpu_id": 2,
    "call_depth": 5,
    "parent_scope": "inference",
    "allocations_bytes": 102400
  }
}
```

#### CPU Syscall Event
```json
{
  "type": "cpu_syscall",
  "name": "read",
  "duration_us": 150,
  "metadata": {
    "syscall_number": 0,
    "fd": 5,
    "bytes": 4096,
    "return_value": 4096
  }
}
```

#### GPU Kernel Event
```json
{
  "type": "gpu_kernel",
  "name": "attention_forward",
  "duration_us": 20300,
  "metadata": {
    "device_id": 0,
    "grid_dim": [32, 1, 1],
    "block_dim": [256, 1, 1],
    "shared_memory_bytes": 8192,
    "registers_per_thread": 64,
    "stream_id": 7,
    "occupancy": 0.87
  }
}
```

#### H2D Copy Event
```json
{
  "type": "h2d_copy",
  "name": "copy_inputs",
  "duration_us": 18400,
  "metadata": {
    "device_id": 0,
    "source_ptr": "host:0x7f1a2b3c",
    "dest_ptr": "device:0xdeadbeef",
    "bytes": 1048576,
    "kind": "pageable|pinned",
    "throughput_gbps": 45.2
  }
}
```

#### D2H Copy Event
```json
{
  "type": "d2h_copy",
  "name": "copy_outputs",
  "duration_us": 12200,
  "metadata": {
    "device_id": 0,
    "source_ptr": "device:0xcafebabe",
    "dest_ptr": "host:0x7f4d5e6f",
    "bytes": 524288,
    "kind": "pageable|pinned",
    "throughput_gbps": 42.8
  }
}
```

#### Memory Event
```json
{
  "type": "memory_event",
  "name": "malloc",
  "duration_us": 45,
  "metadata": {
    "operation": "malloc|free|numa_migration",
    "bytes": 262144,
    "numa_node": 0,
    "numa_pages_migrated": 64
  }
}
```

#### Instant Event
```json
{
  "type": "instant",
  "name": "tokenization_complete",
  "timestamp_us": 1234567900,
  "metadata": {
    "user_metadata": {
      "tokens": 1024,
      "batch_size": 32
    }
  }
}
```

---

## 2. Trace File Format

### 2.1 Trace File Structure

```json
{
  "format_version": "1.0",
  "trace_metadata": {
    "timestamp_created": "2025-12-27T10:35:12.456Z",
    "duration_ms": 87.4,
    "cuda_version": "12.0",
    "python_version": "3.10.5",
    "torch_version": "2.1.0",
    "gpu_model": "NVIDIA A100",
    "hostname": "inference-box-1"
  },
  "system_info": {
    "cpu_count": 16,
    "cpu_model": "Intel Xeon Platinum 8360Y",
    "numa_nodes": 2,
    "ram_gb": 256,
    "gpu_count": 1
  },
  "events": [
    { /* event 1 */ },
    { /* event 2 */ },
    ...
  ],
  "relationships": {
    "scopes": [
      {
        "id": "scope_inference_001",
        "name": "inference",
        "parent_id": null,
        "child_ids": ["scope_prep_001", "scope_gpu_001"],
        "event_range": [0, 120]
      }
    ]
  }
}
```

### 2.2 Schema Versioning

**Version 1.0:**
- Basic event types: cpu_call, gpu_kernel, h2d_copy, d2h_copy
- Metadata: type-specific fields
- Relationships: scope hierarchy

**Future Compatibility:**
- Additive fields only (no removal/rename)
- Version field mandatory
- Readers must skip unknown fields

---

## 3. Analysis Result Model

### 3.1 Bottleneck Analysis Result

```json
{
  "analysis_version": "1.0",
  "summary": {
    "end_to_end_latency_us": 87400,
    "total_cpu_time_us": 31200,
    "total_gpu_time_us": 24100,
    "total_h2d_us": 18400,
    "total_d2h_us": 8700,
    "total_idle_us": 17500,
    "framework_overhead_us": 13700
  },
  "timeline_breakdown": [
    {
      "category": "cpu_preprocessing",
      "duration_us": 31200,
      "percentage": 35.7,
      "examples": ["tokenization", "data_prep"]
    },
    {
      "category": "h2d_copy",
      "duration_us": 18400,
      "percentage": 21.0
    },
    {
      "category": "gpu_compute",
      "duration_us": 24100,
      "percentage": 27.6
    },
    {
      "category": "d2h_copy",
      "duration_us": 8700,
      "percentage": 10.0
    },
    {
      "category": "idle",
      "duration_us": 17500,
      "percentage": 20.0
    }
  ],
  "bottleneck": {
    "type": "cpu_bound|gpu_bound|memory_bound|balanced",
    "primary_cause": "cpu_preprocessing",
    "confidence": 0.92,
    "evidence": [
      "GPU idle for 20% of timeline",
      "Tokenization takes 35.7% of time",
      "GPU compute only 27.6%"
    ]
  },
  "suggestions": [
    {
      "priority": "high",
      "action": "Increase batch size",
      "rationale": "Amortize H2D copy overhead",
      "estimated_improvement_percent": 12
    },
    {
      "priority": "high",
      "action": "Enable pinned memory",
      "rationale": "Improve H2D throughput from 45 to 55 GB/s",
      "estimated_improvement_percent": 8
    }
  ]
}
```

### 3.2 Report Output Model

```json
{
  "report_type": "markdown|html|json",
  "title": "InferScope Performance Report",
  "timestamp": "2025-12-27T10:35:12Z",
  "summary": "User-friendly summary text",
  "analysis": { /* Bottleneck Analysis Result */ },
  "timeline_sections": [
    {
      "title": "Timeline Breakdown",
      "data": [ /* breakdown by category */ ]
    }
  ],
  "recommendations": [
    {
      "title": "Recommendation 1",
      "description": "...",
      "impact": "Estimated improvement: X%"
    }
  ]
}
```

---

## 4. Constraints & Invariants

### 4.1 Event Constraints

| Constraint | Requirement | Rationale |
|-----------|-----------|-----------|
| Event ID uniqueness | Each event has unique ID | Enable references, deduplication |
| Timestamp ordering | start_us < end_us (or instant) | Logical consistency |
| Non-overlapping kernels | Kernels on same GPU don't overlap | GPU scheduling property |
| Parent scope containment | Child events within parent time range | Scope hierarchy validity |

### 4.2 Analysis Constraints

| Constraint | Requirement | Rationale |
|-----------|-----------|-----------|
| Percentage sum | Sum of timeline breakdown ≈ 100% ± 5% | Accounting completeness |
| Confidence range | Confidence in [0.0, 1.0] | Uncertainty quantification |
| Evidence support | Each suggestion backed by ≥1 evidence | Explainability |

---

## 5. Indexing Strategy

**Trace File Indexing (post-collection):**
- Index by event type (for quick filtering)
- Index by timestamp range (for timeline queries)
- Index by scope hierarchy (for scope-level analysis)

**Query Examples:**
- "All GPU kernels between 10-50 ms" → timestamp index
- "All events in scope 'inference'" → scope index
- "Count H2D copies" → event type index

---

## 6. Retention Policies

**Trace Retention:**
- In-memory: Circular ring buffer; oldest events wrap
- Disk: User decides (report generation may trim large traces)

**Report Retention:**
- User owns retention; reports are files
- Suggested: Keep reports; discard raw traces after verification

---

## 7. Example: Complete Trace

See `/docs/6_test_cases/example_trace.json` for a realistic end-to-end trace example.
