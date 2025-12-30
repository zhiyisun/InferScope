# Integration Plan

## Implementation Status (v0.1 - Current)

### Completed Phases ✅

**Phase 1: Foundation** (Weeks 1-2)
- ✅ Trace Buffer: Ring buffer implementation, thread-safe operations
- ✅ Configuration System: Utility patterns established

**Phase 2: Collectors** (Weeks 3-4)
- ✅ CPU Collector: sys.settrace with per-thread TLS buffer
- ✅ GPU Collector: CUPTI callback mechanism (graceful fallback to CPU-only)

**Phase 3: Post-Collection Processing** (Weeks 5-6)
- ✅ Timeline Merger: Clock synchronization, event ordering, <1% error
- ✅ Analyzer Engine: 5 bottleneck detection rules, confidence scoring, suggestions

**Phase 4: Output & Reporting** (Weeks 7-8)
- ✅ Report Generator: Markdown, HTML, and JSON output formats
- ✅ End-to-End Demo: examples/demo_llm_inference.py demonstrating full pipeline

### In-Progress / Planned

**Phase 5: User-Facing Interfaces** (Next)
- ⏳ CLI Interface (`inferscope run`, `inferscope analyze`)
- ⏳ Python API (`scope()`, `mark_event()`)

**Phase 5: System Integration & Testing** (Weeks 9-10)
- ⏳ Real workload examples (LLM, CNN, embeddings)
- ⏳ Kubernetes integration

## Original Integration Order (Weeks 1-10)
1. **Trace Buffer** (utility, no dependencies)
   - Ring buffer implementation
   - Thread-safe enqueue/dequeue
   - Tests: unit tests for capacity, overflow, ordering

2. **Configuration System** (utility, no dependencies)
   - YAML/env var parsing
   - Config validation
   - Tests: unit tests for precedence, validation

### Phase 2: Collectors (Weeks 3-4)
3. **CPU Collector** (depends on: trace buffer, config)
   - sys.settrace hook implementation
   - Per-thread buffer management
   - Tests: unit tests with synthetic call stacks

4. **GPU Collector** (depends on: trace buffer, config)
   - CUPTI initialization and callback handling
   - Activity buffer management
   - Tests: unit tests with mock CUDA events (if GPU unavailable)

### Phase 3: Post-Collection Processing (Weeks 5-6)
5. **Timeline Merger** (depends on: trace buffer, CPU/GPU collectors)
   - Clock synchronization algorithm
   - Event ordering
   - Tests: integration tests with real CPU/GPU events

6. **Analyzer Engine** (depends on: timeline merger)
   - Bottleneck detection rules
   - Suggestion generation
   - Tests: unit tests with synthetic timelines

### Phase 4: Output & CLI (Weeks 7-8)
7. **Report Generator** (depends on: analyzer engine)
   - Markdown/HTML template rendering
   - JSON output
   - Tests: unit tests with synthetic analysis results

8. **CLI Interface** (depends on: all collectors, merger, analyzer, reporter)
   - `inferscope run` command
   - `inferscope analyze` command
   - Tests: integration tests with real scripts

### Phase 5: System Integration & Testing (Weeks 9-10)
9. **End-to-End Pipeline**
   - Full workflow: collect → merge → analyze → report
   - Example workloads (LLM, CNN, embeddings)
   - Tests: system tests with real inference models

10. **Python API** (depends on: collectors, trace buffer)
    - `scope()` context manager
    - `mark_event()` function
    - Tests: integration tests with user code

---

## Dependency Graph

```
┌─────────────────────────────────────┐
│   Configuration System (utility)    │
└─────────────────────────────────────┘
    ↑                ↑
    │                │
┌───┴────────────────┴──────────────────┐
│   Trace Buffer (utility)              │
└───┴──────────────────────────┬────────┘
    │                          │
    ├──────┐                   ├──────┐
    │      │                   │      │
    v      v                   v      v
┌──────┐┌──────────┐      ┌──────────┐┌──────────┐
│ CLI  ││ API      │      │ CPU      ││ GPU      │
│      ││ (scope)  │      │ Collector││Collector│
└──────┘└────┬─────┘      └────┬─────┘└────┬────┘
             │                 │            │
             └─────────────────┼────────────┘
                               │
                               v
                       ┌───────────────┐
                       │ Timeline      │
                       │ Merger        │
                       └───────┬───────┘
                               │
                               v
                       ┌───────────────┐
                       │ Analyzer      │
                       │ Engine        │
                       └───────┬───────┘
                               │
                               v
                       ┌───────────────┐
                       │ Report        │
                       │ Generator     │
                       └───────────────┘
```

---

## Rollback Strategy

| Component | Rollback Trigger | Action |
|-----------|-----------------|--------|
| Trace Buffer | Frequent overflows | Increase buffer size; add disk spilling |
| CPU Collector | High overhead (>5%) | Reduce sampling frequency; profile hotspots |
| GPU Collector | CUPTI API version mismatch | Pin CUDA version; support multiple CUPTI versions |
| Timeline Merger | Clock sync error >5% | Increase calibration samples; warn user |
| Analyzer Engine | False bottleneck classification | Add confidence thresholds; manual validation |
| CLI | Breaking API changes | Maintain backward compatibility; deprecation period |

---

## Cross-Team Integration Points

| Integration Point | Teams | Protocol | Frequency |
|------------------|-------|----------|-----------|
| Trace Buffer API | Collectors, Merger | Python interface (ICD.md) | Per module merge |
| Collector Output Format | Collectors, Merger | JSON schema (schema.json) | Per module merge |
| Timeline Schema | Merger, Analyzer | JSON schema (schema.json) | Per phase |
| Report Format | Analyzer, Reporter | JSON structure (data_model.md) | Per phase |
| CLI Arguments | All, CLI | requirements.yaml | Per PR |

---

## Testing Strategy per Phase

**Phase 1-2:** Unit tests only (isolated modules)
**Phase 3-4:** Integration tests (cross-module workflows)
**Phase 5:** System tests (real inference workloads) + performance tests

**Gate Criteria per Phase:**
- Phase 1-2: >80% unit test coverage
- Phase 3-4: End-to-end pipeline works; <5% profiling overhead
- Phase 5: All system tests pass; performance within targets

---

## Parallel Development

**Can be developed in parallel:**
- CPU Collector & GPU Collector (different subsystems)
- Analyzer Rules & Report Templates (post-merger)
- Documentation (all phases)

**Must be sequential:**
- Trace Buffer → Collectors (dependencies)
- Collectors → Merger (data dependency)
- Merger → Analyzer (data dependency)
- Analyzer → Reporter (data dependency)

---

## Deployment Phases

### MVP (Phase 5 complete)
- Single-GPU support ✓
- PyTorch support ✓
- Markdown reports ✓
- CLI interface ✓

### v0.2 (Post-MVP)
- TensorFlow support
- HTML reports
- Custom bottleneck rules (YAML)
- Multi-GPU warnings (explicit error)

### v0.3+
- ROCm (AMD) GPU support
- Distributed tracing (deferred)
- Real-time dashboard (deferred)
