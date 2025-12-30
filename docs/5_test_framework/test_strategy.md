# Test Strategy Document

## Test Levels

### Unit Tests
- **Scope**: Individual modules (collectors, analyzer, report generator)
- **Approach**: Mock external dependencies (CUDA, sys.settrace)
- **Coverage Goal**: >80% code coverage per module
- **Framework**: pytest

### Integration Tests
- **Scope**: Cross-module workflows (collection → merger → analyzer)
- **Approach**: Real or simulated workloads; minimal GPU/CPU requirements
- **Framework**: pytest with real inference workloads

### System Tests
- **Scope**: End-to-end inference profiling
- **Approach**: Real PyTorch models; validate report quality
- **Coverage**: LLM, CNN, embedding models

### Performance Tests
- **Scope**: Profiling overhead and accuracy
- **Target**: <5% overhead; <1% clock error

---

## Mocking Strategy

| Component | Mock Approach |
|-----------|--------------|
| CUDA/CUPTI | Fake event stream with known timestamps |
| sys.settrace | Inject synthetic call stack |
| PyTorch models | Minimal models (small attention, MLP) |
| perf sampling | Synthetic CPU samples |

---

## Test Coverage Goals

- **Code coverage**: >80% overall; >90% critical paths
- **Requirements coverage**: Each FR/NFR has ≥1 test
- **Bottleneck rules**: Each rule tested with synthetic data

---

## CI/CD Integration

- **Trigger**: On every commit to main branch
- **Parallel jobs**: Unit tests, integration tests, system tests
- **Hardware**: CPU-only for unit/integration; GPU for system tests (optional)
- **Artifacts**: Coverage reports, timing logs, example reports

---

## Known Limitations

- System tests require GPU (deferred if unavailable)
- Performance tests sensitive to system load
- Clock sync tests need real CUDA GPU
