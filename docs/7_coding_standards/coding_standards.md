# Coding Standards & Best Practices

## Python Code Style

### Style Guide
- **Standard**: PEP 8
- **Line length**: 100 characters (pragmatic limit)
- **Imports**: Sorted (isort configuration in `pyproject.toml`)
- **Formatting**: Black (code formatter)

### Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Module | lowercase_with_underscores | `cpu_collector.py` |
| Class | PascalCase | `CpuCollector` |
| Function | lowercase_with_underscores | `synchronize_clocks()` |
| Constant | UPPER_CASE_WITH_UNDERSCORES | `MAX_TRACE_SIZE_MB` |
| Private | _leading_underscore | `_internal_buffer` |

### Docstrings

**Function/Method Docstrings (Google Style):**
```python
def synchronize_clocks(self) -> Dict[str, float]:
    """Synchronize CPU and GPU clocks via calibration.
    
    Establishes a linear mapping between CPU timestamps (rdtsc) and GPU
    timestamps using CUDA events. Runs a calibration routine that samples
    both clocks simultaneously and computes slope/intercept.
    
    Returns:
        Dictionary with keys:
        - "slope": float, CPU-to-GPU timestamp ratio
        - "intercept": float, intercept of linear mapping
        - "error_us": float, estimated synchronization error in microseconds
    
    Raises:
        RuntimeError: If CUDA event query fails or clock skew exceeds 5%
    """
```

### Error Handling

**DO:**
```python
try:
    result = gpu_collector.get_events()
except CudaError as e:
    logger.warn(f"GPU collection failed: {e}; falling back to CPU-only")
    result = []
```

**DON'T:**
```python
try:
    result = gpu_collector.get_events()
except Exception:
    pass  # Silent failure is bad
```

### Logging

**Use structured logging:**
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Trace collection started", extra={
    "duration_ms": 100,
    "events": 500,
    "buffer_size_mb": 50
})
```

**Log levels:**
- `debug`: Detailed info for developers (trace events, timestamps)
- `info`: Milestones (collection started, report generated)
- `warn`: Recoverable issues (GPU unavailable, clock sync uncertainty)
- `error`: Fatal issues (trace format corrupted)

---

## C++ Code Style

### Style Guide
- **Standard**: Google C++ Style Guide
- **Compiler**: C++17
- **Build**: CMake 3.18+

### Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Class | PascalCase | `CpuCollector` |
| Method | PascalCase | `SynchronizeClocks()` |
| Variable | snake_case | `buffer_size` |
| Constant | kPrefixPascalCase | `kMaxTraceSize` |
| Macro | UPPER_CASE | `CUPTI_CHECK(api_call)` |

### CUPTI Integration

**Error handling macro:**
```cpp
#define CUPTI_CHECK(call)                                          \
    do {                                                           \
        CUptiResult status = (call);                               \
        if (CUPTI_SUCCESS != status) {                             \
            const char *errstr = NULL;                             \
            cuptiGetResultString(status, &errstr);                 \
            fprintf(stderr, "CUPTI Error: %s\n", errstr);         \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)
```

---

## Testing Standards

### Test Naming
```python
# Format: test_<module>_<scenario>_<expectation>
def test_cpu_collector_captures_calls_with_correct_timestamps():
    pass

def test_gpu_collector_cuda_unavailable_falls_back_gracefully():
    pass
```

### Fixtures (pytest)
```python
@pytest.fixture
def cpu_collector():
    """Fixture for CpuCollector instance."""
    collector = CpuCollector(trace_buffer=MockTraceBuffer())
    yield collector
    collector.stop()  # cleanup
```

### Assertions
```python
# Good: descriptive assertion
assert analysis["bottleneck_type"] == "cpu_bound", \
    f"Expected CPU-bound, got {analysis['bottleneck_type']}"

# Bad: unclear
assert analysis["gpu_idle"] > 0.1
```

---

## Documentation Standards

### Module Docstrings
```python
"""CPU profiling and timeline collection.

This module hooks into the Python interpreter using sys.settrace() to capture
function call events. It maintains per-thread buffers for lock-free collection
and provides APIs for query and finalization.

Classes:
    CpuCollector: Main entry point for CPU event collection.

Functions:
    _thread_local_hook: Internal sys.settrace hook (private).
"""
```

### Inline Comments
```python
# DO: Explain *why*, not *what*
# We use a ring buffer to avoid allocation during collection
buffer = RingBuffer(capacity_mb=100)

# DON'T: State the obvious
# x = y + 1  # Add 1 to y
```

---

## Configuration Management

### Environment Variables
```
INFERSCOPE_ENABLED=1                    # Enable/disable
INFERSCOPE_LOG_LEVEL=debug              # Verbosity
INFERSCOPE_TRACE_SIZE_MB=100            # Buffer size
```

### Config File (.inferscope.yaml)
```yaml
enabled: true
log_level: info
trace_size_mb: 100
output_format: markdown
cuda_version: 12.0
```

### Precedence
1. CLI arguments (highest priority)
2. Environment variables
3. Config file
4. Hardcoded defaults (lowest priority)

---

## Repository Structure Guide

```
InferScope/
├── src/
│   ├── inferscope/           # Python package
│   │   ├── __init__.py
│   │   ├── cli.py            # CLI entry point
│   │   ├── collectors/
│   │   │   ├── cpu.py
│   │   │   ├── gpu.py
│   │   │   └── __init__.py
│   │   ├── merger.py
│   │   ├── analyzer.py
│   │   ├── reporter.py
│   │   └── trace_buffer.py
│   └── cpp/                  # C++ extensions
│       ├── cupti_wrapper.cpp
│       ├── CMakeLists.txt
│       └── include/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── system/
│   └── fixtures/
├── docs/                      # Documentation (per AI workflow)
├── pyproject.toml            # Python package config
├── setup.py                  # Setup script
└── README.md
```

---

## DO / DON'T Rules

### DO
- ✅ Write unit tests alongside implementation
- ✅ Log errors with full context (not just exception type)
- ✅ Use type hints (Python 3.8+)
- ✅ Document assumptions and limitations
- ✅ Handle CUDA unavailability gracefully
- ✅ Validate user configuration early

### DON'T
- ❌ Silently ignore errors (always log)
- ❌ Use global state (prefer dependency injection)
- ❌ Mix GPU and CPU code without clear boundaries
- ❌ Assume single GPU (validate explicitly)
- ❌ Hardcode paths or configuration
- ❌ Leave TODOs without issue references

---

## Pre-Commit Checks

Run before committing:
```bash
# Linting
flake8 src/inferscope --max-line-length=100

# Formatting (auto-fix)
black src/inferscope --line-length=100

# Type checking
mypy src/inferscope --strict

# Unit tests
pytest tests/unit -v

# Code coverage
pytest tests/unit --cov=src/inferscope --cov-report=html
```

Add to `.pre-commit-config.yaml` for automated checks.
