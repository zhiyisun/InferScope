# Repository Structure Guide

## Directory Layout

```
InferScope/
│
├── README.md                          # Project overview
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Python package config
├── setup.py                           # Setup script
├── setup.cfg                          # Setup configuration
│
├── docs/                               # AI-friendly documentation
│   ├── 1_requirements/
│   │   ├── PRD.md                     # Product Requirement Document
│   │   └── requirements.yaml          # Machine-readable requirements
│   ├── 2_system_architecture/
│   │   ├── SAD.md                     # System Architecture Document
│   │   ├── technology_rationale.md    # Tech choices & tradeoffs
│   │   └── architecture.yaml          # Component definitions
│   ├── 3_module_design/
│   │   ├── ICD.md                     # Interface Control Document
│   │   ├── interfaces.yaml            # Machine-readable APIs
│   │   └── module_specs/
│   │       ├── cpu_collector.md
│   │       ├── gpu_collector.md
│   │       ├── timeline_merger.md
│   │       ├── analyzer_engine.md
│   │       └── report_generator.md
│   ├── 4_data_schema/
│   │   ├── data_model.md              # Data Model Specification
│   │   ├── config_spec.md
│   │   └── schema.json                # JSON Schema for traces
│   ├── 5_test_framework/
│   │   ├── test_strategy.md
│   │   └── test_framework.md
│   ├── 6_test_cases/
│   │   ├── unit_tests.yaml
│   │   ├── integration_tests.yaml
│   │   ├── system_tests.yaml
│   │   ├── edge_cases.yaml
│   │   └── traces/                    # Example trace files
│   ├── 7_coding_standards/
│   │   ├── coding_standards.md
│   │   └── repo_structure.md          # This file
│   ├── 8_system_integration/
│   │   ├── integration_plan.md
│   │   ├── deployment_architecture.md
│   │   └── integration.yaml
│   ├── 9_system_test/
│   │   ├── system_test_spec.md
│   │   ├── performance_benchmarks.md
│   │   └── acceptance_criteria.yaml
│   └── 10_operations/
│       ├── runbooks.md
│       ├── alert_definitions.md
│       ├── logging_metrics.md
│       └── failure_patterns.md
│
├── src/
│   └── inferscope/                    # Main Python package
│       ├── __init__.py                # Package init, version
│       ├── cli.py                     # CLI entry point
│       ├── api.py                     # Public Python API (scope, mark_event)
│       │
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseCollector abstract class
│       │   ├── cpu.py                 # CpuCollector
│       │   ├── gpu.py                 # GpuCollector
│       │   ├── memory.py              # MemoryCollector
│       │   └── trace_buffer.py        # TraceBuffer (ring buffer)
│       │
│       ├── merger/
│       │   ├── __init__.py
│       │   ├── timeline_merger.py     # TimelineMerger
│       │   └── clock_sync.py          # Clock synchronization
│       │
│       ├── analyzer/
│       │   ├── __init__.py
│       │   ├── bottleneck_analyzer.py # BottleneckAnalyzer
│       │   ├── rules.py               # Bottleneck detection rules
│       │   └── suggestions.py         # Suggestion generation
│       │
│       ├── reporter/
│       │   ├── __init__.py
│       │   ├── report_generator.py    # ReportGenerator
│       │   ├── templates/
│       │   │   ├── markdown.jinja2    # Markdown template
│       │   │   └── html.jinja2        # HTML template
│       │   └── formatters.py          # Output formatting
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── config_parser.py       # Config file/env var parsing
│       │   └── defaults.yaml          # Default configuration
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logger.py              # Logging setup
│       │   ├── profiling.py           # Performance profiling
│       │   └── validation.py          # Input validation
│       │
│       ├── ext/                       # C++ extensions
│       │   ├── __init__.py
│       │   └── _cupti_wrapper.so      # Compiled CUPTI wrapper (if available)
│       │
│       └── __version__.py
│
├── cpp/                               # C++ code for CUPTI integration
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── cupti_wrapper.cpp
│   │   ├── gpu_collector.cpp
│   │   └── trace_buffer.cpp
│   ├── include/
│   │   ├── cupti_wrapper.h
│   │   ├── gpu_collector.h
│   │   └── trace_buffer.h
│   └── tests/
│       └── test_cupti_wrapper.cpp
│
├── tests/
│   ├── __init__.py
│   │
│   ├── unit/
│   │   ├── test_cpu_collector.py
│   │   ├── test_gpu_collector.py
│   │   ├── test_timeline_merger.py
│   │   ├── test_analyzer.py
│   │   ├── test_reporter.py
│   │   └── test_cli.py
│   │
│   ├── integration/
│   │   ├── test_collection_pipeline.py
│   │   ├── test_full_workflow.py
│   │   └── test_api.py
│   │
│   ├── system/
│   │   ├── test_llm_inference.py
│   │   ├── test_cnn_inference.py
│   │   └── test_embedding_inference.py
│   │
│   ├── fixtures/
│   │   ├── conftest.py                # pytest fixtures
│   │   ├── mock_gpu.py                # Mock GPU/CUDA
│   │   ├── synthetic_workloads.py     # Synthetic ML workloads
│   │   └── sample_traces.json         # Example trace data
│   │
│   └── performance/
│       ├── test_profiling_overhead.py
│       └── test_clock_sync_accuracy.py
│
├── .github/
│   └── workflows/
│       ├── tests.yml                  # Unit/integration tests CI
│       ├── system-tests.yml           # GPU system tests (optional)
│       └── lint-and-format.yml        # Linting and type checking
│
├── .gitignore
├── .pre-commit-config.yaml            # Pre-commit hooks
├── pyproject.toml                     # Python project metadata
├── setup.py                           # Setup script
├── setup.cfg                          # Setup configuration
├── Makefile                           # Common tasks (build, test, clean)
├── CONTRIBUTING.md                    # Contribution guidelines
├── CHANGELOG.md                       # Version history
└── examples/
    ├── simple_inference.py            # Minimal inference example
    ├── llm_inference.py               # LLM inference example
    ├── cnn_inference.py               # CNN inference example
    └── custom_scopes.py               # Custom API usage example
```

## Naming Conventions

### Module Files
- Use **lowercase with underscores**: `cpu_collector.py`, `timeline_merger.py`
- One class per file preferred (for larger classes)
- Prefix private modules with underscore: `_internal_utils.py`

### Test Files
- Pattern: `test_<module>.py`
- Test functions: `test_<feature>_<scenario>_<expectation>()`
- Fixtures: `@pytest.fixture def <resource>():`

### Configuration Files
- YAML: `*.yaml` (not `.yml`)
- JSON: `*.json`
- Env vars: `INFERSCOPE_*` prefix

### Documentation Files
- Specs: `*_spec.md` or `*_specification.md`
- Guides: `*_guide.md`
- Architecture: `*_architecture.md` or `*_design.md`

## Build & Distribution

### Python Package Distribution
- **Setup**: `setup.py` + `pyproject.toml` (modern)
- **Distribution**: PyPI (inferscope package)
- **Pre-built wheels**: For common CUDA versions (11.8, 12.0)

### Build Process
```bash
# Build Python package
pip install -e ".[dev]"

# Build C++ extensions (if CUDA available)
python setup.py build_ext --inplace

# Create distribution wheels
python -m build
```

## File Ownership & Responsibility

| Directory | Owner | Responsibility |
|-----------|-------|-----------------|
| `src/inferscope/` | Python Lead | Core Python implementation |
| `cpp/` | GPU Lead | CUPTI integration, C++ code |
| `tests/` | QA Lead | Test coverage, test infrastructure |
| `docs/` | Technical Lead | Documentation, specs, design |
| `.github/workflows/` | DevOps | CI/CD pipeline |

## File Size Guidelines

- **Python modules**: <500 lines (prefer breaking into smaller modules)
- **Test files**: <300 lines per test file
- **Documentation files**: <200 lines (break into sections as needed)

## Git Workflow

### Branch Naming
- Features: `feature/cpu-profiling`
- Bugfixes: `bugfix/clock-sync-error`
- Docs: `docs/architecture-update`
- Release: `release/v0.1.0`

### Commit Message Format
```
[COMPONENT] Short description (50 chars max)

Longer explanation (wrap at 72 chars):
- What changed
- Why it changed
- Any side effects

Fixes #123
```

Example:
```
[collectors] Fix CPU thread safety in settrace hook

The previous implementation used a shared dict without locks, causing
race conditions when multiple threads collected events simultaneously.
This commit uses thread-local storage (TLS) via threading.local().

Fixes #45
```
