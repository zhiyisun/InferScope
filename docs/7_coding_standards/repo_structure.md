# Repository Structure Guide

## Directory Layout

```
InferScope/
│
├── README.md                          # Project overview
├── LICENSE                            # Apache 2.0
├── Makefile                           # Development tasks (docs, test, demo, clean)
├── pyproject.toml                     # Python package config
│
├── docs/                              # Documentation (auto-generated from YAML)
│   ├── README.md                      # Documentation structure guide
│   ├── AI-Driven Software Development Workflow.md
│   ├── 1_requirements/
│   │   ├── requirements.yaml          # Machine-readable requirements (source)
│   │   └── PRD.md                     # Product Requirement Document (auto-gen)
│   ├── 2_system_architecture/
│   │   ├── architecture.yaml          # Component definitions (source)
│   │   ├── SAD.md                     # System Architecture Document (auto-gen)
│   │   └── technology_rationale.md    # Tech choices & tradeoffs
│   ├── 3_module_design/
│   │   ├── interfaces.yaml            # Machine-readable APIs (source)
│   │   ├── ICD.md                     # Interface Control Document (auto-gen)
│   │   └── module_specs/
│   │       ├── cpu_collector.md
│   │       ├── gpu_collector.md
│   │       └── timeline_merger.md
│   ├── 4_data_schema/
│   │   ├── data_model.md
│   │   └── schema.json
│   ├── 5_test_framework/
│   │   └── test_strategy.md
│   ├── 6_test_cases/
│   │   └── unit_tests.md
│   ├── 7_coding_standards/
│   │   ├── coding_standards.md
│   │   └── repo_structure.md          # This file
│   ├── 8_system_integration/
│   │   ├── integration_plan.md
│   │   ├── integration_strategy.md
│   │   └── deployment_architecture.md
│   ├── 9_system_test/
│   │   └── system_test_spec.md
│   └── 10_operations/
│       └── runbooks.md
│
├── src/
│   └── inferscope/                    # Main Python package
│       ├── __init__.py
│       ├── api.py                     # Public Python API (scope, mark_event)
│       ├── cli.py                     # CLI interface (inferscope command)
│       ├── profiler.py                # Main Profiler orchestrator
│       │
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── cpu.py                 # CpuCollector (sys.settrace)
│       │   └── gpu.py                 # GpuCollector (CUPTI)
│       │
│       ├── timeline/
│       │   ├── __init__.py
│       │   └── merger.py              # TimelineMerger (clock sync, event ordering)
│       │
│       ├── analyzer/
│       │   ├── __init__.py
│       │   └── bottleneck_analyzer.py # BottleneckAnalyzer (rules + suggestions)
│       │
│       └── reporter/
│           ├── __init__.py
│           └── report_generator.py    # ReportGenerator (MD/HTML output)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures (MockTraceBuffer)
│   └── unit/
│       ├── __init__.py
│       ├── test_cpu_collector.py
│       ├── test_gpu_collector.py
│       ├── test_timeline_merger.py
│       ├── test_analyzer.py
│       ├── test_reporter.py
│       ├── test_profiler.py
│       └── test_api_and_cli.py
│
├── examples/
│   ├── run_profiler_demo.py           # Quick verification demo (CPU/GPU collection)
│   └── demo_llm_inference.py          # Real LLM demo (Qwen3 0.6B)
│
├── scripts/
│   ├── generate_docs.py               # Generate PRD/SAD/ICD from YAML sources
│   ├── inferscope                     # CLI entry point script
│   └── git-hooks/
│       └── pre-commit                 # Git pre-commit hook
│
├── outputs/                           # Generated profiling artifacts (.gitignored)
│   ├── *.json                         # Raw trace files
│   └── *_report.md / *.html           # Analysis reports
│
└── .gitignore                         # Git exclusions (auto-gen docs, __pycache__, etc)
```

## Naming Conventions

### Module Files
- Use **lowercase with underscores**: `cpu_collector.py`, `timeline_merger.py`
- One class per file preferred
- Prefix private modules with underscore: `_internal.py`

### Test Files
- Pattern: `test_<module>.py`
- Test functions: `test_<feature>_<scenario>_<expectation>()`
- Fixtures: `@pytest.fixture def <resource>():`

### Configuration Files
- YAML: `*.yaml` (not `.yml`)
- JSON: `*.json`
- Env vars: `INFERSCOPE_*` prefix

### Documentation Files
- Specs: `*_spec.md`
- Architecture: `*_architecture.md` or `*_design.md`

## File Ownership & Responsibility

| Directory | Responsibility |
|-----------|-----------------|
| `src/inferscope/` | Core Python implementation |
| `tests/` | Test coverage and infrastructure |
| `docs/` | Documentation, specs, design |
| `examples/` | Demo and example code |
| `scripts/` | Utility and build scripts |

## Git Workflow

### Branch Naming
- Features: `feature/cpu-profiling`
- Bugfixes: `bugfix/clock-sync-error`
- Docs: `docs/architecture-update`

### Commit Message Format
```
[COMPONENT] Short description (50 chars max)

Longer explanation:
- What changed
- Why it changed

Fixes #123
```
