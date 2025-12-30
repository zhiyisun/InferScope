# Documentation Structure

This document describes the InferScope documentation organization and generation workflow.

## Overview

InferScope uses a **single source of truth** approach where machine-readable YAML files serve as the authoritative source, and human-readable Markdown files are auto-generated.

## Auto-Generated Documentation

The following Markdown files are **automatically generated** and should NOT be edited manually:

| Source (YAML) | Generated (Markdown) | Generator Script |
|---------------|---------------------|------------------|
| `docs/1_requirements/requirements.yaml` | `docs/1_requirements/PRD.md` | `scripts/generate_docs.py` |
| `docs/2_system_architecture/architecture.yaml` | `docs/2_system_architecture/SAD.md` | `scripts/generate_docs.py` |
| `docs/3_module_design/interfaces.yaml` | `docs/3_module_design/ICD.md` | `scripts/generate_docs.py` |

### Regenerate All Documentation

```bash
make docs
```

This regenerates PRD.md, SAD.md, and ICD.md from their YAML sources.

## Manual Documentation

The following files are **manually maintained** (not auto-generated):

### Architecture
- `docs/2_system_architecture/technology_rationale.md` - Technology choices and trade-offs

### Module Design
- `docs/3_module_design/module_specs/*.md` - Individual module specifications
  - `cpu_collector.md`
  - `gpu_collector.md`
  - `timeline_merger.md`

### Data Schema
- `docs/4_data_schema/data_model.md` - Data structures and formats
- `docs/4_data_schema/schema.json` - JSON Schema for trace validation

### Testing
- `docs/5_test_framework/test_strategy.md` - Overall test strategy
- `docs/6_test_cases/unit_tests.md` - Unit test specifications

### Coding Standards
- `docs/7_coding_standards/coding_standards.md` - Python/C++ style guide
- `docs/7_coding_standards/repo_structure.md` - Repository organization

### Integration
- `docs/8_system_integration/integration_plan.md` - Phase-by-phase roadmap
- `docs/8_system_integration/deployment_architecture.md` - Runtime topology
- `docs/8_system_integration/integration_strategy.md` - Component integration details

### System Testing
- `docs/9_system_test/system_test_spec.md` - End-to-end test scenarios

### Operations
- `docs/10_operations/runbooks.md` - Troubleshooting and deployment

## Workflow

### Updating Requirements

```bash
# 1. Edit YAML source
vim docs/1_requirements/requirements.yaml

# 2. Regenerate all docs
make docs

# 3. Review and commit
git diff docs/1_requirements/
git add docs/1_requirements/
git commit -m "Update requirements"
```

### Updating Architecture

```bash
# 1. Edit YAML source
vim docs/2_system_architecture/architecture.yaml

# 2. Regenerate all docs
make docs

# 3. Review and commit
git add docs/2_system_architecture/
git commit -m "Update architecture"
```

### Updating Interfaces

```bash
# 1. Edit YAML source
vim docs/3_module_design/interfaces.yaml

# 2. Regenerate all docs
make docs

# 3. Review and commit
git add docs/3_module_design/
git commit -m "Update API interfaces"
```

### Updating All Docs

```bash
# After editing any YAML files
make docs

# Review all changes
git diff docs/

# Commit
git add docs/
git commit -m "Update documentation"
```

## Pre-Commit Hook (Optional)

To automatically regenerate docs before each commit, add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Regenerate docs from YAML sources
make docs

# Stage generated files
git add docs/1_requirements/PRD.md
git add docs/2_system_architecture/SAD.md
git add docs/3_module_design/ICD.md
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## CI/CD Integration

Example GitHub Actions workflow to validate docs are up-to-date:

```yaml
name: Validate Documentation

on: [pull_request]

jobs:
  check-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Regenerate docs
        run: make docs
      - name: Check for changes
        run: |
          if ! git diff --quiet docs/; then
            echo "Documentation is out of sync with YAML sources"
            echo "Run 'make docs' and commit the changes"
            exit 1
          fi
```

## File Naming Conventions

- **YAML sources**: `snake_case.yaml` (e.g., `requirements.yaml`)
- **Generated Markdown**: `UPPERCASE.md` for acronyms (e.g., `PRD.md`, `SAD.md`, `ICD.md`)
- **Manual Markdown**: `snake_case.md` (e.g., `test_strategy.md`)
- **Module specs**: `module_name.md` (e.g., `cpu_collector.md`)

## Notes

1. **Auto-generated files have warning headers** - Look for `<!-- AUTO-GENERATED -->` comments
2. **Always edit YAML, not Markdown** - Changes to generated .md files will be overwritten
3. **YAML is machine-readable** - Used by AI tools, scripts, and validation
4. **Markdown is human-readable** - Used for review, documentation, and Git diffs
5. **Single source of truth** - No duplication maintenance burden
