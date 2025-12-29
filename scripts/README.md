# Documentation Generator Scripts

This directory contains scripts for generating documentation from machine-readable sources.

## Quick Start

Generate all documentation:
```bash
make docs
```

This regenerates both PRD.md and SAD.md from their YAML sources.

---

## generate_prd.py

Generates `Doc/1_requirements/PRD.md` from `Doc/1_requirements/requirements.yaml`.

**Usage:**
```bash
python scripts/generate_prd.py
# OR
make prd
```

**Why:** 
- Single source of truth: `requirements.yaml` is authoritative
- PRD.md is auto-generated for human readability
- Prevents duplication and drift between formats

**When to run:**
- After modifying `requirements.yaml`
- Before committing requirement changes
- Can be automated in pre-commit hooks

## generate_sad.py

Generates `Doc/2_system_architecture/SAD.md` from `Doc/2_system_architecture/architecture.yaml`.

**Usage:**
```bash
python scripts/generate_sad.py
# OR
make sad
```

**Why:** 
- Single source of truth: `architecture.yaml` is authoritative
- SAD.md is auto-generated for human readability
- Component definitions, interfaces, and concurrency models stay in sync

**When to run:**
- After modifying `architecture.yaml`
- Before committing architecture changes

## generate_icd.py

Generates `Doc/3_module_design/ICD.md` from `Doc/3_module_design/interfaces.yaml`.

**Usage:**
```bash
python scripts/generate_icd.py
# OR
make icd
```

**Why:** 
- Single source of truth: `interfaces.yaml` is authoritative
- ICD.md is auto-generated for human readability
- API signatures, CLI commands, and error handling stay in sync

**When to run:**
- After modifying `interfaces.yaml`
- Before committing interface changes
- Can be automated in pre-commit hooks
- Can be automated in pre-commit hooks

## Adding to Pre-Commit (Optional)

Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: generate-prd
      name: Generate PRD from requirements.yaml
      entry: python scripts/generate_prd.py
      language: system
      files: Doc/1_requirements/requirements.yaml
      pass_filenames: false
```
