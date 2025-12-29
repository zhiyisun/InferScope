.PHONY: help docs prd sad icd clean test cleanup-temp-docs demo coverage integration

help:
	@echo "InferScope Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make docs     - Generate all documentation from YAML sources"
	@echo "  make prd      - Generate PRD.md from requirements.yaml"
	@echo "  make sad      - Generate SAD.md from architecture.yaml"
		@echo "  make icd      - Generate ICD.md from interfaces.yaml"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make cleanup-temp-docs - Remove docs/_temp temporary documents"
	@echo "  make test     - Run tests (not yet implemented)"
	@echo "  make demo     - Run the profiler demo script"
	@echo "  make coverage - Run unit tests with coverage and generate HTML report"
	@echo "  make integration - Run demo and orchestrator checks (end-to-end)"

docs: prd sad icd
	@echo "✓ All documentation generated"

prd:
	@echo "Generating PRD.md from requirements.yaml..."
	@python scripts/generate_prd.py

sad:
	@echo "Generating SAD.md from architecture.yaml..."
	@python scripts/generate_sad.py

icd:
	@echo "Generating ICD.md from interfaces.yaml..."
	@python scripts/generate_icd.py

clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete"

test:
	@echo "Running unit tests..."
	@.venv/bin/python -m pytest tests/unit -v

cleanup-temp-docs:
	@echo "Removing temporary documents in docs/_temp..."
	@rm -rf docs/_temp
	@echo "✓ Temporary documents removed"

demo:
	@echo "Running profiler demo..."
	@.venv/bin/python scripts/run_profiler_demo.py
	@echo "✓ Demo finished"

coverage:
	@echo "Running unit tests with coverage..."
	@mkdir -p coverage/html
	@.venv/bin/python -m pytest tests/unit -v --cov=src --cov-report=term-missing --cov-report=html:coverage/html
	@echo "✓ Coverage report generated in coverage/html/index.html"

integration:
	@echo "Running integration demo and orchestrator checks..."
	@.venv/bin/python scripts/run_profiler_demo.py
	@.venv/bin/python -m pytest tests/unit/test_profiler.py -v
	@echo "✓ Integration checks completed"

e2e-demo:
	@echo "Running end-to-end pipeline demo..."
	@.venv/bin/python scripts/run_e2e_demo.py
	@echo "✓ End-to-end demo completed (see demo_report.md)"

