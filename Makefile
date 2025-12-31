.PHONY: help docs clean clean-docs distclean test demo

help:
	@echo "InferScope Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make docs       - Generate all documentation from YAML sources"
	@echo "  make clean      - Remove build artifacts and temporary files"
	@echo "  make clean-docs - Remove auto-generated documentation (PRD.md, SAD.md, ICD.md)"
	@echo "  make distclean   - Remove all generated artifacts (build + docs)"
	@echo "  make test       - Run all unit tests with coverage report"
	@echo "  make demo       - Run the profiler demo script"

docs:
	@echo "Generating documentation from YAML sources..."
	@python scripts/generate_docs.py
	@echo "✓ All documentation generated"

clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf docs/_temp 2>/dev/null || true
	@echo "✓ Clean complete"

clean-docs:
	@echo "Removing auto-generated documentation..."
	@rm -f docs/1_requirements/PRD.md
	@rm -f docs/2_system_architecture/SAD.md
	@rm -f docs/3_module_design/ICD.md
	@echo "✓ Auto-generated docs removed (regenerate with: make docs)"

distclean: clean clean-docs
	@echo "Removing all generated artifacts..."
	@rm -rf coverage/
	@rm -rf outputs/
	@echo "✓ Full distclean complete"

test:
	@echo "Running unit tests with coverage..."
	@mkdir -p coverage/html
	@.venv/bin/python -m pytest tests/unit -v --cov=src --cov-report=term-missing --cov-report=html:coverage/html
	@echo "✓ Coverage report generated in coverage/html/index.html"

demo:
	@echo "Running profiler function verification..."
	@.venv/bin/python examples/run_profiler_demo.py
	@echo "✓ Function verification demo finished"
	@echo ""
	@echo "Running LLM inference demo (requires PyTorch)..."
	@.venv/bin/python examples/demo_llm_inference.py --max-new-tokens 16 --output outputs/demo_llm_trace.json 2>&1 || echo "⚠ LLM demo skipped (PyTorch not available or model not found)"
	@echo ""
	@if [ -f outputs/demo_llm_trace.json ]; then \
		echo "Generating analysis reports..."; \
		.venv/bin/python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.md; \
		.venv/bin/python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.html --format html; \
		echo "✓ Reports generated (Markdown and HTML)"; \
	else \
		echo "⚠ No trace file generated, skipping report generation"; \
		echo "  To run the full LLM demo, install PyTorch: pip install torch transformers"; \
	fi
	@echo ""
	@echo "Demo completed. Check outputs/ for traces and reports (if generated)."

