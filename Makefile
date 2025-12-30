.PHONY: help docs clean test demo

help:
	@echo "InferScope Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make docs  - Generate all documentation from YAML sources"
	@echo "  make clean - Remove build artifacts"
	@echo "  make test  - Run all unit tests with coverage report"
	@echo "  make demo  - Run the profiler demo script"

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

test:
	@echo "Running unit tests with coverage..."
	@mkdir -p coverage/html
	@.venv/bin/python -m pytest tests/unit -v --cov=src --cov-report=term-missing --cov-report=html:coverage/html
	@echo "✓ Coverage report generated in coverage/html/index.html"

demo:
	@echo "Running profiler demo..."
	@.venv/bin/python examples/run_profiler_demo.py
	@echo "✓ Function verification demo finished"
	@echo ""
	@echo "Running LLM inference demo..."
	@.venv/bin/python examples/demo_llm_inference.py --model Qwen/Qwen3-0.6B-Base --max-new-tokens 128 --batch-size 1 --stress-mode
	@echo "✓ LLM demo finished"
	@echo ""
	@echo "Generating analysis reports..."
	@.venv/bin/python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.md
	@.venv/bin/python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.html --format html
	@echo "✓ Reports generated (Markdown and HTML)"
	@echo ""
	@echo "All demos completed. Check outputs/ for traces and reports."

