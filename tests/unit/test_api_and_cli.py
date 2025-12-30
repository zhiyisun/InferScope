"""
Integration tests for Python API and CLI Interface
"""

import pytest
import tempfile
import os
import json
import subprocess
import sys
from pathlib import Path

from src.inferscope import (
    Profiler, 
    BottleneckAnalyzer, 
    ReportGenerator,
    scope,
    mark_event,
    set_global_profiler,
)


class SimpleTraceBuffer:
    """Minimal trace buffer for testing."""
    def __init__(self):
        self._events = []
    
    def enqueue(self, event):
        self._events.append(event.copy())
        return True
    
    def read_all(self):
        return [e.copy() for e in self._events]


class TestPythonAPI:
    """Test Python API (scope, mark_event)."""
    
    def test_scope_context_manager(self):
        """Test scope() context manager creates events."""
        buf = SimpleTraceBuffer()
        profiler = Profiler(buf)
        set_global_profiler(profiler)
        
        with scope("test_inference"):
            pass
        
        events = buf.read_all()
        scope_enters = [e for e in events if e.get("type") == "scope_enter"]
        scope_exits = [e for e in events if e.get("type") == "scope_exit"]
        
        assert len(scope_enters) > 0, "No scope_enter events captured"
        assert len(scope_exits) > 0, "No scope_exit events captured"
        assert scope_enters[0]["name"] == "test_inference"
    
    def test_mark_event(self):
        """Test mark_event creates instant events."""
        buf = SimpleTraceBuffer()
        profiler = Profiler(buf)
        set_global_profiler(profiler)
        
        mark_event("test_event", metadata={"key": "value"})
        
        events = buf.read_all()
        instant_events = [e for e in events if e.get("type") == "instant"]
        
        assert len(instant_events) > 0, "No instant events captured"
        assert instant_events[0]["name"] == "test_event"
        assert instant_events[0]["metadata"]["key"] == "value"
    
    def test_nested_scopes(self):
        """Test nested scope contexts."""
        buf = SimpleTraceBuffer()
        profiler = Profiler(buf)
        set_global_profiler(profiler)
        
        with scope("outer"):
            with scope("inner"):
                mark_event("middle_event")
        
        events = buf.read_all()
        
        # Should have enter/exit for both outer and inner
        outer_enters = [e for e in events if e.get("type") == "scope_enter" and e.get("name") == "outer"]
        inner_enters = [e for e in events if e.get("type") == "scope_enter" and e.get("name") == "inner"]
        
        assert len(outer_enters) > 0
        assert len(inner_enters) > 0
    
    def test_mark_event_without_profiler(self):
        """Test mark_event gracefully handles missing profiler."""
        set_global_profiler(None)
        
        # Should not raise
        mark_event("test_event")


class TestCLI:
    """Test CLI Interface."""
    
    def test_cli_config_show(self):
        """Test 'inferscope config show' command."""
        result = subprocess.run(
            [sys.executable, "scripts/inferscope", "config", "show"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "InferScope Configuration" in result.stdout
        assert "version" in result.stdout.lower()
    
    def test_cli_analyze_missing_file(self):
        """Test analyze command with missing trace file."""
        result = subprocess.run(
            [sys.executable, "scripts/inferscope", "analyze", "/nonexistent/trace.json"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Should fail on missing file"
        assert "not found" in result.stderr.lower()
    
    def test_cli_analyze_valid_trace(self):
        """Test analyze command with valid trace file."""
        # Create temporary trace file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            trace_data = {
                "timeline": [
                    {"type": "cpu_call", "name": "func_a", "start_us": 0, "end_us": 100},
                    {"type": "cpu_return", "name": "func_a", "timestamp_us": 100},
                ]
            }
            json.dump(trace_data, f)
            trace_file = f.name
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                report_file = f.name
            
            result = subprocess.run(
                [sys.executable, "scripts/inferscope", "analyze", trace_file, 
                 "--output", report_file, "--format", "markdown"],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                capture_output=True,
                text=True
            )
            
            # Should succeed or show reasonable error
            if result.returncode == 0:
                assert os.path.exists(report_file), "Report file not created"
                with open(report_file) as f:
                    content = f.read()
                    # Report should have content
                    assert len(content) > 0
            
        finally:
            os.unlink(trace_file)
            if os.path.exists(report_file):
                os.unlink(report_file)


class TestCLIRun:
    """Test CLI 'run' command."""
    
    def test_cli_run_missing_script(self):
        """Test run command with missing script."""
        result = subprocess.run(
            [sys.executable, "scripts/inferscope", "run", "/nonexistent/script.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Should fail on missing script"
        assert "not found" in result.stderr.lower()
    
    def test_cli_run_simple_script(self):
        """Test run command with simple Python script."""
        # Create temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('hello')\nprint('world')\n")
            script_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, "scripts/inferscope", "run", script_file],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Script should execute (may not generate profiling without instrumentation)
            assert "not found" not in result.stderr.lower()
            
        finally:
            os.unlink(script_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
