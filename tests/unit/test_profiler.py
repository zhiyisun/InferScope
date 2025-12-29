"""
Unit tests for Profiler orchestrator.
"""

import threading
import sys
import pytest

# Coverage tracing can interfere with sys.settrace-based collection.
# Skip these orchestrator tests when coverage plugin is active.
_cov_active = any(m.startswith('coverage') for m in sys.modules.keys())
pytestmark = pytest.mark.skipif(_cov_active, reason="Skipped under coverage: trace hook interference")


def test_profiler_start_stop(mock_trace_buffer):
    from src.inferscope import Profiler

    profiler = Profiler(mock_trace_buffer)
    profiler.start()
    profiler.stop()

    stats = profiler.get_stats()
    assert stats.cpu['state'] == 'FINALIZED'
    assert stats.gpu['state'] == 'FINALIZED'


def test_profiler_unified_timeline_with_workload_and_gpu(mock_trace_buffer):
    from src.inferscope import Profiler

    profiler = Profiler(mock_trace_buffer)
    profiler.start()

    # Simple workload to generate CPU events
    def workload():
        def inner():
            return 1 + 1
        return inner()

    workload()

    # Inject a GPU event via collector for test visibility
    profiler.gpu._inject_kernel_event(name="test_kernel", duration_us=10)

    profiler.stop()

    unified = profiler.get_unified_timeline()
    assert len(unified) >= 3  # cpu_call, gpu_kernel, cpu_return

    types = [e['type'] for e in unified]
    assert 'cpu_call' in types
    assert 'cpu_return' in types
    assert 'gpu_kernel' in types


def test_profiler_uses_trace_buffer_partition(mock_trace_buffer):
    from src.inferscope import Profiler

    profiler = Profiler(mock_trace_buffer)
    profiler.start()

    # Trigger some CPU activity
    def foo():
        return 42
    foo()

    # Inject GPU copy event
    profiler.gpu._inject_copy_event(name="copy", bytes_transferred=1024, direction='h2d')

    profiler.stop()
    unified = profiler.get_unified_timeline()

    # Ensure events partitioned correctly
    cpu = [e for e in unified if e['type'].startswith('cpu_')]
    gpu = [e for e in unified if not e['type'].startswith('cpu_')]
    assert len(cpu) >= 2
    assert len(gpu) >= 1
