"""
Unit tests for GPU Collector (mock) module.
"""

import pytest
from unittest.mock import patch


def test_gpu_collector_initializes_with_trace_buffer(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector, CollectorState
    collector = GpuCollector(mock_trace_buffer)
    assert collector is not None
    assert collector.state == CollectorState.IDLE


def test_gpu_collector_rejects_none_buffer():
    from src.inferscope.collectors.gpu import GpuCollector
    with pytest.raises(ValueError):
        GpuCollector(None)


def test_gpu_collector_start_sets_collecting(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector, CollectorState
    collector = GpuCollector(mock_trace_buffer)
    collector.start()
    assert collector.state == CollectorState.COLLECTING


def test_gpu_collector_stop_finalizes(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector, CollectorState
    collector = GpuCollector(mock_trace_buffer)
    collector.start()
    collector.stop()
    assert collector.state == CollectorState.FINALIZED


def test_gpu_collector_degrades_without_cuda(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector
    collector = GpuCollector(mock_trace_buffer)
    collector.start()
    # In mock mode, CUPTI is unavailable
    stats = collector.get_statistics()
    assert stats['cupti_available'] is False
    assert collector.get_gpu_events() == []


def test_gpu_collector_records_synthetic_kernel_event(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector
    collector = GpuCollector(mock_trace_buffer)
    collector.start()
    collector._inject_kernel_event(name="gemm", duration_us=150, stream_id=0)
    events = collector.get_gpu_events()
    assert len(events) == 1
    assert events[0]['type'] == 'gpu_kernel'
    assert events[0]['name'] == 'gemm'
    # Trace buffer also receives the event
    assert mock_trace_buffer.get_event_count() >= 1


def test_gpu_collector_records_synthetic_copy_events(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector
    collector = GpuCollector(mock_trace_buffer)
    collector.start()
    collector._inject_copy_event(name="input_copy", bytes_transferred=1024*1024, direction='h2d')
    collector._inject_copy_event(name="output_copy", bytes_transferred=512*1024, direction='d2h')
    events = collector.get_gpu_events()
    types = {e['type'] for e in events}
    assert 'h2d_copy' in types
    assert 'd2h_copy' in types


def test_gpu_collector_statistics_updates(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector
    collector = GpuCollector(mock_trace_buffer)
    collector.start()
    collector._inject_kernel_event(name="relu", duration_us=50)
    collector.stop()
    stats = collector.get_statistics()
    assert stats['event_count'] >= 1
    assert stats['duration_us'] is not None


def test_gpu_collector_stop_idempotent(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector
    collector = GpuCollector(mock_trace_buffer)
    collector.stop()  # Does nothing safely
    collector.stop()  # Still safe


def test_gpu_collector_start_idempotent(mock_trace_buffer):
    from src.inferscope.collectors.gpu import GpuCollector
    collector = GpuCollector(mock_trace_buffer)
    collector.start()
    collector.start()  # No error
