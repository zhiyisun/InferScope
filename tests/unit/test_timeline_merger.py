"""
Unit tests for Timeline Merger module.
"""

import pytest


def test_merger_auto_sync_assumed():
    from src.inferscope.timeline.merger import TimelineMerger

    cpu_events = [
        {
            'type': 'cpu_call',
            'name': 'fn_a',
            'timestamp_start_us': 100,
            'thread_id': 1,
            'metadata': {'module': 'm'}
        },
        {
            'type': 'cpu_return',
            'name': 'fn_a',
            'timestamp_us': 150,
            'thread_id': 1,
            'metadata': {'module': 'm'}
        },
    ]
    gpu_events = [
        {
            'type': 'gpu_kernel',
            'name': 'k1',
            'timestamp_us': 120,
            'stream_id': 0,
            'metadata': {}
        }
    ]

    merger = TimelineMerger(cpu_events, gpu_events)
    unified = merger.get_unified_timeline()

    assert len(unified) == 3
    ts = [e['global_ts_us'] for e in unified]
    assert ts == sorted(ts)
    types = [e['type'] for e in unified]
    assert types == ['cpu_call', 'gpu_kernel', 'cpu_return']


def test_merger_uses_metadata_offset():
    from src.inferscope.timeline.merger import TimelineMerger

    # GPU event provides calibration: cpu_ref_us=1000, gpu_ts=1500 → intercept=500
    gpu_events = [
        {
            'type': 'gpu_kernel',
            'name': 'calib',
            'timestamp_us': 1500,
            'metadata': {'cpu_ref_us': 1000},
        }
    ]
    cpu_events = [
        {
            'type': 'cpu_call',
            'name': 'fn_b',
            'timestamp_start_us': 600,  # mapped to 600 + 500 = 1100
            'thread_id': 2,
            'metadata': {'module': 'm2'}
        }
    ]

    merger = TimelineMerger(cpu_events, gpu_events)
    unified = merger.get_unified_timeline()

    # CPU call should be at 1100, GPU calib at 1500
    assert unified[0]['type'] == 'cpu_call'
    assert unified[0]['global_ts_us'] == 1100
    assert unified[1]['type'] == 'gpu_kernel'
    assert unified[1]['global_ts_us'] == 1500

    meta = merger.get_sync_metadata()
    assert meta['method'] == 'metadata_ref'
    assert meta['state'] == 'Finalized'


def test_merger_excludes_events_without_timestamp():
    from src.inferscope.timeline.merger import TimelineMerger

    cpu_events = [
        {
            'type': 'cpu_call',
            'name': 'no_ts',
            # missing timestamp fields
            'thread_id': 3,
            'metadata': {}
        }
    ]
    gpu_events = []

    merger = TimelineMerger(cpu_events, gpu_events)
    unified = merger.get_unified_timeline()

    assert unified == []


def test_merger_sync_metadata_before_finalize():
    from src.inferscope.timeline.merger import TimelineMerger

    merger = TimelineMerger([], [])
    sync = merger.synchronize_clocks()

    assert 'slope' in sync and 'intercept' in sync and 'error_us' in sync
    meta = merger.get_sync_metadata()
    assert meta['state'] == 'Synchronized'
