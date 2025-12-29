"""
InferScope User-Facing API

Provides context managers and decorators for marking inference regions
and events in user code.
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any
import time
import threading

# Global state for API
_current_profiler = None
_lock = threading.Lock()


def set_global_profiler(profiler):
    """Set the global profiler instance (called by CLI/orchestrator)."""
    global _current_profiler
    with _lock:
        _current_profiler = profiler


def get_global_profiler():
    """Get the global profiler instance."""
    global _current_profiler
    with _lock:
        return _current_profiler


@contextmanager
def scope(name: str):
    """
    Context manager for marking inference regions.
    
    Usage:
        with scope("inference"):
            output = model.forward(inputs)
    
    Args:
        name: Name of the scope region
        
    Yields:
        The scope context
    """
    profiler = get_global_profiler()
    
    # Use monotonic clock to align with collectors
    _mono_ns = time.monotonic_ns()
    timestamp_us = int(_mono_ns // 1000)
    
    if profiler is not None:
        # Emit scope_enter event to trace buffer
        event = {
            "type": "scope_enter",
            "name": name,
            "timestamp_us": timestamp_us,
            "timestamp_ns": _mono_ns,
        }
        try:
            profiler.trace_buffer.enqueue(event)
        except Exception:
            pass  # Silently ignore buffer errors
    
    try:
        yield
    finally:
        _mono_ns_end = time.monotonic_ns()
        timestamp_us_end = int(_mono_ns_end // 1000)
        
        if profiler is not None:
            # Emit scope_exit event
            event = {
                "type": "scope_exit",
                "name": name,
                "timestamp_us": timestamp_us_end,
                "timestamp_ns": _mono_ns_end,
            }
            try:
                profiler.trace_buffer.enqueue(event)
            except Exception:
                pass  # Silently ignore buffer errors


def mark_event(name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Instant event marker with optional metadata.
    
    Usage:
        mark_event("tokenization_complete", metadata={"tokens": 1024})
    
    Args:
        name: Name of the event
        metadata: Optional metadata dictionary
    """
    profiler = get_global_profiler()
    
    _mono_ns = time.monotonic_ns()
    timestamp_us = int(_mono_ns // 1000)
    
    if profiler is not None:
        event = {
            "type": "instant",
            "name": name,
            "timestamp_us": timestamp_us,
            "timestamp_ns": _mono_ns,
            "metadata": metadata or {},
        }
        try:
            profiler.trace_buffer.enqueue(event)
        except Exception:
            pass  # Silently ignore buffer errors


__all__ = ["scope", "mark_event", "set_global_profiler", "get_global_profiler"]
