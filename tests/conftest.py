"""
Pytest configuration and fixtures for InferScope tests.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock


class MockTraceBuffer:
    """Mock trace buffer for unit testing collectors in isolation."""
    
    def __init__(self, capacity_mb: int = 100):
        """Initialize mock trace buffer."""
        self.capacity_mb = capacity_mb
        self.events: List[Dict[str, Any]] = []
        self.overflow_count = 0
        self._is_full = False
    
    def enqueue(self, event: Dict[str, Any]) -> bool:
        """
        Enqueue event to buffer.
        
        Args:
            event: Event dict with type, name, timestamps, metadata
            
        Returns:
            bool: True if enqueued successfully, False if buffer full
        """
        if self._is_full:
            self.overflow_count += 1
            return False
        
        self.events.append(event.copy())
        return True
    
    def read_all(self) -> List[Dict[str, Any]]:
        """Return all events in insertion order."""
        return [e.copy() for e in self.events]
    
    def clear(self) -> None:
        """Reset buffer."""
        self.events.clear()
        self.overflow_count = 0
        self._is_full = False
    
    def set_full(self, is_full: bool = True) -> None:
        """Set buffer full state (for testing overflow behavior)."""
        self._is_full = is_full
    
    def get_event_count(self) -> int:
        """Get number of events in buffer."""
        return len(self.events)


@pytest.fixture
def mock_trace_buffer():
    """Fixture providing a mock trace buffer."""
    return MockTraceBuffer()


@pytest.fixture
def synthetic_call_stack():
    """Fixture providing a synthetic call stack for testing."""
    def outer_function():
        def middle_function():
            def inner_function():
                return 42
            return inner_function()
        return middle_function()
    
    return {
        'outer': outer_function,
        'call_depth': 3
    }


@pytest.fixture
def mock_sys():
    """Fixture providing mock sys module."""
    return Mock()
