"""
CPU Collector Implementation

Captures Python function calls, system calls, and CPU execution time
using sys.settrace() hooks and optional Linux perf sampling.

Maintains per-thread trace buffers for lock-free collection with minimal overhead.
"""

import sys
import threading
import time
import os
import logging
from typing import List, Dict, Any, Optional, Callable, DefaultDict
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class CollectorState(Enum):
    """CPU Collector state machine."""
    UNINITIALIZED = 0
    IDLE = 1
    COLLECTING = 2
    FINALIZED = 3


@dataclass
class ThreadEventBuffer:
    """Per-thread event buffer with metadata."""
    thread_id: int
    events: List[Dict[str, Any]] = field(default_factory=list)
    event_count: int = 0
    overflow_count: int = 0
    start_time_us: Optional[int] = None
    end_time_us: Optional[int] = None
    
    def add_event(self, event: Dict[str, Any]) -> bool:
        """Add event to buffer; return False if overflow."""
        try:
            self.events.append(event.copy())
            self.event_count += 1
            return True
        except MemoryError:
            self.overflow_count += 1
            return False


class CpuCollector:
    """
    Collects CPU execution traces by hooking into Python interpreter.
    
    Uses sys.settrace() to capture function enter/exit events and maintains
    per-thread trace buffers for lock-free collection. Target: <5% CPU overhead.
    
    State Machine:
        [Uninitialized] --init--> [Idle] --start--> [Collecting] --stop--> [Finalized]
    
    Thread Safety:
        - Per-thread TLS buffers (lock-free)
        - No synchronization with main thread during collection
        - Merge at finalization
    """
    
    # Frame skip patterns (avoid instrumenting internal code)
    _SKIP_PATTERNS = {
        'inferscope/collectors',  # Profiling infrastructure
        'sys',                     # Python internals
        'trace',                   # Trace module
        '__',                      # Dunder methods
    }
    
    def __init__(self, trace_buffer):
        """
        Initialize CPU collector.
        
        Args:
            trace_buffer: Shared trace buffer (write-only interface)
            
        Raises:
            ValueError: If trace_buffer is None
        """
        if trace_buffer is None:
            raise ValueError("trace_buffer cannot be None")
        
        self.trace_buffer = trace_buffer
        self.state = CollectorState.UNINITIALIZED
        self._previous_trace_func: Optional[Callable] = None
        self._thread_local_buffers = threading.local()
        self._all_thread_buffers: DefaultDict[int, ThreadEventBuffer] = defaultdict(
            lambda: ThreadEventBuffer(thread_id=threading.get_ident())
        )
        self._start_time_us: Optional[int] = None
        self._end_time_us: Optional[int] = None
        self._frame_skip_count = 0
        self._total_events_captured = 0
        
        self.state = CollectorState.IDLE
        logger.debug(f"CPU Collector initialized (state={self.state})")
    
    def start(self) -> None:
        """
        Enable profiling hooks and begin collecting events.
        
        Installs sys.settrace hook and saves previous trace function
        for chaining. Idempotent: safe to call multiple times.
        
        Raises:
            RuntimeError: If collector already finalized
        """
        if self.state == CollectorState.FINALIZED:
            raise RuntimeError("Cannot restart finalized collector")
        
        if self.state == CollectorState.COLLECTING:
            logger.debug("CPU Collector already started, skipping")
            return
        
        try:
            # Save previous trace function
            self._previous_trace_func = sys.gettrace()
            
            # Install our trace function
            sys.settrace(self._create_trace_function())
            
            self.state = CollectorState.COLLECTING
            self._start_time_us = self._get_time_us()
            
            logger.info(
                f"CPU Collector started (thread_id={threading.get_ident()}, "
                f"prev_trace={self._previous_trace_func is not None})"
            )
            
        except Exception as e:
            logger.error(f"Failed to start CPU Collector: {e}", exc_info=True)
            self.state = CollectorState.IDLE
    
    def stop(self) -> None:
        """
        Disable profiling hooks and finalize collection.
        
        Removes sys.settrace hook and restores previous trace function
        if one was saved. Idempotent: safe to call multiple times.
        """
        if self.state != CollectorState.COLLECTING:
            logger.debug(f"CPU Collector not collecting (state={self.state}), skipping stop")
            return
        
        try:
            # Restore previous trace function
            if self._previous_trace_func is not None:
                sys.settrace(self._previous_trace_func)
            else:
                sys.settrace(None)
            
            self._end_time_us = self._get_time_us()
            self.state = CollectorState.FINALIZED
            
            logger.info(
                f"CPU Collector stopped (duration_us="
                f"{self._end_time_us - self._start_time_us if self._start_time_us else 0}, "
                f"events_captured={self._total_events_captured}, "
                f"frame_skips={self._frame_skip_count})"
            )
            
        except Exception as e:
            logger.error(f"Failed to stop CPU Collector: {e}", exc_info=True)
    
    def _create_trace_function(self) -> Callable:
        """
        Create the trace function for sys.settrace.
        
        The trace function must be fast (executed on every Python operation).
        Optimizations:
        - Inline frame skip logic
        - Minimal allocations
        - Early returns
        
        Returns:
            Callable: Trace function matching sys.settrace signature
        """
        # Capture collector state to avoid repeated attribute lookups
        skip_patterns = self._SKIP_PATTERNS
        should_skip = self._should_skip_frame
        handle_call = self._handle_call_event
        handle_return = self._handle_return_event
        previous_trace = self._previous_trace_func
        
        def trace_function(frame, event, arg):
            """
            Trace function called by Python interpreter.
            
            Called on every Python operation - must be fast.
            
            Args:
                frame: Frame object
                event: 'call', 'return', 'line', 'exception', etc.
                arg: Argument specific to event
            
            Returns:
                Trace function (for line tracing) or None
            """
            try:
                if not frame:
                    return trace_function
                
                # Skip frames quickly
                filename = frame.f_code.co_filename
                if any(pattern in filename for pattern in skip_patterns):
                    if previous_trace:
                        return previous_trace(frame, event, arg)
                    return trace_function
                
                # Handle events
                if event == 'call':
                    handle_call(frame)
                elif event == 'return':
                    handle_return(frame, arg)
                # Ignore 'line', 'exception' events for now (overhead)
                
                # Chain to previous trace function
                if previous_trace:
                    return previous_trace(frame, event, arg)
                
                return trace_function
            
            except Exception as e:
                logger.debug(f"Error in trace function: {e}")
                return trace_function
        
        return trace_function
    
    def _handle_call_event(self, frame) -> None:
        """
        Handle function call event.
        
        Args:
            frame: Current frame object
        """
        try:
            function_name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            
            # Skip internal profiling code
            if self._should_skip_frame(filename, function_name):
                self._frame_skip_count += 1
                return
            
            timestamp_us = self._get_time_us()
            thread_id = threading.get_ident()
            
            event = {
                'type': 'cpu_call',
                'name': function_name,
                'timestamp_start_us': timestamp_us,
                'timestamp_end_us': timestamp_us,  # Placeholder, updated on return
                'filename': filename,
                'lineno': lineno,
                'thread_id': thread_id,
                'metadata': {
                    'event': 'call',
                    'module': frame.f_globals.get('__name__', 'unknown'),
                }
            }
            
            # Store in thread-local buffer
            buffer = self._get_thread_buffer()
            buffer.add_event(event)
            
            # Enqueue to shared buffer
            success = self.trace_buffer.enqueue(event)
            if success:
                self._total_events_captured += 1
            
        except Exception as e:
            logger.debug(f"Error handling call event: {e}")
    
    def _handle_return_event(self, frame, return_value) -> None:
        """
        Handle function return event.
        
        Args:
            frame: Current frame object
            return_value: Return value from function
        """
        try:
            function_name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            
            if self._should_skip_frame(filename, function_name):
                return
            
            timestamp_us = self._get_time_us()
            thread_id = threading.get_ident()
            
            # Represent return value safely (avoid expensive repr)
            return_repr = type(return_value).__name__ if return_value is not None else 'None'
            
            event = {
                'type': 'cpu_return',
                'name': function_name,
                'timestamp_us': timestamp_us,
                'thread_id': thread_id,
                'return_type': return_repr,
                'metadata': {
                    'event': 'return'
                }
            }
            
            # Store in thread-local buffer
            buffer = self._get_thread_buffer()
            buffer.add_event(event)
            
            # Enqueue to shared buffer
            success = self.trace_buffer.enqueue(event)
            if success:
                self._total_events_captured += 1
        
        except Exception as e:
            logger.debug(f"Error handling return event: {e}")
    
    def _should_skip_frame(self, filename: str, function_name: str) -> bool:
        """
        Determine if a frame should be skipped from profiling.
        
        Args:
            filename: Frame's source filename
            function_name: Function name
        
        Returns:
            bool: True if frame should be skipped
        """
        # Skip profiling infrastructure itself
        if 'inferscope' in filename and 'collectors' in filename:
            return True
        
        # Skip Python internals
        if filename.startswith('<'):
            return True
        
        # Skip trace/profiling modules
        if any(x in filename for x in ['pdb', 'trace', 'profile', '__tracemalloc__']):
            return True
        
        # Skip dunder methods (noisy, not interesting)
        if function_name.startswith('__') and function_name.endswith('__'):
            return True
        
        return False
    
    def _get_thread_buffer(self) -> ThreadEventBuffer:
        """
        Get or create thread-local event buffer.
        
        Returns:
            ThreadEventBuffer: Thread-local buffer
        """
        if not hasattr(self._thread_local_buffers, 'buffer'):
            thread_id = threading.get_ident()
            buffer = ThreadEventBuffer(thread_id=thread_id)
            buffer.start_time_us = self._get_time_us()
            self._thread_local_buffers.buffer = buffer
            self._all_thread_buffers[thread_id] = buffer
        
        return self._thread_local_buffers.buffer
    
    @staticmethod
    def _get_time_us() -> int:
        """
        Get current time in microseconds.
        
        Uses monotonic clock for accuracy (not affected by NTP adjustments).
        
        Returns:
            int: Time in microseconds
        """
        return int(time.monotonic_ns() // 1000)
    
    def get_thread_local_buffer(self) -> List[Dict[str, Any]]:
        """
        Return accumulated events for current thread.
        
        Non-blocking operation that returns a copy of the thread-local buffer.
        Safe to call at any time; events already in shared buffer are not modified.
        
        Returns:
            List[Dict]: Copy of accumulated events for this thread
        """
        if not hasattr(self._thread_local_buffers, 'buffer'):
            return []
        
        buffer = self._thread_local_buffers.buffer
        return [e.copy() for e in buffer.events]
    
    def get_all_thread_buffers(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get all thread buffers (for finalization).
        
        Call after stop() to merge all thread-local buffers.
        
        Returns:
            Dict mapping thread_id to list of events
        """
        result = {}
        for thread_id, buffer in self._all_thread_buffers.items():
            result[thread_id] = [e.copy() for e in buffer.events]
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dict with collection metrics (event count, duration, overhead, etc.)
        """
        return {
            'state': self.state.name,
            'total_events_captured': self._total_events_captured,
            'frame_skip_count': self._frame_skip_count,
            'start_time_us': self._start_time_us,
            'end_time_us': self._end_time_us,
            'duration_us': (
                self._end_time_us - self._start_time_us 
                if self._start_time_us and self._end_time_us else None
            ),
            'thread_count': len(self._all_thread_buffers),
            'threads': {
                tid: {
                    'event_count': buf.event_count,
                    'overflow_count': buf.overflow_count,
                    'start_time_us': buf.start_time_us,
                    'end_time_us': buf.end_time_us,
                }
                for tid, buf in self._all_thread_buffers.items()
            }
        }
