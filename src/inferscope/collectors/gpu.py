"""
GPU Collector Implementation (Mock/Stubs)

Captures GPU kernel execution and memory transfers using NVIDIA CUPTI.
This implementation provides a safe, testable stub that degrades gracefully
when CUDA/CUPTI is unavailable.
"""

import logging
import os
import ctypes
import threading
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CollectorState(Enum):
    UNINITIALIZED = 0
    IDLE = 1
    COLLECTING = 2
    FINALIZED = 3


@dataclass
class GpuEvent:
    type: str
    name: str
    timestamp_us: int
    stream_id: Optional[int] = None
    duration_us: Optional[int] = None
    bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GpuCollector:
    """
    GPU Collector facade that integrates with CUPTI when available.
    In unit tests, operates in mock mode.
    """

    def __init__(self, trace_buffer):
        if trace_buffer is None:
            raise ValueError("trace_buffer cannot be None")
        self.trace_buffer = trace_buffer
        self.state = CollectorState.IDLE
        self._cupti_available = False  # Determined at start()
        self._events: List[Dict[str, Any]] = []
        self._start_time_us: Optional[int] = None
        self._end_time_us: Optional[int] = None
        self._activity_buffer_overflow = 0
        self._thread_id = threading.get_ident()
        logger.debug("GPU Collector initialized (IDLE)")

    def start(self) -> None:
        """Enable CUPTI callbacks and activity collection."""
        if self.state == CollectorState.FINALIZED:
            raise RuntimeError("Cannot restart finalized GPU collector")
        if self.state == CollectorState.COLLECTING:
            logger.debug("GPU Collector already collecting; skipping start")
            return
        self._start_time_us = self._now_us()
        # Attempt to initialize CUPTI (mocked/unavailable in tests)
        try:
            # Best-effort CUPTI detection: attempt to load libcupti.so
            cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
            candidates = [
                os.path.join(cuda_home, 'extras', 'CUPTI', 'lib64', 'libcupti.so'),
                os.path.join(cuda_home, 'targets', 'x86_64-linux', 'lib', 'libcupti.so'),
                os.path.join(cuda_home, 'targets', 'sbsa-linux', 'lib', 'libcupti.so'),
            ]
            loaded = False
            for p in candidates:
                if os.path.exists(p):
                    try:
                        ctypes.CDLL(p)
                        loaded = True
                        break
                    except Exception:
                        continue
            self._cupti_available = loaded
            if not self._cupti_available:
                logger.warning("CUPTI unavailable; running in CPU-only mode")
        except Exception as e:
            logger.error(f"Failed to initialize CUPTI: {e}")
            self._cupti_available = False
        self.state = CollectorState.COLLECTING
        logger.info("GPU Collector started")

    def stop(self) -> None:
        """Disable CUPTI collection; finalize GPU buffer."""
        if self.state != CollectorState.COLLECTING:
            logger.debug("GPU Collector not collecting; skipping stop")
            return
        # Flush activity buffers (noop in mock)
        self._flush()
        self._end_time_us = self._now_us()
        self.state = CollectorState.FINALIZED
        logger.info("GPU Collector stopped")

    def get_gpu_events(self) -> List[Dict[str, Any]]:
        """Return collected GPU events with synchronized timestamps."""
        return [e.copy() for e in self._events]

    # ---- Internal helpers ----
    def _flush(self) -> None:
        """Flush pending GPU activity buffers (mock)."""
        # In real implementation, query CUPTI for completed activities.
        pass

    @staticmethod
    def _now_us() -> int:
        return int(time.monotonic_ns() // 1000)

    # The following methods are testing aids for injecting synthetic events
    def _inject_kernel_event(self, name: str, duration_us: int, stream_id: Optional[int] = None) -> None:
        """Inject a synthetic GPU kernel event (for unit tests)."""
        event = {
            'type': 'gpu_kernel',
            'name': name,
            'timestamp_us': self._now_us(),
            'duration_us': duration_us,
            'stream_id': stream_id,
            'metadata': {'source': 'synthetic'},
        }
        self._events.append(event)
        self.trace_buffer.enqueue(event)

    def _inject_copy_event(self, name: str, bytes_transferred: int, direction: str) -> None:
        """Inject a synthetic H2D/D2H copy event (for unit tests)."""
        if direction not in ("h2d", "d2h"):
            raise ValueError("direction must be 'h2d' or 'd2h'")
        event_type = 'h2d_copy' if direction == 'h2d' else 'd2h_copy'
        event = {
            'type': event_type,
            'name': name,
            'timestamp_us': self._now_us(),
            'bytes': bytes_transferred,
            'metadata': {'source': 'synthetic'},
        }
        self._events.append(event)
        self.trace_buffer.enqueue(event)

    def get_statistics(self) -> Dict[str, Any]:
        """Return basic statistics for diagnostics."""
        return {
            'state': self.state.name,
            'cupti_available': self._cupti_available,
            'event_count': len(self._events),
            'activity_buffer_overflow': self._activity_buffer_overflow,
            'start_time_us': self._start_time_us,
            'end_time_us': self._end_time_us,
            'duration_us': (
                self._end_time_us - self._start_time_us
                if self._start_time_us and self._end_time_us else None
            ),
        }
