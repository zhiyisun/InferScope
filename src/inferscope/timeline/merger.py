"""
Timeline Merger Implementation (MVP)

Synchronizes CPU and GPU event streams and produces a unified, ordered timeline.
Clock synchronization uses a simple affine mapping (slope, intercept) with
conservative defaults when calibration data is unavailable.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    slope: float
    intercept: float
    error_us: float
    method: str


class TimelineMerger:
    """
    Merge CPU and GPU timelines into a single, synchronized timeline.
    """

    def __init__(self, cpu_events: List[Dict[str, Any]], gpu_events: List[Dict[str, Any]]):
        self.cpu_events = cpu_events or []
        self.gpu_events = gpu_events or []
        self._sync: Optional[SyncResult] = None
        self._state = 'Ready'

    def synchronize_clocks(self) -> Dict[str, float]:
        """
        Calibrate CPU ↔ GPU timestamp mapping.
        Returns: {"slope": float, "intercept": float, "error_us": float}
        """
        # For MVP/unit tests: assume clocks are already comparable, slope=1
        # and compute intercept via reference pair if available.
        # If unavailable, use intercept=0 and conservative error.
        self._state = 'Synchronized'
        slope = 1.0
        intercept = 0.0
        error_us = 100.0
        method = 'assumed'

        # If we can infer an offset from synthetic event metadata, use it.
        # Example: gpu event metadata may include 'cpu_ref_us' alongside 'timestamp_us'.
        # Find first GPU event with 'cpu_ref_us'.
        for e in self.gpu_events:
            cpu_ref = e.get('metadata', {}).get('cpu_ref_us')
            gpu_ts = e.get('timestamp_us')
            if cpu_ref is not None and gpu_ts is not None:
                intercept = gpu_ts - slope * cpu_ref
                method = 'metadata_ref'
                error_us = 50.0
                break
        
        self._sync = SyncResult(slope=slope, intercept=intercept, error_us=error_us, method=method)
        return {
            'slope': slope,
            'intercept': intercept,
            'error_us': error_us,
            'method': method,
        }

    def get_unified_timeline(self) -> List[Dict[str, Any]]:
        """
        Return all events sorted by global timestamp.
        """
        if self._sync is None:
            # Auto-sync for convenience
            self.synchronize_clocks()
        self._state = 'Finalized'
        slope = self._sync.slope
        intercept = self._sync.intercept

        merged: List[Dict[str, Any]] = []

        def to_global_ts(event: Dict[str, Any]) -> Optional[int]:
            # CPU events: prefer 'timestamp_start_us' (call) else 'timestamp_us'
            ts = event.get('timestamp_start_us') or event.get('timestamp_us')
            if ts is None:
                return None
            # If event is CPU-origin, apply slope/intercept; if GPU, assume already GPU time.
            origin = 'cpu' if event.get('type', '').startswith('cpu_') else 'gpu'
            if origin == 'cpu':
                return int(slope * ts + intercept)
            return int(ts)

        # Normalize and annotate
        for e in self.cpu_events + self.gpu_events:
            ts = to_global_ts(e)
            if ts is None:
                continue
            ev = {**e}
            ev['global_ts_us'] = ts
            ev['sync_error_us'] = self._sync.error_us
            merged.append(ev)

        # Sort by global timestamp, stable sort to preserve relative order
        merged.sort(key=lambda x: x['global_ts_us'])
        return merged

    def get_sync_metadata(self) -> Dict[str, Any]:
        """Return clock sync calibration details for report."""
        if self._sync is None:
            return {}
        return {
            'slope': self._sync.slope,
            'intercept': self._sync.intercept,
            'error_us': self._sync.error_us,
            'method': self._sync.method,
            'state': self._state,
        }
