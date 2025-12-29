"""InferScope main package."""

__version__ = "0.1.0"

from .profiler import Profiler
from .timeline import TimelineMerger, SyncResult
from .collectors.cpu import CpuCollector
from .collectors.gpu import GpuCollector
from .analyzer import BottleneckAnalyzer, BottleneckType
from .reporter import ReportGenerator
from .api import scope, mark_event, set_global_profiler, get_global_profiler

__all__ = [
    "Profiler",
    "TimelineMerger",
    "SyncResult",
    "CpuCollector",
    "GpuCollector",
    "BottleneckAnalyzer",
    "BottleneckType",
    "ReportGenerator",
    "scope",
    "mark_event",
    "set_global_profiler",
    "get_global_profiler",
]
