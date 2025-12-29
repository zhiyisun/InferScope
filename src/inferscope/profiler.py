"""
Profiler orchestrator: coordinates CPU/GPU collectors and produces a unified timeline.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .collectors.cpu import CpuCollector
from .collectors.gpu import GpuCollector
from .timeline.merger import TimelineMerger

logger = logging.getLogger(__name__)


@dataclass
class ProfilerStats:
    cpu: Dict[str, Any]
    gpu: Dict[str, Any]


class Profiler:
    """High-level profiling orchestrator."""

    def __init__(self, trace_buffer):
        if trace_buffer is None:
            raise ValueError("trace_buffer cannot be None")
        
        # Check for multi-GPU (NFR-2: single GPU only for MVP)
        self._check_single_gpu()
        
        self.trace_buffer = trace_buffer
        self.cpu = CpuCollector(trace_buffer)
        self.gpu = GpuCollector(trace_buffer)
        self._unified: Optional[List[Dict[str, Any]]] = None
    
    @staticmethod
    def _check_single_gpu():
        """Enforce single GPU constraint per NFR-2."""
        try:
            import torch
            device_count = torch.cuda.device_count()
            if device_count > 1:
                raise RuntimeError(
                    f"InferScope MVP supports single GPU only. "
                    f"Found {device_count} GPUs. "
                    f"Please set CUDA_VISIBLE_DEVICES to use a single GPU."
                )
        except ImportError:
            # PyTorch not available, skip check
            pass
        except Exception as e:
            # Log warning but don't fail
            logger.warning(f"GPU check failed: {e}")

    def start(self) -> None:
        """Start CPU and GPU collectors."""
        self.cpu.start()
        self.gpu.start()
        logger.info("Profiler started")

    def stop(self) -> None:
        """Stop collectors and finalize."""
        self.cpu.stop()
        self.gpu.stop()
        logger.info("Profiler stopped")

    def get_stats(self) -> ProfilerStats:
        return ProfilerStats(cpu=self.cpu.get_statistics(), gpu=self.gpu.get_statistics())

    def get_unified_timeline(self) -> List[Dict[str, Any]]:
        """
        Read events from the shared trace buffer, split by origin, and merge.
        """
        # Prefer reading from trace buffer to capture both CPU and GPU events
        if not hasattr(self.trace_buffer, 'read_all'):
            # Fallback: use per-collector buffers
            cpu_events = self._collect_cpu_events()
            gpu_events = self.gpu.get_gpu_events()
        else:
            all_events = self.trace_buffer.read_all()
            cpu_events = [e for e in all_events if str(e.get('type', '')).startswith('cpu_')]
            gpu_events = [e for e in all_events if not str(e.get('type', '')).startswith('cpu_')]
            # Under coverage or other tracing tools, CPU events may not be enqueued.
            # If none were found, fallback to per-thread buffers.
            if len(cpu_events) == 0:
                fallback_cpu = self._collect_cpu_events()
                if len(fallback_cpu) > 0:
                    cpu_events = fallback_cpu
        merger = TimelineMerger(cpu_events, gpu_events)
        self._unified = merger.get_unified_timeline()
        return self._unified

    def _collect_cpu_events(self) -> List[Dict[str, Any]]:
        # Merge thread buffers from CpuCollector
        buffers = self.cpu.get_all_thread_buffers()
        merged: List[Dict[str, Any]] = []
        for _, events in buffers.items():
            merged.extend(events)
        return merged
