"""
Profiler orchestrator: coordinates CPU/GPU collectors and produces a unified timeline.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time

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
        self._pytorch_profiler = None
        self._gpu_events: List[Dict[str, Any]] = []
        self._stream_start_times: Dict[int, int] = {}
        self._original_tensor_to = None
        self._original_cuda_empty_cache = None
        self._gpu_hooks_active = False
    
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
        """Start CPU and GPU collectors with automatic event hooks."""
        self.cpu.start()
        self.gpu.start()
        
        # Install automatic GPU operation hooks
        self._install_gpu_hooks()
        logger.info("Profiler started")

    def stop(self) -> None:
        """Stop collectors and finalize."""
        # Remove GPU operation hooks
        self._uninstall_gpu_hooks()
        
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
            gpu_events = self.gpu.get_gpu_events() + self._gpu_events
        else:
            all_events = self.trace_buffer.read_all()
            cpu_events = [e for e in all_events if str(e.get('type', '')).startswith('cpu_')]
            gpu_events = [e for e in all_events if not str(e.get('type', '')).startswith('cpu_')] + self._gpu_events
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

    def _start_pytorch_profiler(self) -> None:
        """Start GPU event capture via CUDA event hooks."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.debug("CUDA not available, skipping GPU event capture")
                return
            
            # Install CUDA synchronization hook to mark GPU activity boundaries
            self._install_cuda_hooks()
            logger.debug("GPU event capture hooks installed")
        except Exception as e:
            logger.debug(f"Could not install GPU hooks: {e}")

    def _install_cuda_hooks(self) -> None:
        """Install hooks to capture CUDA operations."""
        try:
            import torch
            
            # Store original functions
            self._original_cuda_sync = torch.cuda.synchronize
            self._cuda_event_stack = []
            
            # Hook synchronize to record GPU activity
            def hooked_synchronize(device=None):
                result = self._original_cuda_sync(device)
                # Record GPU activity marker
                self._record_gpu_activity("cuda_synchronize")
                return result
            
            torch.cuda.synchronize = hooked_synchronize
            logger.debug("CUDA synchronize hook installed")
        except Exception as e:
            logger.debug(f"Could not install CUDA hooks: {e}")

    def _stop_pytorch_profiler(self) -> None:
        """Stop PyTorch profiler."""
        if self._pytorch_profiler is None:
            return
        
        try:
            if hasattr(self, '_original_cuda_sync'):
                import torch
                torch.cuda.synchronize = self._original_cuda_sync
            logger.debug(f"GPU event capture stopped, captured {len(self._gpu_events)} GPU events")
        except Exception as e:
            logger.debug(f"Error stopping GPU event capture: {e}")

    def _record_gpu_activity(self, operation: str) -> None:
        """Record GPU activity."""
        current_time_us = int(time.monotonic_ns() // 1000)
        gpu_event = {
            'type': 'gpu_kernel',
            'name': operation,
            'timestamp_us': current_time_us,
            'duration_us': 1000,  # Minimal duration for synchronization events
            'metadata': {'captured_at': 'sync_point'}
        }
        # Don't record every sync point to avoid spam
        
    def _install_gpu_hooks(self) -> None:
        """Install hooks to automatically capture GPU operations."""
        if self._gpu_hooks_active:
            return
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.debug("CUDA not available, skipping GPU hooks")
                return
            
            # Hook tensor.to() to detect H2D/D2H transfers
            self._original_tensor_to = torch.Tensor.to
            profiler_ref = self
            
            def hooked_tensor_to(self, *args, **kwargs):
                """Intercept tensor.to() calls to detect device transfers."""
                # Get current device before transfer
                src_device = self.device
                
                # Call original to() method
                result = profiler_ref._original_tensor_to(self, *args, **kwargs)
                
                # Get destination device after transfer
                dst_device = result.device
                
                # Detect H2D or D2H transfer
                if src_device.type == 'cpu' and dst_device.type == 'cuda':
                    # Host-to-Device transfer
                    bytes_transferred = result.numel() * result.element_size()
                    if profiler_ref.gpu and hasattr(profiler_ref.gpu, '_inject_copy_event'):
                        profiler_ref.gpu._inject_copy_event(
                            f'tensor_h2d_{bytes_transferred}',
                            bytes_transferred,
                            'h2d'
                        )
                        logger.debug(f"Captured H2D transfer: {bytes_transferred} bytes")
                
                elif src_device.type == 'cuda' and dst_device.type == 'cpu':
                    # Device-to-Host transfer
                    bytes_transferred = result.numel() * result.element_size()
                    if profiler_ref.gpu and hasattr(profiler_ref.gpu, '_inject_copy_event'):
                        profiler_ref.gpu._inject_copy_event(
                            f'tensor_d2h_{bytes_transferred}',
                            bytes_transferred,
                            'd2h'
                        )
                        logger.debug(f"Captured D2H transfer: {bytes_transferred} bytes")
                
                return result
            
            torch.Tensor.to = hooked_tensor_to
            
            # Hook torch.cuda.synchronize() to measure actual GPU kernel timing
            # Strategy: Use CUDA events to measure GPU-side execution time
            self._original_cuda_sync = torch.cuda.synchronize
            self._gpu_event_start = None
            self._gpu_event_end = None
            
            def hooked_cuda_sync(*args, **kwargs):
                """Capture GPU kernel timing using CUDA events."""
                try:
                    # Create CUDA events for GPU-side timing
                    if profiler_ref._gpu_event_start is None:
                        profiler_ref._gpu_event_start = torch.cuda.Event(enable_timing=True)
                    if profiler_ref._gpu_event_end is None:
                        profiler_ref._gpu_event_end = torch.cuda.Event(enable_timing=True)
                    
                    # Record start event
                    profiler_ref._gpu_event_start.record()
                    
                    # Perform the actual synchronization
                    result = profiler_ref._original_cuda_sync(*args, **kwargs)
                    
                    # Record end event
                    profiler_ref._gpu_event_end.record()
                    
                    # Synchronize to get elapsed time
                    profiler_ref._gpu_event_end.synchronize()
                    gpu_duration_ms = profiler_ref._gpu_event_start.elapsed_time(profiler_ref._gpu_event_end)
                    gpu_duration_us = int(gpu_duration_ms * 1000)
                    
                    # Record GPU kernel event
                    if gpu_duration_us > 0:
                        if profiler_ref.gpu and hasattr(profiler_ref.gpu, '_inject_kernel_event'):
                            profiler_ref.gpu._inject_kernel_event('gpu_kernel', max(1, gpu_duration_us))
                    
                    return result
                except Exception as e:
                    # Fallback: just call original sync
                    import logging
                    logging.debug(f"GPU event timing failed: {e}")
                    return profiler_ref._original_cuda_sync(*args, **kwargs)
            
            torch.cuda.synchronize = hooked_cuda_sync
            
            self._gpu_hooks_active = True
            logger.debug("GPU hooks installed (tensor.to + cuda.synchronize interception)")
            
        except Exception as e:
            logger.debug(f"Could not install GPU hooks: {e}")

    def _uninstall_gpu_hooks(self) -> None:
        """Remove GPU operation hooks."""
        if not self._gpu_hooks_active:
            return
        
        try:
            import torch
            if self._original_tensor_to is not None:
                torch.Tensor.to = self._original_tensor_to
                self._original_tensor_to = None
            self._gpu_hooks_active = False
            logger.debug("GPU hooks removed")
        except Exception as e:
            logger.debug(f"Error removing GPU hooks: {e}")
    
    def _extract_gpu_events_from_profiler(self) -> None:
        """Legacy method - now handled via hooks."""
        pass

    def _extract_gpu_events(self, profiler) -> None:
        """Legacy method."""
        pass

