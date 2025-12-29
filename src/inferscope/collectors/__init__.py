"""InferScope collectors package."""

from .cpu import CpuCollector
from .gpu import GpuCollector

__all__ = ['CpuCollector', 'GpuCollector']
