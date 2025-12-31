"""
Bottleneck Analyzer Engine

Analyzes unified timelines to detect performance bottlenecks and generate
actionable suggestions.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    """Bottleneck classification types."""
    CPU_BOUND = "cpu_bound"
    GPU_BOUND = "gpu_bound"
    MEMORY_BOUND = "memory_bound"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


@dataclass
class BottleneckAnalysis:
    """Result of bottleneck analysis."""
    bottleneck_type: BottleneckType
    confidence: float
    primary_cause: str
    evidence: List[str] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timeline_breakdown: List[Dict[str, Any]] = field(default_factory=list)


class BottleneckAnalyzer:
    """
    Analyzes unified timeline to detect bottlenecks and generate suggestions.
    
    Detection Rules:
    - GPU idle > 10% → CPU-bound
    - CPU idle > 10% and GPU busy → GPU-bound
    - H2D+D2H > 20% → Memory-bound
    - Balanced otherwise
    """
    
    # Thresholds for classification
    IDLE_THRESHOLD = 0.10  # 10%
    MEMORY_OVERHEAD_THRESHOLD = 0.20  # 20%
    CONFIDENCE_HIGH = 0.8
    CONFIDENCE_MEDIUM = 0.6
    
    def __init__(self, timeline: List[Dict[str, Any]]):
        """
        Initialize analyzer with unified timeline.
        
        Args:
            timeline: Sorted list of events with global_ts_us
        """
        self.timeline = timeline
        self._analysis: Optional[BottleneckAnalysis] = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run bottleneck analysis on timeline.
        
        Returns:
            Dict with bottleneck classification, suggestions, and breakdown
        """
        if not self.timeline:
            return self._empty_analysis()
        
        # Compute timeline statistics
        summary = self._compute_summary()
        breakdown = self._compute_breakdown(summary)
        
        # Detect bottleneck
        bottleneck_type, confidence, evidence = self._detect_bottleneck(summary, breakdown)
        primary_cause = self._identify_primary_cause(breakdown, bottleneck_type)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(bottleneck_type, breakdown, summary)
        
        # Build analysis result
        self._analysis = BottleneckAnalysis(
            bottleneck_type=bottleneck_type,
            confidence=confidence,
            primary_cause=primary_cause,
            evidence=evidence,
            suggestions=suggestions,
            summary=summary,
            timeline_breakdown=breakdown,
        )
        
        return {
            'analysis_version': '1.0',
            'bottleneck': {
                'type': bottleneck_type.value,
                'primary_cause': primary_cause,
                'confidence': confidence,
                'evidence': evidence,
            },
            'suggestions': suggestions,
            'summary': summary,
            'timeline_breakdown': breakdown,
        }
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute high-level timeline statistics."""
        if not self.timeline:
            return {}
        
        # Find time range
        start_ts = min(e.get('global_ts_us', e.get('timestamp_start_us', e.get('timestamp_us', 0))) 
                      for e in self.timeline)
        end_ts = max(e.get('global_ts_us', e.get('timestamp_end_us', e.get('timestamp_us', 0))) 
                    for e in self.timeline)
        total_duration_us = end_ts - start_ts if end_ts > start_ts else 1
        
        # Calculate actual CPU wall-clock time by tracking active periods
        # Strategy: Use scope events to identify CPU-active regions
        cpu_active_ranges = []  # List of (start, end) tuples
        
        # Find scope regions which indicate CPU work
        scope_stack = {}  # scope_name -> enter_timestamp
        for event in sorted(self.timeline, key=lambda e: e.get('global_ts_us', e.get('timestamp_us', 0))):
            etype = event.get('type', '')
            
            if etype == 'scope_enter':
                scope_name = event.get('name', '')
                ts = event.get('global_ts_us', event.get('timestamp_us', 0))
                scope_stack[scope_name] = ts
            
            elif etype == 'scope_exit':
                scope_name = event.get('name', '')
                ts = event.get('global_ts_us', event.get('timestamp_us', 0))
                if scope_name in scope_stack:
                    start_ts_scope = scope_stack.pop(scope_name)
                    # Only count scopes that typically have CPU work (tokenization, decoding)
                    # Inference scope may be GPU-dominant
                    if scope_name in ['tokenization', 'decoding', 'preprocessing', 'postprocessing']:
                        cpu_active_ranges.append((start_ts_scope, ts))
        
        # For inference scope, we need to exclude GPU time
        # Find inference scope and compute CPU time as (scope_duration - gpu_kernel_time)
        inference_start = None
        inference_end = None
        for event in self.timeline:
            if event.get('type') == 'scope_enter' and event.get('name') == 'inference':
                inference_start = event.get('global_ts_us', event.get('timestamp_us', 0))
            elif event.get('type') == 'scope_exit' and event.get('name') == 'inference':
                inference_end = event.get('global_ts_us', event.get('timestamp_us', 0))
        
        # Calculate GPU, CPU, and memory transfer times
        gpu_time = 0
        h2d_time = 0
        d2h_time = 0
        cpu_call_time = 0
        
        for event in self.timeline:
            etype = event.get('type', '')
            duration = event.get('duration_us', 0)
            if duration == 0:
                start = event.get('timestamp_start_us', event.get('global_ts_us', 0))
                end = event.get('timestamp_end_us', event.get('global_ts_us', 0))
                duration = max(0, end - start)
            
            if etype.startswith('gpu_'):
                gpu_time += duration
            elif etype in ('h2d_copy', 'H2D'):
                h2d_time += duration
            elif etype in ('d2h_copy', 'D2H'):
                d2h_time += duration
            elif etype in ('cpu_call', 'cpu_function'):
                cpu_call_time += duration
        
        # Merge overlapping CPU ranges
        def merge_ranges(ranges):
            if not ranges:
                return []
            sorted_ranges = sorted(ranges)
            merged = [sorted_ranges[0]]
            for start, end in sorted_ranges[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            return merged
        
        merged_cpu_ranges = merge_ranges(cpu_active_ranges)
        cpu_time = sum(end - start for start, end in merged_cpu_ranges)
        
        # Add direct CPU call time from cpu_call events
        cpu_time += cpu_call_time
        
        # Add CPU overhead during inference (launch overhead, synchronization, etc.)
        # Estimate as inference_scope_duration - gpu_time - h2d_time - d2h_time
        if inference_start and inference_end:
            inference_duration = inference_end - inference_start
            cpu_overhead_in_inference = max(0, inference_duration - gpu_time - h2d_time - d2h_time)
            cpu_time += cpu_overhead_in_inference
        
        return {
            'end_to_end_latency_us': total_duration_us,
            'total_cpu_time_us': cpu_time,
            'total_gpu_time_us': gpu_time,
            'total_h2d_us': h2d_time,
            'total_d2h_us': d2h_time,
            'start_ts_us': start_ts,
            'end_ts_us': end_ts,
        }
    
    def _compute_breakdown(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compute timeline breakdown by category.
        
        Returns mutually exclusive time slices that sum to 100% of wall-clock time.
        Shows what the system was doing during each portion of the end-to-end latency.
        """
        total = summary.get('end_to_end_latency_us', 1)
        
        # Get durations of each activity type
        cpu_time = summary.get('total_cpu_time_us', 0)
        gpu_time = summary.get('total_gpu_time_us', 0)
        h2d_time = summary.get('total_h2d_us', 0)
        d2h_time = summary.get('total_d2h_us', 0)
        
        categories = []
        
        # GPU compute is the dominant activity in most inference workloads
        # Show it as percentage of wall-clock time
        if gpu_time > 0:
            categories.append({
                'category': 'gpu_compute',
                'duration_us': gpu_time,
                'percentage': round(100.0 * gpu_time / total, 1),
            })
        
        # Memory transfers (usually sequential, not overlapping with compute)
        if h2d_time > 0:
            categories.append({
                'category': 'h2d_copy',
                'duration_us': h2d_time,
                'percentage': round(100.0 * h2d_time / total, 1),
            })
        
        if d2h_time > 0:
            categories.append({
                'category': 'd2h_copy',
                'duration_us': d2h_time,
                'percentage': round(100.0 * d2h_time / total, 1),
            })
        
        # CPU preprocessing (may overlap with GPU, so show separately)
        # Only include if significant (>1% of total time)
        if cpu_time > total * 0.01:
            categories.append({
                'category': 'cpu_preprocessing',
                'duration_us': cpu_time,
                'percentage': round(100.0 * cpu_time / total, 1),
            })
        
        # Calculate idle/unaccounted time
        # Sum all activities - if they don't overlap, sum = total
        # If they overlap (concurrent CPU+GPU), sum > total, so idle will be 0
        total_activity = cpu_time + gpu_time + h2d_time + d2h_time
        idle = max(0, total - total_activity)
        
        if idle > total * 0.01:  # Only show if >1% of total time
            categories.append({
                'category': 'idle',
                'duration_us': int(idle),
                'percentage': round(100.0 * idle / total, 1),
            })
        
        # Ensure percentages sum to ~100% by normalizing if needed
        total_pct = sum(c['percentage'] for c in categories)
        if abs(total_pct - 100.0) > 1.0:  # If off by more than 1%
            # Normalize to exactly 100%
            for cat in categories:
                cat['percentage'] = round(100.0 * cat['percentage'] / total_pct, 1)
        
        return sorted(categories, key=lambda x: x['duration_us'], reverse=True)
    
    def _detect_bottleneck(
        self, summary: Dict[str, Any], breakdown: List[Dict[str, Any]]
    ) -> tuple[BottleneckType, float, List[str]]:
        """Detect bottleneck type with confidence and evidence."""
        total = summary.get('end_to_end_latency_us', 1)
        evidence = []
        
        # Compute ratios
        cpu_time_us = summary.get('total_cpu_time_us', 0)
        gpu_time_us = summary.get('total_gpu_time_us', 0)
        h2d_us = summary.get('total_h2d_us', 0)
        d2h_us = summary.get('total_d2h_us', 0)
        
        # Note: CPU ratio may exceed 100% due to nested function calls (call stack accumulation)
        # So we use absolute time comparisons instead of ratios
        gpu_utilization = gpu_time_us / total  # GPU utilization as % of wall-clock time
        memory_ratio = (h2d_us + d2h_us) / total
        
        idle_cat = next((c for c in breakdown if c['category'] == 'idle'), None)
        idle_ratio = (idle_cat['duration_us'] / total) if idle_cat else 0.0
        
        # Rule 1: High memory overhead → Memory-bound
        if memory_ratio > self.MEMORY_OVERHEAD_THRESHOLD:
            evidence.append(f"Memory transfers account for {memory_ratio*100:.1f}% of time")
            evidence.append(f"H2D+D2H overhead is above threshold ({self.MEMORY_OVERHEAD_THRESHOLD*100}%)")
            return BottleneckType.MEMORY_BOUND, self.CONFIDENCE_HIGH, evidence
        
        # Rule 2: High GPU utilization (>80% of wall-clock time) → GPU-bound
        if gpu_utilization > 0.8:
            evidence.append(f"GPU utilization: {gpu_utilization*100:.1f}% of wall-clock time")
            evidence.append(f"GPU is busy for most of the inference pipeline")
            return BottleneckType.GPU_BOUND, self.CONFIDENCE_HIGH, evidence
        
        # Rule 3: High idle time with low GPU usage → CPU-bound
        if idle_ratio > self.IDLE_THRESHOLD and gpu_utilization < 0.5:
            evidence.append(f"GPU idle for {idle_ratio*100:.1f}% of timeline")
            evidence.append(f"GPU utilization only {gpu_utilization*100:.1f}%")
            return BottleneckType.CPU_BOUND, self.CONFIDENCE_HIGH, evidence
        
        # Rule 4: Very low GPU utilization but high CPU activity → CPU-bound
        if gpu_utilization < 0.2 and cpu_time_us > total:
            evidence.append(f"GPU utilization very low: {gpu_utilization*100:.1f}%")
            evidence.append(f"CPU shows significant activity")
            return BottleneckType.CPU_BOUND, self.CONFIDENCE_MEDIUM, evidence
        
        # Rule 5: Moderate GPU utilization (40-80%) → Could be balanced
        if 0.4 < gpu_utilization < 0.8:
            evidence.append(f"GPU utilization: {gpu_utilization*100:.1f}%")
            evidence.append("Workload appears balanced between CPU and GPU")
            return BottleneckType.BALANCED, 0.7, evidence
        
        # Unknown
        evidence.append("Insufficient data for confident classification")
        evidence.append(f"GPU utilization: {gpu_utilization*100:.1f}%, Idle: {idle_ratio*100:.1f}%")
        return BottleneckType.UNKNOWN, 0.3, evidence
    
    def _identify_primary_cause(
        self, breakdown: List[Dict[str, Any]], bottleneck_type: BottleneckType
    ) -> str:
        """Identify primary cause from breakdown."""
        if not breakdown:
            return "unknown"
        
        # Top category by duration
        top_category = breakdown[0]['category']
        
        if bottleneck_type == BottleneckType.CPU_BOUND:
            return "cpu_preprocessing" if top_category == 'cpu_preprocessing' else "cpu_overhead"
        elif bottleneck_type == BottleneckType.MEMORY_BOUND:
            return "h2d_copy" if 'h2d' in top_category else "memory_transfers"
        elif bottleneck_type == BottleneckType.GPU_BOUND:
            return "gpu_compute"
        
        return top_category
    
    def _generate_suggestions(
        self, bottleneck_type: BottleneckType, breakdown: List[Dict[str, Any]], summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable suggestions based on bottleneck type."""
        suggestions = []
        
        if bottleneck_type == BottleneckType.CPU_BOUND:
            suggestions.append({
                'priority': 'high',
                'action': 'Increase batch size',
                'rationale': 'Amortize CPU preprocessing overhead across more samples',
                'estimated_improvement_percent': 15,
            })
            suggestions.append({
                'priority': 'medium',
                'action': 'Move preprocessing off critical path',
                'rationale': 'Use async data loading or separate preprocessing workers',
                'estimated_improvement_percent': 10,
            })
        
        elif bottleneck_type == BottleneckType.MEMORY_BOUND:
            h2d_time = summary.get('total_h2d_us', 0)
            d2h_time = summary.get('total_d2h_us', 0)
            if h2d_time > d2h_time:
                suggestions.append({
                    'priority': 'high',
                    'action': 'Enable pinned memory for inputs',
                    'rationale': 'Improve H2D transfer throughput',
                    'estimated_improvement_percent': 20,
                })
            suggestions.append({
                'priority': 'medium',
                'action': 'Increase batch size',
                'rationale': 'Amortize H2D/D2H copy overhead',
                'estimated_improvement_percent': 12,
            })
        
        elif bottleneck_type == BottleneckType.GPU_BOUND:
            suggestions.append({
                'priority': 'high',
                'action': 'Optimize GPU kernels',
                'rationale': 'Profile kernel execution; consider kernel fusion or quantization',
                'estimated_improvement_percent': 25,
            })
            suggestions.append({
                'priority': 'medium',
                'action': 'Use mixed precision (FP16)',
                'rationale': 'Reduce GPU compute time while maintaining accuracy',
                'estimated_improvement_percent': 30,
            })
        
        elif bottleneck_type == BottleneckType.BALANCED:
            suggestions.append({
                'priority': 'low',
                'action': 'System is relatively balanced',
                'rationale': 'Consider overall throughput optimization or scaling',
                'estimated_improvement_percent': 5,
            })
        
        return suggestions
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for empty timeline."""
        return {
            'analysis_version': '1.0',
            'bottleneck': {
                'type': BottleneckType.UNKNOWN.value,
                'primary_cause': 'no_data',
                'confidence': 0.0,
                'evidence': ['No events in timeline'],
            },
            'suggestions': [],
            'summary': {},
            'timeline_breakdown': [],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return analyzer statistics."""
        return {
            'event_count': len(self.timeline),
            'analysis_complete': self._analysis is not None,
            'bottleneck_type': self._analysis.bottleneck_type.value if self._analysis else None,
        }
