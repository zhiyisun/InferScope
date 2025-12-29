"""
Unit tests for Bottleneck Analyzer.
"""

import pytest


class TestBottleneckAnalyzerInitialization:
    """Tests for analyzer initialization."""
    
    def test_analyzer_initializes_with_empty_timeline(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        analyzer = BottleneckAnalyzer([])
        assert analyzer is not None
        assert analyzer.timeline == []
    
    def test_analyzer_initializes_with_timeline(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 100, 'duration_us': 50},
            {'type': 'gpu_kernel', 'global_ts_us': 200, 'duration_us': 100},
        ]
        analyzer = BottleneckAnalyzer(timeline)
        assert len(analyzer.timeline) == 2


class TestBottleneckDetection:
    """Tests for bottleneck detection rules."""
    
    def test_detects_cpu_bound_workload(self):
        from src.inferscope.analyzer import BottleneckAnalyzer, BottleneckType
        
        # Timeline with high CPU time, low GPU
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'timestamp_start_us': 0, 'timestamp_end_us': 300, 'duration_us': 300},
            {'type': 'gpu_kernel', 'global_ts_us': 400, 'duration_us': 50},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        assert result['bottleneck']['type'] == BottleneckType.CPU_BOUND.value
        assert result['bottleneck']['confidence'] >= 0.6
        assert len(result['bottleneck']['evidence']) > 0
    
    def test_detects_gpu_bound_workload(self):
        from src.inferscope.analyzer import BottleneckAnalyzer, BottleneckType
        
        # Timeline with high GPU time, low CPU
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 50},
            {'type': 'gpu_kernel', 'global_ts_us': 100, 'timestamp_start_us': 100, 'timestamp_end_us': 500, 'duration_us': 400},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        assert result['bottleneck']['type'] == BottleneckType.GPU_BOUND.value
        assert result['bottleneck']['confidence'] >= 0.6
    
    def test_detects_memory_bound_workload(self):
        from src.inferscope.analyzer import BottleneckAnalyzer, BottleneckType
        
        # Timeline with high H2D+D2H overhead
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 50},
            {'type': 'h2d_copy', 'global_ts_us': 100, 'duration_us': 200},
            {'type': 'gpu_kernel', 'global_ts_us': 350, 'duration_us': 100},
            {'type': 'd2h_copy', 'global_ts_us': 500, 'duration_us': 150},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        assert result['bottleneck']['type'] == BottleneckType.MEMORY_BOUND.value
        assert result['bottleneck']['confidence'] >= 0.7
    
    def test_detects_gpu_idle_as_cpu_bound(self):
        from src.inferscope.analyzer import BottleneckAnalyzer, BottleneckType
        
        # Timeline with 30% idle (implying GPU waiting for CPU)
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 200},
            {'type': 'gpu_kernel', 'global_ts_us': 300, 'duration_us': 100},
            # Idle gap: 0-300 has 100us unaccounted after CPU
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        # Should classify as CPU-bound since CPU time dominates
        assert result['bottleneck']['type'] == BottleneckType.CPU_BOUND.value
        assert result['bottleneck']['confidence'] >= 0.5


class TestSuggestionGeneration:
    """Tests for suggestion generation."""
    
    def test_cpu_bound_generates_batch_size_suggestion(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 400},
            {'type': 'gpu_kernel', 'global_ts_us': 500, 'duration_us': 50},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        suggestions = result.get('suggestions', [])
        assert len(suggestions) > 0
        assert any('batch size' in s['action'].lower() for s in suggestions)
    
    def test_memory_bound_generates_pinned_memory_suggestion(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [
            {'type': 'h2d_copy', 'global_ts_us': 0, 'duration_us': 300},
            {'type': 'gpu_kernel', 'global_ts_us': 350, 'duration_us': 100},
            {'type': 'd2h_copy', 'global_ts_us': 500, 'duration_us': 100},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        suggestions = result.get('suggestions', [])
        assert len(suggestions) > 0
        # Should suggest pinned memory or batch size
        actions = [s['action'].lower() for s in suggestions]
        assert any('pinned' in a or 'batch' in a for a in actions)
    
    def test_gpu_bound_generates_optimization_suggestion(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 50},
            {'type': 'gpu_kernel', 'global_ts_us': 100, 'duration_us': 500},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        suggestions = result.get('suggestions', [])
        assert len(suggestions) > 0
        # Should suggest GPU kernel optimization or mixed precision
        actions = ' '.join(s['action'].lower() for s in suggestions)
        assert 'kernel' in actions or 'precision' in actions or 'fp16' in actions


class TestTimelineBreakdown:
    """Tests for timeline breakdown computation."""
    
    def test_computes_breakdown_percentages(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 300},
            {'type': 'gpu_kernel', 'global_ts_us': 400, 'duration_us': 200},
            {'type': 'h2d_copy', 'global_ts_us': 700, 'duration_us': 100},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        breakdown = result.get('timeline_breakdown', [])
        assert len(breakdown) > 0
        
        # Check percentages sum to ~100%
        total_pct = sum(item['percentage'] for item in breakdown)
        assert 95 <= total_pct <= 105  # Allow small rounding error
    
    def test_breakdown_includes_all_categories(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 100},
            {'type': 'gpu_kernel', 'global_ts_us': 200, 'duration_us': 100},
            {'type': 'h2d_copy', 'global_ts_us': 400, 'duration_us': 50},
            {'type': 'd2h_copy', 'global_ts_us': 500, 'duration_us': 50},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        breakdown = result.get('timeline_breakdown', [])
        categories = {item['category'] for item in breakdown}
        
        assert 'cpu_preprocessing' in categories or any('cpu' in c for c in categories)
        assert 'gpu_compute' in categories or any('gpu' in c for c in categories)


class TestEmptyAndEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_timeline_returns_unknown(self):
        from src.inferscope.analyzer import BottleneckAnalyzer, BottleneckType
        
        analyzer = BottleneckAnalyzer([])
        result = analyzer.analyze()
        
        assert result['bottleneck']['type'] == BottleneckType.UNKNOWN.value
        assert result['bottleneck']['confidence'] == 0.0
        assert 'No events' in result['bottleneck']['evidence'][0]
    
    def test_single_event_timeline(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [{'type': 'cpu_call', 'global_ts_us': 100, 'duration_us': 50}]
        
        analyzer = BottleneckAnalyzer(timeline)
        result = analyzer.analyze()
        
        # Should not crash, should produce some result
        assert 'bottleneck' in result
        assert 'summary' in result
    
    def test_get_statistics(self):
        from src.inferscope.analyzer import BottleneckAnalyzer
        
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 0, 'duration_us': 100},
            {'type': 'gpu_kernel', 'global_ts_us': 200, 'duration_us': 100},
        ]
        
        analyzer = BottleneckAnalyzer(timeline)
        stats = analyzer.get_statistics()
        
        assert stats['event_count'] == 2
        assert stats['analysis_complete'] is False
        
        analyzer.analyze()
        stats = analyzer.get_statistics()
        assert stats['analysis_complete'] is True
        assert stats['bottleneck_type'] is not None
