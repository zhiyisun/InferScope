"""
Unit tests for Report Generator.
"""

import pytest
import json
import tempfile
import os


class TestReportGeneratorInitialization:
    """Tests for report generator initialization."""
    
    def test_generator_initializes_with_analysis(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'cpu_bound', 'confidence': 0.9},
            'summary': {'end_to_end_latency_us': 100000},
        }
        
        generator = ReportGenerator(analysis)
        assert generator is not None
        assert generator.analysis == analysis
    
    def test_generator_initializes_with_timeline(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {'bottleneck': {'type': 'cpu_bound'}}
        timeline = [{'type': 'cpu_call', 'global_ts_us': 100}]
        
        generator = ReportGenerator(analysis, timeline)
        assert len(generator.timeline) == 1


class TestMarkdownGeneration:
    """Tests for Markdown report generation."""
    
    def test_generates_markdown_report(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'analysis_version': '1.0',
            'bottleneck': {
                'type': 'cpu_bound',
                'primary_cause': 'cpu_preprocessing',
                'confidence': 0.92,
                'evidence': ['GPU idle for 20%', 'CPU time dominates'],
            },
            'summary': {
                'end_to_end_latency_us': 87400,
                'total_cpu_time_us': 31200,
                'total_gpu_time_us': 24100,
                'total_h2d_us': 18400,
                'total_d2h_us': 8700,
            },
            'timeline_breakdown': [
                {'category': 'cpu_preprocessing', 'duration_us': 31200, 'percentage': 35.7},
                {'category': 'gpu_compute', 'duration_us': 24100, 'percentage': 27.6},
            ],
            'suggestions': [
                {
                    'priority': 'high',
                    'action': 'Increase batch size',
                    'rationale': 'Amortize overhead',
                    'estimated_improvement_percent': 15,
                }
            ],
        }
        
        generator = ReportGenerator(analysis)
        markdown = generator.to_markdown()
        
        assert 'InferScope Performance Report' in markdown
        assert 'Cpu Bound' in markdown or 'cpu bound' in markdown.lower()
        assert '87.4 ms' in markdown  # End-to-end latency
        assert 'Increase batch size' in markdown
        assert 'high priority' in markdown.lower()
    
    def test_markdown_includes_summary(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'summary': {
                'end_to_end_latency_us': 50000,
                'total_cpu_time_us': 20000,
                'total_gpu_time_us': 15000,
            },
            'bottleneck': {'type': 'balanced'},
            'suggestions': [],
            'timeline_breakdown': [],
        }
        
        generator = ReportGenerator(analysis)
        markdown = generator.to_markdown()
        
        assert '## Summary' in markdown
        assert '50.0 ms' in markdown
        assert '20.0 ms' in markdown
    
    def test_markdown_includes_breakdown_table(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'summary': {'end_to_end_latency_us': 1000},
            'bottleneck': {'type': 'cpu_bound'},
            'timeline_breakdown': [
                {'category': 'cpu_preprocessing', 'duration_us': 500, 'percentage': 50.0},
                {'category': 'gpu_compute', 'duration_us': 300, 'percentage': 30.0},
            ],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        markdown = generator.to_markdown()
        
        assert '## Timeline Breakdown' in markdown
        assert '| Category |' in markdown
        assert 'Cpu Preprocessing' in markdown or 'cpu_preprocessing' in markdown
    
    def test_markdown_includes_suggestions(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'summary': {},
            'bottleneck': {'type': 'memory_bound'},
            'timeline_breakdown': [],
            'suggestions': [
                {'priority': 'high', 'action': 'Enable pinned memory', 'rationale': 'Faster H2D', 'estimated_improvement_percent': 20},
                {'priority': 'medium', 'action': 'Reduce batch size', 'rationale': 'Lower memory usage'},
            ],
        }
        
        generator = ReportGenerator(analysis)
        markdown = generator.to_markdown()
        
        assert '## Suggestions' in markdown
        assert 'Enable pinned memory' in markdown
        assert 'high priority' in markdown.lower()
        assert '20%' in markdown


class TestHTMLGeneration:
    """Tests for HTML report generation."""
    
    def test_generates_html_report(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'gpu_bound', 'confidence': 0.85, 'primary_cause': 'gpu_compute', 'evidence': []},
            'summary': {'end_to_end_latency_us': 100000},
            'timeline_breakdown': [],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        html = generator.to_html()
        
        assert '<!DOCTYPE html>' in html
        assert '<html>' in html
        assert 'InferScope Performance Report' in html
        assert '</html>' in html
    
    def test_html_includes_css_styling(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'cpu_bound'},
            'summary': {},
            'timeline_breakdown': [],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        html = generator.to_html()
        
        assert '<style>' in html
        assert 'font-family' in html
        assert '</style>' in html


class TestJSONGeneration:
    """Tests for JSON report generation."""
    
    def test_generates_json_report(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'memory_bound', 'confidence': 0.78},
            'summary': {'end_to_end_latency_us': 120000},
            'timeline_breakdown': [],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        json_str = generator.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert data['report_type'] == 'json'
        assert 'InferScope' in data['title']
        assert data['analysis']['bottleneck']['type'] == 'memory_bound'
    
    def test_json_includes_timeline_count(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {'bottleneck': {'type': 'balanced'}, 'summary': {}, 'timeline_breakdown': [], 'suggestions': []}
        timeline = [
            {'type': 'cpu_call', 'global_ts_us': 100},
            {'type': 'gpu_kernel', 'global_ts_us': 200},
            {'type': 'h2d_copy', 'global_ts_us': 300},
        ]
        
        generator = ReportGenerator(analysis, timeline)
        json_str = generator.to_json()
        
        data = json.loads(json_str)
        assert data['timeline_event_count'] == 3


class TestReportSaving:
    """Tests for saving reports to files."""
    
    def test_saves_markdown_to_file(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'cpu_bound', 'confidence': 0.9, 'primary_cause': 'cpu', 'evidence': []},
            'summary': {'end_to_end_latency_us': 50000},
            'timeline_breakdown': [],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            filepath = f.name
        
        try:
            generator.save(filepath, format='markdown')
            
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
                assert 'InferScope Performance Report' in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_saves_html_to_file(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'gpu_bound'},
            'summary': {},
            'timeline_breakdown': [],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            filepath = f.name
        
        try:
            generator.save(filepath, format='html')
            
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
                assert '<!DOCTYPE html>' in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_saves_json_to_file(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'memory_bound'},
            'summary': {},
            'timeline_breakdown': [],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            generator.save(filepath, format='json')
            
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert data['report_type'] == 'json'
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_invalid_format_raises_error(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {'bottleneck': {'type': 'unknown'}, 'summary': {}, 'timeline_breakdown': [], 'suggestions': []}
        generator = ReportGenerator(analysis)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match='Unknown format'):
                generator.save(filepath, format='invalid')
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestEdgeCases:
    """Tests for edge cases and empty data."""
    
    def test_empty_analysis_generates_minimal_report(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {},
            'summary': {},
            'timeline_breakdown': [],
            'suggestions': [],
        }
        
        generator = ReportGenerator(analysis)
        markdown = generator.to_markdown()
        
        # Should not crash, should produce header
        assert 'InferScope' in markdown
    
    def test_missing_fields_handled_gracefully(self):
        from src.inferscope.reporter import ReportGenerator
        
        analysis = {
            'bottleneck': {'type': 'cpu_bound'},
            # Missing summary, breakdown, suggestions
        }
        
        generator = ReportGenerator(analysis)
        markdown = generator.to_markdown()
        json_str = generator.to_json()
        html = generator.to_html()
        
        # Should not crash
        assert markdown is not None
        assert json_str is not None
        assert html is not None
