#!/usr/bin/env python3
import os
import subprocess
import json
import csv
from datetime import datetime

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
PY = os.path.join(WORKSPACE_ROOT, '.venv', 'bin', 'python')
DEMO = os.path.join(WORKSPACE_ROOT, 'demo_llm_inference.py')
CLI = os.path.join(WORKSPACE_ROOT, 'scripts', 'inferscope')

MODEL = os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base')
PROMPT = os.environ.get('PROMPT', 'Benchmarking InferScope scaling matrix')
TOKENS_LIST = [32, 64, 128]
BATCH_LIST = [1, 4, 8]

results = []

for max_tokens in TOKENS_LIST:
    for batch in BATCH_LIST:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trace_name = os.path.join('outputs', f"trace_{batch}b_{max_tokens}t_{stamp}.json")
        report_md = os.path.join('outputs', f"report_{batch}b_{max_tokens}t_{stamp}.md")
        report_html = os.path.join('outputs', f"report_{batch}b_{max_tokens}t_{stamp}.html")

        demo_cmd = [PY, DEMO, '--model', MODEL, '--prompt', PROMPT, '--max-new-tokens', str(max_tokens), '--batch-size', str(batch), '--stress-mode', '--output', trace_name]
        print('Running:', ' '.join(demo_cmd))
        subprocess.run(demo_cmd, check=True)

        analyze_md = [PY, CLI, 'analyze', trace_name, '--output', report_md]
        analyze_html = [PY, CLI, 'analyze', trace_name, '--output', report_html, '--format', 'html']
        print('Analyze MD:', ' '.join(analyze_md))
        subprocess.run(analyze_md, check=True)
        print('Analyze HTML:', ' '.join(analyze_html))
        subprocess.run(analyze_html, check=True)

        # Extract quick metrics
        with open(os.path.join(WORKSPACE_ROOT, report_md), 'r') as f:
            content = f.read()
        def extract(tag):
            import re
            m = re.search(rf"\*\*{tag}:\*\* ([^\n]+)", content)
            return m.group(1) if m else None
        metrics = {
            'batch': batch,
            'tokens': max_tokens,
            'latency': extract('End-to-end latency'),
            'bottleneck': extract('Bottleneck'),
            'confidence': extract('Confidence'),
        }
        results.append(metrics)

# Save artifacts
summary_json = os.path.join(WORKSPACE_ROOT, 'outputs', 'scaling_summary.json')
with open(summary_json, 'w') as jf:
    json.dump(results, jf, indent=2)

summary_csv = os.path.join(WORKSPACE_ROOT, 'outputs', 'scaling_summary.csv')
with open(summary_csv, 'w', newline='') as cf:
    writer = csv.DictWriter(cf, fieldnames=['batch', 'tokens', 'latency', 'bottleneck', 'confidence'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print('\nSummary (printed and saved):')
for r in results:
    print(f"batch={r['batch']}, tokens={r['tokens']}, latency={r['latency']}, bottleneck={r['bottleneck']}, confidence={r['confidence']}")
print(f"\nSaved: {summary_json}\nSaved: {summary_csv}")
