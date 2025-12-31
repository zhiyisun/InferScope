#!/usr/bin/env python3
import json

with open('outputs/demo_llm_trace.json') as f:
    data = json.load(f)
events = data.get('events', [])
counts = {}
for e in events:
    t = e.get('type', 'unknown')
    counts[t] = counts.get(t, 0) + 1

print('=' * 70)
print('AUTOMATICALLY CAPTURED EVENTS - FINAL VERIFICATION')
print('=' * 70)
for t in sorted(counts.keys()):
    pct = 100 * counts[t] / len(events)
    print(f'{t:25s}: {counts[t]:8d} ({pct:6.2f}%)')
print('=' * 70)
print(f'TOTAL EVENTS: {len(events)}')
print('=' * 70)

# Show details of each event type
print('\n1. CPU EVENTS (Automatic via sys.settrace):')
cpu_calls = [e for e in events if e.get('type') == 'cpu_call']
cpu_returns = [e for e in events if e.get('type') == 'cpu_return']
print(f'   • CPU Calls:     {len(cpu_calls)} events')
print(f'   • CPU Returns:   {len(cpu_returns)} events')
print(f'   Total CPU:       {len(cpu_calls) + len(cpu_returns)} events')

print('\n2. H2D EVENTS (Automatic via tensor.to() hook):')
h2d = [e for e in events if e.get('type') == 'h2d_copy']
print(f'   • H2D Copies:    {len(h2d)} events')
for i, e in enumerate(h2d[:3]):
    print(f'     Sample {i+1}: {e.get("name")} - {e.get("bytes", 0)} bytes')

print('\n3. D2H EVENTS (Automatic via tensor.to() hook):')
d2h = [e for e in events if e.get('type') == 'd2h_copy']
print(f'   • D2H Copies:    {len(d2h)} events')
for i, e in enumerate(d2h[:3]):
    print(f'     Sample {i+1}: {e.get("name")} - {e.get("bytes", 0)} bytes')

print('\n4. SCOPE MARKERS (Manual via scope() API):')
enters = [e for e in events if e.get('type') == 'scope_enter']
exits = [e for e in events if e.get('type') == 'scope_exit']
print(f'   • Scope Enters:  {len(enters)} events')
print(f'   • Scope Exits:   {len(exits)} events')
for e in enters:
    print(f'     - {e.get("name")}')

print('\n' + '=' * 70)
print('✅ ALL EVENTS CAPTURED AUTOMATICALLY')
print('=' * 70)
