# InferScope

**Make inference bottlenecks visible.**

InferScope is a **single-GPU / single-node** AI inference bottleneck analysis tool.
It does not optimize models or tune kernels. Instead, it answers a fundamental but often ignored question:

> **Where does the end-to-end inference time actually go?**

---

## Why InferScope

In real-world inference systems, it is very common to see:

- GPU utilization stuck at **20–40%**
- QPS or latency far below expectations
- The default reaction: **buy more GPUs**

However, the real causes are usually elsewhere:

- CPU preprocessing (tokenization, decoding, data preparation)
- Host ↔ Device (PCIe) memory copies
- Fragmented GPU kernels with high launch overhead
- GPUs **waiting on CPUs**, not computing

Existing tools are either too low-level (perf, Nsight) or too fragmented (nvidia-smi, framework profilers). **None of them provide a system-level, explainable answer.**

InferScope is built to fill this gap.

---

## What InferScope Does / Does Not Do

### What InferScope Does

- Works on **single GPU / single node** systems
- Collects and aligns the following timelines:
  - CPU computation
  - Memory / NUMA-related overhead
  - Host ↔ Device copies (H2D / D2H)
  - GPU kernel execution
  - Framework / runtime overhead
- Merges all events into **one end-to-end timeline**
- Produces **clear bottleneck diagnosis and actionable suggestions**

### What InferScope Explicitly Does Not Do

- ❌ No benchmark rankings
- ❌ No model accuracy or algorithm optimization
- ❌ No manual kernel tuning
- ❌ No multi-GPU or cluster dependency

---

## Example Output

InferScope focuses on conclusions, not raw counters:

```
End-to-end latency: 87.4 ms

Breakdown:
- CPU preprocessing: 31.2 ms (35.7%)
- H2D copy:           18.4 ms (21.0%)
- GPU compute:        24.1 ms (27.6%)
- Framework overhead: 13.7 ms (15.7%)

Diagnosis:
- GPU is idle for 42% of the timeline
- Inference is CPU-bound
- Tokenization dominates CPU time

Suggestions:
- Increase batch size to amortize H2D copy
- Move tokenization off the critical path
- Enable pinned memory for input buffers
```

---

## Use Cases

InferScope is particularly useful for:

- LLM, embedding, and CV inference
- Streaming or pipeline-based inference systems
- Single-GPU performance tuning
- Systems with persistently low GPU utilization
- Grace CPU / EPYC + GPU architecture analysis

InferScope works even with **one GPU or CPU-only workloads**.

---

## Architecture Overview

InferScope consists of four core components:

```
+-------------------+
|   Workload App    |
+-------------------+
          |
          v
+-------------------+
| Trace Collectors  |
|  - CPU / Python   |
|  - GPU kernels    |
|  - H2D / D2H copy |
+-------------------+
          |
          v
+-------------------+
| Timeline Merger   |
|  - clock sync     |
|  - event ordering |
+-------------------+
          |
          v
+-------------------+
| Analyzer Engine   |
|  - bottlenecks    |
|  - root causes    |
|  - suggestions   |
+-------------------+
          |
          v
+-------------------+
| Report Generator  |
|  - Markdown/HTML  |
+-------------------+
```

---

## Quick Start (Current)

### Current Usage (Instrument Your Code)

InferScope works by instrumenting your inference code. Here's a typical workflow:

**1. Create your inference script and add InferScope instrumentation:**

```python
# my_inference.py
from inferscope import Profiler
from inferscope.api import scope, set_global_profiler
import json

class SimpleBuffer:
    def __init__(self):
        self.events = []
    def enqueue(self, event):
        self.events.append(event.copy())
        return True
    def read_all(self):
        return [e.copy() for e in self.events]
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({"events": self.events}, f)

# Initialize profiler
buf = SimpleBuffer()
profiler = Profiler(buf)
set_global_profiler(profiler)

# Start profiling
profiler.start()

# Wrap your inference code with scope()
with scope("my_inference"):
    output = model.generate(inputs)

# Stop profiling and save
profiler.stop()
buf.save("trace.json")
```

**2. Run your script:**

```bash
python my_inference.py
```

**3. Analyze the generated trace:**

```bash
python scripts/inferscope analyze trace.json --output report.md
python scripts/inferscope analyze trace.json --output report.html --format html
```

The generated reports will show:
- Where your inference time is spent (CPU, GPU, memory transfers)
- Bottleneck identification (CPU-bound vs GPU-bound)
- Actionable optimization suggestions

### Quick Demo

To quickly see InferScope in action, run the provided demo:

```bash
python examples/demo_llm_inference.py --model Qwen/Qwen3-0.6B-Base --stress-mode
python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/report.md
```

### Library Usage

You can also instrument your own code using the InferScope Python API:

```python
from inferscope import scope, Profiler

with scope("my_inference"):
    output = model.generate(inputs)
```

### Planned CLI (Future)

The planned `inferscope` command-line tool will provide a simpler interface:

```bash
# Planned (not yet available):
inferscope run python infer.py --report report.md
```

This will be available in a future release.

---

## Design Principles

InferScope follows a few strict principles:

- **System-first**: analyze inference from a system perspective
- **Explainable**: every conclusion is backed by observable evidence
- **Single-node friendly**: no cluster required
- **Engineer-first**: CLI and text reports over heavy UI

---

## Roadmap (Short)

- [ ] CPU / Python timeline v1
- [ ] CUDA / ROCm GPU timeline v1
- [ ] Unified timeline merger
- [ ] Bottleneck rules v1
- [ ] Markdown / HTML report

---

## Who Is This For

- AI infrastructure engineers
- Inference platform owners
- GPU cloud and enterprise AI teams
- Engineers who "feel" a bottleneck but cannot explain it clearly

---

## Project Status

> 🚧 **Active development (MVP phase)**

InferScope is under active development. APIs and internals may change rapidly.

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- CUDA-capable GPU (optional, but recommended for GPU profiling)
- Linux environment (tested on Ubuntu/Debian)

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd InferScope
```

2. **Set up Python virtual environment with uv:**

```bash
uv venv
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
uv pip install -r requirements.txt
```

4. **Verify installation:**

Run the quick function verification demo to confirm everything is set up correctly:

```bash
python scripts/run_profiler_demo.py
```

You should see CPU/GPU profiling statistics and a unified timeline output.

---

## Outputs Folder

Generated artifacts from running profiling and analysis are written to the [outputs/](outputs/) directory:

- **Trace files**: JSON files containing raw profiling events from collectors (CPU calls, GPU kernels, memory transfers)
- **Analysis reports**: Markdown and HTML formatted reports with bottleneck diagnosis and optimization suggestions
- **Output location**: All generated files are saved to `outputs/` for organization and to prevent cluttering the repo root

Example generated files:
- `outputs/*.json` - Raw trace data from profiling
- `outputs/*_report.md` - Human-readable Markdown reports
- `outputs/*_report.html` - Interactive HTML reports

## Profiler Demo

### Quick Function Verification

Run a quick end-to-end demo that collects CPU and GPU events and prints a unified timeline:

```bash
source .venv/bin/activate
python scripts/run_profiler_demo.py
```

Or use the make target:

```bash
make demo
```

**Expected Output:**

You'll see CPU/GPU statistics followed by a unified timeline showing all captured events with microsecond timestamps:

```
Starting profiler...
(GPU profiling will use CUDA/CUPTI if available; otherwise CPU-only mode)

CPU stats: {'state': 'FINALIZED', 'total_events_captured': 22, ...}
GPU stats: {'state': 'FINALIZED', 'cupti_available': True, 'event_count': 0, ...}

Unified timeline (ts_us type name thread/stream):
1231739834082 cpu_call get tid=246575807156128 stream=None
1231739834141 cpu_call encode tid=246575807156128 stream=None
1231739834155 cpu_return encode tid=246575807156128 stream=None
...

Demo complete. Workload result: 30
```

This verifies that InferScope can successfully collect and merge CPU/GPU events into a synchronized timeline.

### Real LLM Inference Demo

Run InferScope on a real LLM inference workload (using Hugging Face Transformers):

```bash
source .venv/bin/activate
python examples/demo_llm_inference.py --model Qwen/Qwen3-0.6B-Base --max-new-tokens 128 --batch-size 1 --stress-mode
```

**What This Demo Does:**

- Loads a real LLM model (Qwen3 0.6B or fallback to DistilGPT2)
- Profiles complete inference pipeline: tokenization → H2D copy → GPU inference → D2H copy → decoding
- Captures CPU function calls, GPU kernels, and memory transfers
- Saves detailed trace to `outputs/demo_llm_trace.json`

**Expected Output:**

```
======================================================================
InferScope Demo: Real LLM Inference (HF Transformers)
======================================================================
[INFO] TORCH_AVAILABLE=True | TRANSFORMERS_AVAILABLE=True
[INFO] Requested model: Qwen/Qwen3-0.6B-Base
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
[INFO] Trace saved: outputs/demo_llm_trace.json
[INFO] Sample output: Deep learning systems deploy attention mechanisms...

Next:
  - Analyze (MD): python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.md
  - Analyze (HTML): python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.html --format html
```

The trace file contains hundreds of thousands of profiling events. To analyze it and generate human-readable reports:

**Analyze the Trace:**

```bash
# Generate Markdown report
python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.md

# Generate HTML report (interactive)
python scripts/inferscope analyze outputs/demo_llm_trace.json --output outputs/llm_report.html --format html
```

**What You'll Find in the Report:**

- **End-to-end latency breakdown**: Total inference time with percentage breakdown by category
- **Time spent analysis**: CPU preprocessing, GPU compute, H2D/D2H memory transfers
- **Bottleneck identification**: Determines if workload is CPU-bound, GPU-bound, or memory-bound
- **Timeline breakdown table**: Detailed breakdown showing duration and percentage for each category
- **Actionable optimization suggestions**: Prioritized recommendations with estimated improvement percentages

**Example Report Output:**

```
## Summary
End-to-end latency: 18632.5 ms

### Breakdown
- CPU time: 3.8 ms (0.0%)
- GPU time: 18622.9 ms (99.9%)
- H2D copy: 0.5 ms (0.0%)
- D2H copy: 0.3 ms (0.0%)

## Diagnosis
Bottleneck: GPU BOUND
Primary cause: GPU compute
Confidence: 60%

## Suggestions
1. Optimize GPU kernels (high priority) - Est. improvement: 25%
2. Use mixed precision (FP16) (medium priority) - Est. improvement: 30%
```

---

## License

Apache-2.0

---

> InferScope is not about showing how fast your GPU is.
> It is about explaining:
>
> **Why your GPU is not running.**
