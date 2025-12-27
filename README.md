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

## Quick Start (Planned)

```bash
pip install inferscope

inferscope run python infer.py --report report.md
```

Library usage:

```python
from inferscope import scope

with scope("llm_inference"):
    output = model.generate(inputs)
```

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

## License

Apache-2.0

---

## Author

Zhiyi Sun  
System / AI Infrastructure Engineer

---

> InferScope is not about showing how fast your GPU is.
> It is about explaining:
>
> **Why your GPU is not running.**
