# System Test Specification

## End-to-End Test Scenarios

### Scenario 1: LLM Token Generation
**Workload:** Small language model generating tokens

```python
# test_llm_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inferscope import scope

model_name = "gpt2"  # Small model for quick testing
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

with scope("llm_inference"):
    inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=50)
    text = tokenizer.decode(outputs[0])
```

**Validation Criteria:**
- Report generated without errors
- End-to-end latency reported (e.g., 150ms)
- CPU preprocessing visible (tokenization, token embedding lookup)
- GPU compute time > 50% total (model forward pass)
- H2D copy visible (input tokens to GPU)
- D2H copy visible (output tokens to CPU)

---

### Scenario 2: CNN Image Classification
**Workload:** ResNet-50 inference on image batch

```python
# test_cnn_inference.py
import torch
import torchvision.models as models
from inferscope import scope

model = models.resnet50(pretrained=True).cuda().eval()
input_batch = torch.randn(4, 3, 224, 224).cuda()  # Batch of 4 images

with scope("cnn_inference"):
    with torch.no_grad():
        output = model(input_batch)
```

**Validation Criteria:**
- Report generated without errors
- GPU compute dominates (>70%)
- Low H2D/D2H overhead (batch already on GPU)
- Framework overhead minimal (<10%)
- Latency scales with batch size

---

### Scenario 3: Embedding Model Inference
**Workload:** Sentence embeddings generation

```python
# test_embedding_inference.py
from sentence_transformers import SentenceTransformer
from inferscope import scope

model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
sentences = ["This is a test sentence"] * 32

with scope("embedding_inference"):
    embeddings = model.encode(sentences, convert_to_tensor=True)
```

**Validation Criteria:**
- Report generated without errors
- Memory transfer overhead visible (multiple batch pieces)
- GPU compute time captured
- CPU preprocessing (tokenization) significant
- Total latency reasonable (<500ms for 32 sentences)

---

## Performance Benchmarks

### Latency Target
| Workload | Target End-to-End | CPU Time | GPU Time | Memory Time |
|----------|------------------|----------|----------|-------------|
| LLM (gpt2, batch=1) | 100-200ms | 10-20% | 50-70% | 20-30% |
| CNN (ResNet50, batch=4) | 50-100ms | 5-10% | 80-90% | 5-10% |
| Embedding (32 sentences) | 200-400ms | 15-25% | 60-70% | 10-15% |

### Profiling Overhead Target
| Metric | Target | Measurement |
|--------|--------|-------------|
| CPU overhead | <5% | (inference_with_profiling - inference_baseline) / inference_baseline |
| GPU overhead | <2% | Same as above for GPU kernels |
| Memory overhead | <200MB | Peak memory during 100ms trace |
| Report generation | <1s | Time from trace complete to report on disk |

---

## Acceptance Criteria

### Report Quality (Must Have)
- ✓ Report generated in Markdown format
- ✓ End-to-end latency clearly stated
- ✓ Timeline breakdown shows all major components
- ✓ Bottleneck classification present (CPU-bound / GPU-bound / balanced)
- ✓ At least 2 actionable suggestions provided

### Timeline Accuracy (Must Have)
- ✓ CPU time + GPU time + Memory time ≈ Total latency ± 10%
- ✓ Clock synchronization error <1% (reported in metadata)
- ✓ No duplicate events
- ✓ Events respect parent-child scope hierarchy

### Robustness (Must Have)
- ✓ Handles GPU out-of-memory gracefully (not caused by profiler)
- ✓ Works with CUDA OOMKiller (trace still valid)
- ✓ Handles interrupted inference (Ctrl+C)
- ✓ Multi-threaded inference profiled correctly

### Documentation (Should Have)
- ✓ Example reports included in repo
- ✓ Known limitations documented
- ✓ Troubleshooting guide available
- ✓ API examples in README

---

## Test Execution Plan

### Phase 1: Pre-MVP System Tests
1. LLM (gpt2) on single GPU
2. CNN (ResNet50) on single GPU
3. Embedding model on single GPU
4. All in isolated test environment

### Phase 2: Extended System Tests (Post-MVP)
5. Multi-model pipeline (sequential)
6. Long-running inference (soak test)
7. High-concurrency inference (many threads, single GPU)
8. Various hardware: V100, A100, H100, RTX

### Phase 3: Regression Tests
- Continuous: Run LLM + CNN tests on each commit
- Weekly: Extended suite
- Monthly: Hardware compatibility matrix

---

## Known Issues & Workarounds

| Issue | Symptom | Workaround |
|-------|---------|-----------|
| GIL contention | High CPU overhead for multi-threaded | Use process pool instead of threads |
| CUPTI buffer overflow | Missing GPU events | Increase INFERSCOPE_TRACE_SIZE_MB |
| Clock sync failure | Large error reported | Reset GPU clock; retry |

---

## Success Metrics

After MVP completion, track:
- Report generation success rate: >99%
- Profiling overhead <5%: 90% of workloads
- User satisfaction (if surveyed): >4/5
- Integration time: <5 minutes per user project
