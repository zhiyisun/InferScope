# Runbooks

## Installation & Setup

### Prerequisites Check
```bash
# Check Python version
python --version  # Must be 3.8+

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"  # Must be 11.8+
```

### Installation
```bash
# From PyPI (recommended)
pip install inferscope

# From source (development)
git clone https://github.com/openai/inferscope.git
cd InferScope
pip install -e ".[dev]"

# Verify installation
python -c "from inferscope import scope; print('InferScope installed successfully')"
```

---

## Basic Usage

### Quick Start: CLI Mode
```bash
# Profile an inference script
inferscope run python my_inference.py --model gpt2

# Report written to: report.md (in current directory)
```

### Quick Start: Library API
```python
from inferscope import scope

with scope("inference"):
    output = model.forward(inputs)

# Traces automatically collected; access report
# (see inference_report.md generated in cwd)
```

---

## Troubleshooting

### Problem: "CUDA not found"
**Symptom:** Error during collection

```
ERROR: CUDA is not available. Falling back to CPU-only mode.
```

**Solution:**
1. Check NVIDIA GPU present: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. If no GPU, CPU-only profiling still works (limited insights)

### Problem: "CUPTI initialization failed"
**Symptom:** GPU events not captured

```
WARN: CUPTI library load failed. GPU profiling disabled.
```

**Solution:**
1. Check CUDA version: `python -c "import torch; print(torch.version.cuda)"`
2. CUDA 11.8+ required
3. Verify CUPTI installed: `find $CUDA_HOME -name libcupti*`
4. If missing, reinstall CUDA toolkit

### Problem: "Multi-GPU setup detected"
**Symptom:** Error on startup

```
ERROR: InferScope supports single-GPU only. Detected 2 GPUs.
```

**Solution:**
```bash
# Option 1: Use only first GPU
export CUDA_VISIBLE_DEVICES=0
inferscope run python my_inference.py

# Option 2: Modify script to use single GPU
# torch.cuda.set_device(0)
```

### Problem: "Profiling overhead exceeds 10%"
**Symptom:** Report shows profiling added significant latency

**Solution:**
1. Increase batch size (amortizes overhead)
2. Profile longer inference (fixed overhead becomes smaller %)
3. Reduce CPU sampling frequency: `export INFERSCOPE_CPU_SAMPLING_HZ=100` (from 1000)
4. Check for contention: `export INFERSCOPE_LOG_LEVEL=debug` for details

### Problem: "Clock synchronization error >5%"
**Symptom:** Report warns about timestamp uncertainty

**Solution:**
1. Reduce background system load (kill other processes)
2. Reset GPU clock: `nvidia-smi -pm 1` (requires root)
3. Increase calibration samples (automatic; report still valid)

### Problem: "Report not generated / Disk full"
**Symptom:** No report.md created

**Solution:**
1. Check disk space: `df -h`
2. Specify alternate path: `export INFERSCOPE_REPORT_PATH=/tmp/report.md`
3. Check file permissions: `ls -l` on output directory

---

## Performance Tuning

### Reduce Profiling Overhead

**1. Increase batch size**
```python
# Bad: Single sample
with scope("inference"):
    output = model(single_input)  # High overhead %

# Good: Batch of 32
with scope("inference"):
    output = model(batch_of_32)  # Lower overhead %
```

**2. Reduce CPU sampling frequency**
```bash
export INFERSCOPE_CPU_SAMPLING_HZ=100  # Default: 1000
```

**3. Profile longer inference**
```python
# Bad: 10ms inference (2% overhead = 0.2ms overhead, significant)
# Good: 1000ms inference (2% overhead = 20ms overhead, acceptable)
```

### Improve Clock Synchronization Accuracy

**1. Run with reduced system load**
```bash
# Kill background processes
sudo killall -9 chrome firefox python  # Adjust as needed

# Run with higher priority
nice -n -10 inferscope run python my_inference.py
```

**2. Stabilize GPU clock**
```bash
# Set GPU to max clock (requires root; not persistent)
sudo nvidia-smi -lgc 1410,1410  # Set GPU and memory clocks
```

---

## Common Configurations

### Development / Debugging
```bash
export INFERSCOPE_ENABLED=1
export INFERSCOPE_LOG_LEVEL=debug
export INFERSCOPE_TRACE_SIZE_MB=200
```

### Production / Deployment
```bash
export INFERSCOPE_ENABLED=1
export INFERSCOPE_LOG_LEVEL=warn
export INFERSCOPE_TRACE_SIZE_MB=100
export INFERSCOPE_REPORT_FORMAT=json  # Machine-readable
```

### Long-Running Inference
```bash
export INFERSCOPE_TRACE_SIZE_MB=500
export INFERSCOPE_SYNC_ERROR_MARGIN_US=2000  # More conservative
```

---

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Profile Inference

on: [push]

jobs:
  profile:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - uses: actions/checkout@v2
      - name: Install InferScope
        run: pip install inferscope
      - name: Profile inference
        run: inferscope run python test_inference.py --report ci_report.md
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: inference-report
          path: ci_report.md
```

---

## Monitoring & Alerts

### Key Metrics to Track
- Profiling overhead: Should stay <5%
- Clock sync error: Should stay <1%
- Report generation time: Should stay <1s
- GPU idle time: Interpret as optimization potential

### Alert Conditions
| Metric | Threshold | Action |
|--------|-----------|--------|
| Profiling overhead | >10% | Investigate; consider sampling adjustments |
| Clock sync error | >5% | Warn; mark with uncertainty flag |
| Report gen time | >5s | Check disk I/O; consider disk spilling |
| Memory usage | >500MB | Increase RAM or reduce trace size |

---

## Data Retention & Cleanup

### Automatic Cleanup
- In-memory traces: Discarded when process exits
- Reports: User's responsibility (default: `./report.md`)

### Manual Cleanup
```bash
# Remove old reports
rm -f report_*.md

# Clear temporary traces
rm -f /tmp/inferscope_*.json
```

### Archive for Analysis
```bash
# Keep important reports for debugging
mkdir -p reports/archive
mv report.md reports/archive/report_$(date +%Y%m%d_%H%M%S).md
```

---

## Getting Help

1. **Check logs:** `export INFERSCOPE_LOG_LEVEL=debug; inferscope run ...`
2. **See example:** `python examples/simple_inference.py`
3. **Read docs:** See `/docs/` folder in repo
4. **File issue:** GitHub Issues with:
   - OS and GPU model
   - CUDA version
   - Python version
   - Full log output (debug level)
   - Minimal reproduction script
