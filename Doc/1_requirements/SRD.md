# System Requirement Document (SRD)

## Hardware Requirements

### Minimum
- **GPU**: 1x NVIDIA GPU (Compute Capability 7.0+, e.g., V100, A100, RTX)
- **CPU**: 4-core x86_64 or ARM64 processor
- **RAM**: 8 GB system memory
- **Storage**: 500 MB for installation

### Recommended
- **GPU**: NVIDIA A100 or H100 (for high-performance inference testing)
- **CPU**: 16+ cores for parallel data preparation
- **RAM**: 32 GB for large batch inference
- **Storage**: 2+ GB for large trace buffers

## Software Requirements

### Operating System
- Linux (x86_64 or ARM64)
- Ubuntu 20.04+ (primary target)
- RHEL 8+ / CentOS 8+

### Runtime Dependencies
- Python 3.8+
- CUDA Toolkit 11.8+
- NVIDIA CUPTI library (bundled with CUDA)
- PyTorch 1.12+ (for application being profiled)

### Build Dependencies
- GCC/Clang C++ compiler
- CMake 3.18+
- Python development headers

## Deployment Environment

### Single-Node Constraint
- One GPU per node
- No distributed memory or process communication
- Local filesystem only (no distributed storage)

### Trace Storage
- In-memory ring buffer (circular, ~100MB default)
- Optional on-disk buffering for long-running traces
- Backward-compatible trace format (JSON + binary)

## Platform-Specific Constraints

### NVIDIA CUDA
- CUPTI event collection required
- NVTX API for framework integration
- Clock synchronization via CUDA driver

### AMD ROCm (Future)
- ROCProfiler API for GPU events
- rocProf CLI integration (deferred to v2)

## Compliance & Constraints

### Security
- No network communication (offline-only)
- No data upload or telemetry
- User workload code remains on-device

### Logging & Observability
- Optional detailed logs to stderr
- Trace format versioning for future compatibility
- No sensitive data in reports

## Known Limitations (MVP)

- Single-GPU only (explicit error on multi-GPU detection)
- No NVIDIA Jetson Orin support (deferred)
- No Windows/macOS support (deferred to v2)
- ROCm support deferred to v2
