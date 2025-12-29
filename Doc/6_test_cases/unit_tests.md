# Test Case Specifications

## Unit Tests

### CPU Collector Tests

| Test | Input | Expected Output | Acceptance Criteria |
|------|-------|-----------------|-------------------|
| test_cpu_collector_starts | CpuCollector instance | Hooks installed | sys.settrace is active |
| test_cpu_collector_captures_calls | Synthetic call stack | Event stream with enter/exit | Event count matches call depth |
| test_cpu_collector_thread_isolation | Two threads calling functions | Per-thread buffers | Events don't cross thread boundaries |
| test_cpu_collector_memory_tracking | malloc/free calls | memory_alloc/free events | Bytes tracked accurately |
| test_cpu_collector_stops | Active collector | Hooks removed | sys.settrace is reset |

### GPU Collector Tests

| Test | Input | Expected Output | Acceptance Criteria |
|------|-------|-----------------|-------------------|
| test_gpu_collector_initializes | CUDA available | CUPTI callbacks loaded | cuEventCreate succeeds |
| test_gpu_collector_captures_kernels | CUDA kernel launch | gpu_kernel event | Kernel name, duration recorded |
| test_gpu_collector_captures_h2d | H2D cudaMemcpy | h2d_copy event | Bytes, throughput recorded |
| test_gpu_collector_captures_d2h | D2H cudaMemcpy | d2h_copy event | Bytes, throughput recorded |
| test_gpu_collector_cuda_unavailable | CUDA not available | Graceful skip | Warning logged; CPU-only mode |

### Timeline Merger Tests

| Test | Input | Expected Output | Acceptance Criteria |
|------|-------|-----------------|-------------------|
| test_merger_syncs_clocks | CPU & GPU events with offset | Unified timeline | Clock error <1% |
| test_merger_sorts_events | Unsorted events | Sorted by timestamp | All events ordered |
| test_merger_preserves_causality | Parent/child scopes | Scope hierarchy maintained | Child events within parent range |
| test_merger_handles_concurrent_kernels | Multiple overlapping kernels | All kernels present | No events dropped |
| test_merger_sync_fails_gracefully | Bad clock sync data | Conservative margin applied | Uncertainty annotated in report |

### Analyzer Tests

| Test | Input | Expected Output | Acceptance Criteria |
|------|-------|-----------------|-------------------|
| test_analyzer_detects_gpu_idle | Timeline with 20% idle | Bottleneck: CPU-bound | idle_time / total > 0.1 |
| test_analyzer_detects_cpu_bound | High CPU time, low GPU | Bottleneck: CPU-bound | Confidence > 0.8 |
| test_analyzer_detects_memory_bound | Large H2D+D2H overhead | Bottleneck: memory-bound | H2D+D2H > 20% total |
| test_analyzer_provides_suggestions | CPU-bound bottleneck | "Increase batch size" suggestion | Suggestion is actionable |

### Report Generator Tests

| Test | Input | Expected Output | Acceptance Criteria |
|------|-------|-----------------|-------------------|
| test_report_gen_markdown | Analysis results | Markdown string | Valid Markdown syntax |
| test_report_gen_html | Analysis results | HTML string | Valid HTML, renders correctly |
| test_report_gen_json | Analysis results | JSON dict | Valid JSON schema |
| test_report_gen_includes_breakdown | Summary data | Percentage breakdown | Sum ≈ 100% ± 5% |
| test_report_gen_includes_suggestions | Bottleneck analysis | Actionable suggestions | Suggestions are concrete |

---

## Integration Tests

| Test | Scenario | Expected Behavior | Acceptance Criteria |
|------|----------|-------------------|-------------------|
| test_collect_to_merge | Simple PyTorch forward pass | Unified timeline generated | Events properly ordered |
| test_collect_merge_analyze | PyTorch forward + backward | Full analysis pipeline works | Report generated without errors |
| test_cli_run_script | `inferscope run simple_infer.py` | Report generated to disk | Report file exists; is valid |
| test_api_scope_context | `with scope("test"):` block | Scope events captured | scope_enter and scope_exit present |
| test_api_mark_event | `mark_event("test")` | Instant event captured | event type is "instant" |

---

## System Tests

| Test | Workload | Validation | Acceptance Criteria |
|------|----------|-----------|------------------|
| test_llm_inference | PyTorch LLM (small) | Report generated; sensible breakdown | CPU/GPU times match expectations |
| test_cnn_inference | PyTorch ResNet-50 | Report generated; GPU compute > 50% | Bottleneck classification reasonable |
| test_embedding_inference | PyTorch embedding model | Report generated; memory transfer visible | H2D, D2H events captured |
| test_batch_size_scaling | Same model, batch 1 vs 32 | H2D overhead scales with batch | Larger batch shows better GPU util |
| test_pinned_memory_effect | With/without pinned memory | H2D throughput improvement visible | Pinned > pageable throughput |

---

## Performance Tests

| Test | Measurement | Target | Acceptance Criteria |
|------|-------------|--------|-------------------|
| test_profiling_overhead | Inference latency with/without profiling | <5% slowdown | overhead_us / inference_us < 0.05 |
| test_clock_sync_error | Sync error vs ground truth | <1% | error_us / trace_duration_us < 0.01 |
| test_trace_buffer_memory | Memory used by 100ms trace | <200 MB | buffer_size_mb < 200 |
| test_report_generation_time | Time to generate report | <1 second | report_time_ms < 1000 |

---

## Edge Cases & Fault Injection

| Case | Input | Expected Behavior | Acceptance Criteria |
|------|-------|-------------------|-------------------|
| test_empty_workload | No inference, just scope | Report generated | Report shows near-zero latency |
| test_cuda_initialization_fails | CUDA unavailable | CPU-only mode | Warning logged; analysis continues |
| test_trace_buffer_overflow | Very long running inference | Events wrap | Oldest events discarded; warning logged |
| test_multi_gpu_rejected | Multi-GPU setup | Error raised | clear error message about single-GPU only |
| test_invalid_config | Bad .inferscope.yaml | Graceful fallback | Defaults used; warning logged |
| test_corrupted_trace_file | Malformed JSON trace | Error handling | Clear error message; report not generated |

---

## Test Data

- Example traces: `/Doc/6_test_cases/traces/`
- Synthetic workloads: `/tests/fixtures/`
- Reference reports: `/tests/expected_outputs/`
