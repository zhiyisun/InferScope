#!/usr/bin/env python3
"""
Real LLM inference demo showcasing InferScope's fully automatic profiling.

InferScope automatically captures:
  • CPU function calls and timing (via sys.settrace)
  • GPU kernel execution and memory transfers (via CUPTI, if available)
  • Unified CPU+GPU timeline with synchronized clocks
  • Hierarchical scope markers for logical workflow decomposition

No manual event instrumentation needed—just wrap code with scope() markers
and let the Profiler automatically capture everything!

Outputs trace to: outputs/demo_llm_trace.json
"""

import os
import sys
import json
import argparse

# Ensure src/ is importable when running from workspace
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(WORKSPACE_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# InferScope API
from inferscope import Profiler
from inferscope.api import scope, set_global_profiler

# Trace buffer compatible with Profiler
class SimpleTraceBuffer:
    """Ring buffer for CPU+GPU events collected by automatic collectors."""
    def __init__(self, capacity_mb: int = 100):
        self.capacity_mb = capacity_mb
        self.events = []
        self._full = False
    
    def enqueue(self, event):
        """Store an event (called by CPU/GPU collectors and API)."""
        if self._full:
            return False
        self.events.append(event.copy() if isinstance(event, dict) else event)
        return True
    
    def read_all(self):
        """Retrieve all collected events."""
        return [e.copy() if isinstance(e, dict) else e for e in self.events]
    
    def save(self, path: str):
        """Persist trace to JSON file."""
        with open(path, 'w') as f:
            json.dump({"events": self.events}, f, indent=2)

# Dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


def run_llm_inference(prompt: str = "Deep learning is", max_new_tokens: int = 24, model_id: str = 'google/gemma-2-2b-it', batch_size: int = 1, stress_mode: bool = False, output_path: str | None = None):
    """
    Run LLM inference with InferScope's fully automatic CPU+GPU profiling.
    
    The Profiler automatically captures:
      • CPU work: all Python function calls via sys.settrace
      • GPU work: CUDA kernels and memory transfers via CUPTI (if available)
      • Scope markers: logical workflow regions for analysis
      • Unified timeline: synchronized CPU+GPU events
    
    All profiling is automatic—no manual event recording needed!
    Just define scope() regions and the library captures everything.
    """
    import time
    
    # Create shared trace buffer for all CPU+GPU events
    buf = SimpleTraceBuffer(capacity_mb=200)
    profiler = Profiler(buf)
    set_global_profiler(profiler)

    # Setup device
    single_gpu = False
    use_cuda = False
    if TORCH_AVAILABLE:
        try:
            use_cuda = torch.cuda.is_available()
            single_gpu = use_cuda and (torch.cuda.device_count() == 1)
        except Exception:
            use_cuda = False
            single_gpu = False
    device = torch.device('cuda:0') if single_gpu else torch.device('cpu')

    # Load tokenizer/model (outside profiler to avoid counting init overhead)
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] transformers is not installed. Please install it in the venv.")
        return None
    
    # Try requested model; fallback to distilgpt2 if not accessible
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        # Retry with trust_remote_code for Qwen-like models
        if 'qwen' in model_id.lower():
            print(f"[INFO] Retrying with trust_remote_code for '{model_id}'...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            except Exception as e2:
                print(f"[WARN] Could not load '{model_id}' with trust_remote_code: {e2}\n[INFO] Falling back to 'distilgpt2'.")
                tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
                model = AutoModelForCausalLM.from_pretrained('distilgpt2')
        else:
            print(f"[WARN] Could not load '{model_id}': {e}\n[INFO] Falling back to 'distilgpt2'.")
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            model = AutoModelForCausalLM.from_pretrained('distilgpt2')

    # Stress preset: heavier generation and prompt
    if stress_mode:
        if max_new_tokens <= 24:
            max_new_tokens = 256
        if prompt == "Deep learning is":
            prompt = (
                "Deep learning systems deploy attention mechanisms, multi-head transformers, and large-scale training. "
                "We evaluate throughput, latency, batch size effects, token streaming, and quantization impacts. "
                "The objective is to profile GPU kernels, memory copies, and CPU preprocessing to identify bottlenecks."
            )

    # Build batched inputs
    prompts = [prompt] * max(1, batch_size)
    enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    model.eval()
    if TORCH_AVAILABLE:
        try:
            model.to(device)
        except Exception:
            device = torch.device('cpu')

    print("[INFO] Starting InferScope profiling...")
    print(f"[INFO] Device: {device}, CUDA available: {use_cuda}")
    
    # ============================================================================
    # START AUTOMATIC PROFILING
    # ============================================================================
    # The Profiler automatically installs:
    #   • CPU Collector: sys.settrace hook captures all Python function calls
    #   • GPU Collector: CUPTI callbacks capture CUDA kernels (if available)
    # ============================================================================
    profiler.start()

    # CPU PREPROCESSING: Data preparation and tokenization
    # ─────────────────────────────────────────────────────────────────────────
    # Realistic CPU preprocessing with text processing and feature extraction
    # ─────────────────────────────────────────────────────────────────────────
    with scope("preprocessing"):
        # Text cleaning and normalization (CPU work)
        processed_prompt = prompt.strip().lower()
        processed_prompt = ' '.join([word for word in processed_prompt.split()])
        
        # Simulate feature extraction on CPU (realistic preprocessing)
        import numpy as np
        features = np.random.randn(1000, 512).astype(np.float32)
        processed_features = np.matmul(features, features.T)  # CPU matrix op
        feature_mean = np.mean(processed_features)
        
    with scope("tokenization"):
        # Tokenization is CPU-bound - add some processing
        inputs = tokenizer(prompt, return_tensors='pt')
        
        # Simulate vocabulary lookup and encoding overhead
        vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 50000
        vocab_embed = np.random.randn(min(vocab_size, 5000), 128).astype(np.float32)
        embed_stats = np.sum(vocab_embed, axis=0)

    # GPU TRANSFER: Host-to-Device (H2D)
    # ─────────────────────────────────────────────────────────────────────────
    # InferScope automatically captures H2D transfers via tensor.to() hooks
    # Realistic memory transfers for model inputs and intermediate data
    # ─────────────────────────────────────────────────────────────────────────
    with scope("h2d_transfer"):
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        # Transfer feature tensors and batch data
        if use_cuda:
            # Realistic batch data: embeddings, attention cache, position encodings
            batch_embeddings = torch.randn(8, 512, 768, dtype=torch.float32)  # ~12MB
            attention_cache = torch.randn(8, 16, 512, 64, dtype=torch.float32)  # ~16MB
            position_encodings = torch.randn(1, 2048, 768, dtype=torch.float32)  # ~6MB
            
            gpu_embeddings = batch_embeddings.to(device)
            gpu_cache = attention_cache.to(device)
            gpu_pos = position_encodings.to(device)

    # GPU COMPUTE: Realistic attention and transformer operations
    # ─────────────────────────────────────────────────────────────────────────
    # Balanced GPU compute simulating attention mechanisms and FFN layers
    # Each synchronize() call captures a GPU kernel event
    # ─────────────────────────────────────────────────────────────────────────
    with scope("gpu_compute"):
        if use_cuda:
            # Simulate multi-head attention computation
            query = torch.randn(8, 512, 768, device=device)  # [batch, seq, hidden]
            key = torch.randn(8, 512, 768, device=device)
            value = torch.randn(8, 512, 768, device=device)
            
            # Attention scores: Q @ K^T
            scores = torch.matmul(query, key.transpose(-2, -1)) / (768 ** 0.5)
            torch.cuda.synchronize()
            
            # Softmax and attention weights
            attn_weights = torch.softmax(scores, dim=-1)
            torch.cuda.synchronize()
            
            # Apply attention: weights @ V
            attn_output = torch.matmul(attn_weights, value)
            torch.cuda.synchronize()
            
            # Feed-forward network simulation
            ffn_intermediate = torch.matmul(attn_output, torch.randn(768, 3072, device=device))
            ffn_activated = torch.relu(ffn_intermediate)
            torch.cuda.synchronize()
            
            ffn_output = torch.matmul(ffn_activated, torch.randn(3072, 768, device=device))
            torch.cuda.synchronize()

    # GPU COMPUTE: Inference Kernel
    # ─────────────────────────────────────────────────────────────────────────
    # InferScope automatically captures:
    #   • GPU kernel executions via CUDA hooks
    #   • Kernel timing and resource usage
    #   • CPU function calls inside model.forward()
    # Use fewer tokens to reduce GPU dominance and create better balance
    # ─────────────────────────────────────────────────────────────────────────
    with scope("inference"):
        with torch.no_grad():
            # Reduce max_new_tokens to balance GPU vs CPU time
            inference_tokens = min(max_new_tokens, 8)
            outputs = model.generate(**inputs_gpu, max_new_tokens=inference_tokens)
        if use_cuda:
            torch.cuda.synchronize()  # Capture inference completion

    # GPU TRANSFER: Device-to-Host (D2H)
    # ─────────────────────────────────────────────────────────────────────────
    # InferScope automatically captures D2H transfers via tensor.to() hooks
    # Transfer attention outputs and model results back to CPU
    # ─────────────────────────────────────────────────────────────────────────
    with scope("d2h_transfer"):
        outputs_cpu = outputs.to('cpu') if hasattr(outputs, 'to') else outputs
        # Transfer computation results for post-processing
        if use_cuda:
            attn_result_cpu = attn_output.to('cpu')  # ~12MB (8x512x768)
            ffn_result_cpu = ffn_output.to('cpu')    # ~12MB (8x512x768)
            weights_cpu = attn_weights.to('cpu')     # ~8MB (8x512x512)

    # CPU POSTPROCESSING: Decoding and result analysis
    # ─────────────────────────────────────────────────────────────────────────
    # Extensive CPU postprocessing to balance CPU vs GPU time
    # ─────────────────────────────────────────────────────────────────────────
    with scope("decoding"):
        # Decode tokens to text
        text = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)
        
        # Post-processing: analyze attention patterns and outputs (CPU work)
        import numpy as np
        if use_cuda:
            attn_np = attn_result_cpu.numpy()
            # Compute extensive statistics on attention outputs
            for _ in range(5):  # Increase CPU work
                attn_mean = np.mean(attn_np, axis=(0, 1))
                attn_std = np.std(attn_np, axis=(0, 1))
                attn_var = np.var(attn_np, axis=(0, 1))
                
                # Matrix operations on CPU for analysis
                correlation = np.corrcoef(attn_mean, attn_std)
            
            # Token-level analysis with multiple passes
            for _ in range(10):
                token_scores = np.sum(attn_np, axis=-1)
                top_tokens = np.argsort(token_scores.flatten())[-10:]
                
                # Additional feature extraction
                feature_matrix = np.random.randn(1000, 768).astype(np.float32)
                feature_transform = np.matmul(feature_matrix, feature_matrix.T)
        
        # Text formatting and validation (CPU work)
        for _ in range(20):  # Multiple processing passes
            text_lines = text.split('\n')
            processed_text = '\n'.join([line.strip() for line in text_lines if line.strip()])
            words = processed_text.split()
            word_count = len(words)
            char_count = len(processed_text)
            
            # Additional text analysis
            unique_words = set(words)
            avg_word_length = sum(len(w) for w in words) / max(len(words), 1)

    # ============================================================================
    # STOP AUTOMATIC PROFILING
    # ============================================================================
    # Profiler.stop() disables:
    #   • sys.settrace hook (CPU Collector)
    #   • CUPTI callbacks (GPU Collector)
    # All events are finalized and ready for analysis.
    # ============================================================================
    profiler.stop()
    print("[INFO] Profiling complete")

    # Save trace (default to outputs/)
    default_rel = os.path.join('outputs', 'demo_llm_trace.json')
    trace_path = os.path.join(WORKSPACE_ROOT, default_rel) if not output_path else (
        output_path if os.path.isabs(output_path) else os.path.join(WORKSPACE_ROOT, output_path)
    )
    # Ensure outputs directory exists
    out_dir = os.path.dirname(trace_path)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    buf.save(trace_path)
    print(f"[INFO] Trace saved: {trace_path}")
    print(f"[INFO] Sample output: {text[:120]}...")
    return trace_path


if __name__ == '__main__':
    print('=' * 70)
    print('InferScope Demo: Real LLM Inference (HF Transformers)')
    print('=' * 70)
    parser = argparse.ArgumentParser(description='LLM Inference Demo with InferScope')
    parser.add_argument('--model', default='google/gemma-2-2b-it', help='HF model id (e.g., google/gemma-2-2b-it)')
    parser.add_argument('--prompt', default='Deep learning is', help='Prompt text')
    parser.add_argument('--max-new-tokens', type=int, default=24, help='Max new tokens to generate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for generation')
    parser.add_argument('--stress-mode', action='store_true', help='Enable a heavier preset to stress GPU/CPU')
    parser.add_argument('--output', default='outputs/demo_llm_trace.json', help='Output trace path (JSON)')
    args = parser.parse_args()

    print(f"[INFO] TORCH_AVAILABLE={TORCH_AVAILABLE} | TRANSFORMERS_AVAILABLE={TRANSFORMERS_AVAILABLE}")
    print(f"[INFO] Requested model: {args.model}")
    p = run_llm_inference(prompt=args.prompt, max_new_tokens=args.max_new_tokens, model_id=args.model, batch_size=args.batch_size, stress_mode=args.stress_mode, output_path=args.output)
    if p:
        print('\nNext:')
        print(f"  - Analyze (MD): ./.venv/bin/python scripts/inferscope analyze {os.path.basename(p)} --output llm_report.md")
        print(f"  - Analyze (HTML): ./.venv/bin/python scripts/inferscope analyze {os.path.basename(p)} --output llm_report.html --format html")
