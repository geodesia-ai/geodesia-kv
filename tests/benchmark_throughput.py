"""Benchmark autoregressive generation throughput (tokens/sec) across different cache policies.

Isolates KV cache attention decoding latency across layers and computes theoretical
maximum tokens/sec for different models (Qwen 3B, Qwen 8B) at long context (16k).
"""

import time
import torch
import torch.nn.functional as F

from geodesia_kv.cuda_ext import get_ext
from geodesia_kv.pack_layout import build_layout


# Helper per simulare quantizzazione senza copiare tutto da test_cuda_kernel
def _quantize_codes_mock(x, bits, group, per_channel):
    nb, pp, d = x.shape
    if per_channel:
        g = min(group, pp)
        xs = x.view(nb, pp // g, g, d)
        lo, hi = xs.amin(2, keepdim=True), xs.amax(2, keepdim=True)
    else:
        g = min(group, d)
        xs = x.view(nb, pp, d // g, g)
        lo, hi = xs.amin(3, keepdim=True), xs.amax(3, keepdim=True)
    step = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    lo16, step16 = lo.to(torch.float16), step.to(torch.float16)
    code = ((xs - lo16.float()) / step16.float()).round().clamp_(0, 2 ** bits - 1)
    return code.view(nb, pp, d).to(torch.uint8), None, lo16, step16

def run_throughput_benchmark():
    dev = "cuda"
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting benchmark.")
        return

    ext = get_ext()
    if ext is None:
        print("geodesia_kv_ext non disponibile, fallback fallito.")
        return

    # Model parameters for 16k context
    context_len = 16384
    batch_size = 1
    
    models = {
        "Qwen2.5-3B": {"num_layers": 36, "num_q_heads": 14, "num_kv_heads": 2, "head_dim": 128},
        "Qwen3-8B":   {"num_layers": 28, "num_q_heads": 28, "num_kv_heads": 4, "head_dim": 128},
    }

    print("================================================================")
    print("      ATTENTION DECODING THROUGHPUT BENCHMARK (TOKENS/SEC)      ")
    print("================================================================\n")

    for model_name, cfg in models.items():
        L = cfg["num_layers"]
        H_q = cfg["num_q_heads"]
        H_kv = cfg["num_kv_heads"]
        D = cfg["head_dim"]
        
        print(f"--- Model: {model_name} (Layers: {L}, Context: {context_len}) ---")
        
        # Dense K and V for one layer
        k = torch.randn(batch_size, H_kv, context_len, D, device=dev, dtype=torch.float16)
        v = torch.randn(batch_size, H_kv, context_len, D, device=dev, dtype=torch.float16)
        
        # Expand KV to match Q heads (GQA simulation)
        num_queries_per_kv = H_q // H_kv
        k_expanded = k.repeat_interleave(num_queries_per_kv, dim=1)
        v_expanded = v.repeat_interleave(num_queries_per_kv, dim=1)

        # ---------------------------------------------------------
        # 1. Full-KV Baseline (PyTorch SDPA)
        # ---------------------------------------------------------
        q = torch.randn(batch_size, H_q, 1, D, device=dev, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k_expanded, v_expanded)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            for _ in range(L): # Simulate going through all layers for 1 token
                out = F.scaled_dot_product_attention(q, k_expanded, v_expanded)
        torch.cuda.synchronize()
        full_kv_time = (time.perf_counter() - start) / iters
        full_kv_tps = 1.0 / full_kv_time
        
        print(f"{'Full-KV (PyTorch SDPA)':<30} : {full_kv_tps:7.2f} tokens/sec")

        # ---------------------------------------------------------
        # 2. StreamingLLM / SnapKV (Sparse Window - 2k tokens)
        # ---------------------------------------------------------
        # Assume they retain roughly 2k tokens out of 16k context
        retained_len = 2048
        k_sparse = k_expanded[:, :, -retained_len:, :]
        v_sparse = v_expanded[:, :, -retained_len:, :]
        
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k_sparse, v_sparse)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iters):
            for _ in range(L):
                out = F.scaled_dot_product_attention(q, k_sparse, v_sparse)
        torch.cuda.synchronize()
        sparse_time = (time.perf_counter() - start) / iters
        sparse_tps = 1.0 / sparse_time
        
        print(f"{'Eviction (StreamingLLM/SnapKV)':<30} : {sparse_tps:7.2f} tokens/sec")

        # ---------------------------------------------------------
        # 3. Geodesia-KV (Packed CUDA Kernel)
        # ---------------------------------------------------------
        P = 64
        group = 32
        nb = context_len // P
        
        # Prepare mixed precision bits for Geodesia-KV (mimicking 5-bit point)
        # 80% 4-bit, 10% 8-bit, 5% 16-bit, 5% centroid
        levels_dist = [4]*int(nb*0.8) + [8]*int(nb*0.1) + [16]*int(nb*0.05) + [1]*int(nb*0.05)
        # Pad if needed
        while len(levels_dist) < nb: levels_dist.append(4)
        levels_dist = levels_dist[:nb]
        
        # Convert K, V for CUDA kernel format
        k_flat = k_expanded[0, 0].view(nb, P, D).float()  # Simplify by using one head for build_layout logic, then expand
        v_flat = v_expanded[0, 0].view(nb, P, D).float()
        
        items = {}
        for b, lv in enumerate(levels_dist):
            if lv == 16:
                items[b] = {"k": k_flat[b].half(), "v": v_flat[b].half()}
            elif lv == 1:
                items[b] = {"k": k_flat[b].mean(0, keepdim=True).half(), "v": v_flat[b].mean(0, keepdim=True).half()}
            else:
                ck, _, klo, kstep = _quantize_codes_mock(k_flat[b:b+1], lv, group, True)
                cv, _, vlo, vstep = _quantize_codes_mock(v_flat[b:b+1], lv, group, False)
                G, gv = (P + group - 1) // group, (D + group - 1) // group
                items[b] = {"k": ck[0], "v": cv[0], "klo": klo.view(G, D), "kstep": kstep.view(G, D), 
                            "vlo": vlo.view(P, gv), "vstep": vstep.view(P, gv)}

        lay = build_layout(items, P, group, D, nb, torch.tensor(levels_dist, device=dev), torch.full((nb,), P, device=dev), dev)
        
        # Expand lay for all H_q heads
        G = (P + group - 1) // group
        gv = (D + group - 1) // group
        
        q_cuda = torch.randn(batch_size, H_q, D, device=dev, dtype=torch.float32)
        data_cuda = lay["data"].view(1, 1, -1).expand(batch_size, H_q, -1).contiguous()
        offs_cuda = lay["offs"]
        klo_cuda = lay["klo"].view(1, 1, nb, G, D).expand(batch_size, H_q, -1, -1, -1).contiguous()
        kstep_cuda = lay["kstep"].view(1, 1, nb, G, D).expand(batch_size, H_q, -1, -1, -1).contiguous()
        vlo_cuda = lay["vlo"].view(1, 1, nb, P, gv).expand(batch_size, H_q, -1, -1, -1).contiguous()
        vstep_cuda = lay["vstep"].view(1, 1, nb, P, gv).expand(batch_size, H_q, -1, -1, -1).contiguous()
        level_cuda = lay["level"]
        valid_cuda = lay["valid"]
        
        # Warmup
        for _ in range(10):
            _ = ext.mixed_attn_decode(q_cuda, data_cuda, offs_cuda, klo_cuda, kstep_cuda, vlo_cuda, vstep_cuda, level_cuda, valid_cuda, P, group)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iters):
            for _ in range(L):
                out = ext.mixed_attn_decode(q_cuda, data_cuda, offs_cuda, klo_cuda, kstep_cuda, vlo_cuda, vstep_cuda, level_cuda, valid_cuda, P, group)
        torch.cuda.synchronize()
        geo_time = (time.perf_counter() - start) / iters
        geo_tps = 1.0 / geo_time
        
        print(f"{'Geodesia-KV (CUDA Packed Kernel)':<30} : {geo_tps:7.2f} tokens/sec")
        
        # Speedup vs Full
        speedup = geo_tps / full_kv_tps
        print(f"  -> Speedup vs Full-KV: {speedup:.2f}x\n")

if __name__ == "__main__":
    run_throughput_benchmark()
