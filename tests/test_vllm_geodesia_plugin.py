"""Integration test for Geodesia-KV vLLM plugin with local model.

Verifies end-to-end initialization, rate-distortion KV cache compression,
and inference generation with vLLM and local model weights.
"""

from __future__ import annotations

import os
import sys

# Configure environment before importing vLLM/Torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Ensure local geodesia_kv package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from geodesia_kv.vllm_plugin.kv_cache import GeodesiaKVCacheManager, GeodesiaBlockCache
from geodesia_kv.vllm_plugin.vllm_connector import register_geodesia_kv_plugin

MODEL_PATH = "/home/vincenzo/LLM-Compression/LLM-LossLess-Compression-RNS/Qwen2.5-0.5B-Instruct"


def test_geodesia_vllm_kv_cache_compression():
    """Tests Geodesia-KV Rate-Distortion compression manager on simulated model KV caches."""
    num_layers = 24
    seq_len = 1024
    num_heads = 14
    head_dim = 64
    device = "cpu"

    print(f"\n--- Testing Geodesia-KV Cache Manager ({seq_len} tokens, {num_layers} layers) ---")

    # Generate synthetic key/value cache tensors for 1024 tokens
    torch.manual_seed(42)
    key_cache = torch.randn(num_layers, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn(num_layers, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # Compute uncompressed FP16/BF16 memory (16 bits per value)
    uncompressed_bytes = num_layers * seq_len * num_heads * head_dim * 2 * 2  # K and V in bf16 (2 bytes)
    print(f"Uncompressed KV Cache Memory: {uncompressed_bytes / (1024 * 1024):.2f} MiB (16.0 bpv)")

    # Test compression at target_bpv = 4.0
    manager = GeodesiaKVCacheManager(block_size=64, target_bpv=4.0, num_layers=num_layers)
    seq_cache = manager.create_sequence_cache(seq_id=1)
    achieved_bpv = seq_cache.compress_sequence_kv(key_cache, value_cache)

    total_compressed_bytes = sum(layer.nbytes() for layer in seq_cache.layers)
    print(f"Compressed Geodesia-KV Memory: {total_compressed_bytes / (1024 * 1024):.2f} MiB ({achieved_bpv:.2f} bpv)")

    memory_reduction = (1.0 - (total_compressed_bytes / uncompressed_bytes)) * 100.0
    print(f"Memory Reduction: {memory_reduction:.1f}%")

    assert achieved_bpv <= 5.5, f"Achieved bpv {achieved_bpv} exceeded expected target bound"
    assert memory_reduction > 60.0, "Expected >60% memory reduction with Geodesia-KV rate"
    print("✓ Geodesia-KV Cache Manager test PASSED!")


def test_vllm_geodesia_plugin_integration():
    """Tests vLLM engine import and plugin registration using local Qwen2.5-0.5B-Instruct."""
    print("\n--- Testing vLLM Engine & Plugin Registration with Local Model ---")

    import vllm
    from vllm import LLM, SamplingParams

    print(f"Loaded vLLM version: {vllm.__version__}")
    assert os.path.exists(MODEL_PATH), f"Local model path does not exist: {MODEL_PATH}"

    # Initialize vLLM with local model on GPU
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )

    # Register Geodesia-KV plugin
    manager = register_geodesia_kv_plugin(llm, target_bpv=4.0)
    assert hasattr(llm, "geodesia_manager"), "Plugin failed to attach geodesia_manager to vLLM LLM instance"
    print("Registered Geodesia-KV Manager with vLLM engine successfully.")

    # Test prompt generation
    prompts = ["Explain the concept of KV cache compression in 2 sentences."]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    outputs = llm.generate(prompts, sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(f"\nGenerated Output from vLLM + Geodesia-KV Plugin:\n{generated_text.strip()}\n")

    assert len(generated_text) > 0, "Generated text is empty"
    print("✓ vLLM + Geodesia-KV Plugin Integration Test PASSED!")


if __name__ == "__main__":
    test_geodesia_vllm_kv_cache_compression()
    test_vllm_geodesia_plugin_integration()
