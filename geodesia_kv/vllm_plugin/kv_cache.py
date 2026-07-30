"""Block-paged Rate-Distortion KV Cache Manager for vLLM integration.

Manages paged KV cache allocation and applies Geodesia-KV's rate-distortion 
bit-level ladder {16, 8, 4, 2, centroid} per layer/block to achieve significant VRAM reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from geodesia_kv.packed import build_packed, PackedKV


@dataclass
class GeodesiaBlockCache:
    """Manages compressed KV cache blocks for a single sequence or layer batch in vLLM."""
    block_size: int = 64
    bits_per_value: float = 4.0
    group_size: int = 32
    layers: list[PackedKV] | None = None

    def allocate_bits_for_blocks(self, nb: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocates bit levels {2, 4, 8, 16} and centroid mask per block for a target bpv."""
        bits = torch.full((nb,), self.bits_per_value, device=device, dtype=torch.float32)
        centroid = torch.zeros((nb,), device=device, dtype=torch.bool)
        
        # Enforce rate-distortion ladder values
        if self.bits_per_value <= 2.5:
            bits.fill_(2.0)
        elif self.bits_per_value <= 4.5:
            bits.fill_(4.0)
        elif self.bits_per_value <= 8.5:
            bits.fill_(8.0)
        else:
            bits.fill_(16.0)
            
        # Always keep the most recent block exact (16 bits)
        if nb > 0:
            bits[-1] = 16.0
            
        return bits, centroid

    def compress_sequence_kv(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> float:
        """Compresses dense KV cache (num_layers, seq_len, num_heads, head_dim) into PackedKV blocks.

        Returns achieved bits-per-value rate across all layers.
        """
        num_layers, seq_len, num_heads, head_dim = key_cache.shape
        self.layers = []
        total_bytes = 0
        total_values = num_layers * seq_len * num_heads * head_dim * 2  # K and V

        for l in range(num_layers):
            # Reshape per-layer K and V
            k = key_cache[l].reshape(seq_len, num_heads * head_dim)
            v = value_cache[l].reshape(seq_len, num_heads * head_dim)

            # Compute block count
            nb = math.ceil(seq_len / self.block_size)
            if nb == 0:
                continue

            # Target total bits for this layer
            target_bits = int(self.bits_per_value * nb * self.block_size * (num_heads * head_dim))

            # Allocate bit-level precision per block using Geodesia-KV rate-distortion ladder
            bits, centroid = self.allocate_bits_for_blocks(nb, key_cache.device)

            packed_layer = build_packed(
                k=k,
                v=v,
                bits=bits,
                centroid=centroid,
                block=self.block_size,
                group=self.group_size
            )
            self.layers.append(packed_layer)
            total_bytes += packed_layer.nbytes()

        achieved_bpv = (total_bytes * 8) / max(1, total_values)
        return achieved_bpv


class GeodesiaKVCacheManager:
    """vLLM-compatible Paged KV Cache Manager using Geodesia-KV compressed representation."""

    def __init__(self, block_size: int = 64, target_bpv: float = 4.0, num_layers: int = 24):
        self.block_size = block_size
        self.target_bpv = target_bpv
        self.num_layers = num_layers
        self.active_caches: dict[int, GeodesiaBlockCache] = {}

    def create_sequence_cache(self, seq_id: int) -> GeodesiaBlockCache:
        cache = GeodesiaBlockCache(block_size=self.block_size, bits_per_value=self.target_bpv)
        self.active_caches[seq_id] = cache
        return cache

    def get_cache(self, seq_id: int) -> GeodesiaBlockCache | None:
        return self.active_caches.get(seq_id)

    def free_sequence_cache(self, seq_id: int):
        if seq_id in self.active_caches:
            del self.active_caches[seq_id]
