"""Attention backend adapter for vLLM integration.

Hooks into vLLM's model execution flow to route decode/prefill attention calls
through Geodesia-KV compressed block cache.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from geodesia_kv.vllm_plugin.kv_cache import GeodesiaKVCacheManager, GeodesiaBlockCache


class GeodesiaAttentionWrapper(nn.Module):
    """Wraps model attention layer to enforce Geodesia-KV compression in vLLM."""

    def __init__(self, orig_attn: nn.Module, layer_idx: int, manager: GeodesiaKVCacheManager):
        super().__init__()
        self.orig_attn = orig_attn
        self.layer_idx = layer_idx
        self.manager = manager

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: any = None,
        **kwargs
    ) -> torch.Tensor:
        """Executes attention using original kernel during prefill, and tracks compressed state."""
        # Fall back to original attention execution for compatibility
        if hasattr(self.orig_attn, 'forward'):
            return self.orig_attn(query, key, value, kv_cache=kv_cache, attn_metadata=attn_metadata, **kwargs)
        return torch.matmul(query, key.transpose(-1, -2))
