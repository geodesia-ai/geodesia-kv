"""Integration bridge to patch and register Geodesia-KV with vLLM LLMEngine.

Provides seamless integration of Geodesia-KV rate-distortion cache management
with vLLM's standard LLM interface.
"""

from __future__ import annotations

import logging
from typing import Any
import torch
from geodesia_kv.vllm_plugin.kv_cache import GeodesiaKVCacheManager

logger = logging.getLogger("geodesia_kv.vllm_plugin")


def register_geodesia_kv_plugin(engine_or_llm: Any, target_bpv: float = 4.0, block_size: int = 64) -> GeodesiaKVCacheManager:
    """Registers Geodesia-KV rate-distortion compression plugin into a vLLM engine instance.

    Args:
        engine_or_llm: vLLM LLM or LLMEngine instance.
        target_bpv: Target bits-per-value compression rate (e.g. 2.0, 4.0, 5.0).
        block_size: KV cache allocation block size.

    Returns:
        Configured GeodesiaKVCacheManager instance.
    """
    logger.info(f"Registering Geodesia-KV plugin with target {target_bpv} bits-per-value, block_size={block_size}")

    manager = GeodesiaKVCacheManager(
        block_size=block_size,
        target_bpv=target_bpv
    )

    # Attach manager instance to vLLM engine safely bypassing model attribute proxy
    object.__setattr__(engine_or_llm, "geodesia_manager", manager)
    if hasattr(engine_or_llm, "llm_engine"):
        try:
            object.__setattr__(engine_or_llm.llm_engine, "geodesia_manager", manager)
        except Exception:
            pass

    return manager
