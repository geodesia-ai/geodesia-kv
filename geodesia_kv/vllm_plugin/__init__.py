"""Geodesia-KV Plugin for vLLM.

Provides Rate-Distortion compressed KV cache management and attention hooks
integrated with vLLM's paged block infrastructure.
"""

from geodesia_kv.vllm_plugin.kv_cache import GeodesiaKVCacheManager, GeodesiaBlockCache
from geodesia_kv.vllm_plugin.vllm_connector import register_geodesia_kv_plugin

__all__ = ["GeodesiaKVCacheManager", "GeodesiaBlockCache", "register_geodesia_kv_plugin"]
