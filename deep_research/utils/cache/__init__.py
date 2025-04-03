"""
Cache module for Docling client.
"""

from .cache_decorator import cache, cached_extract, cached_search, init_cache
from .cache_models import CacheConfig, CacheEntry, clear_cache, clear_expired_cache

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "cache",
    "cached_search",
    "cached_extract",
    "init_cache",
    "clear_cache",
    "clear_expired_cache",
]
