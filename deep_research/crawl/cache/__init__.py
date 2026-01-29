"""
Cache module for web clients.
"""

from .cache_decorator import cache, cached_extract, cached_search, init_cache
from .cache_models import CacheConfig, CacheEntry, clear_cache, clear_expired_cache
from .cache_dump import (
    DumpManager,
    save_dump,
    load_dump,
    list_dumps,
    delete_dump,
)

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "cache",
    "cached_search",
    "cached_extract",
    "init_cache",
    "clear_cache",
    "clear_expired_cache",
    "DumpManager",
    "save_dump",
    "load_dump",
    "list_dumps",
    "delete_dump",
]
