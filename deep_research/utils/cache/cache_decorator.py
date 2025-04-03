"""
Cache decorator for Docling client functions.
"""

import datetime
import functools
import hashlib
import inspect
import json
import pickle
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel
from sqlmodel import Session, select

from .cache_models import CacheConfig, CacheEntry, create_db_and_tables, get_engine

# Global configuration
_config = CacheConfig()

T = TypeVar("T")


def init_cache(config: CacheConfig) -> None:
    """
    Initialize the cache system with the given configuration.

    Args:
        config: Configuration for the cache system.
    """
    global _config
    _config = config

    if _config.create_tables:
        create_db_and_tables(_config.db_url)


def _generate_cache_key(
    func_name: str, args: Any, kwargs: Any, param_model: Optional[BaseModel] = None
) -> str:
    """Generate a unique cache key based on function name and arguments."""
    # If a parameter model is provided, use it to extract and serialize the relevant params
    if param_model:
        # Get function signature
        sig = inspect.signature(func_name)
        param_names = [p.name for p in sig.parameters.values()]

        # Extract relevant params based on model fields
        param_dict = {}
        model_fields = param_model.model_fields

        # Populate from args
        for i, arg in enumerate(args):
            if i < len(param_names) and param_names[i] in model_fields:
                param_dict[param_names[i]] = arg

        # Populate from kwargs
        for key, value in kwargs.items():
            if key in model_fields:
                param_dict[key] = value

        # Create param_model instance
        instance = param_model(**param_dict)
        serialized = instance.model_dump_json()
    else:
        # Fallback to standard serialization
        # Serialize args and kwargs, skip any client instances
        args_list = []
        for arg in args:
            # Skip self references (client instances)
            if arg.__class__.__name__ == "DoclingClient":
                continue
            elif isinstance(arg, BaseModel):
                args_list.append(arg.model_dump())
            else:
                args_list.append(arg)

        kwargs_dict = {}
        for key, value in kwargs.items():
            if isinstance(value, BaseModel):
                kwargs_dict[key] = value.model_dump()
            else:
                kwargs_dict[key] = value

        serialized = json.dumps(
            {"args": args_list, "kwargs": kwargs_dict}, sort_keys=True, default=str
        )

    # Combine function name with serialized args/kwargs for the cache key
    if not isinstance(func_name, str):
        func_name = func_name.__qualname__

    key_data = f"{func_name}:{serialized}"
    # Create a hash to use as the cache key
    return hashlib.md5(key_data.encode("utf-8")).hexdigest()


def cache(
    structure: Optional[Type[BaseModel]] = None, ttl_seconds: Optional[int] = None
):
    """
    Decorator to cache function results. Can be configured with a Pydantic model to define
    which parameters should be used for the cache key.

    Args:
        structure: Optional BaseModel class defining the structure of parameters to cache.
            Only parameters defined in this model will be used for the cache key.
        ttl_seconds: Optional time-to-live for cache entries. Defaults to config value.

    Returns:
        The decorated function.

    Example:
        ```python
        class SearchParams(BaseModel):
            query: str
            max_results: int = 10

        @cache(structure=SearchParams)
        async def search(self, query: str, max_results: int = 10):
            ...
        ```
    """

    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip caching if disabled
            if not _config.enabled:
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = _generate_cache_key(func, args, kwargs, structure)
            func_name = func.__qualname__

            # Use provided TTL or fall back to config value
            actual_ttl = ttl_seconds or _config.ttl_seconds
            expires_at = datetime.datetime.now() + datetime.timedelta(
                seconds=actual_ttl
            )

            # Check for cached result
            engine = get_engine(_config.db_url)
            with Session(engine) as session:
                statement = select(CacheEntry).where(
                    CacheEntry.key == cache_key,
                    CacheEntry.expires_at > datetime.datetime.now(),
                )
                cached_entry = session.exec(statement).first()

                if cached_entry and cached_entry.data:
                    # Deserialize and return cached result
                    return pickle.loads(cached_entry.data)

            # Call the original function
            result = await func(*args, **kwargs)

            # Cache the result
            with Session(engine) as session:
                # Delete any existing entries with this key
                statement = select(CacheEntry).where(CacheEntry.key == cache_key)
                for existing in session.exec(statement):
                    session.delete(existing)

                # Create new cache entry
                cache_entry = CacheEntry(
                    key=cache_key,
                    function_name=func_name,
                    data=pickle.dumps(result),
                    expires_at=expires_at,
                )
                session.add(cache_entry)
                session.commit()

            return result

        return wrapper

    # Allow using @cache without parentheses
    if callable(structure) and not isinstance(structure, type):
        func = structure
        structure = None
        return decorator(func)

    return decorator


# Legacy decorators for backward compatibility
def cached_search(
    func: Callable[..., Any] = None, *, ttl_seconds: Optional[int] = None
):
    """Legacy decorator for backward compatibility."""
    if func is None:
        return lambda f: cache(ttl_seconds=ttl_seconds)(f)
    return cache(ttl_seconds=ttl_seconds)(func)


def cached_extract(
    func: Callable[..., Any] = None, *, ttl_seconds: Optional[int] = None
):
    """Legacy decorator for backward compatibility."""
    if func is None:
        return lambda f: cache(ttl_seconds=ttl_seconds)(f)
    return cache(ttl_seconds=ttl_seconds)(func)
