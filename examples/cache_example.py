"""
Example showing how to use the cache with the Docling client.
"""

import asyncio
import os
import time

from deep_research.utils.cache import (
    CacheConfig,
    clear_cache,
    clear_expired_cache,
    init_cache,
)
from deep_research.utils.docling_client import DoclingClient


async def main():
    """Run a demonstration of the caching functionality."""
    # Get API key from environment variable or use demo key
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    # Create a cache configuration - using SQLite for simplicity
    cache_config = CacheConfig(
        enabled=True,
        ttl_seconds=3600,  # Cache data for 1 hour
        db_url="sqlite:///docling_cache.db",
        create_tables=True,
    )

    # Initialize the DoclingClient with caching enabled
    client = DoclingClient(
        api_key="demo",
        brave_api_key=brave_api_key,
        use_brave_search=brave_api_key is not None,
        cache_config=cache_config,
    )

    # Example search query
    query = "quantum computing impact"

    # First search (not cached yet)
    print(f"Performing first search for: '{query}'")
    start_time = time.time()
    search_result = await client.search(query)
    first_search_time = time.time() - start_time
    print(f"First search took {first_search_time:.3f} seconds")
    print(f"Found {len(search_result.data)} results")

    # Second search (should use cache)
    print(f"\nPerforming second search for: '{query}' (should be cached)")
    start_time = time.time()
    cached_search_result = await client.search(query)
    second_search_time = time.time() - start_time
    print(f"Second search took {second_search_time:.3f} seconds")
    print(f"Found {len(cached_search_result.data)} results")
    print(f"Speed improvement: {first_search_time / second_search_time:.2f}x faster")

    # Verify the results match
    print("\nVerifying that cached results match original results:")
    first_urls = [item.url for item in search_result.data]
    second_urls = [item.url for item in cached_search_result.data]
    print(f"Results match: {first_urls == second_urls}")

    # Example extraction
    if search_result.data:
        # Extract information from the first URL
        url = str(search_result.data[0].url)
        print(f"\nExtracting content from: {url}")

        # First extraction (not cached)
        start_time = time.time()
        extract_result = await client.scrape_url(url)
        first_extract_time = time.time() - start_time
        print(f"First extraction took {first_extract_time:.3f} seconds")
        print(f"Extraction successful: {extract_result.success}")

        # Second extraction (should use cache)
        print("\nPerforming second extraction (should be cached)")
        start_time = time.time()
        cached_extract_result = await client.scrape_url(url)
        second_extract_time = time.time() - start_time
        print(f"Second extraction took {second_extract_time:.3f} seconds")
        print(
            f"Speed improvement: {first_extract_time / second_extract_time:.2f}x faster"
        )

        # Verify extraction results match
        print("\nVerifying that cached extraction matches original:")
        first_len = len(str(extract_result.data))
        second_len = len(str(cached_extract_result.data))
        print(f"Content length matches: {first_len == second_len}")

    # Disable caching and try again
    print("\n--- Testing with caching disabled ---")
    init_cache(CacheConfig(enabled=False))

    # Search with caching disabled
    print(f"Performing search with caching disabled: '{query}'")
    start_time = time.time()
    uncached_result = await client.search(query)
    uncached_time = time.time() - start_time
    print(f"Uncached search took {uncached_time:.3f} seconds")
    print(f"Found {len(uncached_result.data)} results")

    # Test cache clearing functions
    print("\n--- Testing cache management functions ---")
    # Re-enable cache
    init_cache(CacheConfig(enabled=True))
    # Clear specific function cache
    clear_cache(function_name="DoclingClient.search")
    print("Cleared search function cache")

    # Perform search again after clearing cache (should be slow again)
    print("\nPerforming search after clearing cache:")
    start_time = time.time()
    await client.search(query)  # We don't need to store this result
    cleared_time = time.time() - start_time
    print(f"Search after clearing cache took {cleared_time:.3f} seconds")

    # Clear expired cache entries
    clear_expired_cache()
    print("Cleared all expired cache entries")

    # Re-enable caching
    print("\n--- Re-enabling caching ---")
    init_cache(CacheConfig(enabled=True))


if __name__ == "__main__":
    asyncio.run(main())
