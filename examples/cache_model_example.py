"""
Example demonstrating how to use the cache decorator with a structured model.
"""

import asyncio
import time
from typing import List, Optional

from pydantic import BaseModel

from deep_research.utils.cache import CacheConfig, cache, init_cache


class SearchParams(BaseModel):
    """Model defining search parameters to use for caching."""

    query: str
    max_results: int = 10


class ExtractParams(BaseModel):
    """Model defining extract parameters to use for caching."""

    url: str
    prompt: Optional[str] = None


class MockClient:
    """Mock client to demonstrate structured caching."""

    def __init__(self, cache_enabled: bool = True):
        """Initialize the mock client."""
        self.cache_enabled = cache_enabled

        # Initialize cache if enabled
        if cache_enabled:
            init_cache(
                CacheConfig(
                    enabled=True,
                    ttl_seconds=60,  # Short TTL for demo
                    db_url="sqlite:///mock_cache.db",
                    create_tables=True,
                )
            )

    @cache(structure=SearchParams)
    async def search(self, query: str, max_results: int = 10) -> List[str]:
        """
        Perform a mock search operation.
        This function is cached based on the SearchParams model.
        """
        print(
            f"  Performing actual search for '{query}' with max_results={max_results}"
        )
        # Simulate network delay
        await asyncio.sleep(1)
        # Return mock results
        return [f"Result {i} for '{query}'" for i in range(max_results)]

    @cache(structure=ExtractParams)
    async def extract_content(self, url: str, prompt: Optional[str] = None) -> str:
        """
        Extract content from a URL.
        This function is cached based on the ExtractParams model.
        """
        print(f"  Performing actual extraction for '{url}'")
        # Simulate network delay
        await asyncio.sleep(2)
        # Return mock content
        return f"Extracted content from {url}" + (
            f" with prompt '{prompt}'" if prompt else ""
        )


async def main():
    """Demonstrate the structured caching functionality."""
    print("Initializing MockClient with caching enabled")
    client = MockClient(cache_enabled=True)

    # First search - should execute the actual function
    print("\n== First search (not cached) ==")
    start = time.time()
    results1 = await client.search("quantum computing", max_results=3)
    duration1 = time.time() - start
    print(f"  Found {len(results1)} results in {duration1:.3f} seconds")

    # Second search with same parameters - should use cache
    print("\n== Second search (should be cached) ==")
    start = time.time()
    results2 = await client.search("quantum computing", max_results=3)
    duration2 = time.time() - start
    print(f"  Found {len(results2)} results in {duration2:.3f} seconds")
    print(f"  Speed improvement: {duration1 / duration2:.1f}x faster")

    # Third search with different max_results - should NOT use cache
    print("\n== Third search with different max_results (not cached) ==")
    start = time.time()
    results3 = await client.search("quantum computing", max_results=5)
    duration3 = time.time() - start
    print(f"  Found {len(results3)} results in {duration3:.3f} seconds")

    # Extraction example
    print("\n== First extraction (not cached) ==")
    start = time.time()
    content1 = await client.extract_content("https://example.com/page1")
    duration1 = time.time() - start
    print(f"  Extraction completed in {duration1:.3f} seconds")
    print(f"  Content length: {len(content1)} characters")

    # Second extraction - should use cache
    print("\n== Second extraction (should be cached) ==")
    start = time.time()
    # We don't need to store the result - just checking cache performance
    await client.extract_content("https://example.com/page1")
    duration2 = time.time() - start
    print(f"  Extraction completed in {duration2:.3f} seconds")
    print(f"  Speed improvement: {duration1 / duration2:.1f}x faster")

    # Extraction with prompt parameter
    print("\n== Extraction with prompt (not cached) ==")
    start = time.time()
    # We don't need to store the result
    await client.extract_content(
        "https://example.com/page1", prompt="Extract key information"
    )
    duration3 = time.time() - start
    print(f"  Extraction completed in {duration3:.3f} seconds")

    # Disable cache and try again
    print("\n== Testing with cache disabled ==")
    init_cache(CacheConfig(enabled=False))

    start = time.time()
    await client.search(
        "quantum computing", max_results=3
    )  # Don't need to store the result
    duration = time.time() - start
    print(f"  Query with cache disabled completed in {duration:.3f} seconds")

    # Re-enable cache for cleanup demonstration
    print("\n== Cleaning up cache ==")
    init_cache(CacheConfig(enabled=True))

    # Show how to clear specific function cache
    from deep_research.utils.cache import clear_cache

    clear_cache(function_name="MockClient.search")
    print("  Cleared search cache")

    # Clear all expired cache
    from deep_research.utils.cache import clear_expired_cache

    clear_expired_cache()
    print("  Cleared expired cache entries")


if __name__ == "__main__":
    asyncio.run(main())
