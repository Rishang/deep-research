"""
Test script for the search functionality.
"""

import asyncio
import os
import sys

# Add the parent directory to sys.path to allow importing the module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import after path manipulation
from deep_research.crawl.brave_search import BraveSearchClient  # noqa: E402
from deep_research.crawl.markitdown_client import MarkItDownClient  # noqa: E402
from deep_research.crawl.duckduckgo_search import DuckDuckGoSearchClient  # noqa: E402


async def test_search_providers():
    """Test different search providers."""
    # Get Brave API key if available
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    # Search query to test
    query = "quantum computing applications"
    max_results = 3

    # Test Brave Search if API key is available
    if brave_api_key:
        print("Testing Brave Search...")
        brave_client = BraveSearchClient(api_key=brave_api_key)
        brave_results = await brave_client.search(query, max_results)

        print(f"Success: {brave_results.success}")
        if brave_results.success and brave_results.data:
            print(f"Found {len(brave_results.data)} results")
            for i, result in enumerate(brave_results.data, 1):
                print(f"Result {i}: {result.title} - {result.url}")
                print(f"  Provider: {result.provider}, Date: {result.date}")
        else:
            print(f"Error: {brave_results.error}")
        print()

    # Test DuckDuckGo Search (no API key needed)
    print("Testing DuckDuckGo Search...")
    ddg_client = DuckDuckGoSearchClient()
    ddg_results = await ddg_client.search(query, max_results)

    print(f"Success: {ddg_results.success}")
    if ddg_results.success and ddg_results.data:
        print(f"Found {len(ddg_results.data)} results")
        for i, result in enumerate(ddg_results.data, 1):
            print(f"Result {i}: {result.title} - {result.url}")
            print(f"  Provider: {result.provider}, Date: {result.date}")
    else:
        print(f"Error: {ddg_results.error}")
    print()

    # Test MarkItDownClient with fallback mechanism
    print("Testing MarkItDownClient with fallback mechanism...")
    markitdown_client = MarkItDownClient(brave_api_key=brave_api_key)
    markitdown_results = await markitdown_client.search(query, max_results)

    print(f"Success: {markitdown_results.success}")
    if markitdown_results.success and markitdown_results.data:
        print(
            f"Found {len(markitdown_results.data)} results using provider: {markitdown_results.data[0].provider}"
        )
        for i, result in enumerate(markitdown_results.data, 1):
            print(f"Result {i}: {result.title} - {result.url}")
            print(f"  Provider: {result.provider}, Date: {result.date}")
    else:
        print(f"Error: {markitdown_results.error}")


if __name__ == "__main__":
    asyncio.run(test_search_providers())
