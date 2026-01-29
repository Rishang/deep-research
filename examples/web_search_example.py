"""
Example demonstrating the use of different web search providers.
"""

import asyncio
import os

from deep_research.crawl.web import (
    BraveSearchClient,
    DuckDuckGoSearchClient,
    WebSearchProvider,
)


async def demonstrate_brave_search():
    """Demonstrate Brave Search API."""
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    if not api_key:
        print(
            "BRAVE_SEARCH_API_KEY environment variable not set, skipping Brave Search demo"
        )
        return

    print("\n=== Brave Search Example ===")
    client = BraveSearchClient(api_key=api_key)
    result = await client.search("quantum computing advancements", max_results=3)

    if result.success:
        print(f"Found {len(result.data)} results")
        for i, item in enumerate(result.data):
            print(f"\nResult {i + 1}:")
            print(f"Title: {item.title}")
            print(f"URL: {item.url}")
            print(f"Provider: {item.provider}")
            print(f"Relevance: {item.relevance}")
    else:
        print(f"Search failed: {result.error}")


async def demonstrate_duckduckgo_search():
    """Demonstrate DuckDuckGo Search."""
    print("\n=== DuckDuckGo Search Example ===")
    client = DuckDuckGoSearchClient()
    result = await client.search("artificial intelligence ethics", max_results=3)

    if result.success:
        print(f"Found {len(result.data)} results")
        for i, item in enumerate(result.data):
            print(f"\nResult {i + 1}:")
            print(f"Title: {item.title}")
            print(f"URL: {item.url}")
            print(f"Provider: {item.provider}")
    else:
        print(f"Search failed: {result.error}")


async def main():
    """Run all examples."""
    print("Demonstrating different web search providers...\n")

    # Run all examples
    await demonstrate_brave_search()
    await demonstrate_duckduckgo_search()

    # Show Web Search Provider enum
    print("\n=== Web Search Provider Enum ===")
    print("Available providers:")
    for provider in WebSearchProvider:
        print(f" - {provider.name}: {provider.value}")


if __name__ == "__main__":
    asyncio.run(main())
