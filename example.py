"""
Example usage of the Deep Research SDK with different web clients.
"""

import asyncio
import os
import logging
from deep_research import DeepResearch
from deep_research.utils import DoclingClient, DoclingServerClient, FirecrawlClient
from deep_research.utils.cache import CacheConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def example_with_docling():
    """Run example with standard Docling client."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    # Define a research topic
    topic = "The impact of quantum computing on cryptography and security"

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    if not brave_api_key:
        print(
            "Warning: BRAVE_API_KEY environment variable not set. Falling back to DuckDuckGo search."
        )

    print("\n\n==== USING STANDARD DOCLING CLIENT ====")

    # Create a Deep Research instance with DoclingClient
    researcher = DeepResearch(
        web_client=DoclingClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=8,
            cache_config=CacheConfig(enabled=True),
            page_content_max_chars=8000,  # Maximum number of characters to return in the scraped page content
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",  # You can change this to any supported model
        reasoning_model="o3-mini",  # You can change this to any supported model
        max_depth=2,  # Limit depth for this example
        time_limit_minutes=1.5,  # Limit time for this example
    )

    # Run the full research
    print(f"Starting research with DoclingClient on: {topic}")
    # Set max_tokens to avoid context window errors
    result = await researcher.research(topic, max_tokens=8000)

    # Check the result
    if result.success:
        print("\n==== RESEARCH SUCCESSFUL ====")
        print(f"- Found {len(result.data['findings'])} pieces of information")
        print(f"- Used {len(result.data['sources'])} sources")
        print(
            f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
        )

        print("\n==== SOURCES USED ====")
        for i, source in enumerate(result.data["sources"]):
            print(f"{i + 1}. {source['title']} (Relevance: {source['relevance']:.2f})")
            print(f"   URL: {source['url']}")

        print("\n==== FINAL ANALYSIS ====")
        print(result.data["analysis"])
    else:
        print("\n==== RESEARCH FAILED ====")
        print(f"Error: {result.error}")
        print(
            f"- Found {len(result.data['findings'])} pieces of information before failure"
        )
        if "sources" in result.data:
            print(f"- Used {len(result.data['sources'])} sources")
        print(
            f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
        )


async def example_with_docling_server():
    """Run example with DoclingServer client."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    # Define a research topic
    topic = "Advances in machine learning for natural language processing"

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print("\n\n==== USING DOCLING SERVER CLIENT ====")

    # Create a Deep Research instance with DoclingServerClient
    async with DoclingServerClient(
        server_url="http://localhost:8000",  # Update with your Docling server URL
        brave_api_key=brave_api_key,
        max_concurrent_requests=8,
        cache_config=CacheConfig(enabled=True),
        page_content_max_chars=8000,
    ) as docling_server_client:
        researcher = DeepResearch(
            web_client=docling_server_client,
            llm_api_key=openai_api_key,
            research_model="gpt-4o-mini",
            reasoning_model="o3-mini",
            max_depth=2,
            time_limit_minutes=1.5,
        )

        # Run the full research
        print(f"Starting research with DoclingServerClient on: {topic}")
        result = await researcher.research(topic, max_tokens=8000)

        # Check the result
        if result.success:
            print("\n==== RESEARCH SUCCESSFUL ====")
            print(f"- Found {len(result.data['findings'])} pieces of information")
            print(f"- Used {len(result.data['sources'])} sources")
            print(
                f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
            )

            print("\n==== SOURCES USED ====")
            for i, source in enumerate(result.data["sources"]):
                print(
                    f"{i + 1}. {source['title']} (Relevance: {source['relevance']:.2f})"
                )
                print(f"   URL: {source['url']}")

            print("\n==== FINAL ANALYSIS ====")
            print(result.data["analysis"])
        else:
            print("\n==== RESEARCH FAILED ====")
            print(f"Error: {result.error}")
            print(
                f"- Found {len(result.data['findings'])} pieces of information before failure"
            )
            if "sources" in result.data:
                print(f"- Used {len(result.data['sources'])} sources")
            print(
                f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
            )


async def example_with_firecrawl():
    """Run example with Firecrawl client."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")

    # Define a research topic
    topic = "Recent developments in renewable energy technology"

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    if not firecrawl_api_key:
        print("Error: FIRECRAWL_API_KEY environment variable not set")
        return

    print("\n\n==== USING FIRECRAWL CLIENT ====")

    # Create a Deep Research instance with FirecrawlClient
    async with (
        FirecrawlClient(
            api_key=firecrawl_api_key,
            api_url="https://api.firecrawl.dev",  # Update with your Firecrawl API URL if different
            max_concurrent_requests=8,
            cache_config=CacheConfig(enabled=True),
            page_content_max_chars=8000,
        ) as firecrawl_client
    ):
        researcher = DeepResearch(
            web_client=firecrawl_client,
            llm_api_key=openai_api_key,
            research_model="gpt-4o-mini",
            reasoning_model="o3-mini",
            max_depth=2,
            time_limit_minutes=1.5,
        )

        # Run the full research
        print(f"Starting research with FirecrawlClient on: {topic}")
        result = await researcher.research(topic, max_tokens=8000)

        # Check the result
        if result.success:
            print("\n==== RESEARCH SUCCESSFUL ====")
            print(f"- Found {len(result.data['findings'])} pieces of information")
            print(f"- Used {len(result.data['sources'])} sources")
            print(
                f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
            )

            print("\n==== SOURCES USED ====")
            for i, source in enumerate(result.data["sources"]):
                print(
                    f"{i + 1}. {source['title']} (Relevance: {source['relevance']:.2f})"
                )
                print(f"   URL: {source['url']}")

            print("\n==== FINAL ANALYSIS ====")
            print(result.data["analysis"])
        else:
            print("\n==== RESEARCH FAILED ====")
            print(f"Error: {result.error}")
            print(
                f"- Found {len(result.data['findings'])} pieces of information before failure"
            )
            if "sources" in result.data:
                print(f"- Used {len(result.data['sources'])} sources")
            print(
                f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
            )


async def main():
    """Run all examples."""
    # You can comment out the examples you don't want to run
    await example_with_docling()

    # Note: The following examples require additional setup:
    # - DoclingServer example requires a running instance of docling-serve
    # - Firecrawl example requires a Firecrawl API key

    # Uncomment to run these examples if you have the required setup
    # await example_with_docling_server()
    # await example_with_firecrawl()


if __name__ == "__main__":
    asyncio.run(main())
