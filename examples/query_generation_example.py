"""
Example showcasing the query generation feature in Deep Research SDK.
"""

import asyncio
import os
import logging
from deep_research import DeepResearch
from deep_research.utils import DoclingClient
from deep_research.utils.cache import CacheConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def query_generation_example():
    """Run example with query generation enabled."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    # Define a research topic - choose a complex one that benefits from multiple queries
    topic = (
        "The impact of artificial intelligence on healthcare diagnostics and treatment"
    )

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    if not brave_api_key:
        print(
            "Warning: BRAVE_API_KEY environment variable not set. Falling back to DuckDuckGo search."
        )

    print("\n\n==== QUERY GENERATION EXAMPLE ====")
    print(f"Research Topic: {topic}")

    # Create a Deep Research instance with DoclingClient
    researcher = DeepResearch(
        web_client=DoclingClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=8,
            cache_config=CacheConfig(enabled=True),
            page_content_max_chars=8000,
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",
        reasoning_model="o3-mini",  # Using O3 model for reasoning
        max_depth=2,  # Limit depth for this example
        time_limit_minutes=2.0,  # Allow enough time for multiple queries
    )

    # Run the full research with query generation
    print(f"Starting research with generated search queries from topic: {topic}")
    result = await researcher.research(topic, max_tokens=8000)

    # Check the result
    if result.success:
        print("\n==== RESEARCH SUCCESSFUL ====")
        print(f"- Found {len(result.data['findings'])} pieces of information")
        print(f"- Used {len(result.data['sources'])} sources")
        print(
            f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
        )

        print("\n==== SEARCH QUERIES GENERATED ====")
        if "search_queries" in result.data:
            for i, query in enumerate(result.data["search_queries"]):
                print(
                    f"{i + 1}. Query: {query['query']} (Relevance: {query['relevance']:.2f})"
                )
                if "explanation" in query and query["explanation"]:
                    print(f"   Explanation: {query['explanation']}")
        else:
            print("No search queries data available.")

        print("\n==== SOURCES USED ====")
        # Group sources by the queries that led to them (for demonstration)
        unique_sources = {}
        for source in result.data["sources"]:
            if source["url"] not in unique_sources:
                unique_sources[source["url"]] = source

        for i, source in enumerate(unique_sources.values()):
            print(f"{i + 1}. {source['title']} (Relevance: {source['relevance']:.2f})")
            print(f"   URL: {source['url']}")

        print("\n==== FINAL ANALYSIS ====")
        # Print just the first few paragraphs for brevity
        analysis_excerpt = "\n".join(result.data["analysis"].split("\n")[:20])
        print(f"{analysis_excerpt}\n...[Analysis continues]...")
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
    """Run the example."""
    await query_generation_example()


if __name__ == "__main__":
    asyncio.run(main())
