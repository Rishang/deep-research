"""
Example usage of the Deep Research SDK.
"""

import asyncio
import os
import logging
from deep_research import DeepResearch
from deep_research.utils.docling_client import DoclingClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def main():
    """Run the example."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    # Define a research topic
    topic = "What for scope of quantum computing for finance and artifical intelligence in future"

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    if not brave_api_key:
        print(
            "Warning: BRAVE_API_KEY environment variable not set. Falling back to Docling search."
        )

    # Create a Deep Research instance
    researcher = DeepResearch(
        docling_client=DoclingClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=8,
            cache_config=None,
            page_content_max_chars=8000,  # Maximum number of characters to return in the scraped page content
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",  # You can change this to any supported model
        reasoning_model="o3-mini",  # You can change this to any supported model
        max_depth=2,  # Limit depth for this example
        time_limit_minutes=1.5,  # Limit time for this example
    )

    # Run the full research
    print(f"Starting complete research on: {topic}")
    # Set max_tokens to avoid context window errors
    result = await researcher.research(topic, max_tokens=8000)

    # Check the result
    if result.success:
        print("\n\n==== RESEARCH SUCCESSFUL ====")
        print(f"- Found {len(result.data['findings'])} pieces of information")
        print(
            f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
        )
        print("\n==== FINAL ANALYSIS ====")
        print(result.data["analysis"])
    else:
        print("\n==== RESEARCH FAILED ====")
        print(f"Error: {result.error}")
        print(
            f"- Found {len(result.data['findings'])} pieces of information before failure"
        )
        print(
            f"- Completed {result.data['completed_steps']} of {result.data['total_steps']} steps"
        )


if __name__ == "__main__":
    asyncio.run(main())
