"""
Test script for the DeepResearch class.
"""

import asyncio
import os
import sys

# Add the parent directory to sys.path to allow importing the module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import after path manipulation
from deep_research import DeepResearch  # noqa: E402
from deep_research.core.callbacks import PrintCallback  # noqa: E402
from deep_research.utils.docling_client import DoclingClient  # noqa: E402


async def main():
    """Run a simplified test of the DeepResearch class."""
    # Create a Deep Research instance with minimal options
    researcher = DeepResearch(
        docling_client=DoclingClient(
            brave_api_key=os.environ.get(
                "BRAVE_SEARCH_API_KEY", "fake-key-for-testing"
            ),
            max_concurrent_requests=3,
        ),
        llm_api_key=os.environ.get("OPENAI_API_KEY", "fake-key-for-testing"),
        research_model="gpt-4o-mini",  # Use a simpler model for testing
        reasoning_model="o3-mini",
        callback=PrintCallback(),
        max_depth=1,  # Only one level for test
        time_limit_minutes=0.5,  # Very short timeout
    )

    print("DeepResearch instance created successfully.")

    # Test client methods directly for basic functionality
    print("\nTesting search:")
    search_result = await researcher.docling_client.search(
        "quantum computing", max_results=2
    )
    print(f"Search success: {search_result.success}")
    if search_result.success:
        print(f"Found {len(search_result.data)} results")

    # Test the initial phase of research without completing (skip LLM calls if no API key)
    if os.environ.get("OPENAI_API_KEY"):
        print("\nStarting minimal research process (will use OpenAI API):")
        try:
            # This will make actual OpenAI calls if OPENAI_API_KEY is set
            await researcher._add_activity(
                type_="thought",
                status="pending",
                message="Testing activity tracking",
                depth=0,
            )
            print("Activity tracking works.")
        except Exception as e:
            print(f"Error in activity tracking: {str(e)}")
    else:
        print("\nSkipping LLM tests as OPENAI_API_KEY is not set.")

    print("\nTests completed.")


if __name__ == "__main__":
    asyncio.run(main())
