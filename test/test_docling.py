"""
Simple test script for the Docling client.
"""

import asyncio
import os
import sys

# Add the parent directory to sys.path to allow importing the module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import after path manipulation
from deep_research.utils.docling_client import DoclingClient  # noqa: E402


async def main():
    """Test the Docling client."""
    # Create a Docling client
    client = DoclingClient()

    # Test search
    print("Testing search...")
    result = await client.search("quantum computing")
    print(f"Search success: {result.success}")
    if result.success and result.data:
        print(f"Found {len(result.data)} results")
        print(f"First result: {result.data[0]['title']}")

    # Test extraction with a test URL
    print("\nTesting URL extraction...")
    url = "https://en.wikipedia.org/wiki/Quantum_computing"
    prompt = "Extract information about quantum computing"
    extract_result = await client.extract([url], prompt)
    print(f"Extract success: {extract_result.success}")
    if extract_result.success and extract_result.data:
        # Just show the first 200 characters of the extracted content
        content = extract_result.data[0]["data"]
        print(f"Extracted content sample: {content[:200]}...")

    print("\nTests completed.")


if __name__ == "__main__":
    asyncio.run(main())
