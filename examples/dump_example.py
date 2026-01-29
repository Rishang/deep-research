"""
Example showing how to save and load research dumps.
"""

import asyncio
import os

from deep_research import DeepResearch
from deep_research.crawl import MarkItDownClient
from deep_research.crawl.cache import CacheConfig


async def example_save_dump():
    """Example of conducting research with dump saving enabled."""
    print("=" * 70)
    print("EXAMPLE 1: Research with Dump Saving")
    print("=" * 70)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return None

    # Create researcher with dump saving enabled
    researcher = DeepResearch(
        web_client=MarkItDownClient(
            brave_api_key=brave_api_key,
            cache_config=CacheConfig(enabled=True),
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",
        reasoning_model="o3-mini",
        max_depth=2,
        time_limit_minutes=1.0,
        dump_files=True,  # Enable dump saving
        dump_files_path="./dumps",  # Directory for dumps
    )

    # Conduct research
    topic = "Recent advances in graph neural networks"
    print(f"\n🔍 Researching: {topic}\n")

    result = await researcher.research(topic, max_tokens=4000)

    if result.success:
        print("\n✅ Research completed successfully!")
        print(f"   • Findings: {len(result.data['findings'])}")
        print(f"   • Sources: {len(result.data['sources'])}")

        # The dump is automatically saved
        print("\n   📁 Dump automatically saved to ./dumps/")

        # Get the session_id from knowledge graph data if available
        session_id = None
        if "knowledge_graph" in result.data:
            session_id = result.data["knowledge_graph"].get("session_id")

        return session_id
    else:
        print(f"\n❌ Research failed: {result.error}")
        return None


async def example_load_dump(session_id: str):
    """Example of loading a previously saved dump."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Loading Saved Research")
    print("=" * 70)

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Create researcher instance (just for loading dumps)
    researcher = DeepResearch(
        web_client=MarkItDownClient(),
        llm_api_key=openai_api_key,
        dump_files_path="./dumps",
    )

    # Load the dump
    print(f"\n📂 Loading research dump: {session_id}\n")
    result = researcher.load_dump(session_id)

    if result:
        print("\n✅ Dump loaded successfully!")
        print(f"   • Success: {result.success}")
        print(f"   • Findings: {len(result.data.get('findings', []))}")
        print(f"   • Sources: {len(result.data.get('sources', []))}")
        print(f"   • Confirmed facts: {len(result.data.get('confirmed_facts', []))}")

        # Display analysis
        if "analysis" in result.data:
            print("\n📝 Analysis Preview:")
            analysis = result.data["analysis"]
            print(f"   {analysis[:200]}...")
    else:
        print("❌ Failed to load dump")


def example_list_dumps():
    """Example of listing all available dumps."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: List All Dumps")
    print("=" * 70)

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    researcher = DeepResearch(
        web_client=MarkItDownClient(),
        llm_api_key=openai_api_key,
        dump_files_path="./dumps",
    )

    # List all dumps
    dumps = researcher.list_dumps()

    print(f"\n📁 Found {len(dumps)} saved research dumps:\n")
    for i, dump_id in enumerate(dumps[:10], 1):  # Show first 10
        # Get metadata
        metadata = researcher.get_dump_metadata(dump_id)
        if metadata:
            print(f"  {i}. {dump_id}")
            print(f"     Topic: {metadata.get('topic', 'N/A')}")
            print(f"     Timestamp: {metadata.get('timestamp', 'N/A')}")
            print(f"     Success: {metadata.get('success', 'N/A')}")
            if metadata.get("findings_count"):
                print(f"     Findings: {metadata['findings_count']}")
            print()


async def example_delete_dump(session_id: str):
    """Example of deleting a dump file."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Delete Dump")
    print("=" * 70)

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    researcher = DeepResearch(
        web_client=MarkItDownClient(),
        llm_api_key=openai_api_key,
        dump_files_path="./dumps",
    )

    # Delete the dump
    print(f"\n🗑️  Deleting dump: {session_id}\n")
    success = researcher.delete_dump(session_id)

    if success:
        print("✅ Dump deleted successfully")
    else:
        print("❌ Failed to delete dump")


async def example_using_dump_manager_directly():
    """Example of using DumpManager directly for custom use cases."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Using DumpManager Directly")
    print("=" * 70)

    from deep_research.crawl.cache import DumpManager

    # Create a DumpManager instance
    manager = DumpManager(dump_dir="./custom_dumps", format="yaml")

    # Save custom data
    custom_data = {
        "analysis": "Custom research analysis",
        "findings": ["Finding 1", "Finding 2"],
        "sources": ["https://example.com/1", "https://example.com/2"],
    }

    metadata = {
        "topic": "Custom research topic",
        "researcher": "My Research Bot",
    }

    print("\n💾 Saving custom data...")
    filepath = manager.save("custom_session_001", custom_data, metadata)
    print(f"   Saved to: {filepath}")

    # Load it back
    print("\n📂 Loading custom data...")
    loaded = manager.load("custom_session_001")

    if loaded:
        print("✅ Loaded successfully!")
        print(f"   Topic: {loaded.get('topic')}")
        print(f"   Findings: {len(loaded.get('findings', []))}")

    # List all custom dumps
    print("\n📁 All custom dumps:")
    dumps = manager.list()
    for dump_id in dumps:
        print(f"   • {dump_id}")

    # Cleanup
    print("\n🗑️  Cleaning up...")
    manager.delete("custom_session_001")


async def main():
    """Run all examples."""
    print("\n💾 Deep Research SDK - Dump Management Examples\n")

    # Example 1: Save a dump
    session_id = await example_save_dump()

    # Example 2: Load the dump (if research succeeded)
    if session_id:
        await example_load_dump(session_id)

    # Example 3: List all dumps
    example_list_dumps()

    # Example 4: Delete a dump (if you want to clean up)
    # Uncomment to test deletion:
    # if session_id:
    #     await example_delete_dump(session_id)

    # Example 5: Use DumpManager directly
    await example_using_dump_manager_directly()

    print("\n" + "=" * 70)
    print("✅ All dump examples completed!")
    print("=" * 70)
    print("\nTips:")
    print("  • Dumps are saved in YAML format by default (human-readable)")
    print("  • Set dump_files=True when initializing DeepResearch to enable")
    print("  • Use load_dump() to reuse previous research results")
    print("  • Dumps include full research data, metadata, and timestamps")
    print()


if __name__ == "__main__":
    asyncio.run(main())
