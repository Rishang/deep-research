"""
Example showcasing the adaptive query generation in Deep Research SDK.

Demonstrates:
- Phase-aware query generation (Exploration, Deepening, Verification)
- Multiple queries generated per phase
- Query relevance scoring and explanations
- How queries evolve based on findings
"""

import asyncio
import os
import logging
from deep_research import DeepResearch
from deep_research.crawl import MarkItDownClient
from deep_research.crawl.cache import CacheConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def query_generation_example():
    """Run example demonstrating adaptive query generation across phases."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    # Define a research topic - choose a complex one that benefits from multiple phases
    topic = "The role of vector databases in modern AI applications"

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    if not brave_api_key:
        print(
            "Warning: BRAVE_API_KEY environment variable not set. Falling back to DuckDuckGo search."
        )

    print("\n" + "=" * 70)
    print("🔍 ADAPTIVE QUERY GENERATION EXAMPLE")
    print("=" * 70)
    print(f"\n📚 Research Topic: {topic}\n")

    # Create a Deep Research instance with MarkItDownClient
    researcher = DeepResearch(
        web_client=MarkItDownClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=8,
            cache_config=CacheConfig(enabled=True),
            page_content_max_chars=8000,
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",
        reasoning_model="o3-mini",
        max_depth=3,  # Allow all phases to execute
        time_limit_minutes=2.5,  # Enough time for adaptive phases
        enable_graphrag=True,
    )

    # Run the research with adaptive query generation
    print("🚀 Starting adaptive multi-phase research...\n")
    result = await researcher.research(topic, max_tokens=8000)

    # Check the result
    if result.success:
        print("\n" + "=" * 70)
        print("✅ RESEARCH SUCCESSFUL")
        print("=" * 70)

        print("\n📊 Research Statistics:")
        print(f"   • Findings: {len(result.data['findings'])}")
        print(f"   • Sources: {len(result.data['sources'])}")
        print(f"   • Research phases: {result.data.get('research_phases', 0)}")

        # Demonstrate adaptive query generation
        print("\n" + "=" * 70)
        print("🎯 ADAPTIVE QUERY GENERATION - BY PHASE")
        print("=" * 70)

        if "search_queries" in result.data:
            # Group queries by phase
            phase_queries = {}
            for query in result.data["search_queries"]:
                phase = query.get("phase", "unknown")
                if phase not in phase_queries:
                    phase_queries[phase] = []
                phase_queries[phase].append(query)

            # Display each phase's queries
            phase_order = ["exploration", "deepening", "verification"]
            phase_icons = {
                "exploration": "🌍",
                "deepening": "🔬",
                "verification": "✓",
                "unknown": "❓",
            }

            for phase in phase_order:
                if phase in phase_queries:
                    queries = phase_queries[phase]
                    icon = phase_icons.get(phase, "•")
                    print(f"\n{icon} {phase.upper()} Phase ({len(queries)} queries):")
                    print("-" * 70)

                    for i, query in enumerate(queries, 1):
                        print(f"\n   Query {i}: {query['query']}")
                        print(f"   Relevance: {query['relevance']:.2f}")
                        if query.get("explanation"):
                            print(f"   Purpose: {query['explanation']}")

            # Show unknown phase queries if any
            if "unknown" in phase_queries:
                queries = phase_queries["unknown"]
                print(f"\n❓ Other Queries ({len(queries)}):")
                for i, query in enumerate(queries, 1):
                    print(f"   {i}. {query['query']}")

        # Show top sources with quality indicators
        print("\n" + "=" * 70)
        print("📚 TOP SOURCES (Quality Ranked)")
        print("=" * 70)
        for i, source in enumerate(result.data["sources"][:8], 1):
            print(f"\n{i}. {source['title']}")
            print(f"   URL: {source['url']}")
            print(f"   Relevance: {source['relevance']:.2f}")

        print("\n" + "=" * 70)
        print("📝 FINAL ANALYSIS PREVIEW")
        print("=" * 70)
        analysis_lines = result.data["analysis"].split("\n")
        preview = "\n".join(analysis_lines[:25])
        print(
            f"{preview}\n\n... [Full analysis available in result.data['analysis']] ..."
        )

    else:
        print("\n" + "=" * 70)
        print("❌ RESEARCH FAILED")
        print("=" * 70)
        print(f"Error: {result.error}")
        print(f"- Findings collected: {len(result.data.get('findings', []))}")
        print(f"- Sources consulted: {len(result.data.get('sources', []))}")


async def main():
    """Run the example."""
    await query_generation_example()


if __name__ == "__main__":
    asyncio.run(main())
