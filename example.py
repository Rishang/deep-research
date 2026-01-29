"""
Example showcasing the enhanced Deep Research SDK with GraphRAG integration.

Demonstrates:
- Adaptive phase-based research (Exploration → Deepening → Verification)
- GraphRAG knowledge graph building
- Multi-source cross-validation
- Quality scoring and confidence metrics
- Confirmed facts and contradiction detection
"""

import asyncio
import os
import logging
from deep_research import DeepResearch
from deep_research.crawl import MarkItDownClient, FirecrawlClient
from deep_research.crawl.cache import CacheConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def example_with_markitdown(topic: str):
    """Run example with standard MarkItDown client and enhanced GraphRAG features."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "OPENROUTER_API_KEY"
    )
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    if not brave_api_key:
        print(
            "Warning: BRAVE_API_KEY environment variable not set. Falling back to DuckDuckGo search."
        )

    print("\n" + "=" * 70)
    print("🧠 DEEP RESEARCH SDK - ENHANCED WITH GRAPHRAG")
    print("=" * 70)

    # Create a Deep Research instance with GraphRAG enabled
    researcher = DeepResearch(
        web_client=MarkItDownClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=8,
            cache_config=CacheConfig(enabled=True),
            page_content_max_chars=8000,
        ),
        base_url="https://openrouter.ai/api/v1",
        llm_api_key=openai_api_key,
        research_model="openrouter/openai/gpt-4.1-nano",
        reasoning_model="openrouter/google/gemini-3-flash-preview",
        max_depth=3,  # Allow deeper research
        time_limit_minutes=2.0,  # Allow time for phases
        enable_graphrag=True,  # ✨ Enable GraphRAG
        dump_files=True,
    )

    # Run the full research
    print(f"\n🔍 Researching: {topic}\n")
    result = await researcher.research(topic, max_tokens=8000)

    # Check the result
    if result.success:
        print("\n" + "=" * 70)
        print("✅ RESEARCH SUCCESSFUL")
        print("=" * 70)

        # Basic Stats
        print("\n📊 Research Statistics:")
        print(f"   • Findings: {len(result.data['findings'])}")
        print(
            f"   • Sources consulted: {result.data.get('sources_consulted', len(result.data['sources']))}"
        )
        print(
            f"   • Research phases completed: {result.data.get('research_phases', 0)}"
        )
        print(
            f"   • Steps completed: {result.data['completed_steps']}/{result.data['total_steps']}"
        )

        # Research Goals
        if result.data.get("research_goals"):
            print("\n🎯 Research Goals:")
            for i, goal in enumerate(result.data["research_goals"], 1):
                print(f"   {i}. {goal}")

        # Search Queries by Phase
        print("\n🔍 Adaptive Search Queries:")
        if "search_queries" in result.data:
            phase_queries = {}
            for query in result.data["search_queries"]:
                phase = query.get("phase", "unknown")
                if phase not in phase_queries:
                    phase_queries[phase] = []
                phase_queries[phase].append(query)

            for phase, queries in phase_queries.items():
                print(f"\n   📍 {phase.upper()} Phase:")
                for i, query in enumerate(queries, 1):
                    print(f"      {i}. {query['query']}")
                    if query.get("explanation"):
                        print(f"         → {query['explanation']}")

        # Confirmed Facts (NEW!)
        if result.data.get("confirmed_facts"):
            print("\n✅ Confirmed Facts (Multi-Source Validated):")
            for i, fact in enumerate(result.data["confirmed_facts"][:5], 1):
                print(f"   {i}. {fact['claim']}")
                print(
                    f"      Confidence: {fact['confidence']:.1%} | Sources: {len(fact['sources'])}"
                )

        # Contradictions (NEW!)
        if result.data.get("contradictions"):
            print("\n⚠️  Contradictions Detected:")
            for i, contra in enumerate(result.data["contradictions"], 1):
                print(f"   {i}. Topic: {contra['topic']}")
                print(
                    f"      Claim A ({contra['source_a']}): {contra['claim_a'][:80]}..."
                )
                print(
                    f"      Claim B ({contra['source_b']}): {contra['claim_b'][:80]}..."
                )

        # Knowledge Graph (NEW!)
        if result.data.get("knowledge_graph"):
            kg = result.data["knowledge_graph"]
            print("\n🕸️  Knowledge Graph:")
            print(f"   • Session ID: {kg.get('session_id', 'N/A')}")
            print(f"   • Entities extracted: {kg.get('total_entities', 0)}")
            print(f"   • Relationships mapped: {kg.get('total_relationships', 0)}")
            print(f"   • Communities detected: {kg.get('communities', 0)}")

            if kg.get("top_entities"):
                print("\n   🌟 Most Important Entities (by PageRank):")
                for i, entity in enumerate(kg["top_entities"][:5], 1):
                    print(f"      {i}. {entity['name']} ({entity['type']})")
                    print(f"         PageRank: {entity['pagerank']:.4f}")

        # Priority Gaps
        if result.data.get("priority_gaps"):
            print("\n🔍 Remaining Knowledge Gaps:")
            for i, gap in enumerate(result.data["priority_gaps"][:3], 1):
                print(f"   {i}. {gap}")

        print("\n📝 Final Analysis Preview:")
        print("-" * 70)
        print(result.data["analysis"][:600] + "...")
        print("-" * 70)

    else:
        print("\n" + "=" * 70)
        print("❌ RESEARCH FAILED")
        print("=" * 70)
        print(f"Error: {result.error}")
        print(f"- Findings collected: {len(result.data.get('findings', []))}")
        print(f"- Sources consulted: {len(result.data.get('sources', []))}")


async def example_with_firecrawl(topic: str):
    """Run example with Firecrawl client."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "OPENROUTER_API_KEY"
    )
    firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")

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

            print("\n==== SEARCH QUERIES USED ====")
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
    await example_with_markitdown(topic="Why gold prices are increasing in 2026")

    # Note: The following example requires additional setup:
    # - Firecrawl example requires a Firecrawl API key

    # Uncomment to run this example if you have the required setup
    # await example_with_firecrawl()


if __name__ == "__main__":
    asyncio.run(main())
