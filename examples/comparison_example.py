"""
Comparison example: Traditional vs Enhanced Deep Research.

Shows side-by-side comparison of:
- Traditional depth-based iteration vs adaptive phases
- Simple search vs quality-scored selection
- No validation vs multi-source cross-validation
- No knowledge graph vs GraphRAG integration
"""

import asyncio
import os
import logging
from deep_research import DeepResearch
from deep_research.crawl import MarkItDownClient
from deep_research.crawl.cache import CacheConfig

logging.basicConfig(level=logging.WARNING)  # Reduce noise for comparison


async def traditional_approach():
    """Simulates a more traditional RAG approach (simplified)."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    print("\n" + "=" * 80)
    print("📌 APPROACH 1: TRADITIONAL (Simplified)")
    print("=" * 80)
    print("Features:")
    print("  • GraphRAG disabled")
    print("  • Limited depth")
    print("  • Basic search")
    print("  • No validation\n")

    researcher = DeepResearch(
        web_client=MarkItDownClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=5,
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",
        reasoning_model="gpt-4o-mini",  # Use same model
        max_depth=1,  # Single iteration
        time_limit_minutes=1.0,
        enable_graphrag=False,  # Disable GraphRAG
    )

    topic = "How does retrieval-augmented generation improve AI responses"

    print(f"🔍 Researching: {topic}\n")
    result = await researcher.research(topic, max_tokens=3000)

    if result.success:
        print("✅ Results:")
        print(f"   • Findings: {len(result.data['findings'])}")
        print(f"   • Sources: {len(result.data['sources'])}")
        print(f"   • Queries: {len(result.data.get('search_queries', []))}")
        print(f"   • Confirmed facts: {len(result.data.get('confirmed_facts', []))}")
        print(
            f"   • Knowledge graph: {'No' if not result.data.get('knowledge_graph') else 'Yes'}"
        )

        return {
            "findings": len(result.data["findings"]),
            "sources": len(result.data["sources"]),
            "confirmed": len(result.data.get("confirmed_facts", [])),
            "contradictions": len(result.data.get("contradictions", [])),
            "entities": 0,
            "queries": len(result.data.get("search_queries", [])),
        }

    return None


async def enhanced_approach():
    """Uses the enhanced approach with all features."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    print("\n" + "=" * 80)
    print("🚀 APPROACH 2: ENHANCED (GraphRAG + Adaptive Phases)")
    print("=" * 80)
    print("Features:")
    print("  ✅ GraphRAG enabled")
    print("  ✅ Adaptive phases (Exploration → Deepening → Verification)")
    print("  ✅ Quality scoring & diversity")
    print("  ✅ Multi-source validation")
    print("  ✅ Confidence scoring")
    print("  ✅ Contradiction detection")
    print("  ✅ Dynamic termination\n")

    researcher = DeepResearch(
        web_client=MarkItDownClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=10,
            cache_config=CacheConfig(enabled=True),
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",
        reasoning_model="o3-mini",  # Better reasoning model
        max_depth=3,
        time_limit_minutes=2.5,
        enable_graphrag=True,  # Enable GraphRAG
    )

    topic = "How does retrieval-augmented generation improve AI responses"

    print(f"🔍 Researching: {topic}\n")
    result = await researcher.research(topic, max_tokens=6000)

    if result.success:
        print("✅ Results:")
        print(f"   • Findings: {len(result.data['findings'])}")
        print(f"   • Sources: {len(result.data['sources'])}")
        print(f"   • Queries: {len(result.data.get('search_queries', []))}")
        print(f"   • Confirmed facts: {len(result.data.get('confirmed_facts', []))}")
        print(f"   • Contradictions: {len(result.data.get('contradictions', []))}")

        kg = result.data.get("knowledge_graph", {})
        if kg:
            print(f"   • Entities: {kg.get('total_entities', 0)}")
            print(f"   • Relationships: {kg.get('total_relationships', 0)}")
            print(f"   • Communities: {kg.get('communities', 0)}")

        # Show research phases
        if result.data.get("search_queries"):
            phases = set(
                q.get("phase") for q in result.data["search_queries"] if q.get("phase")
            )
            if phases:
                print(f"   • Phases executed: {', '.join(sorted(phases))}")

        # Show top entities
        if kg and kg.get("top_entities"):
            print("\n   🌟 Key Entities Identified:")
            for i, entity in enumerate(kg["top_entities"][:4], 1):
                print(
                    f"      {i}. {entity['name']} (PageRank: {entity['pagerank']:.4f})"
                )

        return {
            "findings": len(result.data["findings"]),
            "sources": len(result.data["sources"]),
            "confirmed": len(result.data.get("confirmed_facts", [])),
            "contradictions": len(result.data.get("contradictions", [])),
            "entities": kg.get("total_entities", 0),
            "queries": len(result.data.get("search_queries", [])),
        }

    return None


async def main():
    """Run comparison demo."""
    print("\n" + "🔬" * 40)
    print("COMPARISON: Traditional RAG vs Enhanced Deep Research")
    print("🔬" * 40)

    # Run both approaches
    traditional_stats = await traditional_approach()
    enhanced_stats = await enhanced_approach()

    # Display comparison
    if traditional_stats and enhanced_stats:
        print("\n" + "=" * 80)
        print("📊 COMPARISON SUMMARY")
        print("=" * 80)
        print(f"\n{'Metric':<25} {'Traditional':<20} {'Enhanced':<20} {'Improvement'}")
        print("-" * 80)

        metrics = [
            ("Findings", "findings"),
            ("Sources", "sources"),
            ("Queries Generated", "queries"),
            ("Confirmed Facts", "confirmed"),
            ("Contradictions", "contradictions"),
            ("Entities", "entities"),
        ]

        for label, key in metrics:
            trad_val = traditional_stats.get(key, 0)
            enh_val = enhanced_stats.get(key, 0)

            if trad_val > 0:
                improvement = (enh_val - trad_val) / trad_val * 100
                improvement_str = (
                    f"+{improvement:.0f}%" if improvement > 0 else f"{improvement:.0f}%"
                )
            else:
                improvement_str = "N/A" if enh_val == 0 else "∞"

            print(f"{label:<25} {trad_val:<20} {enh_val:<20} {improvement_str}")

        print("\n" + "=" * 80)
        print("💡 KEY TAKEAWAYS")
        print("=" * 80)
        print("\nEnhanced Approach Benefits:")
        print("  ✅ More comprehensive findings through adaptive phases")
        print("  ✅ Better source coverage with quality scoring")
        print("  ✅ Fact validation with multi-source confirmation")
        print("  ✅ Contradiction detection for balanced analysis")
        print("  ✅ Knowledge graph for relationship discovery")
        print("  ✅ Entity extraction for structured understanding")
        print("  ✅ Session persistence for knowledge reuse")
        print("\nTraditional Approach Limitations:")
        print("  ❌ Single-pass research (no deepening)")
        print("  ❌ No fact validation")
        print("  ❌ No knowledge structure")
        print("  ❌ No session persistence")
        print("  ❌ Fixed depth (not goal-based)")


if __name__ == "__main__":
    asyncio.run(main())
