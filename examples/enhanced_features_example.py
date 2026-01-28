"""
Comprehensive example showcasing all enhanced Deep Research features.

Demonstrates:
1. Adaptive phase-based research
2. GraphRAG knowledge graph integration
3. Multi-source cross-validation
4. Quality scoring and source ranking
5. Confirmed facts and contradictions
6. Dynamic termination
7. Knowledge persistence and reuse
"""

import asyncio
import os
import logging
from pathlib import Path

from deep_research import DeepResearch
from deep_research.utils import DoclingClient
from deep_research.utils.cache import CacheConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def enhanced_research_demo():
    """Comprehensive demo of enhanced research features."""

    # Get API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Research topic
    topic = "How do vector databases enable semantic search in AI applications"

    print("\n" + "=" * 80)
    print("🧠 DEEP RESEARCH SDK - ENHANCED FEATURES DEMONSTRATION")
    print("=" * 80)
    print(f"\n📚 Topic: {topic}\n")

    # Initialize with all enhanced features
    researcher = DeepResearch(
        web_client=DoclingClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=10,
            cache_config=CacheConfig(
                enabled=True,
                ttl_seconds=3600,
                db_url="sqlite:///enhanced_research_cache.db",
            ),
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",
        reasoning_model="o3-mini",
        max_depth=3,  # Allow all phases
        time_limit_minutes=2.5,
        enable_graphrag=True,  # Enable knowledge graph
        graphrag_storage_path=str(Path.home() / ".deep_research" / "enhanced_demo"),
    )

    print("🚀 Starting enhanced multi-phase research...\n")
    print("Expected flow:")
    print("  1️⃣  Initialization → Extract research goals")
    print("  2️⃣  Exploration → Broad foundational queries (3-5)")
    print("  3️⃣  Deepening → Gap-focused queries (2-4)")
    print("  4️⃣  Verification → Fact-checking queries (2-3)")
    print("  5️⃣  Synthesis → Final analysis with confidence scores\n")

    # Execute research
    result = await researcher.research(topic, max_tokens=6000, temperature=0.7)

    if not result.success:
        print(f"\n❌ Research failed: {result.error}")
        return

    # Display results with all enhanced features
    print("\n" + "=" * 80)
    print("✅ RESEARCH COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # ========== SECTION 1: RESEARCH OVERVIEW ==========
    print("\n📊 SECTION 1: RESEARCH OVERVIEW")
    print("-" * 80)
    print(f"Total findings: {len(result.data['findings'])}")
    print(f"Total sources: {len(result.data['sources'])}")
    print(f"Sources consulted: {result.data.get('sources_consulted', 'N/A')}")
    print(f"Research phases completed: {result.data.get('research_phases', 'N/A')}")
    print(
        f"Steps completed: {result.data['completed_steps']}/{result.data['total_steps']}"
    )

    # ========== SECTION 2: RESEARCH GOALS ==========
    if result.data.get("research_goals"):
        print("\n🎯 SECTION 2: RESEARCH GOALS IDENTIFIED")
        print("-" * 80)
        for i, goal in enumerate(result.data["research_goals"], 1):
            print(f"{i}. {goal}")

    # ========== SECTION 3: ADAPTIVE QUERIES BY PHASE ==========
    print("\n🔍 SECTION 3: ADAPTIVE QUERIES BY PHASE")
    print("-" * 80)

    if result.data.get("search_queries"):
        phase_queries = {}
        for query in result.data["search_queries"]:
            phase = query.get("phase", "unknown")
            if phase not in phase_queries:
                phase_queries[phase] = []
            phase_queries[phase].append(query)

        for phase, queries in sorted(phase_queries.items()):
            print(f"\n📍 {phase.upper()} Phase:")
            for i, q in enumerate(queries, 1):
                print(f"   {i}. {q['query']}")
                print(
                    f"      Relevance: {q['relevance']:.2f} | {q.get('explanation', '')}"
                )

    # ========== SECTION 4: CROSS-VALIDATION RESULTS ==========
    print("\n✅ SECTION 4: CROSS-VALIDATION RESULTS")
    print("-" * 80)

    confirmed = result.data.get("confirmed_facts", [])
    contradictions = result.data.get("contradictions", [])

    print(f"Confirmed facts (multi-source): {len(confirmed)}")
    print(f"Contradictions detected: {len(contradictions)}")

    if confirmed:
        print("\nTop Confirmed Facts:")
        for i, fact in enumerate(confirmed[:3], 1):
            print(f"\n{i}. {fact['claim']}")
            print(f"   Confidence: {fact['confidence']:.1%}")
            print(f"   Validated by {len(fact['sources'])} sources")

    if contradictions:
        print("\nContradictions Found:")
        for i, contra in enumerate(contradictions[:2], 1):
            print(f"\n{i}. Topic: {contra['topic']}")
            print(f"   Source A: {contra['claim_a'][:60]}...")
            print(f"   Source B: {contra['claim_b'][:60]}...")

    # ========== SECTION 5: KNOWLEDGE GRAPH ANALYTICS ==========
    if result.data.get("knowledge_graph"):
        kg = result.data["knowledge_graph"]
        print("\n🕸️  SECTION 5: KNOWLEDGE GRAPH ANALYTICS")
        print("-" * 80)
        print(f"Session ID: {kg.get('session_id')}")
        print(f"Entities extracted: {kg.get('total_entities', 0)}")
        print(f"Relationships mapped: {kg.get('total_relationships', 0)}")
        print(f"Communities detected: {kg.get('communities', 0)}")

        if kg.get("top_entities"):
            print("\nMost Important Entities (by PageRank):")
            for i, entity in enumerate(kg["top_entities"][:8], 1):
                print(f"{i}. {entity['name']} ({entity['type']})")
                print(f"   PageRank: {entity['pagerank']:.4f}")

    # ========== SECTION 6: SOURCE QUALITY ANALYSIS ==========
    print("\n📊 SECTION 6: SOURCE QUALITY ANALYSIS")
    print("-" * 80)

    sources = result.data.get("sources", [])
    if sources:
        # Categorize by domain type
        edu_sources = [s for s in sources if ".edu" in s["url"]]
        gov_sources = [s for s in sources if ".gov" in s["url"]]
        org_sources = [s for s in sources if ".org" in s["url"]]

        print(f"Total sources: {len(sources)}")
        print(f"Academic (.edu): {len(edu_sources)}")
        print(f"Government (.gov): {len(gov_sources)}")
        print(f"Organizations (.org): {len(org_sources)}")
        print(
            f"Other domains: {len(sources) - len(edu_sources) - len(gov_sources) - len(org_sources)}"
        )

        print("\nTop 5 Sources:")
        for i, source in enumerate(sources[:5], 1):
            print(f"{i}. {source['title']}")
            print(f"   Relevance: {source['relevance']:.2f}")
            domain = source["url"].split("/")[2] if "/" in source["url"] else "unknown"
            print(f"   Domain: {domain}")

    # ========== SECTION 7: KNOWLEDGE GAPS ==========
    if result.data.get("priority_gaps"):
        print("\n🔍 SECTION 7: REMAINING KNOWLEDGE GAPS")
        print("-" * 80)
        for i, gap in enumerate(result.data["priority_gaps"][:5], 1):
            print(f"{i}. {gap}")

    # ========== SECTION 8: FINAL ANALYSIS ==========
    print("\n📝 SECTION 8: FINAL ANALYSIS")
    print("=" * 80)
    analysis_lines = result.data["analysis"].split("\n")
    preview = "\n".join(analysis_lines[:30])
    print(preview)
    print("\n... [Continued - see full analysis in result.data['analysis']] ...")
    print("=" * 80)

    # ========== SUMMARY ==========
    print("\n📈 RESEARCH SUMMARY")
    print("-" * 80)
    print(f"✅ Research Goals: {len(result.data.get('research_goals', []))}")
    print(f"✅ Confirmed Facts: {len(result.data.get('confirmed_facts', []))}")
    print(f"⚠️  Contradictions: {len(result.data.get('contradictions', []))}")
    print(
        f"🕸️  Entities: {result.data.get('knowledge_graph', {}).get('total_entities', 0)}"
    )
    print(
        f"🔗 Relationships: {result.data.get('knowledge_graph', {}).get('total_relationships', 0)}"
    )
    print(f"🌍 Sources: {len(result.data.get('sources', []))}")
    print(f"📊 Queries: {len(result.data.get('search_queries', []))}")

    if result.data.get("knowledge_graph", {}).get("session_id"):
        session_id = result.data["knowledge_graph"]["session_id"]
        print(
            f"\n💾 Knowledge graph saved to: ~/.deep_research/enhanced_demo/{session_id}.json"
        )
        print(
            f"   You can reload it later with: researcher.load_knowledge_graph('{session_id}')"
        )


async def main():
    """Run the enhanced features demo."""
    await enhanced_research_demo()


if __name__ == "__main__":
    asyncio.run(main())
