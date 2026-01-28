"""
Example demonstrating GraphRAG (Graph Retrieval-Augmented Generation) usage.

This example shows how to:
1. Enable GraphRAG knowledge graph during research
2. Build and persist knowledge graphs
3. Query the knowledge graph
4. Load and reuse previous research sessions
"""

import asyncio
import os
from pathlib import Path

from deep_research import DeepResearch
from deep_research.graphrag import GraphQuery, KnowledgeGraphManager
from deep_research.utils import DoclingClient
from deep_research.utils.cache import CacheConfig


async def main():
    # Get API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print("\n🧠 GraphRAG Deep Research Example\n")
    print("=" * 60)

    # Define research topic
    topic = "GraphRAG and knowledge graphs in AI systems"

    # ========== PART 1: Research with GraphRAG ==========
    print("\n📊 Part 1: Conducting research with GraphRAG enabled\n")

    # Initialize DeepResearch with GraphRAG enabled
    researcher = DeepResearch(
        web_client=DoclingClient(
            brave_api_key=brave_api_key,
            max_concurrent_requests=8,
            cache_config=CacheConfig(enabled=True),
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",
        reasoning_model="o3-mini",
        max_depth=2,
        time_limit_minutes=1.5,
        enable_graphrag=True,  # Enable GraphRAG
        graphrag_storage_path=str(Path.home() / ".deep_research" / "graphs"),
    )

    # Perform research
    print(f"🔍 Researching: {topic}\n")
    result = await researcher.research(topic, max_tokens=4000)

    if result.success:
        print("\n✅ Research completed successfully!\n")
        print(f"📚 Findings: {len(result.data['findings'])}")
        print(f"🌐 Sources: {len(result.data['sources'])}")
        print(f"✓ Confirmed facts: {len(result.data['confirmed_facts'])}")

        # Display GraphRAG info
        if "knowledge_graph" in result.data:
            kg_data = result.data["knowledge_graph"]
            print("\n🕸️  Knowledge Graph Statistics:")
            print(f"   • Session ID: {kg_data.get('session_id', 'N/A')}")
            print(f"   • Entities: {kg_data.get('total_entities', 0)}")
            print(f"   • Relationships: {kg_data.get('total_relationships', 0)}")
            print(f"   • Communities: {kg_data.get('communities', 0)}")

            print("\n🌟 Top Entities by PageRank:")
            for i, entity in enumerate(kg_data.get("top_entities", [])[:5], 1):
                print(f"   {i}. {entity['name']} ({entity['type']})")
                print(f"      PageRank: {entity['pagerank']:.4f}")

            session_id = kg_data.get("session_id")
        else:
            print("\n⚠️  GraphRAG data not available")
            session_id = None

        print("\n📝 Analysis Preview:")
        print("-" * 60)
        print(result.data["analysis"][:500] + "...")
        print("-" * 60)

    else:
        print(f"\n❌ Research failed: {result.error}")
        return

    # ========== PART 2: Query the Knowledge Graph ==========
    if session_id:
        print("\n📊 Part 2: Querying the knowledge graph\n")

        # Create a new researcher instance and load the saved graph
        graph_researcher = DeepResearch(
            web_client=DoclingClient(brave_api_key=brave_api_key),
            llm_api_key=openai_api_key,
            research_model="gpt-4o-mini",
            enable_graphrag=True,
        )

        # Load the knowledge graph from the previous session
        if graph_researcher.load_knowledge_graph(session_id):
            print(f"✅ Loaded knowledge graph from session: {session_id}\n")

            # Query the graph
            query_text = "What are the key benefits of GraphRAG?"

            if graph_researcher.graphrag_retriever:
                print(f"🔍 Query: {query_text}\n")

                graph_query = GraphQuery(
                    query_text=query_text,
                    max_hops=2,
                    max_results=5,
                    use_semantic_search=True,
                )

                retrieval_result = await graph_researcher.graphrag_retriever.retrieve(
                    graph_query
                )

                print(f"📊 Retrieved {len(retrieval_result.entities)} entities")
                print(f"🔗 Found {len(retrieval_result.relationships)} relationships")
                print(f"💯 Confidence: {retrieval_result.confidence:.2f}\n")

                print("🔍 Retrieved Entities:")
                for entity in retrieval_result.entities[:5]:
                    print(f"   • {entity.name} ({entity.type})")
                    print(f"     {entity.description[:100]}...")

                print("\n🔗 Key Relationships:")
                for rel in retrieval_result.relationships[:3]:
                    if (
                        graph_researcher.graphrag_manager
                        and graph_researcher.graphrag_manager.current_graph
                    ):
                        graph = graph_researcher.graphrag_manager.current_graph
                        source = graph.get_entity(rel.source_id)
                        target = graph.get_entity(rel.target_id)
                        if source and target:
                            print(f"   • {source.name} --[{rel.type}]--> {target.name}")

                print("\n🧠 Reasoning Path:")
                for step in retrieval_result.reasoning_path:
                    print(f"   {step}")

        else:
            print(f"❌ Could not load knowledge graph from session: {session_id}")

    # ========== PART 3: List Available Sessions ==========
    print("\n📊 Part 3: Available research sessions\n")

    graph_manager = KnowledgeGraphManager()
    sessions = graph_manager.list_sessions()

    print(f"Found {len(sessions)} saved research sessions:")
    for i, sess_id in enumerate(sessions[-5:], 1):  # Show last 5
        print(f"   {i}. {sess_id}")

    print("\n✅ GraphRAG example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
