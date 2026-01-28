"""
Graph-based retrieval for GraphRAG.
"""

import json
from typing import List, Optional

import litellm

from .knowledge_graph import KnowledgeGraph
from .models import (
    Entity,
    GraphQuery,
    GraphRetrievalResult,
    Relationship,
)


class GraphRetriever:
    """
    Retrieves relevant information from the knowledge graph.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        """
        Initialize the graph retriever.

        Args:
            knowledge_graph: Knowledge graph to query.
            model: LLM model for query understanding.
            api_key: API key for LLM.
            base_url: Base URL for API requests.
        """
        self.graph = knowledge_graph
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        if api_key:
            litellm.api_key = api_key

    async def retrieve(self, query: GraphQuery) -> GraphRetrievalResult:
        """
        Retrieve relevant entities and relationships from the graph.

        Args:
            query: Graph query specification.

        Returns:
            Retrieval result with entities, relationships, and subgraph.
        """
        # Step 1: Identify relevant entities using semantic understanding
        relevant_entity_ids = await self._identify_relevant_entities(query)

        # Step 2: Expand with multi-hop traversal if requested
        if query.max_hops > 1 and query.include_neighbors:
            expanded_ids = set(relevant_entity_ids)
            for entity_id in relevant_entity_ids:
                neighbors = self.graph.get_neighbors(entity_id, query.max_hops - 1)
                expanded_ids.update(neighbors)
            relevant_entity_ids = list(expanded_ids)

        # Step 3: Filter by confidence
        filtered_ids = [
            eid
            for eid in relevant_entity_ids
            if eid in self.graph.nodes
            and self.graph.nodes[eid].entity.confidence >= query.min_confidence
        ]

        # Step 4: Rank by importance (PageRank)
        ranked_ids = sorted(
            filtered_ids,
            key=lambda eid: self.graph.nodes[eid].pagerank,
            reverse=True,
        )[: query.max_results]

        # Step 5: Extract entities and relationships
        entities = [
            self.graph.nodes[eid].entity
            for eid in ranked_ids
            if eid in self.graph.nodes
        ]

        # Get relationships between selected entities
        relevant_rel_ids = set(ranked_ids)
        relationships = []
        for edge in self.graph.edges.values():
            rel = edge.relationship
            if rel.source_id in relevant_rel_ids and rel.target_id in relevant_rel_ids:
                if not query.relationship_types or rel.type in query.relationship_types:
                    relationships.append(rel)

        # Step 6: Extract subgraph
        subgraph = self.graph.get_subgraph(set(ranked_ids))

        # Step 7: Generate reasoning path
        reasoning_path = self._generate_reasoning_path(entities, relationships)

        # Calculate confidence
        avg_confidence = (
            sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        )

        return GraphRetrievalResult(
            entities=entities,
            relationships=relationships,
            subgraph=subgraph,
            confidence=avg_confidence,
            reasoning_path=reasoning_path,
        )

    async def _identify_relevant_entities(self, query: GraphQuery) -> List[str]:
        """
        Identify entities relevant to the query using LLM.

        Args:
            query: Graph query.

        Returns:
            List of relevant entity IDs.
        """
        try:
            # Get all entities for LLM to consider
            all_entities = []
            if query.entity_types:
                for entity_type in query.entity_types:
                    all_entities.extend(self.graph.get_entities_by_type(entity_type))
            else:
                all_entities = [node.entity for node in self.graph.nodes.values()]

            if not all_entities:
                return []

            # Prepare entity list for LLM
            entity_list = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description[:100],
                }
                for e in all_entities[:100]  # Limit for context
            ]

            prompt = f"""Given a query and a list of entities from a knowledge graph, identify the most relevant entities.

<query>
{query.query_text}
</query>

<entities>
{json.dumps(entity_list, indent=2)}
</entities>

<task>
Select up to {query.max_results} most relevant entity IDs that would help answer the query.
Consider:
- Direct relevance to the query topic
- Potential to provide useful information
- Connections to other important entities
</task>

<response_format>
{{
  "relevant_entity_ids": ["entity_id1", "entity_id2", ...]
}}
</response_format>
"""

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                drop_params=True,
                base_url=self.base_url,
            )

            result_text = response.choices[0].message.content

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            parsed = json.loads(result_text)
            return parsed.get("relevant_entity_ids", [])

        except Exception as e:
            print(f"Entity identification error: {str(e)}")
            # Fallback: return top PageRank entities
            top_entities = self.graph.get_most_important_entities(query.max_results)
            return [e.id for e in top_entities]

    def _generate_reasoning_path(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> List[str]:
        """
        Generate a reasoning path explaining the retrieval.

        Args:
            entities: Retrieved entities.
            relationships: Retrieved relationships.

        Returns:
            List of reasoning steps.
        """
        path = []

        if entities:
            path.append(
                f"Found {len(entities)} relevant entities: "
                + ", ".join([e.name for e in entities[:5]])
            )

        if relationships:
            path.append(
                f"Identified {len(relationships)} relationships between entities"
            )

            # Describe a few key relationships
            for rel in relationships[:3]:
                source = self.graph.get_entity(rel.source_id)
                target = self.graph.get_entity(rel.target_id)
                if source and target:
                    path.append(
                        f"{source.name} {rel.type.replace('_', ' ')} {target.name}"
                    )

        return path

    async def retrieve_context_for_query(
        self, query_text: str, max_context_length: int = 2000
    ) -> str:
        """
        Retrieve and format context for an LLM query.

        Args:
            query_text: User query.
            max_context_length: Maximum length of context.

        Returns:
            Formatted context string.
        """
        graph_query = GraphQuery(
            query_text=query_text,
            max_hops=2,
            max_results=10,
            use_semantic_search=True,
        )

        result = await self.retrieve(graph_query)

        # Format context
        context_parts = []

        # Add entities
        if result.entities:
            context_parts.append("## Relevant Entities\n")
            for entity in result.entities[:5]:
                context_parts.append(
                    f"- **{entity.name}** ({entity.type}): {entity.description}"
                )
                if entity.properties:
                    key_props = list(entity.properties.items())[:2]
                    for k, v in key_props:
                        context_parts.append(f"  - {k}: {v}")

        # Add relationships
        if result.relationships:
            context_parts.append("\n## Relationships\n")
            for rel in result.relationships[:5]:
                source = self.graph.get_entity(rel.source_id)
                target = self.graph.get_entity(rel.target_id)
                if source and target:
                    context_parts.append(
                        f"- {source.name} **{rel.type.replace('_', ' ')}** {target.name}"
                    )
                    if rel.description:
                        context_parts.append(f"  {rel.description}")

        context = "\n".join(context_parts)

        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        return context

    def get_entity_context(self, entity_id: str, include_neighbors: bool = True) -> str:
        """
        Get formatted context for a specific entity.

        Args:
            entity_id: Entity ID.
            include_neighbors: Whether to include neighbor entities.

        Returns:
            Formatted context string.
        """
        entity = self.graph.get_entity(entity_id)
        if not entity:
            return ""

        context_parts = [
            f"# {entity.name}",
            f"**Type**: {entity.type}",
            f"**Description**: {entity.description}",
        ]

        if entity.properties:
            context_parts.append("\n**Properties**:")
            for k, v in entity.properties.items():
                context_parts.append(f"- {k}: {v}")

        if include_neighbors and entity_id in self.graph.nodes:
            neighbors = self.graph.get_neighbors(entity_id, max_distance=1)
            if neighbors:
                context_parts.append("\n**Connected to**:")
                for neighbor_id in list(neighbors)[:5]:
                    neighbor = self.graph.get_entity(neighbor_id)
                    if neighbor:
                        context_parts.append(f"- {neighbor.name} ({neighbor.type})")

        return "\n".join(context_parts)
