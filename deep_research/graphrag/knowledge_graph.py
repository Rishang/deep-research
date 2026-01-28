"""
Knowledge Graph management and operations.
"""

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import (
    CommunityInfo,
    Entity,
    EntityType,
    GraphEdge,
    GraphMemory,
    GraphNode,
    Relationship,
)


class KnowledgeGraph:
    """
    In-memory knowledge graph for storing and querying research findings.
    """

    def __init__(self):
        """Initialize the knowledge graph."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # Type -> entity IDs
        self.communities: Dict[int, CommunityInfo] = {}

    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph.

        Args:
            entity: Entity to add.
        """
        if entity.id in self.nodes:
            # Merge with existing entity
            existing = self.nodes[entity.id].entity
            existing.description = (
                entity.description
                if len(entity.description) > len(existing.description)
                else existing.description
            )
            existing.sources.extend(
                [s for s in entity.sources if s not in existing.sources]
            )
            existing.properties.update(entity.properties)
            existing.confidence = max(existing.confidence, entity.confidence)
            existing.updated_at = datetime.now()
        else:
            # Add new entity
            node = GraphNode(entity=entity)
            self.nodes[entity.id] = node
            self.entity_index[entity.type].add(entity.id)

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the graph.

        Args:
            relationship: Relationship to add.
        """
        # Ensure entities exist
        if (
            relationship.source_id not in self.nodes
            or relationship.target_id not in self.nodes
        ):
            return

        # Add edge
        if relationship.id not in self.edges:
            self.edges[relationship.id] = GraphEdge(relationship=relationship)

        # Update node neighbors
        self.nodes[relationship.source_id].neighbors.add(relationship.target_id)
        self.nodes[relationship.target_id].neighbors.add(relationship.source_id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        node = self.nodes.get(entity_id)
        return node.entity if node else None

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_index.get(entity_type, set())
        return [self.nodes[eid].entity for eid in entity_ids if eid in self.nodes]

    def find_path(
        self, start_id: str, end_id: str, max_hops: int = 3
    ) -> Optional[List[str]]:
        """
        Find path between two entities using BFS.

        Args:
            start_id: Starting entity ID.
            end_id: Target entity ID.
            max_hops: Maximum path length.

        Returns:
            List of entity IDs in path, or None if no path found.
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        if start_id == end_id:
            return [start_id]

        visited = {start_id}
        queue = deque([(start_id, [start_id])])

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_hops:
                continue

            for neighbor_id in self.nodes[current_id].neighbors:
                if neighbor_id == end_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_neighbors(self, entity_id: str, max_distance: int = 1) -> Set[str]:
        """
        Get all neighbors within max_distance hops.

        Args:
            entity_id: Entity ID.
            max_distance: Maximum distance in hops.

        Returns:
            Set of neighbor entity IDs.
        """
        if entity_id not in self.nodes:
            return set()

        if max_distance == 1:
            return self.nodes[entity_id].neighbors

        visited = {entity_id}
        current_level = {entity_id}

        for _ in range(max_distance):
            next_level = set()
            for node_id in current_level:
                for neighbor_id in self.nodes[node_id].neighbors:
                    if neighbor_id not in visited:
                        next_level.add(neighbor_id)
                        visited.add(neighbor_id)
            current_level = next_level

        visited.remove(entity_id)  # Remove starting node
        return visited

    def get_subgraph(self, entity_ids: Set[str]) -> Dict[str, any]:
        """
        Extract a subgraph containing specified entities and their connections.

        Args:
            entity_ids: Set of entity IDs to include.

        Returns:
            Subgraph representation.
        """
        subgraph_entities = {}
        subgraph_relationships = {}

        for entity_id in entity_ids:
            if entity_id in self.nodes:
                subgraph_entities[entity_id] = self.nodes[entity_id].entity

        for edge_id, edge in self.edges.items():
            rel = edge.relationship
            if rel.source_id in entity_ids and rel.target_id in entity_ids:
                subgraph_relationships[edge_id] = rel

        return {
            "entities": subgraph_entities,
            "relationships": subgraph_relationships,
        }

    def calculate_pagerank(self, iterations: int = 20, damping: float = 0.85) -> None:
        """
        Calculate PageRank scores for all entities.

        Args:
            iterations: Number of iterations.
            damping: Damping factor.
        """
        n = len(self.nodes)
        if n == 0:
            return

        # Initialize PageRank scores
        initial_score = 1.0 / n
        for node in self.nodes.values():
            node.pagerank = initial_score

        # Iterative calculation
        for _ in range(iterations):
            new_scores = {}

            for entity_id, node in self.nodes.items():
                rank_sum = 0.0
                for neighbor_id in node.neighbors:
                    neighbor_node = self.nodes[neighbor_id]
                    neighbor_degree = len(neighbor_node.neighbors)
                    if neighbor_degree > 0:
                        rank_sum += neighbor_node.pagerank / neighbor_degree

                new_scores[entity_id] = (1 - damping) / n + damping * rank_sum

            for entity_id, score in new_scores.items():
                self.nodes[entity_id].pagerank = score

    def detect_communities_simple(self) -> Dict[int, List[str]]:
        """
        Simple community detection using connected components.

        Returns:
            Dictionary mapping community ID to list of entity IDs.
        """
        visited = set()
        communities = {}
        community_id = 0

        for entity_id in self.nodes:
            if entity_id not in visited:
                # BFS to find connected component
                component = []
                queue = deque([entity_id])
                visited.add(entity_id)

                while queue:
                    current_id = queue.popleft()
                    component.append(current_id)

                    for neighbor_id in self.nodes[current_id].neighbors:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append(neighbor_id)

                communities[community_id] = component
                for eid in component:
                    self.nodes[eid].community_id = community_id

                community_id += 1

        # Create community info
        for comm_id, entity_ids in communities.items():
            # Find central entities (highest PageRank)
            sorted_entities = sorted(
                entity_ids,
                key=lambda eid: self.nodes[eid].pagerank,
                reverse=True,
            )
            central = sorted_entities[:3]

            self.communities[comm_id] = CommunityInfo(
                community_id=comm_id,
                entity_ids=entity_ids,
                central_entities=central,
                size=len(entity_ids),
            )

        return communities

    def get_most_important_entities(self, top_n: int = 10) -> List[Entity]:
        """
        Get most important entities based on PageRank.

        Args:
            top_n: Number of entities to return.

        Returns:
            List of most important entities.
        """
        sorted_nodes = sorted(
            self.nodes.values(), key=lambda n: n.pagerank, reverse=True
        )
        return [node.entity for node in sorted_nodes[:top_n]]

    def to_dict(self) -> Dict:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": {
                eid: {
                    "entity": node.entity.dict(),
                    "neighbors": list(node.neighbors),
                    "community_id": node.community_id,
                    "pagerank": node.pagerank,
                }
                for eid, node in self.nodes.items()
            },
            "edges": {
                eid: {"relationship": edge.relationship.dict()}
                for eid, edge in self.edges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        graph = cls()

        # Restore nodes
        for entity_id, node_data in data.get("nodes", {}).items():
            entity = Entity(**node_data["entity"])
            node = GraphNode(
                entity=entity,
                neighbors=set(node_data.get("neighbors", [])),
                community_id=node_data.get("community_id"),
                pagerank=node_data.get("pagerank", 0.0),
            )
            graph.nodes[entity_id] = node
            graph.entity_index[entity.type].add(entity_id)

        # Restore edges
        for edge_id, edge_data in data.get("edges", {}).items():
            relationship = Relationship(**edge_data["relationship"])
            graph.edges[edge_id] = GraphEdge(relationship=relationship)

        return graph


class KnowledgeGraphManager:
    """
    Manages knowledge graph persistence and high-level operations.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the knowledge graph manager.

        Args:
            storage_path: Path to store graph data.
        """
        self.storage_path = storage_path or Path.home() / ".deep_research" / "graphs"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_graph: Optional[KnowledgeGraph] = None
        self.current_memory: Optional[GraphMemory] = None

    def create_graph(self, session_id: str, research_topic: str) -> KnowledgeGraph:
        """
        Create a new knowledge graph for a research session.

        Args:
            session_id: Unique session identifier.
            research_topic: Topic of research.

        Returns:
            New knowledge graph.
        """
        self.current_graph = KnowledgeGraph()
        self.current_memory = GraphMemory(
            session_id=session_id, research_topic=research_topic
        )
        return self.current_graph

    def save_graph(self, session_id: str) -> None:
        """
        Save the current graph to disk.

        Args:
            session_id: Session identifier.
        """
        if not self.current_graph or not self.current_memory:
            return

        graph_data = self.current_graph.to_dict()
        memory_data = self.current_memory.dict()
        memory_data["graph_data"] = graph_data

        file_path = self.storage_path / f"{session_id}.json"
        with open(file_path, "w") as f:
            json.dump(memory_data, f, indent=2, default=str)

    def load_graph(self, session_id: str) -> Optional[KnowledgeGraph]:
        """
        Load a graph from disk.

        Args:
            session_id: Session identifier.

        Returns:
            Loaded knowledge graph or None if not found.
        """
        file_path = self.storage_path / f"{session_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        graph_data = data.pop("graph_data", {})
        self.current_graph = KnowledgeGraph.from_dict(graph_data)
        self.current_memory = GraphMemory(**data)

        return self.current_graph

    def list_sessions(self) -> List[str]:
        """List all saved session IDs."""
        return [f.stem for f in self.storage_path.glob("*.json")]
