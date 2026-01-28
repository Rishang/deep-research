"""
GraphRAG data models for knowledge representation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    TECHNOLOGY = "technology"
    EVENT = "event"
    FACT = "fact"
    METRIC = "metric"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    ENABLES = "enables"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    DEVELOPED_BY = "developed_by"
    USED_IN = "used_in"
    INFLUENCES = "influences"
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"


class Entity(BaseModel):
    """An entity in the knowledge graph."""

    id: str
    name: str
    type: EntityType
    description: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)  # URLs of sources
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None  # Vector embedding for semantic search

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class Relationship(BaseModel):
    """A relationship between two entities."""

    id: str
    source_id: str  # Entity ID
    target_id: str  # Entity ID
    type: RelationshipType
    description: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    weight: float = Field(default=1.0, ge=0.0)  # Relationship strength

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class GraphNode(BaseModel):
    """A node in the knowledge graph (wrapper for Entity)."""

    entity: Entity
    neighbors: Set[str] = Field(default_factory=set)  # Connected entity IDs
    community_id: Optional[int] = None  # Community detection result
    pagerank: float = 0.0  # PageRank score
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0


class GraphEdge(BaseModel):
    """An edge in the knowledge graph (wrapper for Relationship)."""

    relationship: Relationship
    traversal_count: int = 0  # How often this edge has been traversed


class GraphMemory(BaseModel):
    """Persistent memory structure for the knowledge graph."""

    session_id: str
    user_id: Optional[str] = None
    research_topic: str
    entities: Dict[str, Entity] = Field(default_factory=dict)
    relationships: Dict[str, Relationship] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}
        arbitrary_types_allowed = True


class GraphQuery(BaseModel):
    """A query for graph-based retrieval."""

    query_text: str
    entity_types: Optional[List[EntityType]] = None
    relationship_types: Optional[List[RelationshipType]] = None
    max_hops: int = 2  # Maximum path length for multi-hop retrieval
    max_results: int = 10
    min_confidence: float = 0.5
    use_semantic_search: bool = True
    include_neighbors: bool = True


class GraphRetrievalResult(BaseModel):
    """Result from graph-based retrieval."""

    entities: List[Entity]
    relationships: List[Relationship]
    subgraph: Dict[str, Any]  # Subgraph structure
    confidence: float
    reasoning_path: List[str] = Field(default_factory=list)  # Path through graph


class CommunityInfo(BaseModel):
    """Information about a detected community in the graph."""

    community_id: int
    entity_ids: List[str]
    central_entities: List[str]  # Most important entities
    description: str = ""
    topics: List[str] = Field(default_factory=list)
    size: int = 0
