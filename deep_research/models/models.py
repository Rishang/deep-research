"""
Pydantic models used in the Deep Research SDK.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl


class ActivityStatus(str, Enum):
    """Status of a research activity."""

    PENDING = "pending"
    COMPLETE = "complete"
    ERROR = "error"


class ActivityType(str, Enum):
    """Type of research activity."""

    SEARCH = "search"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"
    THOUGHT = "thought"


class ActivityItem(BaseModel):
    """A single activity item in the research process."""

    type: ActivityType
    status: ActivityStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    depth: Optional[int] = None


class SourceItem(BaseModel):
    """A source of information used in the research."""

    url: HttpUrl | str
    title: str
    relevance: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None


class WebSearchItem(BaseModel):
    """A standard format for search results from any provider."""

    url: HttpUrl | str
    title: str
    description: str = ""
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    provider: str = "unknown"  # This should match a value from WebSearchProvider enum
    date: str = ""


class SearchResult(BaseModel):
    """Result from a search operation."""

    success: bool
    data: Optional[List[WebSearchItem]] = None
    error: Optional[str] = None


class ExtractResult(BaseModel):
    """Result from an extraction operation."""

    success: bool
    data: Optional[Union[str, List[Dict]]] = None
    error: Optional[str] = None


class AnalysisResult(BaseModel):
    """Result from an analysis operation."""

    summary: str
    gaps: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    should_continue: bool = True
    next_search_topic: Optional[str] = None
    url_to_search: Optional[str] = None


class ResearchState(BaseModel):
    """Current state of research process."""

    findings: List[Dict[str, str]] = Field(default_factory=list)
    summaries: List[str] = Field(default_factory=list)
    sources: List[SourceItem] = Field(default_factory=list)
    next_search_topic: str = ""
    url_to_search: str = ""
    current_depth: int = 0
    failed_attempts: int = 0
    max_failed_attempts: int = 3
    completed_steps: int = 0
    total_expected_steps: int = 0


class ResearchResult(BaseModel):
    """Final result of the research process."""

    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
