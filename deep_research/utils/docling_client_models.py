"""
Pydantic models for Docling client cache parameters.
"""

from typing import Optional

from pydantic import BaseModel


class SearchParams(BaseModel):
    """Parameters for search operation caching."""

    query: str
    max_results: int = 10
    provider: Optional[str] = None


class ExtractParams(BaseModel):
    """Parameters for URL extraction caching."""

    url: str
    prompt: str = ""


class ScrapeParams(BaseModel):
    """Parameters for URL scraping caching."""

    url: str
