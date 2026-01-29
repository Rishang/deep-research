"""Crawl utilities for the Deep Research SDK."""

from .base_client import BaseWebClient

# Optional imports - these may fail if dependencies aren't installed
try:
    from .markitdown_client import MarkItDownClient
except ImportError:
    MarkItDownClient = None  # type: ignore

try:
    from .firecrawl_client import FirecrawlClient
except ImportError:
    FirecrawlClient = None  # type: ignore

# Re-export web module contents
from .web import (
    BraveSearchClient,
    DuckDuckGoSearchClient,
    BaseSearchClient,
    WebSearchProvider,
)

__all__ = [
    "BaseWebClient",
    "MarkItDownClient",
    "FirecrawlClient",
    "BaseSearchClient",
    "BraveSearchClient",
    "DuckDuckGoSearchClient",
    "WebSearchProvider",
]
