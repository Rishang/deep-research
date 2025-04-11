"""Utility functions for the Deep Research SDK."""

from .base_client import BaseWebClient
from .docling_client import DoclingClient
from .docling_server_client import DoclingServerClient
from .firecrawl_client import FirecrawlClient

# Re-export web module contents
from .web import (
    BraveSearchClient,
    DuckDuckGoSearchClient,
    BaseSearchClient,
    WebSearchProvider,
)

__all__ = [
    "BaseWebClient",
    "DoclingClient",
    "DoclingServerClient",
    "FirecrawlClient",
    "BaseSearchClient",
    "BraveSearchClient",
    "DuckDuckGoSearchClient",
    "WebSearchProvider",
]
