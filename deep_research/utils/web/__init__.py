"""
Web search and web content retrieval modules.
"""

from .base_search import BaseSearchClient
from .brave_search import BraveSearchClient
from .duckduckgo_search import DuckDuckGoSearchClient
from .search_provider import WebSearchProvider

__all__ = [
    "BaseSearchClient",
    "BraveSearchClient",
    "DuckDuckGoSearchClient",
    "WebSearchProvider",
]
