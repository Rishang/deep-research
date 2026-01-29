"""
Web search and web content retrieval modules.
"""

from .base_search import BaseSearchClient
from .search_provider import WebSearchProvider

# Optional imports - these may fail if dependencies aren't installed
try:
    from .brave_search import BraveSearchClient
except ImportError:
    BraveSearchClient = None  # type: ignore

try:
    from .duckduckgo_search import DuckDuckGoSearchClient
except ImportError:
    DuckDuckGoSearchClient = None  # type: ignore

__all__ = [
    "BaseSearchClient",
    "BraveSearchClient",
    "DuckDuckGoSearchClient",
    "WebSearchProvider",
]
