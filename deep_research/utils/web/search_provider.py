"""
Enum for web search providers.
"""

from enum import Enum


class WebSearchProvider(str, Enum):
    """
    Enumeration of web search providers.
    """

    BRAVE = "brave"
    DUCKDUCKGO = "duckduckgo"
    MOCK = "mock"
