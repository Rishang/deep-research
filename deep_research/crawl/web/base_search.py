"""
Base search client interface for all web search providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ...models import SearchResult


class BaseSearchClient(ABC):
    """
    Abstract base class for web search clients.
    """

    @abstractmethod
    async def search(
        self, query: str, max_results: int = 10, search_params: Optional[Dict] = None
    ) -> SearchResult:
        """
        Search for web pages using the provider's API.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.
            search_params (Optional[Dict], optional): Additional search parameters. Defaults to None.

        Returns:
            SearchResult: The search results.
        """
        pass
