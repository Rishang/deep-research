"""
DuckDuckGo search client for web searches as a fallback when Brave API is not available.
"""

import asyncio
from typing import List

from duckduckgo_search import DDGS

from ..models import SearchResult, WebSearchItem


class DuckDuckGoSearchClient:
    """
    Client for using DuckDuckGo search.
    Used as a fallback when Brave Search API key is not provided.
    """

    def __init__(self, region: str = "us-en", safesearch: str = "moderate"):
        """
        Initialize the DuckDuckGo Search client.

        Args:
            region (str, optional): Region for search results. Defaults to "us-en".
            safesearch (str, optional): SafeSearch option. Defaults to "moderate".
        """
        self.region = region
        self.safesearch = safesearch

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for web pages using DuckDuckGo.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            SearchResult: The search results.
        """
        try:
            # DuckDuckGo search is synchronous, so we need to run it in a thread
            loop = asyncio.get_event_loop()

            # Create a function to run in the thread pool
            def run_search():
                with DDGS() as ddgs:
                    ddg_results = list(
                        ddgs.text(
                            query,
                            region=self.region,
                            safesearch=self.safesearch,
                            max_results=max_results,
                        )
                    )
                    return ddg_results

            # Run the search in a thread pool to avoid blocking
            search_results = await loop.run_in_executor(None, run_search)

            # Format the results using our Pydantic model
            formatted_results: List[WebSearchItem] = []

            for result in search_results:
                try:
                    search_item = WebSearchItem(
                        url=result.get("href", ""),
                        title=result.get("title", ""),
                        description=result.get("body", ""),
                        relevance=1.0,  # DuckDuckGo doesn't provide relevance score
                        provider="duckduckgo",
                        date=result.get("published_date", ""),
                    )
                    formatted_results.append(search_item)
                except Exception:
                    # Skip invalid results
                    continue

            return SearchResult(success=True, data=formatted_results)

        except Exception as e:
            return SearchResult(
                success=False, error=f"DuckDuckGo search failed: {str(e)}"
            )
