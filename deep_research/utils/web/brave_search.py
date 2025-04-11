"""
Brave Search API client for web searches.
"""

from typing import Dict, Optional

import aiohttp

from ...models import SearchResult, WebSearchItem
from .base_search import BaseSearchClient


class BraveSearchClient(BaseSearchClient):
    """
    Client for interacting with the Brave Search API.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, country: str = "US"):
        """
        Initialize the Brave Search client.

        Args:
            api_key (str): Brave Search API key.
            country (str, optional): Country code for search results. Defaults to "US".
        """
        self.api_key = api_key
        self.country = country
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }

    async def search(
        self, query: str, max_results: int = 10, search_params: Optional[Dict] = None
    ) -> SearchResult:
        """
        Search for web pages using the Brave Search API.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.
            search_params (Optional[Dict], optional): Additional search parameters. Defaults to None.

        Returns:
            SearchResult: The search results.
        """
        try:
            # Prepare parameters
            params = {
                "q": query,
                "count": max_results,
                "country": self.country,
                "search_lang": "en",
                "extra_snippets": True,  # Include extra snippets for more context
                "related": True,  # Include related queries
                "rich_data": True,  # Include additional data like entity information
            }

            # Add any additional parameters
            if search_params:
                params.update(search_params)

            # Execute search request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, headers=self.headers, params=params
                ) as response:
                    if response.status != 200:
                        return SearchResult(
                            success=False,
                            error=f"Brave Search API returned status code {response.status}",
                        )

                    data = await response.json()

                    # Check for errors in the response
                    if "error" in data:
                        return SearchResult(
                            success=False,
                            error=f"Brave Search API error: {data['error']}",
                        )

                    # Format the search results using our Pydantic model
                    formatted_results = []
                    if "web" in data and "results" in data["web"]:
                        for result in data["web"]["results"]:
                            try:
                                # Get extra snippets if available
                                extra_snippets = ""
                                if "extra_snippets" in result:
                                    extra_snippets = " ".join(result["extra_snippets"])

                                # Combine regular description with extra snippets
                                full_description = result.get("description", "")
                                if extra_snippets:
                                    full_description = (
                                        f"{full_description} {extra_snippets}"
                                    )

                                search_item = WebSearchItem(
                                    url=result.get("url", ""),
                                    title=result.get("title", ""),
                                    description=full_description,
                                    relevance=result.get("relevance", 1.0),
                                    provider="brave",
                                    date=result.get("published_date", ""),
                                )
                                formatted_results.append(search_item)
                            except Exception as e:
                                # Skip invalid results
                                print(f"Error processing search result: {str(e)}")
                                continue

                    # Add related searches if available
                    if "related" in data and "results" in data["related"]:
                        for related_result in data["related"]["results"]:
                            try:
                                # Create a search result for each related query with a slightly lower relevance
                                search_item = WebSearchItem(
                                    url=f"https://search.brave.com/search?q={related_result.get('query', '').replace(' ', '+')}",
                                    title=f"Related search: {related_result.get('query', '')}",
                                    description=f"A related search query for additional information on {query}",
                                    relevance=0.7,  # Lower relevance for related searches
                                    provider="brave_related",
                                    date="",
                                )
                                formatted_results.append(search_item)
                            except Exception:
                                # Skip invalid results
                                continue

                    return SearchResult(success=True, data=formatted_results)

        except Exception as e:
            return SearchResult(success=False, error=f"Brave search failed: {str(e)}")
