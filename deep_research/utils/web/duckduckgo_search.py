"""
DuckDuckGo search client for web searches as a fallback when Brave API is not available.
"""

import asyncio
from typing import Dict, List, Optional

from duckduckgo_search import DDGS

from ...models import SearchResult, WebSearchItem
from .base_search import BaseSearchClient


class DuckDuckGoSearchClient(BaseSearchClient):
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

    async def search(
        self, query: str, max_results: int = 10, search_params: Optional[Dict] = None
    ) -> SearchResult:
        """
        Search for web pages using DuckDuckGo.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.
            search_params (Optional[Dict], optional): Additional search parameters. Defaults to None.

        Returns:
            SearchResult: The search results.
        """
        try:
            # DuckDuckGo search is synchronous, so we need to run it in a thread
            loop = asyncio.get_event_loop()

            # Create a function to run in the thread pool
            def run_search():
                results_data = []

                with DDGS() as ddgs:
                    # Get main search results
                    main_results = list(
                        ddgs.text(
                            query,
                            region=self.region,
                            safesearch=self.safesearch,
                            max_results=max_results,
                        )
                    )
                    results_data.extend(main_results)

                    # Get related searches by looking at the "suggestions" field
                    # First, try to get related searches directly
                    try:
                        related_queries = ddgs.suggestions(query)
                        if related_queries:
                            for related in related_queries[
                                :5
                            ]:  # Limit to top 5 related queries
                                # Create a pseudo-result for the related query
                                related_result = {
                                    "href": f"https://duckduckgo.com/?q={related.replace(' ', '+')}",
                                    "title": f"Related search: {related}",
                                    "body": f"A related search query for additional information on {query}",
                                    "published_date": "",
                                    "is_related": True,
                                }
                                results_data.append(related_result)
                    except Exception:
                        # If getting suggestions fails, try with a slightly modified query
                        pass

                    # Get some results from a small variation of the query for more breadth
                    if len(query.split()) > 2:
                        words = query.split()
                        # Try with just the first few words
                        shortened_query = " ".join(words[:2]) + " information"
                        try:
                            additional_results = list(
                                ddgs.text(
                                    shortened_query,
                                    region=self.region,
                                    safesearch=self.safesearch,
                                    max_results=3,  # Just a few additional results
                                )
                            )
                            # Mark these as additional context
                            for res in additional_results:
                                res["is_additional"] = True
                            results_data.extend(additional_results)
                        except Exception:
                            pass

                    return results_data

            # Run the search in a thread pool to avoid blocking
            search_results = await loop.run_in_executor(None, run_search)

            # Format the results using our Pydantic model
            formatted_results: List[WebSearchItem] = []

            for result in search_results:
                try:
                    # Determine the relevance score based on the type of result
                    relevance = 1.0  # Default for main results
                    provider = "duckduckgo"

                    if result.get("is_related", False):
                        relevance = 0.7  # Lower relevance for related searches
                        provider = "duckduckgo_related"
                    elif result.get("is_additional", False):
                        relevance = (
                            0.8  # Slightly lower relevance for additional context
                        )
                        provider = "duckduckgo_additional"

                    # Create a description that indicates the type of result
                    description = result.get("body", "")
                    if result.get("is_additional", False):
                        description = f"[Additional context] {description}"

                    search_item = WebSearchItem(
                        url=result.get("href", ""),
                        title=result.get("title", ""),
                        description=description,
                        relevance=relevance,
                        provider=provider,
                        date=result.get("published_date", ""),
                    )
                    formatted_results.append(search_item)
                except Exception as e:
                    # Skip invalid results
                    print(f"Error processing DDG result: {str(e)}")
                    continue

            return SearchResult(success=True, data=formatted_results)

        except Exception as e:
            return SearchResult(
                success=False, error=f"DuckDuckGo search failed: {str(e)}"
            )
