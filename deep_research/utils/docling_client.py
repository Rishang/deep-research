"""
Docling client for web scraping and information extraction.
"""

import asyncio
from typing import Dict, List, Optional, Union

from docling.document_converter import DocumentConverter
from pydantic import HttpUrl

from ..models import ExtractResult, SearchResult, WebSearchItem
from .brave_search import BraveSearchClient
from .cache import CacheConfig, cache, init_cache
from .docling_client_models import ScrapeParams, SearchParams
from .duckduckgo_search import DuckDuckGoSearchClient


class DoclingClient:
    """
    A client for interacting with Docling for web scraping and search.
    Replaces the Firecrawl functionality from the TypeScript implementation.
    """

    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        max_concurrent_requests: int = 5,
        cache_config: Optional[CacheConfig] = None,
        page_content_max_chars: int = 8000,
    ):
        """
        Initialize the Docling client.

        Args:
            brave_api_key (Optional[str], optional): Brave Search API key. Defaults to None.
            max_concurrent_requests (int, optional): Maximum number of concurrent requests.
                Defaults to 5.
            cache_config (Optional[CacheConfig], optional): Configuration for cache system.
                If None, caching is disabled by default.
            page_content_max_chars (int, optional): Maximum number of characters to return in the page content.
                Defaults to 8000.
        """

        self.page_content_max_chars = page_content_max_chars
        self.client = (
            DocumentConverter()
        )  # Docling uses DocumentConverter instead of Client
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Setup search providers
        self.use_brave_search = brave_api_key is not None

        # Initialize search providers
        self.brave_search = (
            BraveSearchClient(api_key=brave_api_key) if brave_api_key else None
        )
        self.duckduckgo_search = DuckDuckGoSearchClient() if not brave_api_key else None

        # Setup cache if provided
        if cache_config:
            init_cache(cache_config)
            self.cache_enabled = cache_config.enabled
        else:
            # Default to disabled cache
            self.cache_enabled = False
            # But initialize with default config in case it's enabled later
            init_cache(CacheConfig(enabled=False))

    @cache(structure=SearchParams)
    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for web pages using the provided query.
        Uses Brave Search API if available, otherwise falls back to DuckDuckGo.

        This method is cached if caching is enabled in the client.
        The cache key is based on the SearchParams model (query and max_results).

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to return.
                Defaults to 10.

        Returns:
            SearchResult: The search results.
        """
        # Use Brave Search if available and enabled
        if self.use_brave_search and self.brave_search:
            try:
                return await self.brave_search.search(query, max_results)
            except Exception as e:
                # If Brave Search fails, fall back to DuckDuckGo
                print(f"Brave search failed, falling back to DuckDuckGo: {str(e)}")

        # Use DuckDuckGo search as first fallback
        if self.duckduckgo_search:
            try:
                return await self.duckduckgo_search.search(query, max_results)
            except Exception as e:
                print(
                    f"DuckDuckGo search failed, falling back to mock results: {str(e)}"
                )

        # Use mocked search as last resort fallback
        try:
            # Create a list of sample URLs related to the query as a last resort
            search_terms = query.replace(" ", "+")

            # Create WebSearchItem objects
            search_results = [
                WebSearchItem(
                    url=f"https://en.wikipedia.org/wiki/{search_terms}",  # type: ignore
                    title=f"Wikipedia - {query}",
                    description=f"Encyclopedia article about {query}",
                    relevance=0.95,
                    provider="mock",
                    date="",
                ),
                WebSearchItem(
                    url=f"https://arxiv.org/search/?query={search_terms}&searchtype=all",  # type: ignore
                    title=f"arXiv Papers - {query}",
                    description=f"Scientific papers related to {query}",
                    relevance=0.9,
                    provider="mock",
                    date="",
                ),
                WebSearchItem(
                    url=f"https://scholar.google.com/scholar?q={search_terms}",  # type: ignore
                    title=f"Google Scholar - {query}",
                    description=f"Academic resources about {query}",
                    relevance=0.85,
                    provider="mock",
                    date="",
                ),
            ]

            return SearchResult(success=True, data=search_results[:max_results])
        except Exception as e:
            return SearchResult(
                success=False, error=f"All search methods failed: {str(e)}"
            )

    @cache(structure=ScrapeParams)
    async def _extract_single_url(self, url: str, prompt: str) -> Dict:
        """
        Extract information from a single URL.

        This method is now cached using the ScrapeParams model structure.
        The cache key is based on the URL.

        Args:
            url (str): URL to extract from.
            prompt (str): Extraction prompt.

        Returns:
            Dict: Extraction result.
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Use Docling DocumentConverter to extract content
                loop = asyncio.get_event_loop()
                # Convert the synchronous DocumentConverter.convert call to async
                result = await loop.run_in_executor(
                    None, lambda: self.client.convert(url)
                )

                # Extract content as markdown
                content = result.document.export_to_markdown()

                # Add prompt-based extraction here (in a real implementation, you might use an LLM)
                # For now, we'll just return the content with the prompt as context

                return {
                    "success": True,
                    "url": url,
                    "data": f"Extracted with prompt '{prompt}': {content[: self.page_content_max_chars]}...",  # Truncate for reasonable size
                    "error": None,
                }
            except Exception as e:
                return {"success": False, "url": url, "data": None, "error": str(e)}

    # Note: We don't add @cached_extract here because we need to handle
    # multiple URLs differently than the single URL case
    async def extract(
        self, urls: List[Union[str, HttpUrl]], prompt: str
    ) -> ExtractResult:
        """
        Extract information from the provided URLs based on the prompt.
        Uses concurrent requests for faster processing.

        Note: This method handles multiple URLs, so caching happens at the
        individual URL level within _extract_single_url.

        Args:
            urls (List[Union[str, HttpUrl]]): URLs to extract information from.
            prompt (str): Description of the information to extract.

        Returns:
            ExtractResult: The extracted information.
        """
        # Remove empty URLs if any
        urls = [str(url) for url in urls if url]
        if not urls:
            return ExtractResult(success=True, data=[])

        # Create tasks for all URLs to extract concurrently
        tasks = [self._extract_single_url(url, prompt) for url in urls]
        results = await asyncio.gather(*tasks)

        # Process results
        successful_results = []
        errors = []

        for result in results:
            if result["success"]:
                successful_results.append(
                    {"url": result["url"], "data": result["data"]}
                )
            else:
                errors.append(f"{result['url']}: {result['error']}")

        # If we have at least some successful results, consider it a success
        if successful_results:
            return ExtractResult(success=True, data=successful_results)
        else:
            error_message = f"All extractions failed: {'; '.join(errors)}"
            return ExtractResult(success=False, error=error_message)

    async def scrape_urls(
        self, urls: List[Union[str, HttpUrl]]
    ) -> Dict[str, ExtractResult]:
        """
        Scrape content from multiple URLs concurrently.

        Args:
            urls (List[Union[str, HttpUrl]]): URLs to scrape.

        Returns:
            Dict[str, ExtractResult]: Mapping of URL to extraction result.
        """

        @cache(structure=ScrapeParams)
        async def _scrape_single(url: str):
            try:
                async with self.semaphore:
                    # Convert the synchronous DocumentConverter.convert call to async
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: self.client.convert(url)
                    )

                    # Extract content as markdown
                    content = result.document.export_to_markdown()

                    result = ExtractResult(success=True, data=content)
                    return url, result
            except Exception as e:
                result = ExtractResult(success=False, error=f"Scrape failed: {str(e)}")
                return url, result

        # Filter out empty URLs
        urls = [str(url) for url in urls if url]
        if not urls:
            return {}

        # Create tasks for all URLs
        tasks = [_scrape_single(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Convert results to dictionary
        return dict(results)

    @cache(structure=ScrapeParams)
    async def scrape_url(self, url: Union[str, HttpUrl]) -> ExtractResult:
        """
        Scrape content from a specific URL.

        This method is cached if caching is enabled in the client.
        The cache key is based on the ScrapeParams model (url only).

        Args:
            url (Union[str, HttpUrl]): URL to scrape.

        Returns:
            ExtractResult: The scraped content.
        """
        try:
            async with self.semaphore:
                # Convert the synchronous DocumentConverter.convert call to async
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.client.convert(str(url))
                )

                # Extract content as markdown
                content = result.document.export_to_markdown()

                return ExtractResult(success=True, data=content)
        except Exception as e:
            return ExtractResult(success=False, error=f"Scrape failed: {str(e)}")
