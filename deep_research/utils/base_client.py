"""
Base client interface for web crawling, scraping and information extraction.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from pydantic import HttpUrl

from ..models import ExtractResult, SearchResult
from .cache import CacheConfig, init_cache


class BaseWebClient(ABC):
    """
    Abstract base class for web clients that provide search and extraction functionality.
    All implementations (Docling, DoclingServer, Firecrawl) should inherit from this.
    """

    def __init__(
        self,
        max_concurrent_requests: int = 5,
        cache_config: Optional[CacheConfig] = None,
        page_content_max_chars: int = 8000,
    ):
        """
        Initialize the base web client.

        Args:
            max_concurrent_requests (int, optional): Maximum number of concurrent requests.
                Defaults to 5.
            cache_config (Optional[CacheConfig], optional): Configuration for cache system.
                If None, caching is disabled by default.
            page_content_max_chars (int, optional): Maximum number of characters to return in page content.
                Defaults to 8000.
        """
        self.page_content_max_chars = page_content_max_chars
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Setup cache if provided
        if cache_config:
            init_cache(cache_config)
            self.cache_enabled = cache_config.enabled
        else:
            # Default to disabled cache
            self.cache_enabled = False
            # But initialize with default config in case it's enabled later
            init_cache(CacheConfig(enabled=False))

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for web pages using the provided query.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to return.
                Defaults to 10.

        Returns:
            SearchResult: The search results.
        """
        pass

    @abstractmethod
    async def extract(
        self, urls: List[Union[str, HttpUrl]], prompt: str
    ) -> ExtractResult:
        """
        Extract information from the provided URLs based on the prompt.

        Args:
            urls (List[Union[str, HttpUrl]]): URLs to extract information from.
            prompt (str): Description of the information to extract.

        Returns:
            ExtractResult: The extracted information.
        """
        pass

    @abstractmethod
    async def scrape_url(self, url: Union[str, HttpUrl]) -> ExtractResult:
        """
        Scrape content from a specific URL.

        Args:
            url (Union[str, HttpUrl]): URL to scrape.

        Returns:
            ExtractResult: The scraped content.
        """
        pass

    @abstractmethod
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
        pass
