"""
Tests for the Docling cache functionality.
"""

import asyncio
import datetime
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sqlmodel import Session, select

from deep_research.models import ExtractResult, SearchResult
from deep_research.utils.cache import (
    CacheConfig,
    ExtractCache,
    SearchCache,
    cached_extract,
    cached_search,
    init_cache,
)
from deep_research.utils.cache.cache_models import get_engine


class TestCache(unittest.TestCase):
    """Test the caching functionality."""

    def setUp(self):
        # Create a temporary database for testing
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db_url = f"sqlite:///{self.db_path}"

        # Initialize cache with test config
        self.config = CacheConfig(
            enabled=True,
            ttl_seconds=3600,
            db_url=self.db_url,
            create_tables=True,
        )
        init_cache(self.config)

    def tearDown(self):
        # Clean up the temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_search_cache_model(self):
        """Test the SearchCache model."""
        engine = get_engine(self.db_url)

        # Create a test cache entry
        entry = SearchCache(
            query="test query",
            max_results=10,
            results='{"success": true, "data": [], "error": null}',
            provider="test",
            expires_at=datetime.datetime.now() + datetime.timedelta(hours=1),
        )

        # Add it to the database
        with Session(engine) as session:
            session.add(entry)
            session.commit()
            session.refresh(entry)

            # Verify it was added
            self.assertIsNotNone(entry.id)

            # Query it back
            stmt = select(SearchCache).where(SearchCache.query == "test query")
            result = session.exec(stmt).first()

            # Verify the result
            self.assertEqual(result.query, "test query")
            self.assertEqual(result.max_results, 10)
            self.assertEqual(result.provider, "test")

    def test_extract_cache_model(self):
        """Test the ExtractCache model."""
        engine = get_engine(self.db_url)

        # Create a test cache entry
        entry = ExtractCache(
            url="https://example.com",
            prompt="test prompt",
            content='{"success": true, "data": "test content", "error": null}',
            expires_at=datetime.datetime.now() + datetime.timedelta(hours=1),
        )

        # Add it to the database
        with Session(engine) as session:
            session.add(entry)
            session.commit()
            session.refresh(entry)

            # Verify it was added
            self.assertIsNotNone(entry.id)

            # Query it back
            stmt = select(ExtractCache).where(ExtractCache.url == "https://example.com")
            result = session.exec(stmt).first()

            # Verify the result
            self.assertEqual(result.url, "https://example.com")
            self.assertEqual(result.prompt, "test prompt")

    @patch("deep_research.utils.cache.cache_decorator._serialize_search_result")
    @patch("deep_research.utils.cache.cache_decorator._deserialize_search_result")
    def test_cached_search_decorator(self, mock_deserialize, mock_serialize):
        """Test the cached_search decorator."""
        # Set up the mocks
        mock_serialize.return_value = '{"success": true, "data": [], "error": null}'
        mock_deserialize.return_value = SearchResult(success=True, data=[], error=None)

        # Create an example search function
        @cached_search
        async def example_search(self, query, max_results=10):
            self.called = True
            return SearchResult(success=True, data=[], error=None)

        # Create a mock instance to use with the function
        instance = MagicMock()
        instance.use_brave_search = True
        instance.brave_search = True
        instance.called = False

        # Run the test
        async def run_test():
            # First call should run the function
            result1 = await example_search(instance, "test query")
            self.assertTrue(instance.called)
            self.assertTrue(result1.success)

            # Reset the called flag
            instance.called = False

            # Second call should use the cache
            result2 = await example_search(instance, "test query")
            # The actual function should not be called again
            self.assertFalse(instance.called)
            self.assertTrue(result2.success)

        # Run the async test
        asyncio.run(run_test())

    @patch("deep_research.utils.cache.cache_decorator._serialize_extract_result")
    @patch("deep_research.utils.cache.cache_decorator._deserialize_extract_result")
    def test_cached_extract_decorator(self, mock_deserialize, mock_serialize):
        """Test the cached_extract decorator."""
        # Set up the mocks
        mock_serialize.return_value = (
            '{"success": true, "data": "content", "error": null}'
        )
        mock_deserialize.return_value = ExtractResult(
            success=True, data="content", error=None
        )

        # Create an example extract function
        @cached_extract
        async def example_extract(self, url, prompt=""):
            self.called = True
            return ExtractResult(success=True, data="content", error=None)

        # Create a mock instance to use with the function
        instance = MagicMock()
        instance.called = False

        # Run the test
        async def run_test():
            # First call should run the function
            result1 = await example_extract(instance, "https://example.com")
            self.assertTrue(instance.called)
            self.assertTrue(result1.success)

            # Reset the called flag
            instance.called = False

            # Second call should use the cache
            result2 = await example_extract(instance, "https://example.com")
            # The actual function should not be called again
            self.assertFalse(instance.called)
            self.assertTrue(result2.success)

        # Run the async test
        asyncio.run(run_test())

    def test_cache_disabled(self):
        """Test that cache can be disabled."""
        # Disable cache
        init_cache(CacheConfig(enabled=False))

        # Create an example function to use with the decorator
        @cached_search
        async def example_search(self, query, max_results=10):
            self.called_count += 1
            return SearchResult(success=True, data=[], error=None)

        # Create a mock instance to use with the function
        instance = MagicMock()
        instance.use_brave_search = True
        instance.brave_search = True
        instance.called_count = 0

        # Run the test
        async def run_test():
            # First call should run the function
            await example_search(instance, "test query")
            self.assertEqual(instance.called_count, 1)

            # Second call should also run the function since cache is disabled
            await example_search(instance, "test query")
            self.assertEqual(instance.called_count, 2)

        # Run the async test
        asyncio.run(run_test())

        # Re-enable cache for other tests
        init_cache(self.config)


if __name__ == "__main__":
    unittest.main()
