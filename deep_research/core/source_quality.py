"""
Source quality assessment module.

Provides methods for evaluating source reliability, authority, and relevance.
"""

from datetime import datetime
from typing import List, Tuple
from urllib.parse import urlparse

from ..models.models import SourceQualityMetrics, WebSearchItem


class SourceQualityAssessor:
    """Assesses the quality of information sources."""

    @staticmethod
    async def assess_source_quality(search_item: WebSearchItem) -> SourceQualityMetrics:
        """
        Assess the quality of a source based on various metrics.

        Args:
            search_item: The search result to assess.

        Returns:
            SourceQualityMetrics with quality scores.
        """
        # Domain authority (based on TLD and known authoritative domains)
        domain_authority = SourceQualityAssessor._calculate_domain_authority(
            str(search_item.url)
        )

        # Recency score (based on date if available)
        recency_score = SourceQualityAssessor._calculate_recency_score(search_item.date)

        # Content depth (based on description length and relevance)
        content_depth = SourceQualityAssessor._calculate_content_depth(search_item)

        # Cross-reference score (will be updated as we find multiple sources)
        cross_reference_score = 0.0

        return SourceQualityMetrics(
            domain_authority=domain_authority,
            recency_score=recency_score,
            content_depth=content_depth,
            citation_count=0,
            cross_reference_score=cross_reference_score,
        )

    @staticmethod
    def _calculate_domain_authority(url: str) -> float:
        """Calculate domain authority score based on the URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # High authority domains
            high_authority = [
                "edu",
                "gov",
                "arxiv.org",
                "scholar.google",
                "ieee.org",
                "acm.org",
                "nature.com",
                "science.org",
                "nih.gov",
                "who.int",
            ]

            # Medium authority domains
            medium_authority = [
                "org",
                "wikipedia.org",
                "medium.com",
                "stackoverflow.com",
                "github.com",
            ]

            # Check for high authority
            for auth_domain in high_authority:
                if auth_domain in domain:
                    return 0.9

            # Check for medium authority
            for auth_domain in medium_authority:
                if auth_domain in domain:
                    return 0.7

            # Default score for other domains
            return 0.5
        except Exception:
            return 0.5

    @staticmethod
    def _calculate_recency_score(date_str: str) -> float:
        """Calculate recency score based on publication date."""
        if not date_str:
            return 0.5  # No date information

        try:
            # Try to parse various date formats
            from dateutil import parser

            date = parser.parse(date_str)
            now = datetime.now()
            days_old = (now - date).days

            # Score based on age
            if days_old < 30:  # Less than a month
                return 1.0
            elif days_old < 90:  # Less than 3 months
                return 0.9
            elif days_old < 180:  # Less than 6 months
                return 0.8
            elif days_old < 365:  # Less than a year
                return 0.7
            elif days_old < 730:  # Less than 2 years
                return 0.6
            else:
                return 0.5
        except Exception:
            return 0.5  # Can't parse date

    @staticmethod
    def _calculate_content_depth(search_item: WebSearchItem) -> float:
        """Calculate content depth score based on description."""
        description = search_item.description
        if not description:
            return 0.5

        # Longer descriptions generally indicate more detailed content
        length = len(description)
        if length > 300:
            return 0.9
        elif length > 200:
            return 0.8
        elif length > 100:
            return 0.7
        else:
            return 0.6

    @staticmethod
    def calculate_diversity_score(
        result: WebSearchItem, existing_results: List[Tuple]
    ) -> float:
        """
        Calculate diversity bonus for selecting sources from different domains.

        Args:
            result: The search result to evaluate.
            existing_results: List of already selected (result, score, quality) tuples.

        Returns:
            Diversity score between 0.0 and 1.0.
        """
        try:
            parsed = urlparse(str(result.url))
            domain = parsed.netloc.lower()

            # Count how many existing results are from the same domain
            same_domain_count = 0
            for existing_result, _, _ in existing_results:
                existing_domain = urlparse(str(existing_result.url)).netloc.lower()
                if domain == existing_domain:
                    same_domain_count += 1

            # Penalize multiple sources from same domain
            if same_domain_count == 0:
                return 1.0
            elif same_domain_count == 1:
                return 0.7
            elif same_domain_count == 2:
                return 0.4
            else:
                return 0.2
        except Exception:
            return 0.5

    @staticmethod
    def calculate_gap_relevance(
        result: WebSearchItem, priority_gaps: List[str]
    ) -> float:
        """
        Calculate how relevant a result is to current knowledge gaps.

        Args:
            result: The search result to evaluate.
            priority_gaps: List of priority knowledge gaps.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        if not priority_gaps:
            return 1.0  # No gaps specified, all results equally relevant

        # Check if title or description contains keywords from gaps
        text = (result.title + " " + result.description).lower()

        relevance_scores = []
        for gap in priority_gaps:
            gap_keywords = gap.lower().split()
            matches = sum(1 for keyword in gap_keywords if keyword in text)
            score = min(matches / len(gap_keywords), 1.0) if gap_keywords else 0.0
            relevance_scores.append(score)

        # Return the highest relevance score
        return max(relevance_scores) if relevance_scores else 0.5
