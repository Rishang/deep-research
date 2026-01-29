"""
Adaptive query generation module.

Generates search queries based on research phase and current knowledge state.
"""

import json
from typing import List

import litellm

from ..utils import logger
from ..models.models import EnhancedResearchState, SearchQuery, UnconfirmedClaim


class QueryGenerator:
    """Generates adaptive search queries for different research phases."""

    def __init__(
        self,
        reasoning_model: str,
        base_url: str,
    ):
        """
        Initialize the query generator.

        Args:
            reasoning_model: Model to use for query generation.
            base_url: Base URL for API requests.
        """
        self.reasoning_model = reasoning_model
        self.base_url = base_url

    async def generate_adaptive_queries(
        self, topic: str, state: EnhancedResearchState, context: str = ""
    ) -> List[SearchQuery]:
        """
        Generate queries adaptively based on current research phase and knowledge state.

        Args:
            topic: The main research topic.
            state: Current research state.
            context: Additional context for query generation.

        Returns:
            List of search queries appropriate for the current phase.
        """
        from ..models.models import ResearchPhase

        if state.current_phase == ResearchPhase.EXPLORATION:
            return await self.generate_exploratory_queries(topic)
        elif state.current_phase == ResearchPhase.DEEPENING:
            return await self.generate_deepening_queries(topic, state.priority_gaps[:3])
        elif state.current_phase == ResearchPhase.VERIFICATION:
            return await self.generate_verification_queries(
                state.unconfirmed_claims[:5]
            )
        else:
            # Fallback to basic query generation
            return await self.generate_exploratory_queries(topic)

    async def generate_exploratory_queries(self, topic: str) -> List[SearchQuery]:
        """
        Generate broad exploratory queries for initial research phase.

        Args:
            topic: The main research topic.

        Returns:
            List of 3-5 broad search queries.
        """
        try:
            prompt = f"""You are an expert research assistant conducting initial exploration of a topic.

<topic>
{topic}
</topic>

<task>
Generate 3-5 exploratory search queries that provide comprehensive coverage from different angles.
These queries should help establish a foundational understanding of the topic.
</task>

<guidelines>
- Generate exactly 3-5 queries
- Cover different aspects: fundamentals, current state, applications, challenges, future trends
- Each query should be specific but broad enough to gather substantial information
- Optimize for search engines (clear, concise, keyword-rich)
- Include technical terms where appropriate
- Prioritize authoritative sources
</guidelines>

<response_format>
Respond with a JSON array:
[
  {{
    "query": "search query text",
    "relevance": 0.0-1.0,
    "explanation": "what aspect this covers"
  }}
]
</response_format>
"""

            model_temp = 1 if "o3" in self.reasoning_model.lower() else 0
            response = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                drop_params=True,
                base_url=self.base_url,
                stream=False,
                stream_options=None,  # Prevent usage tracking parameters for OpenRouter compatibility
            )

            result_text = response.choices[0].message.content
            parsed = json.loads(result_text)
            search_queries = [
                SearchQuery(
                    query=q.get("query", ""),
                    relevance=q.get("relevance", 1.0),
                    explanation=q.get("explanation", ""),
                    phase="exploration",
                )
                for q in parsed
            ]

            if len(search_queries) > 5:
                search_queries = search_queries[:5]
            elif not search_queries:
                search_queries = [SearchQuery(query=topic, phase="exploration")]

            return search_queries
        except Exception as e:
            logger.error(f"Error generating exploratory queries: {str(e)}")
            return [SearchQuery(query=topic, phase="exploration")]

    async def generate_deepening_queries(
        self, topic: str, priority_gaps: List[str]
    ) -> List[SearchQuery]:
        """
        Generate focused queries to address specific knowledge gaps.

        Args:
            topic: The main research topic.
            priority_gaps: List of top priority knowledge gaps.

        Returns:
            List of targeted search queries.
        """
        if not priority_gaps:
            return [SearchQuery(query=topic, phase="deepening")]

        try:
            gaps_text = "\n".join([f"- {gap}" for gap in priority_gaps])
            prompt = f"""You are an expert research assistant conducting deep investigation.

<topic>
{topic}
</topic>

<knowledge_gaps>
{gaps_text}
</knowledge_gaps>

<task>
Generate 2-4 highly targeted search queries to address these specific knowledge gaps.
Each query should focus on filling one specific gap with detailed, technical information.
</task>

<guidelines>
- Generate 2-4 queries maximum (one or two per major gap)
- Be very specific and technical
- Target authoritative, detailed sources
- Include precise terminology
- Formulate to find data, specifications, implementations, or expert analysis
</guidelines>

<response_format>
Respond with a JSON array:
[
  {{
    "query": "specific targeted search query",
    "relevance": 0.0-1.0,
    "explanation": "which gap this addresses"
  }}
]
</response_format>
"""

            model_temp = 1 if "o3" in self.reasoning_model.lower() else 0
            response = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                drop_params=True,
                base_url=self.base_url,
                stream=False,
                stream_options=None,  # Prevent usage tracking parameters for OpenRouter compatibility
            )

            result_text = response.choices[0].message.content
            parsed = json.loads(result_text)
            search_queries = [
                SearchQuery(
                    query=q.get("query", ""),
                    relevance=q.get("relevance", 1.0),
                    explanation=q.get("explanation", ""),
                    phase="deepening",
                )
                for q in parsed
            ]

            if len(search_queries) > 4:
                search_queries = search_queries[:4]

            return (
                search_queries
                if search_queries
                else [SearchQuery(query=priority_gaps[0], phase="deepening")]
            )
        except Exception as e:
            logger.error(f"Error generating deepening queries: {str(e)}")
            return [SearchQuery(query=priority_gaps[0], phase="deepening")]

    async def generate_verification_queries(
        self, unconfirmed_claims: List[UnconfirmedClaim]
    ) -> List[SearchQuery]:
        """
        Generate queries to verify or refute unconfirmed claims.

        Args:
            unconfirmed_claims: List of claims that need verification.

        Returns:
            List of verification-focused queries.
        """
        if not unconfirmed_claims:
            return []

        try:
            claims_text = "\n".join(
                [
                    f"- {claim.claim} (from {claim.source})"
                    for claim in unconfirmed_claims[:3]
                ]
            )
            prompt = f"""You are an expert fact-checker tasked with verifying claims.

<claims_to_verify>
{claims_text}
</claims_to_verify>

<task>
Generate 2-3 search queries designed to find corroborating or contradicting evidence.
Focus on finding authoritative, credible sources that can confirm or refute these claims.
</task>

<guidelines>
- Generate 2-3 queries
- Formulate queries to find independent verification
- Target authoritative sources (academic, government, established institutions)
- Include specific facts or figures mentioned in claims
- Look for both supporting and contradicting evidence
</guidelines>

<response_format>
Respond with a JSON array:
[
  {{
    "query": "verification search query",
    "relevance": 0.0-1.0,
    "explanation": "which claim this verifies"
  }}
]
</response_format>
"""

            model_temp = 1 if "o3" in self.reasoning_model.lower() else 0
            response = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                drop_params=True,
                base_url=self.base_url,
                stream=False,
                stream_options=None,  # Prevent usage tracking parameters for OpenRouter compatibility
            )

            result_text = response.choices[0].message.content
            parsed = json.loads(result_text)
            search_queries = [
                SearchQuery(
                    query=q.get("query", ""),
                    relevance=q.get("relevance", 1.0),
                    explanation=q.get("explanation", ""),
                    phase="verification",
                )
                for q in parsed
            ]

            return search_queries[:3] if search_queries else []
        except Exception as e:
            logger.error(f"Error generating verification queries: {str(e)}")
            return []

    async def extract_research_goals(self, topic: str) -> List[str]:
        """
        Extract specific research goals from the topic using LLM.

        Args:
            topic: The research topic.

        Returns:
            List of specific research goals.
        """
        try:
            prompt = f"""Given this research topic, identify 3-5 specific research goals or questions that should be answered.

<topic>
{topic}
</topic>

<task>
Break down the topic into specific, answerable research goals.
Each goal should be a clear question or objective that can be satisfied with concrete information.
</task>

<response_format>
Respond with a JSON array of strings:
["Goal 1: specific question", "Goal 2: specific question", ...]
</response_format>
"""

            model_temp = 1 if "o3" in self.reasoning_model.lower() else 0
            response = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                drop_params=True,
                base_url=self.base_url,
                stream=False,
                stream_options=None,  # Prevent usage tracking parameters for OpenRouter compatibility
            )

            result_text = response.choices[0].message.content
            goals = json.loads(result_text)
            return goals if isinstance(goals, list) else []
        except Exception as e:
            logger.error(f"Error extracting research goals: {str(e)}")
            return []
