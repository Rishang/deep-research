"""
Main Deep Research implementation.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import litellm

from .core.callbacks import PrintCallback, ResearchCallback
from .models.models import (
    ActivityItem,
    ActivityStatus,
    ActivityType,
    ConfirmedFact,
    Contradiction,
    EnhancedResearchState,
    ResearchPhase,
    ResearchResult,
    SearchQuery,
    SourceItem,
    SourceQualityMetrics,
    UnconfirmedClaim,
    WebSearchItem,
)
from .utils.base_client import BaseWebClient
from .utils.docling_client import DoclingClient
from .utils.docling_server_client import DoclingServerClient
from .utils.firecrawl_client import FirecrawlClient


class DeepResearch:
    """
    Main class for the Deep Research functionality.
    Implements the core research logic described in the TypeScript code.
    """

    def __init__(
        self,
        web_client: Union[
            BaseWebClient, DoclingClient, DoclingServerClient, FirecrawlClient
        ],
        llm_api_key: Optional[str] = None,
        research_model: str = "gpt-4o-mini",
        reasoning_model: str = "o3-mini",
        callback: Optional[ResearchCallback] = PrintCallback(),
        base_url: str = "https://api.openai.com/v1",
        max_depth: int = 7,
        time_limit_minutes: float = 4.5,
        max_concurrent_requests: int = 5,
        enable_graphrag: bool = True,
        graphrag_storage_path: Optional[str] = None,
    ):
        """
        Initialize the Deep Research instance.

        Args:
            web_client (Union[BaseWebClient, DoclingClient, DoclingServerClient, FirecrawlClient]):
                An initialized web client instance. Can be any client that implements BaseWebClient interface.
            llm_api_key (Optional[str], optional): API key for LLM. Defaults to None.
            research_model (str, optional): Model to use for research. Defaults to "gpt-4o-mini".
            reasoning_model (str, optional): Model to use for reasoning. Defaults to "o3-mini".
            callback (Optional[ResearchCallback], optional): Callback for research updates.
                Defaults to PrintCallback().
            base_url (str, optional): Base URL for API requests. Defaults to "https://openai.com/api/v1".
            max_depth (int, optional): Maximum research depth. Defaults to 7.
            time_limit_minutes (float, optional): Time limit in minutes. Defaults to 4.5.
            max_concurrent_requests (int, optional): Maximum number of concurrent web requests.
                Defaults to 5.
            enable_graphrag (bool, optional): Enable GraphRAG knowledge graph. Defaults to True.
            graphrag_storage_path (Optional[str], optional): Path for GraphRAG storage. Defaults to None.
        """
        self.web_client = web_client
        self.llm_api_key = llm_api_key
        self.base_url = base_url
        self.research_model = research_model
        self.reasoning_model = reasoning_model
        self.callback = callback
        self.max_depth = max_depth
        self.time_limit_seconds = time_limit_minutes * 60
        self.max_concurrent_requests = max_concurrent_requests

        # GraphRAG initialization
        self.enable_graphrag = enable_graphrag
        self.graphrag_manager = None
        self.graphrag_extractor = None
        self.graphrag_retriever = None

        if enable_graphrag:
            from pathlib import Path
            from .graphrag.extraction import EntityExtractor
            from .graphrag.knowledge_graph import KnowledgeGraphManager

            storage_path = (
                Path(graphrag_storage_path) if graphrag_storage_path else None
            )
            self.graphrag_manager = KnowledgeGraphManager(storage_path)
            self.graphrag_extractor = EntityExtractor(
                model=research_model,
                api_key=llm_api_key,
                base_url=base_url,
            )

        # Initialize litellm
        if llm_api_key:
            # Set the API key for OpenAI models
            litellm.api_key = llm_api_key

            # Configure models to use OpenAI
            litellm.set_verbose = False  # Disable verbose output

            # Set model configuration for both research and reasoning models
            if "gpt" in self.research_model.lower():
                # If model is a GPT model, use openai provider
                self.research_model = f"openai/{self.research_model}"

            if "gpt" in self.reasoning_model.lower():
                # If model is a GPT model, use openai provider
                self.reasoning_model = f"openai/{self.reasoning_model}"

    # ============= GRAPHRAG INTEGRATION =============

    async def _build_knowledge_graph_from_findings(
        self, findings: List[Dict[str, str]], topic: str
    ) -> None:
        """
        Build knowledge graph from research findings using GraphRAG.

        Args:
            findings: List of findings with text and source.
            topic: Research topic for context.
        """
        if (
            not self.enable_graphrag
            or not self.graphrag_manager
            or not self.graphrag_extractor
        ):
            return

        graph = self.graphrag_manager.current_graph
        if not graph:
            return

        await self._add_activity(
            ActivityType.REASONING,
            ActivityStatus.PENDING,
            f"Building knowledge graph from {len(findings)} findings",
            0,
        )

        extracted_count = 0
        for finding in findings:
            text = finding.get("text", "")
            source = finding.get("source", "")

            if not text or len(text) < 50:  # Skip very short texts
                continue

            try:
                # Extract entities and relationships
                (
                    entities,
                    relationships,
                ) = await self.graphrag_extractor.extract_entities_and_relationships(
                    text, source, context=topic
                )

                # Add to graph
                for entity in entities:
                    graph.add_entity(entity)
                    extracted_count += 1

                for relationship in relationships:
                    graph.add_relationship(relationship)

            except Exception as e:
                print(f"Error extracting from finding: {str(e)}")
                continue

        if extracted_count > 0:
            # Calculate PageRank
            graph.calculate_pagerank()

            # Detect communities
            graph.detect_communities_simple()

            await self._add_activity(
                ActivityType.REASONING,
                ActivityStatus.COMPLETE,
                f"Knowledge graph built: {len(graph.nodes)} entities, {len(graph.edges)} relationships",
                0,
            )

    async def _retrieve_graph_context(self, query: str, max_length: int = 1500) -> str:
        """
        Retrieve relevant context from knowledge graph for a query.

        Args:
            query: Query text.
            max_length: Maximum context length.

        Returns:
            Formatted context from graph.
        """
        if not self.enable_graphrag or not self.graphrag_retriever:
            return ""

        try:
            context = await self.graphrag_retriever.retrieve_context_for_query(
                query, max_length
            )
            return context
        except Exception as e:
            print(f"Error retrieving graph context: {str(e)}")
            return ""

    def _save_knowledge_graph(self, session_id: str) -> None:
        """
        Save the knowledge graph to disk.

        Args:
            session_id: Session identifier.
        """
        if not self.enable_graphrag or not self.graphrag_manager:
            return

        try:
            self.graphrag_manager.save_graph(session_id)
        except Exception as e:
            print(f"Error saving knowledge graph: {str(e)}")

    def load_knowledge_graph(self, session_id: str) -> bool:
        """
        Load a previously saved knowledge graph.

        Args:
            session_id: Session identifier.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.enable_graphrag or not self.graphrag_manager:
            return False

        try:
            graph = self.graphrag_manager.load_graph(session_id)
            if graph:
                from .graphrag.retrieval import GraphRetriever

                self.graphrag_retriever = GraphRetriever(
                    graph,
                    model=self.research_model,
                    api_key=self.llm_api_key,
                    base_url=self.base_url,
                )
                return True
            return False
        except Exception as e:
            print(f"Error loading knowledge graph: {str(e)}")
            return False

    # ============= SOURCE QUALITY ASSESSMENT =============

    async def _assess_source_quality(
        self, search_item: WebSearchItem
    ) -> SourceQualityMetrics:
        """
        Assess the quality of a source based on various metrics.

        Args:
            search_item: The search result to assess.

        Returns:
            SourceQualityMetrics with quality scores.
        """
        # Domain authority (based on TLD and known authoritative domains)
        domain_authority = self._calculate_domain_authority(str(search_item.url))

        # Recency score (based on date if available)
        recency_score = self._calculate_recency_score(search_item.date)

        # Content depth (based on description length and relevance)
        content_depth = self._calculate_content_depth(search_item)

        # Cross-reference score (will be updated as we find multiple sources)
        cross_reference_score = 0.0

        return SourceQualityMetrics(
            domain_authority=domain_authority,
            recency_score=recency_score,
            content_depth=content_depth,
            citation_count=0,
            cross_reference_score=cross_reference_score,
        )

    def _calculate_domain_authority(self, url: str) -> float:
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

    def _calculate_recency_score(self, date_str: str) -> float:
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

    def _calculate_content_depth(self, search_item: WebSearchItem) -> float:
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

    def _calculate_diversity_score(
        self, result: WebSearchItem, existing_results: List[Tuple]
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

    def _calculate_gap_relevance(
        self, result: WebSearchItem, priority_gaps: List[str]
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

    # ============= ADAPTIVE QUERY GENERATION =============

    async def _add_activity(
        self, type_: ActivityType, status: ActivityStatus, message: str, depth: int
    ) -> ActivityItem:
        """
        Add an activity to the research process.

        Args:
            type_ (ActivityType): Type of activity.
            status (ActivityStatus): Status of activity.
            message (str): Activity message.
            depth (int): Current depth.
        """
        activity = ActivityItem(
            type=type_,
            status=status,
            message=message,
            timestamp=datetime.now(),
            depth=depth,
        )
        if self.callback:
            await self.callback.on_activity(activity)
        return activity

    async def _add_source(
        self, source: Union[WebSearchItem, Dict], state: EnhancedResearchState
    ) -> SourceItem:
        """
        Add a source to the research process and track it in state.

        Args:
            source: Source information (Dict or WebSearchItem).
            state: Current research state to track the source.
        """
        if isinstance(source, WebSearchItem):
            # It's a WebSearchItem
            source_item = SourceItem(
                url=source.url,
                title=source.title,
                relevance=getattr(source, "relevance", 1.0),
                description=getattr(source, "description", ""),
                # Note: date and provider from WebSearchItem aren't currently used in SourceItem
                # but are stored in the WebSearchItem for reference
            )
        else:
            # It's a dictionary
            source_item = SourceItem(
                url=source["url"],
                title=source["title"],
                relevance=source.get("relevance", 1.0),
                description=source.get("description", ""),
            )

        # Add to state
        state.sources.append(source_item)

        # Notify via callback
        if self.callback:
            await self.callback.on_source(source_item)

        return source_item

    async def _adaptive_query_generation(
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
        if state.current_phase == ResearchPhase.EXPLORATION:
            return await self._generate_exploratory_queries(topic)
        elif state.current_phase == ResearchPhase.DEEPENING:
            return await self._generate_deepening_queries(
                topic, state.priority_gaps[:3]
            )
        elif state.current_phase == ResearchPhase.VERIFICATION:
            return await self._generate_verification_queries(
                state.unconfirmed_claims[:5]
            )
        else:
            # Fallback to basic query generation
            return await self._generate_exploratory_queries(topic)

    async def _generate_exploratory_queries(self, topic: str) -> List[SearchQuery]:
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
            print(f"Error generating exploratory queries: {str(e)}")
            return [SearchQuery(query=topic, phase="exploration")]

    async def _generate_deepening_queries(
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
            print(f"Error generating deepening queries: {str(e)}")
            return [SearchQuery(query=priority_gaps[0], phase="deepening")]

    async def _generate_verification_queries(
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
            print(f"Error generating verification queries: {str(e)}")
            return []

    async def _generate_search_queries(self, topic: str) -> List[SearchQuery]:
        """
        Legacy method for backward compatibility.
        Generates exploratory queries by default.
        """
        return await self._generate_exploratory_queries(topic)

    # ============= INCREMENTAL KNOWLEDGE BUILDING =============

    async def _incremental_analysis(
        self,
        new_findings: List[Dict[str, str]],
        state: EnhancedResearchState,
        topic: str,
    ) -> Optional[dict]:
        """
        Analyze new findings in context of existing knowledge.
        Identifies confirmations, contradictions, and new information.

        Args:
            new_findings: Newly extracted findings.
            state: Current research state with existing knowledge.
            topic: Research topic.

        Returns:
            Analysis result with updated knowledge categorization.
        """
        try:
            confirmed_facts_text = json.dumps(
                [
                    {"claim": f.claim, "sources": len(f.sources)}
                    for f in state.confirmed_facts
                ],
                indent=2,
            )
            unconfirmed_text = json.dumps(
                [
                    {"claim": c.claim, "source": c.source}
                    for c in state.unconfirmed_claims
                ],
                indent=2,
            )
            new_findings_text = json.dumps(new_findings, indent=2)

            prompt = f"""You are an expert research analyst evaluating new findings in context of existing knowledge.

<topic>
{topic}
</topic>

<existing_confirmed_facts>
{confirmed_facts_text}
</existing_confirmed_facts>

<existing_unconfirmed_claims>
{unconfirmed_text}
</existing_unconfirmed_claims>

<new_findings>
{new_findings_text}
</new_findings>

<tasks>
1. Identify which new findings CONFIRM existing unconfirmed claims (multiple independent sources)
2. Identify NEW claims from the findings that need verification
3. Identify CONTRADICTIONS between sources (conflicting information)
4. Update knowledge gaps - what critical information is still missing?
5. Determine if we have sufficient information or need deeper investigation
</tasks>

<response_format>
Respond with a JSON object:
{{
  "confirmed_facts": [
    {{"claim": "fact now confirmed", "sources": ["url1", "url2"], "confidence": 0.0-1.0}}
  ],
  "new_unconfirmed": [
    {{"claim": "new claim", "source": "url", "needs_verification": true}}
  ],
  "contradictions": [
    {{"topic": "what contradicts", "claim_a": "claim from source A", "source_a": "url", "claim_b": "conflicting claim", "source_b": "url"}}
  ],
  "knowledge_gaps": ["specific gap 1", "specific gap 2"],
  "should_continue": true/false,
  "recommended_phase": "exploration|deepening|verification|synthesis",
  "summary": "brief summary of new insights"
}}
</response_format>
"""

            model_temp = 1 if "o3" in self.reasoning_model.lower() else 0
            response = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                drop_params=True,
                base_url=self.base_url,
            )

            result_text = response.choices[0].message.content
            return json.loads(result_text)
        except Exception as e:
            print(f"Incremental analysis error: {str(e)}")
            return None

    async def _update_research_state(
        self, analysis: dict, state: EnhancedResearchState
    ) -> None:
        """
        Update research state based on incremental analysis.

        Args:
            analysis: Analysis results from _incremental_analysis.
            state: Research state to update.
        """
        # Add confirmed facts
        for fact_data in analysis.get("confirmed_facts", []):
            fact = ConfirmedFact(
                claim=fact_data["claim"],
                sources=fact_data.get("sources", []),
                confidence=fact_data.get("confidence", 0.9),
            )
            state.confirmed_facts.append(fact)

            # Remove from unconfirmed if it was there
            state.unconfirmed_claims = [
                c for c in state.unconfirmed_claims if c.claim != fact.claim
            ]

        # Add new unconfirmed claims
        for claim_data in analysis.get("new_unconfirmed", []):
            claim = UnconfirmedClaim(
                claim=claim_data["claim"],
                source=claim_data.get("source", ""),
                needs_verification=claim_data.get("needs_verification", True),
            )
            state.unconfirmed_claims.append(claim)

        # Add contradictions
        for contra_data in analysis.get("contradictions", []):
            contradiction = Contradiction(
                topic=contra_data["topic"],
                claim_a=contra_data["claim_a"],
                source_a=contra_data["source_a"],
                claim_b=contra_data["claim_b"],
                source_b=contra_data["source_b"],
            )
            state.contradictions.append(contradiction)

        # Update priority gaps
        state.priority_gaps = analysis.get("knowledge_gaps", [])

        # Update phase if recommended
        recommended_phase = analysis.get("recommended_phase", "")
        if recommended_phase:
            try:
                state.current_phase = ResearchPhase(recommended_phase)
            except ValueError:
                pass  # Keep current phase if invalid

    async def _cross_validate_claims(
        self, state: EnhancedResearchState, topic: str
    ) -> None:
        """
        Actively search for evidence to verify unconfirmed claims.

        Args:
            state: Current research state.
            topic: Research topic.
        """
        if not state.unconfirmed_claims:
            return

        # Generate verification queries
        verification_queries = await self._generate_verification_queries(
            state.unconfirmed_claims[:3]  # Top 3 unconfirmed claims
        )

        if not verification_queries:
            return

        await self._add_activity(
            ActivityType.REASONING,
            ActivityStatus.PENDING,
            f"Verifying {len(state.unconfirmed_claims)} claims",
            state.current_depth,
        )

        # Search for verification evidence
        for query in verification_queries:
            search_result = await self.web_client.search(query.query)

            if not search_result.success or not search_result.data:
                continue

            # Analyze if results support or contradict claims
            # This is a simplified version - could be enhanced with LLM analysis
            for claim in state.unconfirmed_claims[:3]:
                matching_sources = [
                    str(result.url)
                    for result in search_result.data
                    if any(
                        keyword.lower() in result.description.lower()
                        for keyword in claim.claim.split()[:5]
                    )
                ]

                # If we find multiple sources mentioning the claim, increase confidence
                if len(matching_sources) >= 2:
                    # Convert to confirmed fact
                    confirmed = ConfirmedFact(
                        claim=claim.claim,
                        sources=matching_sources,
                        confidence=min(len(matching_sources) / 3.0, 1.0),
                    )
                    state.confirmed_facts.append(confirmed)

        # Remove verified claims from unconfirmed
        confirmed_claims = {f.claim for f in state.confirmed_facts}
        state.unconfirmed_claims = [
            c for c in state.unconfirmed_claims if c.claim not in confirmed_claims
        ]

        await self._add_activity(
            ActivityType.REASONING,
            ActivityStatus.COMPLETE,
            f"Verified claims, now have {len(state.confirmed_facts)} confirmed facts",
            state.current_depth,
        )

    # ============= INTELLIGENT URL SELECTION =============

    async def _select_urls_intelligently(
        self,
        search_results: List[WebSearchItem],
        state: EnhancedResearchState,
        max_urls: int = 15,
    ) -> List[str]:
        """
        Select URLs intelligently based on quality metrics, diversity, and research needs.

        Args:
            search_results: List of search results from queries.
            state: Current research state.
            max_urls: Maximum number of URLs to select.

        Returns:
            List of selected URLs.
        """
        scored_results: List[Tuple[WebSearchItem, float, SourceQualityMetrics]] = []

        for result in search_results:
            url_str = str(result.url)

            # Skip already visited URLs
            if url_str in state.visited_urls:
                continue

            # Calculate quality score
            quality = await self._assess_source_quality(result)

            # Calculate diversity bonus (prefer different domains)
            diversity_bonus = self._calculate_diversity_score(result, scored_results)

            # Calculate relevance to current gaps
            gap_relevance = self._calculate_gap_relevance(result, state.priority_gaps)

            # Calculate final score with weighted components
            final_score = (
                quality.composite_score() * 0.4
                + diversity_bonus * 0.3
                + gap_relevance * 0.3
            )

            scored_results.append((result, final_score, quality))

        # Sort by score and take top N
        scored_results.sort(key=lambda x: x[1], reverse=True)

        selected = []
        for result, score, quality in scored_results[:max_urls]:
            url_str = str(result.url)
            selected.append(url_str)
            state.visited_urls.add(url_str)
            state.source_quality_map[url_str] = quality

        return selected

    # ============= PARALLEL PROCESSING =============

    async def _parallel_research_wave(
        self, queries: List[SearchQuery], state: EnhancedResearchState, topic: str
    ) -> List[Dict[str, str]]:
        """
        Process multiple queries in parallel, extract and analyze concurrently.

        Args:
            queries: List of search queries to process.
            state: Current research state.
            topic: Research topic.

        Returns:
            List of findings from all queries.
        """
        # Search all queries in parallel
        search_tasks = [self.web_client.search(query.query) for query in queries]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Aggregate all successful results
        all_results = []
        for idx, result_or_exc in enumerate(search_results):
            if isinstance(result_or_exc, (Exception, BaseException)):
                await self._add_activity(
                    ActivityType.SEARCH,
                    ActivityStatus.ERROR,
                    f"Query failed: {queries[idx].query}",
                    state.current_depth,
                )
                continue

            # Now we know it's a SearchResult - add type ignore for mypy
            if (
                hasattr(result_or_exc, "success")
                and result_or_exc.success
                and result_or_exc.data
            ):  # type: ignore
                all_results.extend(result_or_exc.data)  # type: ignore
                # Add sources
                for result in result_or_exc.data:  # type: ignore
                    await self._add_source(result, state)

        if not all_results:
            return []

        # Select best URLs across all queries
        selected_urls = await self._select_urls_intelligently(
            all_results, state, max_urls=15
        )

        if not selected_urls:
            return []

        # Extract in parallel
        findings = await self._extract_from_urls(
            selected_urls, topic, state.current_depth
        )

        return findings

    async def _extract_from_urls(
        self, urls: List[str], topic: str, current_depth: int
    ) -> List[Dict[str, str]]:
        """
        Extract information from URLs concurrently.

        Args:
            urls (List[str]): URLs to extract from.
            topic (str): Research topic.
            current_depth (int): Current research depth.

        Returns:
            List[Dict[str, str]]: Extracted information.
        """
        # Filter out empty URLs
        urls = [url for url in urls if url]
        if not urls:
            return []

        # Add pending activities for all URLs
        for url in urls:
            await self._add_activity(
                ActivityType.EXTRACT,
                ActivityStatus.PENDING,
                f"Analyzing {url}",
                current_depth,
            )

        # Extract from all URLs concurrently
        prompt = f"Extract key information about {topic}. Focus on facts, data, and expert opinions. Analysis should be full of details and very comprehensive."
        extract_result = await self.web_client.extract(urls=urls, prompt=prompt)

        results = []

        # Process extraction results
        if extract_result.success and extract_result.data:
            for item in extract_result.data:
                if isinstance(item, dict):
                    extracted_url: str = item.get("url", "")
                    extracted_data: str = item.get("data", "")

                    # Update activity status
                    await self._add_activity(
                        ActivityType.EXTRACT,
                        ActivityStatus.COMPLETE,
                        f"Extracted from {extracted_url}",
                        current_depth,
                    )

                    # Add to results
                    results.append({"text": extracted_data, "source": extracted_url})

        # Mark failed URLs as errors if any
        if not extract_result.success:
            await self._add_activity(
                ActivityType.EXTRACT,
                ActivityStatus.ERROR,
                f"Some extractions failed: {extract_result.error}",
                current_depth,
            )

        return results

    # ============= DYNAMIC RESEARCH TERMINATION =============

    def _should_continue_research(
        self, state: EnhancedResearchState, start_time: float
    ) -> Tuple[bool, str]:
        """
        Decide whether to continue research based on multiple factors.

        Args:
            state: Current research state.
            start_time: Research start time.

        Returns:
            Tuple of (should_continue, reason).
        """
        time_elapsed = time.time() - start_time
        time_remaining = self.time_limit_seconds - time_elapsed

        # Time check - need at least 30 seconds for synthesis
        if time_remaining < 30:
            return False, "Time limit approaching"

        # Minimum findings check
        if len(state.findings) < 3:
            return True, "Need more findings"

        # Goal completion check (if goals were set)
        if state.research_goals:
            goal_coverage = len(state.confirmed_facts) / max(
                len(state.research_goals), 1
            )
            if goal_coverage >= 0.85:
                return False, "Research goals substantially met"

        # Gap analysis
        if not state.priority_gaps and len(state.confirmed_facts) >= 5:
            return False, "No significant gaps remain and sufficient facts gathered"

        # Diminishing returns check
        novelty_rate = self._calculate_novelty_rate(state)
        if novelty_rate < 0.1 and len(state.findings) >= 10:
            return False, "Diminishing returns - little new information"

        # Max depth check (safety limit)
        if state.current_depth >= self.max_depth:
            return False, f"Maximum depth of {self.max_depth} reached"

        # Failed attempts check
        if state.failed_attempts >= state.max_failed_attempts:
            return False, "Too many failed attempts"

        # Continue if we have time and work to do
        return True, f"Continuing - {len(state.priority_gaps)} gaps remain"

    def _calculate_novelty_rate(self, state: EnhancedResearchState) -> float:
        """
        Calculate the rate of new information in recent findings.

        Args:
            state: Current research state.

        Returns:
            Novelty rate between 0.0 and 1.0.
        """
        if not state.recent_findings_count or len(state.recent_findings_count) < 2:
            return 1.0  # Not enough data, assume high novelty

        # Track the last few iterations
        if len(state.recent_findings_count) >= 3:
            recent = state.recent_findings_count[-3:]
            avg_new = sum(recent) / len(recent)
            # Compare to earlier average
            if len(state.recent_findings_count) > 3:
                earlier = state.recent_findings_count[-6:-3]
                if earlier:
                    avg_earlier = sum(earlier) / len(earlier)
                    if avg_earlier > 0:
                        return min(avg_new / avg_earlier, 1.0)

        return 0.5  # Default moderate novelty

    async def _extract_research_goals(self, topic: str) -> List[str]:
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
            )

            result_text = response.choices[0].message.content
            goals = json.loads(result_text)
            return goals if isinstance(goals, list) else []
        except Exception as e:
            print(f"Error extracting research goals: {str(e)}")
            return []

    # ============= MAIN RESEARCH METHOD =============

    async def research(
        self, topic: str, max_tokens: int = 8000, temperature: float = 0.5
    ) -> ResearchResult:
        """
        Perform deep research on a topic.

        Args:
            topic (str): The topic to research.

        Returns:
            ResearchResult: The research results.
        """
        start_time = time.time()

        # Generate unique session ID
        session_id = (
            f"research_{int(time.time())}_{hashlib.md5(topic.encode()).hexdigest()[:8]}"
        )

        # Extract research goals from topic
        research_goals = await self._extract_research_goals(topic)

        # Initialize GraphRAG if enabled
        if self.enable_graphrag and self.graphrag_manager:
            graph = self.graphrag_manager.create_graph(session_id, topic)
            from .graphrag.retrieval import GraphRetriever

            self.graphrag_retriever = GraphRetriever(
                graph,
                model=self.research_model,
                api_key=self.llm_api_key,
                base_url=self.base_url,
            )

        # Initialize ENHANCED research state
        state = EnhancedResearchState(
            current_phase=ResearchPhase.INITIALIZATION,
            research_goals=research_goals,
            findings=[],
            summaries=[],
            next_search_topic="",
            url_to_search="",
            current_depth=0,
            failed_attempts=0,
            max_failed_attempts=3,
            completed_steps=0,
            total_expected_steps=self.max_depth * 5,
        )

        # Initialize progress tracking
        if self.callback:
            await self.callback.on_progress_init(
                max_depth=self.max_depth, total_steps=state.total_expected_steps
            )

        try:
            # PHASE 1: EXPLORATION
            state.current_phase = ResearchPhase.EXPLORATION
            state.current_depth += 1

            await self._add_activity(
                ActivityType.THOUGHT,
                ActivityStatus.PENDING,
                "Phase: EXPLORATION - Gathering broad foundational knowledge",
                state.current_depth,
            )

            if self.callback:
                await self.callback.on_depth_change(
                    current=state.current_depth,
                    maximum=self.max_depth,
                    completed_steps=state.completed_steps,
                    total_steps=state.total_expected_steps,
                )

            # Generate exploratory queries
            exploration_queries = await self._adaptive_query_generation(topic, state)

            # Execute parallel research wave
            new_findings = await self._parallel_research_wave(
                exploration_queries, state, topic
            )
            state.findings.extend(new_findings)
            state.recent_findings_count.append(len(new_findings))

            # Build knowledge graph from findings
            if new_findings and self.enable_graphrag:
                await self._build_knowledge_graph_from_findings(new_findings, topic)

            # Perform incremental analysis
            if new_findings:
                analysis_result = await self._incremental_analysis(
                    new_findings, state, topic
                )
                if analysis_result:
                    await self._update_research_state(analysis_result, state)
                    state.summaries.append(analysis_result.get("summary", ""))

            state.completed_steps += 1

            # MAIN RESEARCH LOOP with adaptive phases
            while True:
                # Check if we should continue research
                should_continue, reason = self._should_continue_research(
                    state, start_time
                )
                if not should_continue:
                    await self._add_activity(
                        ActivityType.THOUGHT,
                        ActivityStatus.COMPLETE,
                        f"Research terminating: {reason}",
                        state.current_depth,
                    )
                    break

                # PHASE 2: DEEPENING (if we have gaps)
                if (
                    state.priority_gaps
                    and state.current_phase != ResearchPhase.DEEPENING
                ):
                    state.current_phase = ResearchPhase.DEEPENING
                    state.current_depth += 1

                    await self._add_activity(
                        ActivityType.THOUGHT,
                        ActivityStatus.PENDING,
                        f"Phase: DEEPENING - Investigating {len(state.priority_gaps)} knowledge gaps",
                        state.current_depth,
                    )

                    if self.callback:
                        await self.callback.on_depth_change(
                            current=state.current_depth,
                            maximum=self.max_depth,
                            completed_steps=state.completed_steps,
                            total_steps=state.total_expected_steps,
                        )

                    # Generate deepening queries
                    deepening_queries = await self._adaptive_query_generation(
                        topic, state
                    )

                    if deepening_queries:
                        # Execute parallel research wave
                        new_findings = await self._parallel_research_wave(
                            deepening_queries, state, topic
                        )
                        state.findings.extend(new_findings)
                        state.recent_findings_count.append(len(new_findings))

                        # Build knowledge graph from deepening findings
                        if new_findings and self.enable_graphrag:
                            await self._build_knowledge_graph_from_findings(
                                new_findings, topic
                            )

                        # Incremental analysis
                        if new_findings:
                            analysis_result = await self._incremental_analysis(
                                new_findings, state, topic
                            )
                            if analysis_result:
                                await self._update_research_state(
                                    analysis_result, state
                                )
                                state.summaries.append(
                                    analysis_result.get("summary", "")
                                )

                        state.completed_steps += 1

                # PHASE 3: VERIFICATION (if we have unconfirmed claims)
                if (
                    len(state.unconfirmed_claims) >= 2
                    and state.current_phase != ResearchPhase.VERIFICATION
                    and state.completed_steps < state.total_expected_steps - 2
                ):
                    state.current_phase = ResearchPhase.VERIFICATION
                    state.current_depth += 1

                    await self._add_activity(
                        ActivityType.THOUGHT,
                        ActivityStatus.PENDING,
                        f"Phase: VERIFICATION - Validating {len(state.unconfirmed_claims)} claims",
                        state.current_depth,
                    )

                    if self.callback:
                        await self.callback.on_depth_change(
                            current=state.current_depth,
                            maximum=self.max_depth,
                            completed_steps=state.completed_steps,
                            total_steps=state.total_expected_steps,
                        )

                    # Cross-validate claims
                    await self._cross_validate_claims(state, topic)

                    state.completed_steps += 1

                # Check termination conditions again
                should_continue, reason = self._should_continue_research(
                    state, start_time
                )
                if not should_continue:
                    await self._add_activity(
                        ActivityType.THOUGHT,
                        ActivityStatus.COMPLETE,
                        f"Research complete: {reason}",
                        state.current_depth,
                    )
                    break

                # If we've exhausted all phases and still have time, do another deepening round
                if not state.priority_gaps and not state.unconfirmed_claims:
                    break

                # Safety: prevent infinite loop
                if state.current_depth >= self.max_depth:
                    break

            # FINAL SYNTHESIS
            state.current_phase = ResearchPhase.SYNTHESIS

            await self._add_activity(
                ActivityType.SYNTHESIS,
                ActivityStatus.PENDING,
                "Preparing final analysis with cross-validation",
                state.current_depth,
            )

            findings_text = "\n".join(
                [f"[From {f['source']}]: {f['text']}" for f in state.findings]
            )

            summaries_text = "\n".join([f"[Summary]: {s}" for s in state.summaries])

            # Include confirmed facts and contradictions in synthesis
            confirmed_facts_text = "\n".join(
                [
                    f"[CONFIRMED - {f.confidence:.1%} confidence]: {f.claim} (Sources: {len(f.sources)})"
                    for f in state.confirmed_facts
                ]
            )

            contradictions_text = "\n".join(
                [
                    f"[CONTRADICTION]: {c.topic}\n  - {c.source_a}: {c.claim_a}\n  - {c.source_b}: {c.claim_b}"
                    for c in state.contradictions
                ]
            )

            quality_summary = (
                f"\n\nResearch Quality Metrics:\n"
                f"- Total sources consulted: {len(state.visited_urls)}\n"
                f"- Confirmed facts: {len(state.confirmed_facts)}\n"
                f"- Contradictions found: {len(state.contradictions)}\n"
                f"- Research phases completed: {state.current_depth}\n"
            )

            # <required_elements>
            # 1. Executive Summary (250-300 words):
            #    <executive_summary_guidelines>
            #    - Begin with a compelling hook that highlights the significance of the topic
            #    - Summarize key findings, threats, and opportunities
            #    - Include a precise statement of current state of knowledge
            #    - Mention key stakeholders and their interests
            #    - End with impactful concluding statement on broader implications
            #    </executive_summary_guidelines>

            # 2. Main Analysis (5-7 clear thematic sections):
            #    <section_structure>
            #    - Each section must start with a clear, bold thesis statement
            #    - Each thesis must make a specific, defensible claim
            #    - Provide 3-5 pieces of supporting evidence from the primary sources
            #    - Include at least one direct quotation per section with source attribution
            #    - Include technical specifications, numerical data, and implementation details where available
            #    - Compare conflicting viewpoints where they exist
            #    - End each section with implications of the findings in that section
            #    </section_structure>

            # 3. Future Implications:
            #    <future_implications_guidelines>
            #    - Project 3-5 year developments based on current trends
            #    - Identify critical uncertainties that could alter projections
            #    - Discuss potential breakthrough technologies and their impact
            #    - Address cross-disciplinary effects (e.g., economic, social, policy)
            #    - Include both optimistic and cautious perspectives
            #    </future_implications_guidelines>

            # 4. Citations and Evidence Evaluation:
            #    <citation_guidelines>
            #    - Evaluate the credibility of each major source
            #    - Identify potential biases or limitations in the evidence
            #    - Note currency of information and how quickly it may become outdated
            #    - Address any disagreements between authoritative sources
            #    - Highlight the strongest and weakest elements of the available evidence
            #    </citation_guidelines>

            # 5. Knowledge Gaps:
            #    <knowledge_gap_analysis>
            #    - Precisely identify specific missing information
            #    - Explain why each gap matters to the overall understanding
            #    - Prioritize gaps by importance and urgency
            #    - Suggest specific research approaches to address each gap
            #    - Note any areas where expert consensus is lacking
            #    </knowledge_gap_analysis>

            # 6. Recommendations:
            #    <recommendation_guidelines>
            #    - Provide strategic recommendations for different stakeholders (researchers, industry, policymakers)
            #    - Include both immediate actions and long-term strategies
            #    - Recommend specific technologies, approaches, or standards where appropriate
            #    - Address implementation challenges and how to overcome them
            #    - Include considerations of cost, timeline, and resource requirements
            #    </recommendation_guidelines>
            # </required_elements>

            synthesis_prompt = f"""You are an expert academic researcher creating a comprehensive, structured analysis of: {topic}

            <evidence_and_primary_sources>
            {findings_text}
            </evidence_and_primary_sources>
            
            <interim_summaries>
            {summaries_text}
            </interim_summaries>
            
            <confirmed_facts>
            {confirmed_facts_text if confirmed_facts_text else "No facts confirmed across multiple sources."}
            </confirmed_facts>
            
            <contradictions_and_uncertainties>
            {contradictions_text if contradictions_text else "No major contradictions found."}
            </contradictions_and_uncertainties>
            
            <research_metadata>
            {quality_summary}
            </research_metadata>
            
            <task_description>
            Create a thorough, detailed analysis that synthesizes all information into a cohesive, authoritative report. 
            This should be your most comprehensive, detailed work - structured with clear sections and subsections.
            
            IMPORTANT: 
            - Give higher weight to confirmed facts (those verified across multiple sources)
            - Explicitly address any contradictions found
            - Note the confidence level for claims based on source agreement
            - Distinguish between well-established facts and emerging information
            </task_description>
            
            <formatting_guidelines>
            - Use "--------------------" as section dividers
            - Create a logical hierarchy with numbered sections and subsections
            - Use bullet points for lists of related items
            - Include direct quotations where particularly insightful
            - Highlight key terms or concepts in context
            - Mark confidence levels where appropriate (HIGH, MEDIUM, LOW)
            </formatting_guidelines>
            
            <scholarly_standards>
            - Maintain a formal, analytical tone
            - Present balanced coverage of conflicting viewpoints
            - Achieve depth rather than breadth in analysis
            - Prioritize precision and accuracy in all technical explanations
            - Connect specific findings to broader theoretical frameworks
            - Synthesize information across sources rather than summarizing each separately
            - Explicitly note areas of uncertainty or disagreement
            </scholarly_standards>
            
            <output_quality>
            Your analysis should be the definitive resource on this topic - comprehensive, authoritative, and insightful.
            Include a section on limitations and areas requiring further research based on identified gaps.
            </output_quality>"""

            # For O-series models, we need to use temperature=1 (only supported value)
            model_temp = 1 if "o3" in self.reasoning_model.lower() else temperature

            final_analysis = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=max_tokens,  # Reduced to avoid context window limits
                temperature=model_temp,
                drop_params=True,  # Drop unsupported params for certain models
                base_url=self.base_url,
            )

            final_text = final_analysis.choices[0].message.content

            await self._add_activity(
                ActivityType.SYNTHESIS,
                ActivityStatus.COMPLETE,
                "Research completed",
                state.current_depth,
            )

            if self.callback:
                await self.callback.on_finish(final_text)

            # Save knowledge graph
            if self.enable_graphrag:
                self._save_knowledge_graph(session_id)

            # Prepare knowledge graph data
            graphrag_data = {}
            if (
                self.enable_graphrag
                and self.graphrag_manager
                and self.graphrag_manager.current_graph
            ):
                graph = self.graphrag_manager.current_graph
                top_entities = graph.get_most_important_entities(10)
                graphrag_data = {
                    "session_id": session_id,
                    "total_entities": len(graph.nodes),
                    "total_relationships": len(graph.edges),
                    "communities": len(graph.communities),
                    "top_entities": [
                        {
                            "name": e.name,
                            "type": e.type,
                            "pagerank": graph.nodes[e.id].pagerank,
                        }
                        for e in top_entities
                    ],
                }

            # Convert objects to dictionaries for JSON serialization
            sources_data = [source.dict() for source in state.sources]
            confirmed_facts_data = [fact.dict() for fact in state.confirmed_facts]
            contradictions_data = [contra.dict() for contra in state.contradictions]
            search_queries_data = [
                {
                    "query": q.query,
                    "relevance": q.relevance,
                    "explanation": q.explanation,
                    "phase": q.phase,
                }
                for q in state.search_queries
            ]

            return ResearchResult(
                success=True,
                data={
                    "findings": state.findings,
                    "analysis": final_text,
                    "sources": sources_data,
                    "confirmed_facts": confirmed_facts_data,
                    "contradictions": contradictions_data,
                    "search_queries": search_queries_data,
                    "research_goals": state.research_goals,
                    "priority_gaps": state.priority_gaps,
                    "completed_steps": state.completed_steps,
                    "total_steps": state.total_expected_steps,
                    "sources_consulted": len(state.visited_urls),
                    "research_phases": state.current_depth,
                    "knowledge_graph": graphrag_data,
                },
            )

        except Exception as e:
            await self._add_activity(
                ActivityType.THOUGHT,
                ActivityStatus.ERROR,
                f"Research failed: {str(e)}",
                state.current_depth,
            )

            # Convert objects to dictionaries for JSON serialization
            sources_data = [source.dict() for source in state.sources]
            confirmed_facts_data = [fact.dict() for fact in state.confirmed_facts]
            contradictions_data = [contra.dict() for contra in state.contradictions]
            search_queries_data = [
                {
                    "query": q.query,
                    "relevance": q.relevance,
                    "explanation": q.explanation,
                    "phase": getattr(q, "phase", None),
                }
                for q in state.search_queries
            ]

            return ResearchResult(
                success=False,
                error=str(e),
                data={
                    "findings": state.findings,
                    "sources": sources_data,
                    "confirmed_facts": confirmed_facts_data,
                    "contradictions": contradictions_data,
                    "search_queries": search_queries_data,
                    "research_goals": state.research_goals,
                    "completed_steps": state.completed_steps,
                    "total_steps": state.total_expected_steps,
                    "sources_consulted": len(state.visited_urls),
                },
            )
