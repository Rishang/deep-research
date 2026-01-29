"""
Main Deep Research implementation.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm

from .core.callbacks import PrintCallback, ResearchCallback
from .core.prompts import render_prompt
from .core.source_quality import SourceQualityAssessor
from .core.query_generation import QueryGenerator
from .utils import logger
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
from .crawl import BaseWebClient, MarkItDownClient, FirecrawlClient
from .crawl.cache import DumpManager


class DeepResearch:
    """
    Main class for the Deep Research functionality.
    Implements the core research logic described in the TypeScript code.
    """

    def __init__(
        self,
        web_client: Union[BaseWebClient, MarkItDownClient, FirecrawlClient],
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
        dump_files: bool = False,
        dump_files_path: Optional[str] = "./dumps",
    ):
        """
        Initialize the Deep Research instance.

        Args:
            web_client (Union[BaseWebClient, MarkItDownClient, FirecrawlClient]):
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
            dump_files (bool, optional): Enable saving research dumps to disk. Defaults to False.
            dump_files_path (Optional[str], optional): Directory path for saving dumps. Defaults to './dumps'.
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

        # Dump files configuration
        self.dump_files = dump_files
        self.dump_files_path = dump_files_path
        self.dump_manager = None
        if dump_files and dump_files_path:
            self.dump_manager = DumpManager(dump_dir=dump_files_path, format="yaml")

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

        # Configure litellm for OpenRouter compatibility
        litellm.drop_params = True  # Drop unsupported parameters globally
        litellm.modify_params = True  # Allow parameter modification
        litellm.enable_json_schema_validation = False  # Disable JSON schema validation

        # Disable stream_options globally for OpenRouter
        import os

        os.environ["LITELLM_DROP_PARAMS"] = "True"

        # Initialize query generator
        self.query_generator = QueryGenerator(
            reasoning_model=reasoning_model,
            base_url=base_url,
        )

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
                logger.error(f"Error extracting from finding: {str(e)}")
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
            logger.error(f"Error retrieving graph context: {str(e)}")
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
            logger.error(f"Error saving knowledge graph: {str(e)}")

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
            logger.error(f"Error loading knowledge graph: {str(e)}")
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
        return await SourceQualityAssessor.assess_source_quality(search_item)

    # ============= ACTIVITY AND SOURCE TRACKING =============

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
        return await self.query_generator.generate_adaptive_queries(
            topic, state, context
        )

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

            prompt = render_prompt(
                "incremental_analysis",
                topic=topic,
                confirmed_facts_text=confirmed_facts_text,
                unconfirmed_text=unconfirmed_text,
                new_findings_text=new_findings_text,
            )

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
            return json.loads(result_text)
        except Exception as e:
            logger.error(f"Incremental analysis error: {str(e)}")
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
        verification_queries = await self.query_generator.generate_verification_queries(
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
            diversity_bonus = SourceQualityAssessor.calculate_diversity_score(
                result, scored_results
            )

            # Calculate relevance to current gaps
            gap_relevance = SourceQualityAssessor.calculate_gap_relevance(
                result, state.priority_gaps
            )

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
        prompt = render_prompt("url_extraction", topic=topic)
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
        research_goals = await self.query_generator.extract_research_goals(topic)

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

            synthesis_prompt = render_prompt(
                "final_synthesis",
                topic=topic,
                findings_text=findings_text,
                summaries_text=summaries_text,
                confirmed_facts_text=confirmed_facts_text
                if confirmed_facts_text
                else "No facts confirmed across multiple sources.",
                contradictions_text=contradictions_text
                if contradictions_text
                else "No major contradictions found.",
                quality_summary=quality_summary,
            )

            # For O-series models, we need to use temperature=1 (only supported value)
            model_temp = 1 if "o3" in self.reasoning_model.lower() else temperature

            final_analysis = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=max_tokens,  # Reduced to avoid context window limits
                temperature=model_temp,
                drop_params=True,  # Drop unsupported params for certain models
                base_url=self.base_url,
                stream=False,
                stream_options=None,  # Prevent usage tracking parameters for OpenRouter compatibility
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

            result = ResearchResult(
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

            # Save dump if enabled
            if self.dump_files:
                self._save_dump(topic, session_id, result)

            return result

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

    # ============= DUMP MANAGEMENT =============

    def _save_dump(self, topic: str, session_id: str, result: ResearchResult) -> None:
        """
        Save research result to a dump file.

        Args:
            topic: Research topic.
            session_id: Session identifier.
            result: Research result to save.
        """
        if not self.dump_manager:
            return

        try:
            # Prepare data and metadata
            data = {
                "success": result.success,
                "data": result.data,
                "error": result.error,
            }
            metadata = {"topic": topic}

            self.dump_manager.save(session_id, data, metadata)

        except Exception as e:
            logger.warning(f"Failed to save dump: {str(e)}")

    def load_dump(self, session_id: str) -> Optional[ResearchResult]:
        """
        Load a research result from a dump file (supports both YAML and JSON).

        Args:
            session_id: Session identifier or filename.

        Returns:
            ResearchResult if found, None otherwise.
        """
        if not self.dump_manager:
            self.dump_manager = DumpManager(dump_dir=self.dump_files_path)

        try:
            dump_data = self.dump_manager.load(session_id)

            if dump_data is None:
                return None

            # Reconstruct ResearchResult
            result = ResearchResult(
                success=dump_data.get("success", False),
                data=dump_data.get("data", {}),
                error=dump_data.get("error"),
            )

            return result

        except Exception as e:
            logger.error(f"Error loading dump: {str(e)}")
            return None

    def list_dumps(self) -> List[str]:
        """
        List all available dump files (both YAML and JSON).

        Returns:
            List of session IDs (filenames without extension).
        """
        if not self.dump_manager:
            self.dump_manager = DumpManager(dump_dir=self.dump_files_path)

        return self.dump_manager.list()

    def delete_dump(self, session_id: str) -> bool:
        """
        Delete a dump file.

        Args:
            session_id: Session identifier or filename.

        Returns:
            True if deleted successfully, False otherwise.
        """
        if not self.dump_manager:
            self.dump_manager = DumpManager(dump_dir=self.dump_files_path)

        return self.dump_manager.delete(session_id)

    def get_dump_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata from a dump file without loading all data.

        Args:
            session_id: Session identifier or filename.

        Returns:
            Metadata dictionary or None if not found.
        """
        if not self.dump_manager:
            self.dump_manager = DumpManager(dump_dir=self.dump_files_path)

        return self.dump_manager.get_metadata(session_id)
