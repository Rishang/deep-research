"""
Main Deep Research implementation.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import litellm

from .core.callbacks import PrintCallback, ResearchCallback
from .models import (
    ActivityItem,
    ActivityStatus,
    ActivityType,
    AnalysisResult,
    ResearchResult,
    ResearchState,
    SourceItem,
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

    async def _add_activity(
        self, type_: ActivityType, status: ActivityStatus, message: str, depth: int
    ) -> None:
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
        await self.callback.on_activity(activity)
        return activity

    async def _add_source(self, source, state) -> None:
        """
        Add a source to the research process and track it in state.

        Args:
            source: Source information (Dict or WebSearchItem).
            state: Current research state to track the source.
        """
        if hasattr(source, "url"):
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
        await self.callback.on_source(source_item)

        return source_item

    async def _analyze_and_plan(
        self, findings: List[Dict[str, str]], topic: str, time_remaining_minutes: float
    ) -> Optional[AnalysisResult]:
        """
        Analyze findings and plan next steps.

        Args:
            findings (List[Dict[str, str]]): Current findings.
            topic (str): Research topic.
            time_remaining_minutes (float): Time remaining in minutes.

        Returns:
            Optional[AnalysisResult]: Analysis results or None if analysis failed.
        """
        try:
            findings_text = "\n".join(
                [f"[From {f['source']}]: {f['text']}" for f in findings]
            )

            prompt = f"""You are an expert research analyst evaluating findings on: {topic}
            
            <research_parameters>
            - You have {time_remaining_minutes:.1f} minutes remaining (but don't need to use all of it)
            - Your task is to critically analyze the current findings and determine the most strategic next steps
            </research_parameters>
            
            <current_findings>
            {findings_text}
            </current_findings>
            
            <analysis_instructions>
            1. Thoroughly evaluate what has been discovered so far
            2. Identify specific knowledge gaps that need to be addressed
            3. Determine if these gaps are significant enough to warrant additional research
            4. If more research is needed:
               - Specify the most important next search topic (be specific and targeted)
               - If relevant, suggest a specific URL to investigate deeper
            5. If findings are sufficient or time is limited (< 1 minute), recommend concluding the research
            </analysis_instructions>
            
            <evaluation_criteria>
            - Depth: Have we explored the core concepts thoroughly?
            - Breadth: Are we missing important related aspects?
            - Balance: Do we have diverse perspectives and sources?
            - Reliability: Are our sources authoritative and credible?
            - Technical depth: Do we have sufficient technical details and implementations?
            - Currency: Is the information up-to-date?
            </evaluation_criteria>
            
            <response_format>
            Respond in this exact JSON format:
            {{
              "analysis": {{
                "summary": "Precise, insightful summary of current findings with key insights",
                "gaps": ["Specific gap 1 - be precise", "Specific gap 2", "Specific gap 3"],
                "nextSteps": ["Detailed next step 1", "Detailed next step 2"],
                "shouldContinue": true/false,
                "nextSearchTopic": "Precisely formulated search query",
                "urlToSearch": "specific URL if needed"
              }}
            }}
            </response_format>"""

            # For O-series models, we need to use temperature=1 (only supported value)
            # For other models, we can use temperature=0
            model_temp = 1 if "o3" in self.reasoning_model.lower() else 0

            response = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                drop_params=True,  # Drop unsupported params for certain models
                base_url=self.base_url,
            )

            result_text = response.choices[0].message.content

            # Parse the JSON response
            import json

            try:
                parsed = json.loads(result_text)
                analysis = parsed.get("analysis", {})

                return AnalysisResult(
                    summary=analysis.get("summary", ""),
                    gaps=analysis.get("gaps", []),
                    next_steps=analysis.get("nextSteps", []),
                    should_continue=analysis.get("shouldContinue", False),
                    next_search_topic=analysis.get("nextSearchTopic", ""),
                    url_to_search=analysis.get("urlToSearch", ""),
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract key information
                # This is a fallback mechanism
                return None
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return None

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
                url: str = item.get("url", "")
                data: str = item.get("data", "")

                # Update activity status
                await self._add_activity(
                    ActivityType.EXTRACT,
                    ActivityStatus.COMPLETE,
                    f"Extracted from {url}",
                    current_depth,
                )

                # Add to results
                results.append({"text": data, "source": url})

        # Mark failed URLs as errors if any
        if not extract_result.success:
            await self._add_activity(
                ActivityType.EXTRACT,
                ActivityStatus.ERROR,
                f"Some extractions failed: {extract_result.error}",
                current_depth,
            )

        return results

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

        # Initialize research state
        state = ResearchState(
            findings=[],
            summaries=[],
            next_search_topic="",
            url_to_search="",
            current_depth=0,
            failed_attempts=0,
            max_failed_attempts=3,
            completed_steps=0,
            total_expected_steps=self.max_depth * 5,  # Each depth has about 5 steps
        )

        # Initialize progress tracking
        await self.callback.on_progress_init(
            max_depth=self.max_depth, total_steps=state.total_expected_steps
        )

        try:
            while state.current_depth < self.max_depth:
                # Check time limit
                time_elapsed = time.time() - start_time
                if time_elapsed >= self.time_limit_seconds:
                    break

                # Increment depth
                state.current_depth += 1

                # Update depth information
                await self.callback.on_depth_change(
                    current=state.current_depth,
                    maximum=self.max_depth,
                    completed_steps=state.completed_steps,
                    total_steps=state.total_expected_steps,
                )

                # SEARCH PHASE
                await self._add_activity(
                    ActivityType.SEARCH,
                    ActivityStatus.PENDING,
                    f'Searching for "{topic}"',
                    state.current_depth,
                )

                search_topic = state.next_search_topic or topic
                search_result = await self.web_client.search(search_topic)

                if not search_result.success:
                    await self._add_activity(
                        ActivityType.SEARCH,
                        ActivityStatus.ERROR,
                        f'Search failed for "{search_topic}"',
                        state.current_depth,
                    )

                    state.failed_attempts += 1
                    if state.failed_attempts >= state.max_failed_attempts:
                        break
                    continue

                await self._add_activity(
                    ActivityType.SEARCH,
                    ActivityStatus.COMPLETE,
                    f"Found {len(search_result.data)} relevant results",
                    state.current_depth,
                )

                # Add sources from search results
                for result in search_result.data:
                    await self._add_source(result, state)

                # EXTRACT PHASE
                top_urls = [result.url for result in search_result.data[:3]]
                if state.url_to_search:
                    top_urls = [state.url_to_search] + top_urls

                new_findings = await self._extract_from_urls(
                    top_urls, topic, state.current_depth
                )
                state.findings.extend(new_findings)

                # ANALYSIS PHASE
                await self._add_activity(
                    ActivityType.ANALYZE,
                    ActivityStatus.PENDING,
                    "Analyzing findings",
                    state.current_depth,
                )

                time_remaining = self.time_limit_seconds - (time.time() - start_time)
                time_remaining_minutes = time_remaining / 60

                analysis = await self._analyze_and_plan(
                    state.findings, topic, time_remaining_minutes
                )

                if not analysis:
                    await self._add_activity(
                        ActivityType.ANALYZE,
                        ActivityStatus.ERROR,
                        "Failed to analyze findings",
                        state.current_depth,
                    )

                    state.failed_attempts += 1
                    if state.failed_attempts >= state.max_failed_attempts:
                        break
                    continue

                state.next_search_topic = analysis.next_search_topic or ""
                state.url_to_search = analysis.url_to_search or ""
                state.summaries.append(analysis.summary)

                await self._add_activity(
                    ActivityType.ANALYZE,
                    ActivityStatus.COMPLETE,
                    analysis.summary,
                    state.current_depth,
                )

                # Increment completed steps
                state.completed_steps += 1

                # Check if we should continue
                if not analysis.should_continue or not analysis.gaps:
                    break

                # Update topic based on gaps
                topic = analysis.gaps[0] if analysis.gaps else topic

            # FINAL SYNTHESIS
            await self._add_activity(
                ActivityType.SYNTHESIS,
                ActivityStatus.PENDING,
                "Preparing final analysis",
                state.current_depth,
            )

            findings_text = "\n".join(
                [f"[From {f['source']}]: {f['text']}" for f in state.findings]
            )

            summaries_text = "\n".join([f"[Summary]: {s}" for s in state.summaries])

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
            
            <task_description>
            Create a thorough, detailed analysis that synthesizes all information into a cohesive, authoritative report. This should be your most comprehensive, detailed work - structured with clear sections and subsections.
            </task_description>
                        
            
            <formatting_guidelines>
            - Use "--------------------" as section dividers
            - Create a logical hierarchy with numbered sections and subsections
            - Use bullet points for lists of related items
            - Include direct quotations where particularly insightful
            - Highlight key terms or concepts in context
            </formatting_guidelines>
            
            <scholarly_standards>
            - Maintain a formal, analytical tone
            - Present balanced coverage of conflicting viewpoints
            - Achieve depth rather than breadth in analysis
            - Prioritize precision and accuracy in all technical explanations
            - Connect specific findings to broader theoretical frameworks
            - Synthesize information across sources rather than summarizing each separately
            </scholarly_standards>
            
            <output_quality>
            Your analysis should be the definitive resource on this topic - comprehensive, authoritative, and insightful.
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

            await self.callback.on_finish(final_text)

            # Convert SourceItem objects to dictionaries for JSON serialization
            sources_data = [source.dict() for source in state.sources]

            return ResearchResult(
                success=True,
                data={
                    "findings": state.findings,
                    "analysis": final_text,
                    "sources": sources_data,
                    "completed_steps": state.completed_steps,
                    "total_steps": state.total_expected_steps,
                },
            )

        except Exception as e:
            await self._add_activity(
                ActivityType.THOUGHT,
                ActivityStatus.ERROR,
                f"Research failed: {str(e)}",
                state.current_depth,
            )

            # Convert SourceItem objects to dictionaries for JSON serialization
            sources_data = [source.dict() for source in state.sources]

            return ResearchResult(
                success=False,
                error=str(e),
                data={
                    "findings": state.findings,
                    "sources": sources_data,
                    "completed_steps": state.completed_steps,
                    "total_steps": state.total_expected_steps,
                },
            )
