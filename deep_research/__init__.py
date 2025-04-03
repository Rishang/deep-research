"""
Main Deep Research implementation.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

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
from .utils.docling_client import DoclingClient


class DeepResearch:
    """
    Main class for the Deep Research functionality.
    Implements the core research logic described in the TypeScript code.
    """

    def __init__(
        self,
        docling_client: DoclingClient,
        llm_api_key: Optional[str] = None,
        research_model: str = "gpt-4o-mini",
        reasoning_model: str = "o3-mini",
        callback: Optional[ResearchCallback] = PrintCallback(),
        max_depth: int = 7,
        time_limit_minutes: float = 4.5,
        max_concurrent_requests: int = 5,
    ):
        """
        Initialize the Deep Research instance.

        Args:
            docling_client (DoclingClient): Initialized DoclingClient instance.
            llm_api_key (Optional[str], optional): API key for LLM. Defaults to None.
            research_model (str, optional): Model to use for research. Defaults to "gpt-4o".
            reasoning_model (str, optional): Model to use for reasoning. Defaults to "gpt-4o".
            callback (Optional[ResearchCallback], optional): Callback for research updates.
                Defaults to None.
            max_depth (int, optional): Maximum research depth. Defaults to 7.
            time_limit_minutes (float, optional): Time limit in minutes. Defaults to 4.5.
            max_concurrent_requests (int, optional): Maximum number of concurrent web requests.
                Defaults to 5.
        """
        self.docling_client = docling_client
        self.llm_api_key = llm_api_key
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

    async def _add_source(self, source) -> None:
        """
        Add a source to the research process.

        Args:
            source: Source information (Dict or WebSearchItem).
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

            prompt = f"""You are a research agent analyzing findings about: {topic}
            You have {time_remaining_minutes:.1f} minutes remaining to complete the research but you don't need to use all of it.
            Current findings: {findings_text}
            What has been learned? What gaps remain? What specific aspects should be investigated next if any?
            If you need to search for more information, include a nextSearchTopic.
            If you need to search for more information in a specific URL, include a urlToSearch.
            Important: If less than 1 minute remains, set shouldContinue to false to allow time for final synthesis.
            If I have enough information, set shouldContinue to false.

            Respond in this exact JSON format:
            {{
              "analysis": {{
                "summary": "summary of findings",
                "gaps": ["gap1", "gap2"],
                "nextSteps": ["step1", "step2"],
                "shouldContinue": true/false,
                "nextSearchTopic": "optional topic",
                "urlToSearch": "optional url"
              }}
            }}"""

            # For O-series models, we need to use temperature=1 (only supported value)
            # For other models, we can use temperature=0
            model_temp = 1 if "o3" in self.reasoning_model.lower() else 0

            response = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                drop_params=True,  # Drop unsupported params for certain models
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
        extract_result = await self.docling_client.extract(urls=urls, prompt=prompt)

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
                search_result = await self.docling_client.search(search_topic)

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
                    await self._add_source(result)

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

            synthesis_prompt = f"""Create a comprehensive long analysis of {topic} based on these findings:
            {findings_text}
            {summaries_text}
            Provide all the thoughts processes including findings details, key insights, conclusions, and any remaining uncertainties. Include citations to sources where appropriate. This analysis should be very comprehensive and full of details. It is expected to be very long, detailed and comprehensive."""

            # For O-series models, we need to use temperature=1 (only supported value)
            model_temp = 1 if "o3" in self.reasoning_model.lower() else temperature

            final_analysis = await litellm.acompletion(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=max_tokens,  # Reduced to avoid context window limits
                temperature=model_temp,
                drop_params=True,  # Drop unsupported params for certain models
            )

            final_text = final_analysis.choices[0].message.content

            await self._add_activity(
                ActivityType.SYNTHESIS,
                ActivityStatus.COMPLETE,
                "Research completed",
                state.current_depth,
            )

            await self.callback.on_finish(final_text)

            return ResearchResult(
                success=True,
                data={
                    "findings": state.findings,
                    "analysis": final_text,
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

            return ResearchResult(
                success=False,
                error=str(e),
                data={
                    "findings": state.findings,
                    "completed_steps": state.completed_steps,
                    "total_steps": state.total_expected_steps,
                },
            )
