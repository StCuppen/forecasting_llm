import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import dotenv

dotenv.load_dotenv()

from .planner import make_search_plan
from .retrieval import perform_research
from .utils import (
    ExaClient,
    call_openrouter_llm,
    clean_indents,
    reset_token_usage,
    get_token_usage,
    record_token_usage,
    extract_search_queries,
    extract_market_probabilities,
)
from .prompts import FORECASTER_CONTEXT, REACT_SYSTEM_PROMPT

from forecasting_tools import MetaculusApi

logger = logging.getLogger(__name__)


DEFAULT_AGENT_MODEL = os.getenv("AGENT_MODEL") or "x-ai/grok-4.1-fast:free"


@dataclass
class AgentForecastResult:
    """Container for a single agent run."""

    question: str
    planner_text: str
    research_memo: str
    final_forecast: str
    decomposition: Optional[str] = None
    outside_view: Optional[str] = None
    inside_view: Optional[str] = None
    scenarios_and_probs: Optional[str] = None


@dataclass
class AgentForecastOutput:
    """Simplified output for external consumers."""
    probability: float
    explanation: str
    # Optional structured metrics
    search_count: int = 0
    sources: list = None  # List of SourceInfo dicts
    token_usage: dict = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.token_usage is None:
            self.token_usage = {}


@dataclass
class SourceInfo:
    """Tracking info for a consulted source."""
    url: str
    title: str
    date: str = ""
    quality: str = "Medium"  # High, Medium, Low
    snippet: str = ""


class ForecastingAgent:
    """
    Lightweight experimental agent that stitches together:

    1. Question decomposition
    2. Web research (brave search api)
    3. Outside view (reference classes, historical analogues)
    4. Inside view adjustments
    5. Scenario decomposition + probabilistic forecasting
    6. Final forecast write‑up

    All LLM steps default to a Grok‑style model via OpenRouter.
    You can override the model id via the AGENT_MODEL environment
    variable or by passing `model_name=...` to the constructor.
    """

    def __init__(self, model_name: str = "x-ai/grok-4.1-fast:free", reasoning_effort: Optional[str] = None):
        """Initialize the agent."""
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        
        # Exa is now the sole search provider
        exa_key = os.getenv("EXA_API_KEY")
        if exa_key:
            self.exa_client: Optional[ExaClient] = ExaClient(
                api_key=exa_key,
                max_results=10,
            )
            logger.info(">>> Exa.ai ENABLED as sole search provider. <<<")
        else:
            logger.warning("EXA_API_KEY not set; Exa disabled.")
            self.exa_client = None

        self.serper_client = None

        # Deprecated providers - kept for backwards compatibility
        self.brave_client = None
        self.tavily_client = None
        self.langsearch_client = None
        self.sonar_client = None

        logger.info(f"ForecastingAgent initialized with model={self.model_name}")

    async def _llm(
        self,
        prompt: str,
        *,
        max_tokens: int = 4000,
        temperature: float = 0.6,
        usage_label: Optional[str] = None,
    ) -> str:
        """Thin wrapper around call_openrouter_llm with agent defaults."""
        text = await call_openrouter_llm(
            prompt=prompt,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            usage_label=usage_label,
            reasoning_effort=self.reasoning_effort,
        )
        return text.strip()

    async def decompose_question(self, question: str, today_str: str) -> str:
        """Step 1: Decompose the question into sub‑problems and info needs."""
        prompt = clean_indents(
            f"""
            You are a forecasting assistant.

            Decompose the question into:
            - Clarified resolution criteria
            - Time horizon
            - Key sub‑questions / uncertainties
            - Information needs, split into:
              - Outside view (reference classes, base rates, historical analogues)
              - Inside view (current status, mechanisms, actors, constraints)

            Output in concise Markdown with these sections:

            # QUESTION DECOMPOSITION

            ## 1. Clarified Question
            - Paraphrase: ...
            - Resolution criteria: ...
            - Time horizon: ...

            ## 2. Key Uncertainties
            - U1: ...
            - U2: ...
            - U3: ...

            ## 3. Information Needs
            ### 3.1 Outside View
            - Need 1: ...

            ### 3.2 Inside View
            - Need 1: ...

            Question: {question}
            Today: {today_str}
            """
        )
        return await self._llm(
            prompt, max_tokens=2000, temperature=0.4, usage_label="agent:decompose"
        )

    async def find_information(
        self,
        question: str,
        today_str: str,
    ) -> tuple[str, str]:
        """
        Step 2: Find information via your search tool.

        Returns:
            planner_text: The PLANNER markdown.
            research_memo: The RESEARCH MEMO markdown.
        """
        logger.info("Agent: planning searches...")
        queries, planner_text = await make_search_plan(
            question_text=question,
            today_str=today_str,
            llm_model=self.model_name,
        )

        logger.info(f"Agent: executing retrieval over {len(queries)} queries...")
        research_memo = await perform_research(
            question_text=question,
            queries=queries,
            today_str=today_str,
            serper_client=self.serper_client,
            exa_client=self.exa_client,
            llm_model=self.model_name,
        )

        return planner_text, research_memo

    async def develop_outside_view(
        self,
        question: str,
        research_memo: str,
        today_str: str,
    ) -> str:
        """Step 3: Develop an outside view (reference classes, base rates, analogues)."""
        prompt = clean_indents(
            f"""
            You are building an OUTSIDE VIEW for a forecasting question.

            Use ONLY the provided research memo. Focus on:
            - Reference classes and historical analogues
            - Structural base rates and frequencies
            - Typical dynamics and swing sizes
            - A rough prior probability range (0‑1) based purely on outside view

            Output in Markdown with this structure:

            # OUTSIDE VIEW

            ## 1. Reference Classes
            - RC1: ...
            - RC2: ...

            ## 2. Historical Patterns
            - Pattern 1: ...
            - Pattern 2: ...

            ## 3. Structural Base Rates
            - Narrative summary (no invented data): ...

            ## 4. Outside‑View Prior
            - P_outside (0‑1, rough band): ...
            - Rationale: ...

            Question: {question}
            Today: {today_str}

            Research memo:
            {research_memo}
            """
        )
        return await self._llm(
            prompt, max_tokens=2500, temperature=0.4, usage_label="agent:outside_view"
        )

    async def make_inside_view_adjustment(
        self,
        question: str,
        research_memo: str,
        outside_view: str,
        today_str: str,
    ) -> str:
        """Step 4: Make inside‑view adjustments relative to the outside view."""
        prompt = clean_indents(
            f"""
            You are building an INSIDE VIEW to adjust an outside‑view prior.

            Use ONLY:
            - The forecasting question
            - The OUTSIDE VIEW summary
            - The RESEARCH MEMO

            Your job:
            - Identify upward pressures vs the outside‑view prior
            - Identify downward pressures vs the outside‑view prior
            - Propose an adjusted probability band P_inside (0‑1, rough)

            Output in Markdown with this structure:

            # INSIDE VIEW

            ## 1. Upward Pressures
            - Factor 1: ...

            ## 2. Downward Pressures
            - Factor 1: ...

            ## 3. Net Adjustment
            - Narrative: ...
            - P_inside (0‑1, rough band): ...

            Question: {question}
            Today: {today_str}

            Outside view:
            {outside_view}

            Research memo:
            {research_memo}
            """
        )
        return await self._llm(
            prompt, max_tokens=2500, temperature=0.5, usage_label="agent:inside_view"
        )

    async def scenario_decomposition_and_forecast(
        self,
        question: str,
        research_memo: str,
        outside_view: str,
        inside_view: str,
        today_str: str,
    ) -> tuple[str, str]:
        """
        Step 5: Scenario decomposition + probabilistic forecasting,
        and Step 6: Final forecast write‑up.

        Returns:
            scenarios_md: Scenario decomposition with probabilities.
            final_forecast_md: Short final forecast summary.
        """
        prompt = clean_indents(
            f"""
            You are a probabilistic forecaster.

            You are given:
            - A forecasting question
            - An OUTSIDE VIEW summary
            - An INSIDE VIEW adjustment
            - A RESEARCH MEMO (no probabilities)

            First, perform scenario decomposition and assign probabilities.
            Then, produce a short final forecast summary.

            Output in Markdown with EXACTLY this structure:

            # SCENARIOS
            - Scenario 1 (p = ...): ...
            - Scenario 2 (p = ...): ...
            - Scenario 3 (p = ...): ...
            (Add up to 5 scenarios total; probabilities between 0 and 1 and must sum to 1.0.)

            # FINAL FORECAST
            - FINAL_PROBABILITY (0-1): ...
            - One-sentence rationale: ...

            Rules:
            - All probabilities must be numeric between 0 and 1.
            - Scenario probabilities must sum to 1.0 (within rounding).
            - FINAL_PROBABILITY should be consistent with the scenarios.
            - Do not call external tools; reason only with the provided text.

            Question:
            {question}

            Today: {today_str}

            Outside view:
            {outside_view}

            Inside view:
            {inside_view}

            Research memo:
            {research_memo}
            """
        )
        text = await self._llm(
            prompt, max_tokens=3500, temperature=0.5, usage_label="agent:scenarios_final"
        )

        # Naive split into scenarios vs final forecast based on headings.
        scenarios_part = text
        final_part = ""
        if "# FINAL FORECAST" in text:
            parts = text.split("# FINAL FORECAST", 1)
            scenarios_part = parts[0].strip()
            final_part = "# FINAL FORECAST" + parts[1]

        return scenarios_part.strip(), final_part.strip()

    async def run(self, question: str) -> AgentForecastResult:
        """
        Run the full agent pipeline for a free‑form question.

        This is intentionally self‑contained so you can experiment
        with different orchestration styles without touching the
        main Metaculus pipeline.
        """
        today_str = datetime.utcnow().date().isoformat()

        decomposition = await self.decompose_question(question, today_str)
        planner_text, research_memo = await self.find_information(question, today_str)
        outside_view = await self.develop_outside_view(
            question, research_memo, today_str
        )
        inside_view = await self.make_inside_view_adjustment(
            question, research_memo, outside_view, today_str
        )
        scenarios_md, final_forecast_md = await self.scenario_decomposition_and_forecast(
            question, research_memo, outside_view, inside_view, today_str
        )

        return AgentForecastResult(
            question=question,
            decomposition=decomposition,
            planner_text=planner_text,
            research_memo=research_memo,
            outside_view=outside_view,
            inside_view=inside_view,
            scenarios_and_probs=scenarios_md,
            final_forecast=final_forecast_md,
        )

    async def _iterative_forecast(
        self,
        question: str,
        today_str: str,
        planner_text: str,
        research_memo: str,
        market_priors: list[dict] = None,
    ) -> str:
        """
        Single-call, lightly-structured iterative forecast.

        The model is told what good forecasting structure looks like
        (decomposition, outside view, inside view, scenarios, final
        probability) but is free to move between these in whatever
        order it finds useful. The only hard requirements are that it
        is explicit about its reasoning and that it ends with a
        FINAL_PROBABILITY line in section 5. It may also flag where
        additional web retrieval would be valuable.
        """
        # Format market priors as informational context (not anchoring rules)
        market_priors_text = ""
        if market_priors:
            priors_lines = []
            for p in market_priors:
                priors_lines.append(f"- {p['source'].title()}: {p['probability']:.1%}")
            market_priors_text = f"""
            CONTEXT - Prediction market/community forecasts found in research:
            {chr(10).join(priors_lines)}
            
            Note: These are provided as additional data points. You should consider them
            but form your own independent judgment based on the evidence in the research memo.
            If your analysis leads to a different conclusion, that's fine - explain your reasoning.
            """
        
        prompt = clean_indents(
            f"""
            You are a superforecaster making a probabilistic prediction.

            You have:
            - A forecasting question
            - Today's date: {today_str}
            - Detailed research gathered from web searches
            {market_priors_text}

            YOUR TASK: Analyze the research thoroughly and produce your best probability estimate.
            
            Key principles:
            1. BASE YOUR FORECAST ON THE EVIDENCE in the research memo, not on market priors
            2. Think step-by-step through the key factors that determine the outcome
            3. Be explicit about your reasoning - show your work
            4. Consider what would need to happen for the event to occur vs not occur
            5. Estimate probabilities based on the actual state of the world described in the research

            Structure your response as follows:

            # FORECAST

            ## 1. Question Understanding
            - What exactly needs to happen for YES?
            - What is the time horizon?
            - What is the current state?

            ## 2. Key Evidence from Research
            - List the most important facts from the research memo
            - What does the evidence tell us about the current situation?

            ## 3. Analysis
            - What are the key factors that will determine the outcome?
            - What would need to happen for YES? How likely is each step?
            - What would need to happen for NO? How likely is that path?

            ## 4. Probability Estimate
            - FINAL_PROBABILITY (0-1): [your estimate]
            - Reasoning: [1-3 sentences explaining your number]

            Question:
            {question}

            Research memo (this is your primary evidence source):
            {research_memo}
            """
        )
        text = await self._llm(
            prompt, max_tokens=25000, temperature=0.5, usage_label="agent:iterative_forecast"
        )
        return text

    async def run_iterative(self, question: str, max_iterations: int = 2, community_prior: float = None) -> AgentForecastResult:
        """
        Run the agent with iterative research capability.

        The agent can loop back to searching after initial analysis:
        1. Do initial research
        2. Analyze and identify what else is needed (section 6)
        3. Extract those queries and search again
        4. Repeat up to max_iterations times
        5. Generate final forecast

        Args:
            question: The forecasting question
            max_iterations: Maximum research iterations (default 2, reduced from 3 to minimize failures)
            community_prior: Metaculus/market community prediction (0-1) to use as anchor
        """
        today_str = datetime.utcnow().date().isoformat()

        # Initial research pass
        planner_text, research_memo = await self.find_information(question, today_str)
        all_research = [research_memo]
        
        # Track first successful forecast for fallback
        first_successful_forecast: Optional[str] = None
        first_successful_probability: Optional[float] = None

        # Iterative research loop
        for iteration in range(max_iterations - 1):  # -1 because we already did one pass
            logger.info(f"Iteration {iteration + 1}/{max_iterations - 1}: Checking if more research needed...")

            # Ask agent what else it needs
            combined_research = "\n\n---\n\n".join(all_research)
            followup_prompt = clean_indents(f"""
                You are analyzing a forecasting question and have done some research.
                Review the research below and determine if you need MORE information.

                Question: {question}
                Today: {today_str}

                Research so far:
                {combined_research}

                Do you need additional information to make a well-calibrated forecast?
                If YES, provide 1-3 specific web search queries (one per line).
                If NO, respond with exactly: "READY_FOR_FORECAST"

                Your response (be brief, max 100 words):
            """)

            try:
                response = await self._llm(
                    followup_prompt,
                    max_tokens=10000,  # Generous limit to give reasoning models plenty of room
                    temperature=0.3,
                    usage_label="agent:check_research_needs"
                )
                
                # Handle empty response (common with reasoning models hitting token limits)
                if not response or not response.strip():
                    logger.warning("Empty response from research check, proceeding to forecast")
                    break
                    
            except Exception as e:
                logger.warning(f"Research check call failed: {e}, proceeding to forecast")
                break

            # Check if agent is ready
            if "READY_FOR_FORECAST" in response.upper():
                logger.info("Agent indicates research is sufficient")
                break

            # Extract queries from response using robust parser
            queries = extract_search_queries(response)

            if not queries:
                logger.info("No valid follow-up queries found, proceeding to forecast")
                break

            logger.info(f"Agent requested {len(queries)} follow-up queries: {queries}")

            # Execute follow-up research using Exa
            try:
                if self.exa_client is None:
                    logger.warning("Exa client not available for follow-up research")
                    break
                    
                followup_results = []
                for query in queries:
                    exa_results = await self.exa_client.search(query, num_results=5)
                    # Format results nicely
                    formatted = []
                    for r in exa_results[:5]:
                        title = r.get("title", "Untitled")
                        url = r.get("url", "")
                        content = r.get("content", "")[:500]  # Truncate for brevity
                        formatted.append(f"- {title} ({url})\n  {content}")
                    followup_results.append(f"Query: {query}\n" + "\n".join(formatted))

                followup_memo = "\n\n".join(followup_results)
                all_research.append(f"# Follow-up Research (Iteration {iteration + 2})\n\n{followup_memo}")
                logger.info(f"Completed follow-up research iteration {iteration + 2}")

            except Exception as e:
                logger.warning(f"Follow-up research failed: {e}")
                break

        # Combine all research
        final_research_memo = "\n\n".join(all_research)
        self._last_research_memo = final_research_memo # Store for logging

        # Extract market/community priors from research for anchoring
        market_priors = extract_market_probabilities(final_research_memo)
        
        # If we have a community prior passed directly from Metaculus, add it (higher priority)
        if community_prior is not None:
            # Add/overwrite with the direct Metaculus prior
            metaculus_priors = [p for p in market_priors if p["source"] != "metaculus"]
            metaculus_priors.append({"source": "metaculus", "probability": community_prior})
            market_priors = metaculus_priors
            logger.info(f"Using direct Metaculus community prior: {community_prior:.1%}")
        
        if market_priors:
            logger.info(f"Found market/community priors for anchoring: {market_priors}")
        else:
            logger.info("No market/community priors found for anchoring")

        # Generate final forecast with all research
        try:
            agent_output = await self._iterative_forecast(
                question=question,
                today_str=today_str,
                planner_text=planner_text,
                research_memo=final_research_memo,
                market_priors=market_priors,
            )
            
            # Validate we got a real response
            if agent_output and agent_output.strip():
                # Extract and store the first successful probability for potential fallback
                if first_successful_forecast is None:
                    first_successful_forecast = agent_output
                    first_successful_probability = extract_probability_from_forecast(agent_output)
                    if first_successful_probability != 0.5:  # 0.5 is the default fallback
                        logger.info(f"First successful forecast captured: {first_successful_probability:.1%}")
            else:
                logger.warning("Empty forecast response received")
                # Use fallback if available
                if first_successful_forecast:
                    logger.info(f"Using fallback from first successful iteration: {first_successful_probability:.1%}")
                    agent_output = first_successful_forecast
                else:
                    agent_output = f"Forecast generation failed. Community prior: {community_prior:.1%}" if community_prior else "Forecast generation failed."
                    
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            # Use fallback if available
            if first_successful_forecast:
                logger.info(f"Using fallback from first successful iteration: {first_successful_probability:.1%}")
                agent_output = first_successful_forecast
            else:
                agent_output = f"Forecast generation failed: {e}. Community prior: {community_prior:.1%}" if community_prior else f"Forecast generation failed: {e}"

        return AgentForecastResult(
            question=question,
            planner_text=planner_text,
            research_memo=final_research_memo,
            final_forecast=agent_output,
        )

    async def run_forecast(self, question: str, community_prior: float = None) -> "AgentForecastOutput":
        """
        Runs the iterative forecasting agent and extracts the final probability and explanation.
        
        Args:
            question: The forecasting question text.
            community_prior: Metaculus/market community prediction (0-1) to use as anchor.
        """
        forecast_result = await self.run_iterative(question, community_prior=community_prior)
        
        probability = extract_probability_from_forecast(forecast_result.final_forecast)
        
        # The explanation is the full final_forecast text from the iterative agent
        explanation = forecast_result.final_forecast

        return AgentForecastOutput(probability=probability, explanation=explanation)

    async def _llm_conversation(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 8000,
        temperature: float = 0.5,
        usage_label: Optional[str] = None,
    ) -> tuple[str, dict]:
        """
        Call LLM with conversation history (list of messages).
        
        Returns:
            tuple of (response_text, usage_dict)
        """
        import httpx
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        
        response.raise_for_status()
        result = response.json()
        
        usage = result.get("usage", {})
        content = ""
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {}) or {}
            content = (message.get("content") or "").strip()
            # Fallback to reasoning if content empty
            if not content:
                content = (message.get("reasoning") or "").strip()
        
        return content, usage

    async def run_react_agent(
        self,
        question: str,
        community_prior: float = None,
        max_searches: int = 30,
        max_tokens_total: int = 75000,
    ) -> AgentForecastOutput:
        """
        ReAct-style iterative agent: search, reason, forecast in unified loop.
        
        The model can issue SEARCH("query") actions at any time.
        It iterates until it outputs FINAL_FORECAST or hits limits.
        
        Args:
            question: Forecasting question text
            community_prior: Optional community/market prior (0-1)
            max_searches: Maximum number of search actions (default 30)
            max_tokens_total: Token budget per model (default 75k)
        
        Returns:
            AgentForecastOutput with probability and full explanation
        """
        from .utils import clean_indents
        import json
        
        today_str = datetime.utcnow().date().isoformat()
        
        # Set up logging directory
        logs_dir = "forecastingoutput/react_logs"
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_model = self.model_name.replace("/", "_").replace(":", "_")
        log_file = f"{logs_dir}/{timestamp}_{safe_model}.md"
        
        # Build the system prompt using structured template from prompts.py
        prior_text = ""
        if community_prior is not None:
            prior_text = f"\nCONTEXT: Metaculus community prediction is {community_prior:.1%}. Consider this as one data point but form your own judgment.\n"
        
        system_prompt = REACT_SYSTEM_PROMPT.format(
            forecaster_context=FORECASTER_CONTEXT,
            question=question,
            today=today_str,
            prior_text=prior_text,
        )
        
        messages = [{"role": "user", "content": system_prompt}]
        
        search_count = 0
        total_tokens_used = 0
        prompt_tokens_used = 0
        completion_tokens_used = 0
        all_searches = []
        all_sources = []  # Track all sources with full metadata
        iteration = 0
        max_iterations = 5  # Capped at 5 iterations for efficiency
        
        # Initialize log content
        log_content = f"""# ReAct Forecasting Agent Log

**Model**: {self.model_name}
**Question**: {question}
**Timestamp**: {timestamp}
**Community Prior**: {community_prior if community_prior else 'None'}

---

## System Prompt

{system_prompt}

---

## Agent Conversation

"""
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check token budget
            if total_tokens_used >= max_tokens_total:
                logger.warning(f"Token budget exhausted ({total_tokens_used}/{max_tokens_total})")
                log_content += f"\n\n**[SYSTEM] Token budget exhausted ({total_tokens_used}/{max_tokens_total})**\n"
                break
            
            # Check search limit
            if search_count >= max_searches:
                logger.info(f"Search limit reached ({search_count}/{max_searches}), prompting for final forecast")
                messages.append({
                    "role": "user",
                    "content": "You have used all available searches. Please provide your FINAL_FORECAST now based on the information gathered."
                })
                log_content += f"\n### User (Search limit)\n\nYou have used all available searches. Please provide your FINAL_FORECAST now.\n"
            
            try:
                response, usage = await self._llm_conversation(
                    messages,
                    max_tokens=8000,
                    temperature=0.5,
                    usage_label=f"react:iter{iteration}"
                )
                
                # Track token usage
                total_tokens_used += usage.get("total_tokens", 0)
                prompt_tokens_used += usage.get("prompt_tokens", 0)
                completion_tokens_used += usage.get("completion_tokens", 0)
                logger.info(f"Iteration {iteration}: {usage.get('total_tokens', 0)} tokens, total: {total_tokens_used}")
                
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                log_content += f"\n\n**[ERROR] LLM call failed: {e}**\n"
                break
            
            if not response:
                logger.warning("Empty response from model")
                messages.append({"role": "user", "content": "Please continue with your analysis."})
                continue
            
            # Log the assistant response
            log_content += f"\n### Assistant (Iteration {iteration}, {usage.get('total_tokens', 0)} tokens)\n\n{response}\n"
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            
            # Check for FINAL_FORECAST
            if "FINAL_FORECAST" in response:
                logger.info(f"Final forecast received after {iteration} iterations, {search_count} searches")
                probability = extract_probability_from_forecast(response)
                
                # Build full explanation from conversation
                full_explanation = self._build_explanation_from_conversation(messages, all_searches)
                
                # Finalize and save log
                log_content += f"""
---

## Summary

- **Iterations**: {iteration}
- **Searches**: {search_count}
- **Total Tokens**: {total_tokens_used}
- **Final Probability**: {probability:.1%}

## Search Queries Used

"""
                for i, s in enumerate(all_searches, 1):
                    log_content += f"{i}. {s['query']}\n"
                
                # Save log file
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(log_content)
                logger.info(f"Log saved to: {log_file}")
                
                return AgentForecastOutput(
                    probability=probability,
                    explanation=full_explanation,
                    search_count=search_count,
                    sources=all_sources,
                    token_usage={
                        "prompt": prompt_tokens_used,
                        "completion": completion_tokens_used,
                        "total": total_tokens_used,
                    }
                )
            
            # Check for SEARCH action
            search_match = re.search(r'SEARCH\s*\(\s*["\']([^"\']+)["\']\s*\)', response)
            if search_match and search_count < max_searches:
                query = search_match.group(1)
                search_count += 1
                logger.info(f"Search {search_count}/{max_searches}: {query}")
                log_content += f"\n### Search {search_count}: `{query}`\n"
                
                try:
                    if self.exa_client:
                        results = await self.exa_client.search(query, num_results=8)
                        # Track sources with full metadata
                        for r in results:
                            all_sources.append({
                                "url": r.get("url", ""),
                                "title": r.get("title", "Untitled")[:80],
                                "date": r.get("published_date", "")[:10] if r.get("published_date") else "",
                                "quality": "Medium",  # Default, could be enhanced
                                "snippet": r.get("content", "")[:200] if r.get("content") else "",
                            })
                        formatted_results = self._format_search_results(results, query)
                        all_searches.append({"query": query, "results": formatted_results, "source_count": len(results)})
                        log_content += f"\n{formatted_results[:2000]}...\n" if len(formatted_results) > 2000 else f"\n{formatted_results}\n"
                        messages.append({
                            "role": "user",
                            "content": f"Search results for '{query}':\n\n{formatted_results}\n\nContinue your analysis. You may search again or provide your FINAL_FORECAST when ready."
                        })
                    else:
                        log_content += "\n**[Search failed - no client available]**\n"
                        messages.append({
                            "role": "user",
                            "content": f"Search failed (no search client available). Continue with available information or try a different approach."
                        })
                except Exception as e:
                    logger.warning(f"Search failed: {e}")
                    log_content += f"\n**[Search failed: {str(e)[:100]}]**\n"
                    messages.append({
                        "role": "user",
                        "content": f"Search for '{query}' failed: {str(e)[:100]}. Continue with available information or try a different query."
                    })
                continue
            
            # No action found - prompt model to continue
            # Force completion more aggressively as iterations increase
            if iteration >= max_iterations - 1:
                # LAST ITERATION - force completion NOW
                messages.append({
                    "role": "user",
                    "content": """STOP. You have run out of iterations. You MUST output your FINAL_FORECAST NOW.

Based on all information gathered, output EXACTLY this format:

### 5. PROBABILITY COMPUTATION
| Pathway | Probability |
|---------|-------------|
| [Path1] | X% |
| [Path2] | Y% |
| **TOTAL P(YES)** | **Z%** |

FINAL_FORECAST
Probability: [Z - same as TOTAL above]
One-line summary: [Your reasoning]

OUTPUT NOW. Do not search again. Do not explain. Just give the table and FINAL_FORECAST."""
                })
            elif search_count >= max_searches:
                messages.append({
                    "role": "user",
                    "content": "You've reached the search limit. Please provide your FINAL_FORECAST now with the pathway probability table."
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Continue your analysis. Use SEARCH(\"query\") to find more information, or provide your FINAL_FORECAST when ready."
                })
        
        # Fallback: extract best probability from conversation
        logger.warning(f"Agent did not produce final forecast after {iteration} iterations")
        full_text = "\n".join([m["content"] for m in messages if m["role"] == "assistant"])
        probability = extract_probability_from_forecast(full_text)
        
        # Save log even on fallback
        log_content += f"""
---

## Summary (FALLBACK - No explicit FINAL_FORECAST)

- **Iterations**: {iteration}
- **Searches**: {search_count}
- **Total Tokens**: {total_tokens_used}
- **Extracted Probability**: {probability:.1%}

## Search Queries Used

"""
        for i, s in enumerate(all_searches, 1):
            log_content += f"{i}. {s['query']}\n"
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)
        logger.info(f"Log saved to: {log_file}")
        
        return AgentForecastOutput(
            probability=probability,
            explanation=f"[Agent reached iteration limit]\n\n{full_text[-5000:]}",
            search_count=search_count,
            sources=all_sources,
            token_usage={
                "prompt": prompt_tokens_used,
                "completion": completion_tokens_used,
                "total": total_tokens_used,
            }
        )
    
    def _format_search_results(self, results: list, query: str) -> str:
        """Format Exa search results for the agent."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, r in enumerate(results[:8], 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("text", r.get("content", ""))[:1500]  # Truncate long content
            formatted.append(f"**[{i}] {title}**\nURL: {url}\n{content}\n")
        
        return "\n---\n".join(formatted)
    
    def _build_explanation_from_conversation(self, messages: list, searches: list) -> str:
        """Build a coherent explanation from the agent's conversation history."""
        parts = []
        
        # Add search summary
        if searches:
            parts.append(f"## Research ({len(searches)} searches)")
            for s in searches:
                parts.append(f"- {s['query']}")
            parts.append("")
        
        # Add key reasoning from assistant messages
        parts.append("## Agent Reasoning")
        for msg in messages:
            if msg["role"] == "assistant":
                content = msg["content"]
                # NO TRUNCATION - show full reasoning
                parts.append(content)
                parts.append("\n---\n")
        
        return "\n".join(parts)

    async def run_forecast_react(self, question: str, community_prior: float = None) -> AgentForecastOutput:
        """
        Convenience wrapper: run ReAct agent for forecasting.
        This is the new preferred entry point.
        """
        return await self.run_react_agent(question, community_prior=community_prior)


def extract_probability_from_forecast(forecast_text: str) -> float:
    """Extract probability from agent forecast text.

    Priority order:
    1. TOTAL P(YES) from pathway computation table
    2. FINAL_FORECAST Probability line
    3. Various other probability formats
    """
    # FIRST: Look for TOTAL P(YES) from pathway computation table
    # Match patterns like "**TOTAL P(YES)** | **13%**" or "TOTAL | 0.9%"
    total_match = re.search(r'\*?\*?TOTAL[^|]*\*?\*?\s*\|\s*\*?\*?([0-9]+(?:\.[0-9]+)?)\s*(%?)\*?\*?', forecast_text, re.IGNORECASE)
    if total_match:
        prob = float(total_match.group(1))
        has_percent = total_match.group(2) == '%'
        logger.info(f"Extracted probability from TOTAL row: {prob}{'%' if has_percent else ''}")
        # If it has a % sign, it's definitely a percentage (divide by 100)
        # If no % and value > 1, it's also a percentage
        # If no % and value <= 1, it's already a decimal
        if has_percent or prob > 1:
            return prob / 100
        else:
            return prob
    
    # SECOND: Look for "Probability: X" after FINAL_FORECAST  
    final_section_match = re.search(r'FINAL_FORECAST.*?Probability:\s*\[?([0-9.]+)\s*(%?)\]?', forecast_text, re.IGNORECASE | re.DOTALL)
    if final_section_match:
        prob = float(final_section_match.group(1))
        has_percent = final_section_match.group(2) == '%'
        logger.info(f"Extracted probability from FINAL_FORECAST section: {prob}{'%' if has_percent else ''}")
        if has_percent or prob > 1:
            return prob / 100
        else:
            return prob
    
    # Try FINAL_PROBABILITY (0-1): format
    match = re.search(r'FINAL_PROBABILITY\s*\(0-1\)\s*:\s*([0-9.]+)', forecast_text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Try FINAL_PROBABILITY: format (with optional ** markdown bold)
    match = re.search(r'\*?\*?FINAL_PROBABILITY\*?\*?\s*:\s*([0-9.]+)', forecast_text, re.IGNORECASE)
    if match:
        prob = float(match.group(1))
        return prob / 100 if prob > 1 else prob

    # Try "Final Probability" with optional markdown bold and various separators
    match = re.search(r'\*?\*?Final\s+Probability\*?\*?\s*[:\-=]\s*\*?\*?([0-9.]+)\s*%?\*?\*?', forecast_text, re.IGNORECASE)
    if match:
        prob = float(match.group(1))
        return prob / 100 if prob > 1 else prob

    # Try percentage with "probability" nearby (e.g., "35% probability", "probability is 35%")
    match = re.search(r'(?:probability[:\s]+|probability\s+is\s+)?([0-9.]+)\s*%\s*(?:probability)?', forecast_text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100

    # Try "P = X" or "P: X" format
    match = re.search(r'\bP\s*[=:]\s*([0-9.]+)', forecast_text)
    if match:
        prob = float(match.group(1))
        return prob / 100 if prob > 1 else prob

    # Try to find any decimal between 0 and 1 after "forecast" or "probability"
    match = re.search(r'(?:forecast|probability|estimate)[:\s]+(?:\*\*)?([0]\.[0-9]+)(?:\*\*)?', forecast_text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Look for bracketed probability like [0.35] or (0.35) near end of text
    match = re.search(r'[\[\(]([0]\.[0-9]+)[\]\)]', forecast_text[-2000:])
    if match:
        return float(match.group(1))

    # Last resort: look for "## 5. Final" section and extract number
    section_match = re.search(r'##\s*5\.?\s*Final[^\n]*\n(.*?)(?:##|$)', forecast_text, re.IGNORECASE | re.DOTALL)
    if section_match:
        section_text = section_match.group(1)
        # Look for any decimal 0.X in this section
        prob_match = re.search(r'\b(0\.[0-9]+)\b', section_text)
        if prob_match:
            return float(prob_match.group(1))
        # Look for percentage
        pct_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\s*%', section_text)
        if pct_match:
            return float(pct_match.group(1)) / 100

    # Default fallback
    logger.warning("Could not extract probability from forecast, defaulting to 0.5")
    return 0.5


async def run_ensemble_forecast(
    question: str,
    models: list[dict] = None,
    publish_to_metaculus: bool = False,
    community_prior: float = None,
    use_react: bool = True,
) -> dict:
    """
    Run an ensemble of forecasting agents on a question.
    
    Args:
        question: The forecasting question text.
        models: List of model configs. If None, uses default ensemble.
        publish_to_metaculus: Whether to post the result to Metaculus.
        community_prior: Metaculus community prediction (0-1) to use as anchor.
        use_react: If True, use new ReAct-style iterative agent. If False, use legacy agent.
        
    Returns:
        dict containing:
        - final_probability: float
        - summary_text: str
        - full_log: str
        - individual_results: list[dict]
    """
    if models is None:
        # Two-model ensemble with reasoning enabled
        models = [
            {
                "name": "openai/gpt-4o-mini",
                "reasoning_effort": None, 
                "max_tokens": 16000,
                "label": "GPT-4o Mini"
            },
            {
                "name": "google/gemini-3-flash-preview",
                "reasoning_effort": "medium",  # Enable reasoning for better forecasts
                "max_tokens": 100000,
                "label": "Gemini 3 Flash (Reasoning)"
            },
        ]

    agent_type = "ReAct" if use_react else "Legacy"
    print(f"Starting Multi-Model Ensemble Forecast ({agent_type}) for: {question}")
    print("-" * 60)

    results = []
    logs_dir = "forecastingoutput"
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    full_log_content = f"# FORECAST LOG - {timestamp}\nAgent Type: {agent_type}\n\nQuestion: {question}\n\n"

    for config in models:
        print(f"\n>>> Running {config['label']} ({config['name']}) with {agent_type} agent...")
        try:
            agent = ForecastingAgent(
                model_name=config["name"],
                reasoning_effort=config.get("reasoning_effort")
            )
            
            # Use ReAct agent (new) or legacy agent
            if use_react:
                result = await agent.run_react_agent(question, community_prior=community_prior)
            else:
                result = await agent.run_forecast(question, community_prior=community_prior)
            
            results.append({
                "config": config,
                "result": result
            })
            
            # Build sources table for this model
            sources_table = ""
            if hasattr(result, 'sources') and result.sources:
                sources_table = "\n--- SOURCES CONSULTED ---\n"
                sources_table += "| # | Title | URL | Date |\n|---|-------|-----|------|\n"
                for i, src in enumerate(result.sources[:15], 1):  # Limit to 15 sources
                    title = src.get('title', 'N/A')[:50]
                    url = src.get('url', 'N/A')
                    date = src.get('date', 'N/A')
                    sources_table += f"| {i} | {title} | {url} | {date} |\n"
            
            # Build metrics block
            metrics_block = "\n--- METRICS ---\n"
            if hasattr(result, 'search_count'):
                metrics_block += f"Searches: {result.search_count}\n"
            if hasattr(result, 'sources') and result.sources:
                metrics_block += f"Sources: {len(result.sources)}\n"
            if hasattr(result, 'token_usage') and result.token_usage:
                tu = result.token_usage
                metrics_block += f"Tokens: prompt={tu.get('prompt', 0):,}, completion={tu.get('completion', 0):,}, total={tu.get('total', 0):,}\n"
            
            log_section = f"""
{'='*60}
MODEL: {config['label']} ({config['name']})
REASONING: {config.get('reasoning_effort') or 'Default'}
MAX_TOKENS: {config['max_tokens']}
{'='*60}

PREDICTION: {result.probability:.1%}
{sources_table}
{metrics_block}
--- EXPLANATION ---
{result.explanation}
"""
            full_log_content += log_section
            print(f"   Result: {result.probability:.1%}")

        except Exception as e:
            logger.error(f"Error running {config['label']}: {e}")
            full_log_content += f"\n\nERROR running {config['label']}: {e}\n\n"

    # Save Log File
    safe_q = "".join(c for c in question[:50] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    log_filename = f"{logs_dir}/forecast_{timestamp}_{safe_q}.txt"
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(full_log_content)
    print(f"\nDetailed log saved to: {log_filename}")

    # Compute Average
    if not results:
        print("No successful forecasts.")
        # Use community prior as fallback if available, otherwise default to 0.5
        fallback_prob = community_prior if community_prior is not None else 0.5
        summary = f"All models failed. Using community prior: {fallback_prob:.1%}" if community_prior else "All models failed."
        logger.warning(f"All models failed, using fallback probability: {fallback_prob:.1%}")
        return {
            "final_probability": fallback_prob,
            "summary_text": summary,
            "full_reasoning": summary,
            "full_log": full_log_content,
            "individual_results": []
        }

    probs = [r["result"].probability for r in results]
    avg_prob = sum(probs) / len(probs)
    
    print(f"\n{'='*60}")
    print(f"FINAL ENSEMBLE PREDICTION: {avg_prob:.1%}")
    print(f"{'='*60}")

    # Create compact summary
    summary_lines = [f"Ensemble Prediction: {avg_prob:.1%}"]
    summary_lines.append("Individual Models:")
    for r in results:
        summary_lines.append(f"- {r['config']['label']}: {r['result'].probability:.1%}")
    
    summary_text = "\n".join(summary_lines)
    
    # Create full reasoning text including all model explanations
    full_reasoning_parts = [f"# Ensemble Forecast: {avg_prob:.1%}\n"]
    for r in results:
        model_name = r['config']['label']
        model_prob = r['result'].probability
        model_explanation = r['result'].explanation or "No explanation available"
        full_reasoning_parts.append(f"## {model_name}: {model_prob:.1%}\n\n{model_explanation}\n")
    
    full_reasoning = "\n---\n\n".join(full_reasoning_parts)
    
    full_log_content += f"\n\n{'='*60}\nSUMMARY\n{'='*60}\n{summary_text}\n"

    # Log file already saved above

    # Post to Metaculus
    if publish_to_metaculus:
        metaculus_token = os.getenv("METACULUS_TOKEN")
        if metaculus_token:
            try:
                from forecasting_tools import MetaculusApi
                found_question = MetaculusApi.get_question_by_url(question)
                if found_question:
                    # Handle different attribute names (.id, .question_id, .id_of_question, .post_id)
                    q_id = (
                        getattr(found_question, 'id', None) or 
                        getattr(found_question, 'question_id', None) or 
                        getattr(found_question, 'id_of_question', None) or
                        getattr(found_question, 'post_id', None)
                    )
                    if q_id:
                        print(f"Posting prediction {avg_prob:.1%} to Metaculus (ID: {q_id})...")
                        
                        # Post the prediction
                        MetaculusApi.post_binary_question_prediction(
                            question_id=q_id,
                            prediction_in_decimal=avg_prob,
                        )
                        print("Prediction posted successfully.")
                        
                        # Post a comment with the rationale
                        comment_text = f"## Automated Ensemble Forecast\n\n{summary_text}\n\n---\n*Generated by forecasting bot*"
                        try:
                            MetaculusApi.post_question_comment(
                                question_id=q_id,
                                comment_text=comment_text,
                            )
                            print("Comment with rationale posted successfully.")
                        except Exception as comment_err:
                            logger.warning(f"Prediction posted but comment failed: {comment_err}")
                    else:
                        logger.error(f"Could not find question ID attribute. Available: {dir(found_question)}")
                else:
                    print("Could not resolve Metaculus question for posting.")
            except Exception as e:
                logger.error(f"Failed to post to Metaculus: {e}")

    return {
        "final_probability": avg_prob,
        "summary_text": summary_text,
        "full_reasoning": full_reasoning,
        "full_log": full_log_content,
        "individual_results": results
    }


async def _demo():
    """Run the agent on a single question from CLI args or env."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", type=str, help="Question text or URL (positional)")
    parser.add_argument("--question", dest="question_flag", type=str, help="Question text or URL (flag)")
    args = parser.parse_args()
    
    question = args.question or args.question_flag or os.getenv("QUESTION")
    if not question:
        print("No question provided. Use --question or set QUESTION env var.")
        return

    await run_ensemble_forecast(question, publish_to_metaculus=True)
    # Print Token Usage
    print("\nToken Usage Summary:")
    for model, usage in get_token_usage().items():
        print(f"  {model}: {usage}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_demo())
