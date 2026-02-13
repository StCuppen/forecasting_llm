import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse
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
    extract_json_from_response,
)
from .prompts import (
    FORECASTER_CONTEXT,
    REACT_SYSTEM_PROMPT,
    LEAN_BINARY_FORECAST_PROMPT,
    SPEC_EXTRACTION_PROMPT,
    SPEC_CONSISTENCY_PROMPT,
    OUTLIER_CROSSEXAM_PROMPT,
)
from bot.aggregation import AggregatedForecast, ForecastRun, aggregate_forecasts
from bot.publish_gate import (
    EvidenceItem as GateEvidenceItem,
    SpecLockResult,
    evaluate_publish_gate,
    shrink_probability,
)

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
    warnings: list = None
    risk_flags: list = None
    diagnostics: dict = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.token_usage is None:
            self.token_usage = {}
        if self.warnings is None:
            self.warnings = []
        if self.risk_flags is None:
            self.risk_flags = []
        if self.diagnostics is None:
            self.diagnostics = {}


@dataclass
class SourceInfo:
    """Tracking info for a consulted source."""
    url: str
    title: str
    date: str = ""
    quality: str = "Medium"  # High, Medium, Low
    snippet: str = ""


@dataclass
class CanonicalSpec:
    target: str
    yes_condition: str
    no_condition: str
    time_window: str
    threshold: str = ""
    metric: str = ""
    canonical_one_line: str = ""


@dataclass
class FeatureFlags:
    spec_lock: bool = True
    evidence_ledger: bool = True
    numeric_provenance: bool = True
    market_snapshot: bool = True
    outlier_xexam: bool = True


def _flags_from_dict(raw: Optional[dict[str, Any]]) -> FeatureFlags:
    defaults = FeatureFlags()
    if not raw:
        return defaults
    return FeatureFlags(
        spec_lock=bool(raw.get("spec_lock", defaults.spec_lock)),
        evidence_ledger=bool(raw.get("evidence_ledger", defaults.evidence_ledger)),
        numeric_provenance=bool(raw.get("numeric_provenance", defaults.numeric_provenance)),
        market_snapshot=bool(raw.get("market_snapshot", defaults.market_snapshot)),
        outlier_xexam=bool(raw.get("outlier_xexam", defaults.outlier_xexam)),
    )


def _format_canonical_spec_text(spec: Optional[CanonicalSpec], question: str) -> str:
    if spec is None:
        return f"- One-line: {question}"
    one_line = spec.canonical_one_line or question
    return (
        f"- One-line: {one_line}\n"
        f"- YES: {spec.yes_condition or 'See question text'}\n"
        f"- NO: {spec.no_condition or 'Complement of YES'}\n"
        f"- Window: {spec.time_window or 'Not specified'}\n"
        f"- Threshold: {spec.threshold or 'None'}\n"
        f"- Metric: {spec.metric or 'None'}"
    )


def _directness_tag_for_url(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    if not host:
        return "CONTEXT"

    if any(x in host for x in [".gov", ".edu", ".int", "europa.eu", "census.gov", "bls.gov", "federalreserve.gov", "olympics.com"]):
        return "DIRECT"
    if any(x in host for x in ["wikipedia.org", "reuters.com", "apnews.com", "ft.com", "wsj.com", "bloomberg.com", "bbc.com"]):
        return "PROXY"
    return "CONTEXT"


def _extract_numeric_claims(text: str) -> list[str]:
    claims = []
    for m in re.finditer(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?\b', text):
        token = m.group(0).strip()
        if len(token) <= 1:
            continue
        claims.append(token)
    # Keep first 40 claims max to avoid huge validator payloads.
    return claims[:40]


def _numeric_provenance_report(model_text: str, evidence_text: str) -> dict[str, Any]:
    claims = _extract_numeric_claims(model_text)
    evidence_lower = evidence_text.lower()
    report_items: list[dict[str, str]] = []
    orphan_count = 0
    assumption_count = 0
    sourced_count = 0

    for claim in claims:
        c = claim.lower()
        if c and c in evidence_lower:
            status = "sourced"
            sourced_count += 1
        else:
            # Approximate assumption detection from local context sentence.
            ctx_match = re.search(rf'[^.\n]*{re.escape(claim)}[^.\n]*', model_text, re.IGNORECASE)
            ctx = ctx_match.group(0).lower() if ctx_match else ""
            if any(k in ctx for k in ["assume", "assumption", "pathway", "chance", "estimate", "roughly", "about"]):
                status = "assumption"
                assumption_count += 1
            else:
                status = "orphan"
                orphan_count += 1
        report_items.append({"claim": claim, "status": status})

    return {
        "total_claims": len(claims),
        "sourced": sourced_count,
        "assumption": assumption_count,
        "orphan": orphan_count,
        "items": report_items,
    }


def _mentions_market_odds(text: str) -> bool:
    return bool(
        re.search(
            r'(metaculus|manifold|polymarket|kalshi|predictit)[^\n]{0,100}\d{1,3}(?:\.\d+)?\s*%',
            text,
            re.IGNORECASE,
        )
    )


async def extract_canonical_spec(
    question: str,
    model: str = "google/gemini-2.5-flash",
) -> CanonicalSpec:
    prompt = SPEC_EXTRACTION_PROMPT.format(question=question)
    try:
        raw = await call_openrouter_llm(
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=1200,
            usage_label="spec_extract",
        )
        obj = extract_json_from_response(raw)
    except Exception as exc:
        logger.warning(f"Canonical spec extraction failed, using fallback: {exc}")
        obj = {
            "target": question,
            "yes_condition": "As stated in question",
            "no_condition": "Complement of YES in question",
            "time_window": "As stated in question",
            "threshold": "",
            "metric": "",
            "canonical_one_line": question,
        }
    return CanonicalSpec(
        target=str(obj.get("target", question)),
        yes_condition=str(obj.get("yes_condition", "As stated in question")),
        no_condition=str(obj.get("no_condition", "Complement of YES in question")),
        time_window=str(obj.get("time_window", "As stated in question")),
        threshold=str(obj.get("threshold", "")),
        metric=str(obj.get("metric", "")),
        canonical_one_line=str(obj.get("canonical_one_line", question)),
    )


async def check_spec_consistency(
    canonical_spec_text: str,
    model_answer: str,
    model: str = "google/gemini-2.5-flash",
) -> tuple[str, str]:
    prompt = SPEC_CONSISTENCY_PROMPT.format(
        canonical_spec_text=canonical_spec_text,
        model_answer=model_answer[-6000:],
    )
    try:
        raw = await call_openrouter_llm(
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=500,
            usage_label="spec_check",
        )
        obj = extract_json_from_response(raw)
        status = str(obj.get("status", "MINOR_DRIFT")).upper()
        if status not in {"OK", "MINOR_DRIFT", "MAJOR_DRIFT"}:
            status = "MINOR_DRIFT"
        reason = str(obj.get("reason", "No reason provided"))
        return status, reason
    except Exception as exc:
        logger.warning(f"Spec consistency check failed, defaulting to MINOR_DRIFT: {exc}")
        return "MINOR_DRIFT", f"Spec check unavailable: {exc}"


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
        max_searches_per_iteration: int = 3,
        max_tokens_total: int = 75000,
        canonical_spec: Optional[CanonicalSpec] = None,
        feature_flags: Optional[FeatureFlags] = None,
    ) -> AgentForecastOutput:
        """
        ReAct-style iterative agent: search, reason, forecast in unified loop.
        
        The model can issue SEARCH("query") actions at any time.
        It iterates until it outputs FINAL_FORECAST or hits limits.
        
        Args:
            question: Forecasting question text
            community_prior: Optional community/market prior (0-1)
            max_searches: Maximum number of search actions (default 30)
            max_searches_per_iteration: Maximum SEARCH() calls executed from a single model response
            max_tokens_total: Token budget per model (default 75k)
            canonical_spec: Canonical resolution lens used for drift checking
            feature_flags: Soft guardrail feature toggles
        
        Returns:
            AgentForecastOutput with probability and full explanation
        """
        from .utils import clean_indents
        import json
        
        today_str = datetime.utcnow().date().isoformat()
        flags = feature_flags or FeatureFlags()
        canonical_spec_text = _format_canonical_spec_text(canonical_spec, question)
        
        # Set up logging directory
        logs_dir = "forecasts/react_logs"
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
            canonical_spec_text=canonical_spec_text,
        )
        
        messages = [{"role": "user", "content": system_prompt}]
        
        search_count = 0
        total_tokens_used = 0
        prompt_tokens_used = 0
        completion_tokens_used = 0
        all_searches = []
        all_sources = []  # Track all sources with full metadata
        evidence_ledger = []
        ledger_next_id = 1
        market_snapshot = {"found": False, "items": []}
        warning_messages = []
        risk_flags = []
        spec_status = "NOT_CHECKED"
        spec_reason = ""
        iteration = 0
        max_iterations = 5  # Capped at 5 iterations for efficiency
        
        # Initialize log content
        log_content = f"""# ReAct Forecasting Agent Log

**Model**: {self.model_name}
**Question**: {question}
**Timestamp**: {timestamp}
**Community Prior**: {community_prior if community_prior else 'None'}
**Canonical Spec**:
{canonical_spec_text}
**Feature Flags**: {flags}

---

## System Prompt

{system_prompt}

---

## Agent Conversation

"""

        def _run_soft_validators(final_text: str) -> tuple[list[str], list[str], dict[str, Any]]:
            warnings_local: list[str] = []
            risk_local: list[str] = []
            diagnostics: dict[str, Any] = {
                "spec_status": spec_status,
                "spec_reason": spec_reason,
                "canonical_spec": canonical_spec_text,
                "evidence_ledger": evidence_ledger if flags.evidence_ledger else [],
                "market_snapshot": market_snapshot if flags.market_snapshot else {"found": False, "items": []},
            }

            if flags.numeric_provenance:
                evidence_text = "\n".join([s.get("results", "") for s in all_searches])[:30000]
                numeric_report = _numeric_provenance_report(final_text, evidence_text)
                diagnostics["numeric_provenance"] = numeric_report
                orphan = numeric_report.get("orphan", 0)
                if orphan > 0:
                    warnings_local.append(f"Numeric provenance warning: {orphan} orphan numeric claim(s).")
                if orphan >= 4:
                    risk_local.append("ORPHAN_NUMERICS_HIGH")

            if flags.market_snapshot:
                found = bool(market_snapshot.get("found"))
                if not found and _mentions_market_odds(final_text):
                    warnings_local.append("Market odds mentioned without direct market snapshot (found=false).")
                    risk_local.append("MARKET_HALLUCINATION")

            if flags.evidence_ledger and evidence_ledger:
                direct_count = sum(1 for e in evidence_ledger if e.get("directness_tag") == "DIRECT")
                proxy_count = sum(1 for e in evidence_ledger if e.get("directness_tag") == "PROXY")
                diagnostics["direct_source_count"] = direct_count
                diagnostics["proxy_source_count"] = proxy_count
                if direct_count == 0 and proxy_count > 0:
                    warnings_local.append("Evidence quality warning: no direct sources, proxy-heavy evidence.")
                    risk_local.append("PROXY_HEAVY")

            return warnings_local, risk_local, diagnostics
        
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
                    max_tokens=16000,
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
                # Don't burn this iteration — decrement so we retry
                iteration -= 1
                messages.append({"role": "user", "content": "Your previous response was empty. Please continue with your analysis or provide your FINAL_FORECAST."})
                continue
            
            # Log the assistant response
            log_content += f"\n### Assistant (Iteration {iteration}, {usage.get('total_tokens', 0)} tokens)\n\n{response}\n"
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            
            # Check for FINAL_FORECAST
            min_searches = 3
            if "FINAL_FORECAST" in response and search_count < min_searches and iteration < max_iterations - 1:
                # Model tried to conclude too early — reject and nudge for more research
                needed = min_searches - search_count
                logger.info(f"Rejecting early FINAL_FORECAST: only {search_count}/{min_searches} searches done")
                messages.append({
                    "role": "user",
                    "content": f"You have only done {search_count} search(es) — that is not enough for a calibrated forecast. "
                    f"Issue at least {needed} more SEARCH calls for current news, recent developments, and prediction market data before providing your FINAL_FORECAST."
                })
                continue

            if "FINAL_FORECAST" in response:
                logger.info(f"Final forecast received after {iteration} iterations, {search_count} searches")
                if flags.spec_lock:
                    spec_status, spec_reason = await check_spec_consistency(
                        canonical_spec_text=canonical_spec_text,
                        model_answer=response,
                    )
                    if spec_status == "MINOR_DRIFT":
                        warning_messages.append(f"Spec consistency MINOR_DRIFT: {spec_reason}")
                    elif spec_status == "MAJOR_DRIFT":
                        warning_messages.append(f"Spec consistency MAJOR_DRIFT: {spec_reason}")
                        risk_flags.append("SPEC_MAJORDRIFT")
                        # Single repair pass, then continue even if still drifting.
                        repair_prompt = (
                            "Repair pass required.\n"
                            "Your previous answer drifted from the canonical spec.\n"
                            f"Canonical spec:\n{canonical_spec_text}\n\n"
                            "Provide FINAL_FORECAST again, strictly matching this spec.\n"
                            "Do not add new SEARCH calls.\n"
                            "Use exact final format with Probability and One-line summary."
                        )
                        messages.append({"role": "user", "content": repair_prompt})
                        try:
                            repair_response, repair_usage = await self._llm_conversation(
                                messages,
                                max_tokens=2000,
                                temperature=0.2,
                                usage_label="react:spec_repair",
                            )
                            total_tokens_used += repair_usage.get("total_tokens", 0)
                            prompt_tokens_used += repair_usage.get("prompt_tokens", 0)
                            completion_tokens_used += repair_usage.get("completion_tokens", 0)
                            log_content += f"\n### User (Spec Repair)\n\n{repair_prompt}\n"
                            log_content += (
                                f"\n### Assistant (Spec Repair, {repair_usage.get('total_tokens', 0)} tokens)\n\n"
                                f"{repair_response}\n"
                            )
                            if repair_response and "FINAL_FORECAST" in repair_response:
                                messages.append({"role": "assistant", "content": repair_response})
                                repair_status, repair_reason = await check_spec_consistency(
                                    canonical_spec_text=canonical_spec_text,
                                    model_answer=repair_response,
                                )
                                spec_status = repair_status
                                spec_reason = repair_reason
                                if repair_status != "MAJOR_DRIFT":
                                    response = repair_response
                                else:
                                    warning_messages.append(f"Spec repair failed: {repair_reason}")
                                    risk_flags.append("SPEC_REPAIR_FAILED")
                            else:
                                warning_messages.append("Spec repair pass did not return FINAL_FORECAST.")
                                risk_flags.append("SPEC_REPAIR_EMPTY")
                        except Exception as spec_exc:
                            warning_messages.append(f"Spec repair call failed: {spec_exc}")
                            risk_flags.append("SPEC_REPAIR_ERROR")

                probability = extract_probability_from_forecast(response)
                validator_warnings, validator_risks, diagnostics = _run_soft_validators(response)
                warning_messages.extend(validator_warnings)
                risk_flags.extend(validator_risks)
                
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
- **Spec Consistency**: {spec_status}

## Search Queries Used

"""
                for i, s in enumerate(all_searches, 1):
                    log_content += f"{i}. {s['query']}\n"
                if warning_messages:
                    log_content += "\n## Warnings\n\n"
                    for w in warning_messages:
                        log_content += f"- {w}\n"
                if risk_flags:
                    log_content += "\n## Risk Flags\n\n"
                    for rf in sorted(set(risk_flags)):
                        log_content += f"- {rf}\n"
                
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
                    },
                    warnings=warning_messages,
                    risk_flags=sorted(set(risk_flags)),
                    diagnostics=diagnostics,
                )
            
            # Determine if model should be forced to conclude (no more searching)
            should_force = (
                iteration >= max_iterations - 1
                or search_count >= max_searches
                or (search_count >= min_searches and iteration >= 3)  # enough data after min searches
            )

            # Check for SEARCH actions — extract ALL queries from the response
            search_matches = re.findall(r'SEARCH\s*\(\s*["\']([^"\']+)["\']\s*\)', response)
            if search_matches and should_force:
                logger.info(
                    f"Ignoring SEARCH actions because model should finalize now "
                    f"(iteration={iteration}, searches={search_count})."
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Do not search again. You already have enough information. "
                        "Provide FINAL_FORECAST now in the required format."
                    )
                })
                continue

            if search_matches and search_count < max_searches:
                # Process up to remaining budget and per-iteration cap to prevent query spam
                remaining_budget = max_searches - search_count
                allowed_this_turn = min(max_searches_per_iteration, remaining_budget)
                queries_to_run = search_matches[:allowed_this_turn]
                if len(search_matches) > allowed_this_turn:
                    logger.info(
                        f"Capping SEARCH calls this iteration: requested={len(search_matches)}, "
                        f"running={allowed_this_turn}, max_per_iteration={max_searches_per_iteration}"
                    )
                    log_content += (
                        f"\n**[SYSTEM] SEARCH cap applied: model requested {len(search_matches)} "
                        f"queries, executed {allowed_this_turn} this iteration.**\n"
                    )
                all_formatted = []

                for query in queries_to_run:
                    if search_count >= max_searches:
                        break
                    search_count += 1
                    logger.info(f"Search {search_count}/{max_searches}: {query}")
                    log_content += f"\n### Search {search_count}: `{query}`\n"

                    try:
                        if self.exa_client:
                            results = await self.exa_client.search(query, num_results=8)
                            # Track sources with dedup by URL
                            seen_urls = {s["url"] for s in all_sources}
                            for r in results:
                                url = r.get("url", "")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    all_sources.append({
                                        "url": url,
                                        "title": r.get("title", "Untitled")[:80],
                                        "date": r.get("published_date", "")[:10] if r.get("published_date") else "",
                                        "quality": "Medium",
                                        "snippet": r.get("content", "")[:200] if r.get("content") else "",
                                    })
                                    if flags.evidence_ledger:
                                        evidence_ledger.append({
                                            "id": f"LEDGER-{ledger_next_id}",
                                            "source_type": "web_search",
                                            "url_ref": url,
                                            "retrieved_at": datetime.utcnow().isoformat(),
                                            "snippet": (r.get("content", "") or "")[:220],
                                            "directness_tag": _directness_tag_for_url(url),
                                        })
                                        ledger_next_id += 1
                            formatted_results = self._format_search_results(results, query)
                            all_searches.append({"query": query, "results": formatted_results, "source_count": len(results)})
                            if flags.market_snapshot:
                                extracted = extract_market_probabilities(formatted_results)
                                if extracted:
                                    market_snapshot["found"] = True
                                    for item in extracted:
                                        if item not in market_snapshot["items"]:
                                            market_snapshot["items"].append(item)
                            all_formatted.append(f"### Results for '{query}':\n{formatted_results}")
                            log_content += f"\n{formatted_results[:2000]}...\n" if len(formatted_results) > 2000 else f"\n{formatted_results}\n"
                        else:
                            all_formatted.append(f"### Results for '{query}': Search failed (no client available)")
                            log_content += "\n**[Search failed - no client available]**\n"
                    except Exception as e:
                        logger.warning(f"Search failed: {e}")
                        all_formatted.append(f"### Results for '{query}': Search failed: {str(e)[:100]}")
                        log_content += f"\n**[Search failed: {str(e)[:100]}]**\n"

                combined_results = "\n\n".join(all_formatted)
                remaining = max_searches - search_count
                messages.append({
                    "role": "user",
                    "content": (
                        f"{combined_results}\n\nYou have {remaining} searches remaining. "
                        f"At most {max_searches_per_iteration} SEARCH calls are executed per response. "
                        "Continue analysis or provide your FINAL_FORECAST when ready."
                    )
                })
                continue
            
            # No action found - prompt model to continue or conclude
            # If searches are below minimum, nudge for more research before allowing conclusion
            if search_count < min_searches and iteration < max_iterations - 1:
                needed = min_searches - search_count
                messages.append({
                    "role": "user",
                    "content": f"You have only done {search_count} search(es). Issue at least {needed} more SEARCH calls "
                    f"to gather current news, recent developments, and prediction market odds before concluding."
                })
                continue

            # Force conclusion aggressively once we have enough searches or iterations
            if should_force:
                messages.append({
                    "role": "user",
                    "content": """You have enough information. Provide your FINAL_FORECAST NOW.

Use this EXACT format:

FINAL_FORECAST
Probability: [your probability as a percentage, e.g. 25%]
One-line summary: [single sentence explaining your reasoning]

Do not search again. Output FINAL_FORECAST now."""
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Continue your analysis. Use SEARCH(\"query\") to find more information, or provide your FINAL_FORECAST when ready."
                })
        
        # Fallback: make one last dedicated call to force a probability out
        logger.warning(f"Agent did not produce final forecast after {iteration} iterations, attempting forced extraction")

        # Last-ditch: ask the model directly for just a number
        try:
            assistant_text = "\n".join([m["content"] for m in messages if m["role"] == "assistant"])
            # Trim to last 3000 chars to stay within context
            summary_text = assistant_text[-3000:] if len(assistant_text) > 3000 else assistant_text
            force_messages = [
                {"role": "user", "content": f"A forecaster was analyzing this question: {question}\n\nHere is their research and analysis so far:\n{summary_text}\n\nBased on this analysis, what is the probability that the answer is YES? Reply with ONLY a number between 0 and 100 (representing the percentage). Just the number, nothing else."}
            ]
            force_response, _ = await self._llm_conversation(
                force_messages,
                max_tokens=100,
                temperature=0.2,
                usage_label="react:forced_extraction"
            )
            # Try to extract a number from the response
            num_match = re.search(r'(\d{1,3}(?:\.\d+)?)', force_response.strip())
            if num_match:
                forced_prob = float(num_match.group(1))
                if forced_prob > 1:
                    forced_prob = forced_prob / 100
                if 0.01 <= forced_prob <= 0.99:
                    logger.info(f"Forced extraction got probability: {forced_prob:.1%}")
                    full_text = "\n".join([m["content"] for m in messages if m["role"] == "assistant"])
                    if flags.spec_lock:
                        spec_status, spec_reason = await check_spec_consistency(
                            canonical_spec_text=canonical_spec_text,
                            model_answer=full_text,
                        )
                        if spec_status != "OK":
                            warning_messages.append(f"Spec consistency {spec_status}: {spec_reason}")
                            if spec_status == "MAJOR_DRIFT":
                                risk_flags.append("SPEC_MAJORDRIFT")
                    validator_warnings, validator_risks, diagnostics = _run_soft_validators(full_text)
                    warning_messages.extend(validator_warnings)
                    risk_flags.extend(validator_risks)

                    log_content += f"\n\n**[SYSTEM] Forced probability extraction: {forced_prob:.1%}**\n"

                    # Save log
                    log_content += f"""
---

## Summary (FORCED EXTRACTION)

- **Iterations**: {iteration}
- **Searches**: {search_count}
- **Total Tokens**: {total_tokens_used}
- **Extracted Probability**: {forced_prob:.1%}

## Search Queries Used

"""
                    for i, s in enumerate(all_searches, 1):
                        log_content += f"{i}. {s['query']}\n"
                    if warning_messages:
                        log_content += "\n## Warnings\n\n"
                        for w in warning_messages:
                            log_content += f"- {w}\n"
                    if risk_flags:
                        log_content += "\n## Risk Flags\n\n"
                        for rf in sorted(set(risk_flags)):
                            log_content += f"- {rf}\n"

                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(log_content)

                    return AgentForecastOutput(
                        probability=forced_prob,
                        explanation=f"[Forced extraction after iteration limit]\n\n{full_text[-5000:]}",
                        search_count=search_count,
                        sources=all_sources,
                        token_usage={
                            "prompt": prompt_tokens_used,
                            "completion": completion_tokens_used,
                            "total": total_tokens_used,
                        },
                        warnings=warning_messages,
                        risk_flags=sorted(set(risk_flags)),
                        diagnostics=diagnostics,
                    )
        except Exception as e:
            logger.warning(f"Forced extraction failed: {e}")

        full_text = "\n".join([m["content"] for m in messages if m["role"] == "assistant"])
        probability = extract_probability_from_forecast(full_text)
        if flags.spec_lock:
            spec_status, spec_reason = await check_spec_consistency(
                canonical_spec_text=canonical_spec_text,
                model_answer=full_text,
            )
            if spec_status != "OK":
                warning_messages.append(f"Spec consistency {spec_status}: {spec_reason}")
                if spec_status == "MAJOR_DRIFT":
                    risk_flags.append("SPEC_MAJORDRIFT")
        validator_warnings, validator_risks, diagnostics = _run_soft_validators(full_text)
        warning_messages.extend(validator_warnings)
        risk_flags.extend(validator_risks)

        # If still at default 0.5, try extracting any percentage from assistant messages
        if probability == 0.5:
            # Look for any explicit probability statements in reverse order (latest first)
            for msg in reversed(messages):
                if msg["role"] != "assistant":
                    continue
                # Look for patterns like "I estimate 35%", "probability of 70%", "around 20%", "likely ~60%"
                prob_patterns = [
                    r'(?:estimat|predict|assess|believ|think|probability|likelihood|chance|forecast)[^\n]{0,40}?(\d{1,2}(?:\.\d+)?)\s*%',
                    r'(\d{1,2}(?:\.\d+)?)\s*%\s*(?:probability|likely|chance|likelihood)',
                    r'(?:around|approximately|roughly|about|~)\s*(\d{1,2}(?:\.\d+)?)\s*%',
                ]
                for pat in prob_patterns:
                    m = re.search(pat, msg["content"], re.IGNORECASE)
                    if m:
                        candidate = float(m.group(1)) / 100
                        if 0.01 <= candidate <= 0.99:
                            probability = candidate
                            logger.info(f"Extracted fallback probability {probability:.1%} from conversation")
                            break
                if probability != 0.5:
                    break
        
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
        if warning_messages:
            log_content += "\n## Warnings\n\n"
            for w in warning_messages:
                log_content += f"- {w}\n"
        if risk_flags:
            log_content += "\n## Risk Flags\n\n"
            for rf in sorted(set(risk_flags)):
                log_content += f"- {rf}\n"
        
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
            },
            warnings=warning_messages,
            risk_flags=sorted(set(risk_flags)),
            diagnostics=diagnostics,
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
    feature_flags: Optional[dict[str, Any]] = None,
    outlier_threshold_pp: float = 15.0,
    question_type: str = "binary",
    options: Optional[list[str]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    unit: Optional[str] = None,
) -> dict:
    """
    Run an ensemble of forecasting agents on a question.
    
    Args:
        question: The forecasting question text.
        models: List of model configs. If None, uses default ensemble.
        publish_to_metaculus: Whether to post the result to Metaculus.
        community_prior: Metaculus community prediction (0-1) to use as anchor.
        use_react: If True, use new ReAct-style iterative agent. If False, use legacy agent.
        feature_flags: Optional module toggles (spec_lock, evidence_ledger, numeric_provenance, market_snapshot, outlier_xexam).
        outlier_threshold_pp: Outlier threshold in percentage points for cross-exam.
        
    Returns:
        dict containing:
        - final_probability: float
        - summary_text: str
        - full_log: str
        - individual_results: list[dict]
    """
    from .lean_ensemble import run_lean_ensemble_forecast

    return await run_lean_ensemble_forecast(
        question=question,
        models=models,
        publish_to_metaculus=publish_to_metaculus,
        community_prior=community_prior,
        use_react=use_react,
        feature_flags=feature_flags,
        outlier_threshold_pp=outlier_threshold_pp,
        question_type=question_type,
        options=options,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
        extract_probability_fn=extract_probability_from_forecast,
        canonical_spec_extractor=extract_canonical_spec,
        canonical_spec_formatter=_format_canonical_spec_text,
        spec_consistency_checker=check_spec_consistency,
    )

    flags = _flags_from_dict(feature_flags)
    original_question_input = question.strip()
    question_url_for_post: Optional[str] = None
    if original_question_input.lower().startswith("http") and "metaculus.com/questions/" in original_question_input:
        question_url_for_post = original_question_input
        try:
            meta_q = MetaculusApi.get_question_by_url(question_url_for_post)
            if meta_q is not None:
                rich_parts = [
                    getattr(meta_q, "question_text", "") or "",
                    getattr(meta_q, "resolution_criteria", "") or "",
                    getattr(meta_q, "fine_print", "") or "",
                    getattr(meta_q, "background_info", "") or "",
                ]
                question = "\n\n".join([p.strip() for p in rich_parts if p and p.strip()])[:12000]
                cp = getattr(meta_q, "community_prediction_at_access_time", None)
                if community_prior is None and isinstance(cp, (float, int)) and 0 <= cp <= 1:
                    community_prior = float(cp)
                logger.info("Loaded rich Metaculus question content from URL input.")
        except Exception as meta_exc:
            logger.warning(f"Could not enrich question from Metaculus URL: {meta_exc}")

    if models is None:
        # Default production trio requested by user.
        models = [
            {
                "name": "moonshotai/kimi-k2.5",
                "reasoning_effort": None,  # Kimi has built-in reasoning
                "max_tokens": 100000,
                "label": "Kimi K2.5"
            },
            {
                "name": "google/gemini-2.5-flash",
                "reasoning_effort": "medium",
                "max_tokens": 100000,
                "label": "Gemini 2.5 Flash"
            },
            {
                "name": "openai/gpt-5-mini",
                "reasoning_effort": "medium",
                "max_tokens": 32000,
                "label": "GPT-5 Mini"
            },
        ]

    agent_type = "ReAct" if use_react else "Legacy"
    print(f"Starting Multi-Model Ensemble Forecast ({agent_type}) for: {question}")
    print("-" * 60)

    logs_dir = "forecasts"
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    canonical_spec = await extract_canonical_spec(question) if flags.spec_lock else None
    canonical_spec_text = _format_canonical_spec_text(canonical_spec, question)

    full_log_content = (
        f"# FORECAST LOG - {timestamp}\n"
        f"Agent Type: {agent_type}\n\n"
        f"Question: {question}\n\n"
        f"Feature Flags: {flags}\n\n"
        f"Canonical Spec:\n{canonical_spec_text}\n\n"
    )

    # --- Run all models in parallel ---
    async def _run_one(config: dict) -> dict:
        """Run a single model and return config + result or error."""
        agent = ForecastingAgent(
            model_name=config["name"],
            reasoning_effort=config.get("reasoning_effort")
        )
        if use_react:
            result = await agent.run_react_agent(
                question,
                community_prior=community_prior,
                canonical_spec=canonical_spec,
                feature_flags=flags,
            )
        else:
            result = await agent.run_forecast(question, community_prior=community_prior)
        return {"config": config, "result": result}

    tasks = [_run_one(cfg) for cfg in models]
    print(f"Running {len(models)} models in parallel...")
    settled = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, outcome in enumerate(settled):
        config = models[i]
        if isinstance(outcome, Exception):
            logger.error(f"Error running {config['label']}: {outcome}")
            full_log_content += f"\n\nERROR running {config['label']}: {outcome}\n\n"
            print(f"  {config['label']}: FAILED ({outcome})")
        else:
            results.append(outcome)
            result = outcome["result"]

            # Build sources table for this model
            sources_table = ""
            if hasattr(result, 'sources') and result.sources:
                sources_table = "\n--- SOURCES CONSULTED ---\n"
                sources_table += "| # | Title | URL | Date |\n|---|-------|-----|------|\n"
                for j, src in enumerate(result.sources[:15], 1):
                    title = src.get('title', 'N/A')[:50]
                    url = src.get('url', 'N/A')
                    date = src.get('date', 'N/A')
                    sources_table += f"| {j} | {title} | {url} | {date} |\n"

            # Build metrics block
            metrics_block = "\n--- METRICS ---\n"
            if hasattr(result, 'search_count'):
                metrics_block += f"Searches: {result.search_count}\n"
            if hasattr(result, 'sources') and result.sources:
                metrics_block += f"Sources: {len(result.sources)}\n"
            if hasattr(result, 'token_usage') and result.token_usage:
                tu = result.token_usage
                metrics_block += f"Tokens: prompt={tu.get('prompt', 0):,}, completion={tu.get('completion', 0):,}, total={tu.get('total', 0):,}\n"
            if hasattr(result, "warnings") and result.warnings:
                metrics_block += f"Warnings: {len(result.warnings)}\n"
            if hasattr(result, "risk_flags") and result.risk_flags:
                metrics_block += f"Risk Flags: {', '.join(sorted(set(result.risk_flags)))}\n"

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
            if getattr(result, "warnings", None):
                log_section += "\n--- WARNINGS ---\n" + "\n".join([f"- {w}" for w in result.warnings]) + "\n"
            if getattr(result, "risk_flags", None):
                log_section += "\n--- RISK FLAGS ---\n" + "\n".join([f"- {rf}" for rf in sorted(set(result.risk_flags))]) + "\n"
            full_log_content += log_section
            print(f"  {config['label']}: {result.probability:.1%} ({result.search_count} searches)")

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

    # Filter out failed models: exclude results with 0 searches AND 0.5 probability
    # (these are models that returned errors and fell back to the default)
    valid_results = [
        r for r in results
        if not (r["result"].probability == 0.5 and r["result"].search_count == 0)
    ]
    if not valid_results:
        # All models failed — fall back to all results
        logger.warning("All models appear to have failed (0.5 with 0 searches). Using unfiltered results.")
        valid_results = results

    if len(valid_results) < len(results):
        failed_labels = [r["config"]["label"] for r in results if r not in valid_results]
        logger.warning(f"Filtered out failed models from ensemble: {failed_labels}")

    def _is_forced_extraction(result: AgentForecastOutput) -> bool:
        explanation = (result.explanation or "").lower()
        return (
            "[forced extraction" in explanation
            or "[agent reached iteration limit]" in explanation
        )

    def _reliability_label(result: AgentForecastOutput) -> str:
        if _is_forced_extraction(result):
            return "LOW"
        total_tokens = (result.token_usage or {}).get("total", 0)
        if total_tokens >= 60000:
            return "MEDIUM"
        return "HIGH"

    def _quality_weight(result: AgentForecastOutput) -> float:
        # Cap search contribution so query spam cannot dominate.
        search_component = float(max(1, min(result.search_count, 8)))
        reliability_multiplier = 1.0
        if _is_forced_extraction(result):
            reliability_multiplier *= 0.35
        total_tokens = (result.token_usage or {}).get("total", 0)
        if total_tokens >= 70000:
            reliability_multiplier *= 0.85
        return max(0.5, search_component * reliability_multiplier)

    # --- Judge / Synthesis Step ---
    # Build summaries of each model's key evidence and probability for the judge
    model_summaries_parts = []
    for r in valid_results:
        label = r["config"]["label"]
        prob = r["result"].probability
        searches = r["result"].search_count
        reliability = _reliability_label(r["result"])
        warn_text = "; ".join((r["result"].warnings or [])[:3]) if hasattr(r["result"], "warnings") else ""
        risk_text = ", ".join(sorted(set(r["result"].risk_flags or []))) if hasattr(r["result"], "risk_flags") else ""
        market_found = False
        if hasattr(r["result"], "diagnostics") and r["result"].diagnostics:
            market_found = bool((r["result"].diagnostics.get("market_snapshot") or {}).get("found"))
        explanation = r["result"].explanation or "No explanation available"
        # Trim explanation to key parts (last 1500 chars contains the reasoning/forecast)
        if len(explanation) > 2000:
            explanation = explanation[-2000:]
        model_summaries_parts.append(
            f"### {label} - {prob:.0%} (based on {searches} searches, reliability={reliability}, market_found={market_found})\n"
            f"Warnings: {warn_text or 'None'}\n"
            f"Risk flags: {risk_text or 'None'}\n\n"
            f"{explanation}"
        )
    model_summaries = "\n\n---\n\n".join(model_summaries_parts)

    # Compute weighted average as fallback
    weights = [_quality_weight(r["result"]) for r in valid_results]
    probs = [r["result"].probability for r in valid_results]

    # Outlier cross-exam: interrogate large deviations instead of hard-dropping.
    if flags.outlier_xexam and len(valid_results) >= 3:
        med = median(probs)
        for idx, r in enumerate(valid_results):
            pp_diff = abs(r["result"].probability - med) * 100
            if pp_diff <= outlier_threshold_pp:
                continue
            ledger = []
            if getattr(r["result"], "diagnostics", None):
                ledger = r["result"].diagnostics.get("evidence_ledger", []) or []
            ledger_summary = "\n".join(
                [
                    f"{item.get('id')} | {item.get('directness_tag')} | {item.get('url_ref')}"
                    for item in ledger[:8]
                ]
            ) or "No ledger items"
            try:
                xexam_prompt = OUTLIER_CROSSEXAM_PROMPT.format(
                    question=question,
                    canonical_spec_text=canonical_spec_text,
                    model_answer=(r["result"].explanation or "")[-5000:],
                    ledger_summary=ledger_summary,
                )
                xexam_response = await call_openrouter_llm(
                    prompt=xexam_prompt,
                    model="google/gemini-2.5-flash",
                    temperature=0.2,
                    max_tokens=1200,
                    usage_label="outlier_xexam",
                )
                if r["result"].diagnostics is not None:
                    r["result"].diagnostics["outlier_cross_exam"] = xexam_response
                # Require explicit grounding syntax.
                grounding_hits = len(
                    re.findall(r'Evidence:\s*(LEDGER-\d+|ASSUMPTION)', xexam_response, re.IGNORECASE)
                )
                if grounding_hits < 3:
                    weights[idx] *= 0.60
                    r["result"].warnings.append(
                        f"Outlier cross-exam ungrounded (diff {pp_diff:.1f}pp, grounding_hits={grounding_hits})."
                    )
                    r["result"].risk_flags.append("OUTLIER_UNGROUNDED")
                else:
                    r["result"].warnings.append(
                        f"Outlier cross-exam passed (diff {pp_diff:.1f}pp)."
                    )
            except Exception as xexc:
                r["result"].warnings.append(f"Outlier cross-exam failed: {xexc}")
                r["result"].risk_flags.append("OUTLIER_XEXAM_ERROR")

    total_weight = sum(weights)
    weighted_avg = sum(p * w for p, w in zip(probs, weights)) / total_weight

    # Run the judge synthesis via LLM
    judge_model = "google/gemini-2.5-flash"
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    judge_prompt = JUDGE_SYNTHESIS_PROMPT.format(
        question=question,
        today=today_str,
        model_summaries=model_summaries,
    )

    avg_prob = weighted_avg  # default to weighted average
    judge_reasoning = ""
    try:
        print("\nRunning judge synthesis...")
        judge_response = await call_openrouter_llm(
            prompt=judge_prompt,
            model=judge_model,
            temperature=0.3,
            max_tokens=4000,
            usage_label="judge_synthesis",
            reasoning_effort="medium",
        )
        # Extract probability from judge response
        judge_response_plain = judge_response.replace("*", "")
        judge_prob_match = re.search(r'FINAL_PROBABILITY\s*:\s*(\d{1,3}(?:\.\d+)?)\s*%', judge_response_plain, re.IGNORECASE)
        if not judge_prob_match:
            judge_prob_match = re.search(r'FINAL_PROBABILITY[^0-9]{0,12}(\d{1,3}(?:\.\d+)?)\s*%', judge_response_plain, re.IGNORECASE)
        if not judge_prob_match:
            judge_prob_match = re.search(r'Final probability[^0-9]{0,12}(\d{1,3}(?:\.\d+)?)\s*%', judge_response_plain, re.IGNORECASE)
        if judge_prob_match:
            judge_prob = float(judge_prob_match.group(1)) / 100.0
            if 0.01 <= judge_prob <= 0.99:
                avg_prob = judge_prob
                judge_reasoning = judge_response
                logger.info(f"Judge synthesis probability: {avg_prob:.1%}")
            else:
                logger.warning(f"Judge probability out of range: {judge_prob}, using weighted average")
        else:
            logger.warning("Could not extract probability from judge response, using weighted average")
            judge_reasoning = judge_response
    except Exception as e:
        logger.error(f"Judge synthesis failed: {e}, using weighted average")

    # Log weighting details
    for r, w in zip(valid_results, weights):
        label = r["config"]["label"]
        p = r["result"].probability
        reliability = _reliability_label(r["result"])
        pct_w = w / total_weight * 100
        logger.info(f"Individual: {label} p={p:.1%} weight={w:.2f} ({pct_w:.0f}%), reliability={reliability}")
    logger.info(f"Weighted average: {weighted_avg:.1%} | Judge synthesis: {avg_prob:.1%}")

    print(f"\n{'='*60}")
    print(f"WEIGHTED AVERAGE: {weighted_avg:.1%}")
    print(f"JUDGE SYNTHESIS:  {avg_prob:.1%}")
    print(f"FINAL PREDICTION: {avg_prob:.1%}")
    print(f"{'='*60}")

    # Create compact summary
    summary_lines = [f"Final Prediction: {avg_prob:.1%} (Judge Synthesis)"]
    summary_lines.append(f"Weighted Average: {weighted_avg:.1%}")
    summary_lines.append("Individual Models:")
    for r in results:
        rel = _reliability_label(r["result"])
        warn_count = len(r["result"].warnings or []) if hasattr(r["result"], "warnings") else 0
        risk_count = len(set(r["result"].risk_flags or [])) if hasattr(r["result"], "risk_flags") else 0
        summary_lines.append(
            f"- {r['config']['label']}: {r['result'].probability:.1%} "
            f"({r['result'].search_count} searches, reliability={rel}, warnings={warn_count}, risks={risk_count})"
        )

    summary_text = "\n".join(summary_lines)

    # Create full reasoning text including all model explanations + judge
    full_reasoning_parts = [f"# Ensemble Forecast: {avg_prob:.1%}\n"]
    for r in results:
        model_name = r['config']['label']
        model_prob = r['result'].probability
        model_explanation = r['result'].explanation or "No explanation available"
        full_reasoning_parts.append(f"## {model_name}: {model_prob:.1%}\n\n{model_explanation}\n")
    if judge_reasoning:
        full_reasoning_parts.append(f"## Judge Synthesis: {avg_prob:.1%}\n\n{judge_reasoning}\n")

    full_reasoning = "\n---\n\n".join(full_reasoning_parts)

    # Add judge section to log
    judge_log = ""
    if judge_reasoning:
        judge_log = f"""

{'='*60}
JUDGE SYNTHESIS (Model: {judge_model})
{'='*60}

{judge_reasoning}
"""
    full_log_content += judge_log
    full_log_content += f"\n\n{'='*60}\nSUMMARY\n{'='*60}\n{summary_text}\n"

    # Re-save log file with judge synthesis and summary appended
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(full_log_content)
    print(f"Updated log saved to: {log_filename}")

    # Post to Metaculus
    if publish_to_metaculus:
        metaculus_token = os.getenv("METACULUS_TOKEN")
        if metaculus_token:
            try:
                if question_url_for_post:
                    found_question = MetaculusApi.get_question_by_url(question_url_for_post)
                else:
                    found_question = MetaculusApi.get_question_by_url(question)
                if found_question:
                    # Handle different attribute names (.id, .question_id, .id_of_question, .post_id)
                    q_id = (
                        getattr(found_question, 'id', None) or 
                        getattr(found_question, 'question_id', None) or 
                        getattr(found_question, 'id_of_question', None) or
                        getattr(found_question, 'id_of_post', None) or
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
                            post_id = (
                                getattr(found_question, 'id_of_post', None) or
                                getattr(found_question, 'post_id', None) or
                                (getattr(found_question, 'api_json', {}) or {}).get('id')
                            )
                            MetaculusApi.post_question_comment(
                                post_id=int(post_id),
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
        "individual_results": results,
        "feature_flags": {
            "spec_lock": flags.spec_lock,
            "evidence_ledger": flags.evidence_ledger,
            "numeric_provenance": flags.numeric_provenance,
            "market_snapshot": flags.market_snapshot,
            "outlier_xexam": flags.outlier_xexam,
        },
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
