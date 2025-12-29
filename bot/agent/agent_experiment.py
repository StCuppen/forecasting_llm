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
    extract_search_queries,
    extract_market_probabilities,
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


def extract_probability_from_forecast(forecast_text: str) -> float:
    """Extract probability from agent forecast text.

    Looks for patterns like:
    - FINAL_PROBABILITY (0-1): 0.57
    - FINAL_PROBABILITY: 0.57
    - Final probability: 57%
    - **Final Probability**: 0.35
    - Probability: 35%
    - P = 0.35
    - 35% probability
    """
    # Try FINAL_PROBABILITY (0-1): format first
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
) -> dict:
    """
    Run an ensemble of forecasting agents on a question.
    
    Args:
        question: The forecasting question text.
        models: List of model configs. If None, uses default ensemble.
        publish_to_metaculus: Whether to post the result to Metaculus.
        community_prior: Metaculus community prediction (0-1) to use as anchor.
        
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
                "name": "openai/gpt-5-mini",
                "reasoning_effort": "medium",  # Enable reasoning for better forecasts
                "max_tokens": 100000,
                "label": "GPT-5 Mini (Reasoning)"
            },
            {
                "name": "google/gemini-3-flash-preview",
                "reasoning_effort": "medium",  # Enable reasoning for better forecasts
                "max_tokens": 100000,
                "label": "Gemini 3 Flash (Reasoning)"
            },
        ]

    print(f"Starting Multi-Model Ensemble Forecast for: {question}")
    print("-" * 60)

    results = []
    logs_dir = "forecasts"
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    full_log_content = f"# FORECAST LOG - {timestamp}\n\nQuestion: {question}\n\n"

    for config in models:
        print(f"\n>>> Running {config['label']} ({config['name']})...")
        try:
            agent = ForecastingAgent(
                model_name=config["name"],
                reasoning_effort=config.get("reasoning_effort")
            )
            
            result = await agent.run_forecast(question, community_prior=community_prior)
            
            results.append({
                "config": config,
                "result": result
            })
            
            log_section = f"""
{'='*60}
MODEL: {config['label']} ({config['name']})
REASONING: {config.get('reasoning_effort') or 'Default'}
MAX_TOKENS: {config['max_tokens']}
{'='*60}

PREDICTION: {result.probability:.1%}

--- EXPLANATION ---
{result.explanation}

--- RESEARCH MEMO ---
{getattr(agent, '_last_research_memo', 'Not available')}
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
