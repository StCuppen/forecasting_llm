"""
Clean Tavily-based forecasting pipeline.

This file implements a simplified forecasting approach:
- Tavily is used ONLY for research (outside view + inside view)
- NO personas, NO synthesizer, NO base-rate caching
- 3 models (GPT-5-mini, Gemini 2.5 Flash, Claude 3.5 Haiku) each make ONE forecast
- Aggregate via median
"""

import argparse
import asyncio
import hashlib
import logging
import os
import json
from typing import Literal, Optional
from datetime import datetime
import numpy as np

import dotenv
dotenv.load_dotenv()

# OpenAI model selector
def get_openai_model() -> str:
    """Return preferred OpenAI model, defaulting to GPT-5 mini."""
    value = os.getenv("PREFERRED_OPENAI_MODEL")
    return value.strip() if value and value.strip() else "gpt-5-mini"

# Add OpenAI package for direct calls
import openai

# Configure OpenAI client if key exists
if os.getenv("OPENAI_API_KEY"):
    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
# Or use the Metaculus proxy if OpenAI key doesn't exist
elif os.getenv("METACULUS_TOKEN"):
    openai_client = openai.AsyncOpenAI(
        base_url="https://llm-proxy.metaculus.com/proxy/openai/v1",
        default_headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {os.getenv('METACULUS_TOKEN')}",
        },
        api_key="not-used",  # Required by client but not actually used
    )

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)

from bot.agent.utils import ExaClient, SerperClient, TavilyClient, SonarClient, BraveClient, reset_token_usage, get_token_usage
from bot.agent.agent_experiment import run_ensemble_forecast

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ========================================
# Helper Functions
# ========================================

def build_report(
    question_text: str,
    today_str: str,
    model_results: list[dict],
    token_usage: dict | None = None,
) -> str:
    """Build a transparent forecast report (no embedded research).

    Research text is handled separately by run_research; this report
    contains token usage (if provided), a summary, and per-model forecasts.
    """
    probs = [r["probability"] for r in model_results if r.get("probability") is not None]
    agg = np.median(probs) if probs else 0.5
    lo = min(probs) if probs else 0.5
    hi = max(probs) if probs else 0.5

    summary = f"""SUMMARY
Question: {question_text}
Date: {today_str}
Final Prediction: {round(agg * 100, 1)}%
Range Across Models: {round(lo * 100, 1)}% - {round(hi * 100, 1)}%
"""

    # Compact forecasts summary block (per-model + aggregate)
    forecasts_summary_lines = ["FORECASTS"]
    for r in model_results:
        forecasts_summary_lines.append(
            f"- {r['model_name']}: {round(r['probability'] * 100, 1)}%"
        )
    forecasts_summary_lines.append(
        f"- Final aggregate (median): {round(agg * 100, 1)}%"
    )
    forecasts_summary = "\n".join(forecasts_summary_lines)

    forecast_sections = []
    for i, r in enumerate(model_results, 1):
        forecast_sections.append(f"## Forecast {i} - {r['model_name']}\n\n{r.get('forecast_text', '')}")
    forecasts_block = "# FORECASTS\n\n" + "\n\n".join(forecast_sections)

    token_block = ""
    if token_usage:
        lines = []
        for label, stats in token_usage.items():
            lines.append(
                f"- {label}: prompt={stats.get('prompt', 0)}, "
                f"completion={stats.get('completion', 0)}, total={stats.get('total', 0)}"
            )
        if lines:
            token_block = "TOKENS\n" + "\n".join(lines) + "\n\n"

    report = token_block + summary + "\n\n" + forecasts_summary + "\n\n\n" + forecasts_block
    return report


# ========================================
# TemplateForecaster
# ========================================

# ========================================
# TemplateForecaster
# ========================================

class TemplateForecaster(ForecastBot):
    """Clean forecasting bot using Tavily for research and 3-model ensemble."""

    def __init__(self, **kwargs):
        """Initialize the forecaster.
        
        Args:
            **kwargs: Passed to ForecastBot parent class
        """
        super().__init__(**kwargs)
        
        # Cache for research text blocks
        self._research_cache: dict[str, str] = {}
        
        # Exa is now the sole search provider
        exa_key = os.getenv("EXA_API_KEY")
        if exa_key:
            self.exa_client = ExaClient(api_key=exa_key)
            logger.info("Exa.ai enabled as sole search provider.")
        else:
            self.exa_client = None
            logger.warning("EXA_API_KEY not set; Exa disabled.")

        # Disable others as per user request
        self.serper_client = None
        self.brave_client = None
        self.tavily_client = None
        self.sonar_client = None

        logger.info("TemplateForecaster initialized (Search provider: Exa)")


    @staticmethod
    def log_report_summary(
        forecast_reports,  # type: ignore[override]
        raise_errors: bool = True,
    ) -> None:
        """
        Override the base class summary logger.

        The default implementation prints an additional "Report 1 Summary" /
        "Research Summary" block which duplicates the custom SUMMARY / FORECASTS
        sections that this bot already writes into the explanation. To avoid
        that noise, this override suppresses the per-report summary and only
        raises/logs aggregate errors if present.
        """
        from forecasting_tools.data_models.forecast_report import ForecastReport  # type: ignore
        from typing import Sequence

        exceptions = [
            report
            for report in forecast_reports  # type: ignore[assignment]
            if not isinstance(report, ForecastReport)
            and isinstance(report, BaseException)  # type: ignore[name-defined]
        ]
        if exceptions and raise_errors:
            raise RuntimeError(
                f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
            )

    async def summarize_research(
        self,
        question: MetaculusQuestion,
        research: str,
    ) -> str:
        """
        Override base summarizer to avoid duplicating the full planner+memo.

        Returns a compact summary focused on the RESEARCH MEMO portion
        (if present), trimmed to a reasonable length.
        """
        logger.info(f"TemplateForecaster.summarize_research for: {getattr(question, 'page_url', None)}")
        marker = "RESEARCH MEMO"
        idx = research.find(marker)
        memo = research[idx:] if idx != -1 else research
        max_chars = 2500
        if len(memo) > max_chars:
            memo = memo[:max_chars] + "..."
        return memo

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Execute the research pipeline.
        
        NOTE: Since we are using the ensemble agent which does its own research,
        this method is largely a placeholder to satisfy the ForecastBot interface.
        """
        logger.info(f"Skipping standalone research for {question.question_text} (handled by ensemble agents)")
        return "Research handled by individual agents in ensemble."

    async def _run_forecast_on_binary(
        self,
        question: BinaryQuestion,
        research: str,
    ) -> ReasonedPrediction:
        """Run forecast on binary question using the agent ensemble."""
        logger.info(f"Running ensemble forecast for: {question.question_text}")
        
        # Extract community prediction from Metaculus if available
        community_prior = None
        if hasattr(question, 'community_prediction_at_access_time'):
            community_prior = question.community_prediction_at_access_time
            if community_prior is not None:
                logger.info(f"Using Metaculus community prior: {community_prior:.1%}")
        
        try:
            # Call the ensemble function from agent_experiment.py
            # We do NOT publish to Metaculus here because the ForecastBot framework handles that.
            result = await run_ensemble_forecast(
                question=question.question_text,
                models=None, # Use default ensemble
                publish_to_metaculus=False,
                community_prior=community_prior,
            )
            
            final_prob = result["final_probability"]
            summary_text = result["summary_text"]
            
            # We can append the full log or a link to it if we want, but for now just the summary
            # The full log is already saved to disk by run_ensemble_forecast
            
            return ReasonedPrediction(
                prediction_value=final_prob,
                reasoning=summary_text + "\n\n(Generated by Multi-Model Agent Ensemble)"
            )
            
        except Exception as e:
            logger.error(f"Ensemble forecast failed: {e}")
            return ReasonedPrediction(
                prediction_value=0.5,
                reasoning=f"Ensemble forecast failed: {e}"
            )

    async def _run_forecast_on_multiple_choice(
        self,
        question: MultipleChoiceQuestion,
        research: str,
    ) -> PredictedOptionList:
        """Stub implementation for multiple choice questions."""
        logger.warning("Multiple choice questions not supported in clean pipeline")
        raise NotImplementedError("Multiple choice forecasting not implemented in clean pipeline")

    async def _run_forecast_on_numeric(
        self,
        question: NumericQuestion,
        research: str,
    ) -> NumericDistribution:
        """Stub implementation for numeric questions."""
        logger.warning("Numeric questions not supported in clean pipeline")
        raise NotImplementedError("Numeric forecasting not implemented in clean pipeline")



# ========================================
# CLI and Main
# ========================================

def main():
    """Main entrypoint for the forecasting bot."""
    parser = argparse.ArgumentParser(description="Clean Tavily-based forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "urls", "test_questions"],
        default="tournament",
        help="Run mode: tournament, urls, or test_questions"
    )
    parser.add_argument(
        "--urls",
        type=str,
        help="Comma-separated list of Metaculus question URLs (for urls mode)"
    )
    parser.add_argument(
        "--force-repost",
        action="store_true",
        help="Force re-post on previously forecasted questions"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./forecast_reports.json",
        help="Path to save forecast reports"
    )

    args = parser.parse_args()
    run_mode = args.mode

    # Initialize bot
    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # 3 models, but treated as 1 ensemble
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="./forecast_reports",
        skip_previously_forecasted_questions=True,
    )

    # Startup banner
    logger.info(
        f"RUN MODE={run_mode} | publish={template_bot.publish_reports_to_metaculus} | "
        f"OPENROUTER={bool(os.getenv('OPENROUTER_API_KEY'))} | "
        f"METACULUS={bool(os.getenv('METACULUS_TOKEN'))} | "
        f"TAVILY={bool(os.getenv('TAVILY_API_KEY'))}"
    )

    # Apply CLI overrides
    if getattr(args, "force_repost", False):
        template_bot.skip_previously_forecasted_questions = False

    # Run based on mode
    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions for testing
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Binary
        ]
        template_bot.skip_previously_forecasted_questions = False
        template_bot.publish_reports_to_metaculus = False
        
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    elif run_mode == "urls":
        if not args.urls:
            raise SystemExit("--urls is required when --mode urls")
        
        if getattr(args, "force_repost", False):
            template_bot.skip_previously_forecasted_questions = False
        
        template_bot.publish_reports_to_metaculus = True
        
        url_list = [u.strip() for u in args.urls.split(",") if u.strip()]
        questions = [MetaculusApi.get_question_by_url(u) for u in url_list]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    else:
        raise SystemExit(f"Unknown mode: {run_mode}")

    # Post-process explanations to drop the default "Report 1 Summary" section
    # and keep only the full RESEARCH block + forecast rationales.
    # This prevents duplicated "Research Summary" while preserving detailed
    # research and reasoning.
    try:
        from forecasting_tools.data_models.forecast_report import ForecastReport as _FR  # type: ignore
        for idx, r in enumerate(forecast_reports):
            if isinstance(r, _FR):
                try:
                    new_expl = r.research + "\n\n" + r.forecast_rationales
                    r.explanation = new_expl
                except Exception:
                    # If anything goes wrong, leave the original explanation
                    continue
    except Exception as e:
        logger.warning(f"Could not post-process explanations to strip summary sections: {e}")

    # Write forecast reports
    try:
        out_path = args.output_file
        serializable = []
        for r in forecast_reports:
            try:
                if hasattr(r, 'to_dict'):
                    serializable.append(r.to_dict())
                else:
                    serializable.append(repr(r))
            except Exception:
                serializable.append(repr(r))
        with open(out_path, 'w', encoding='utf-8') as wf:
            json.dump(serializable, wf, indent=2, ensure_ascii=False)
        logging.info(f"Wrote forecast reports to {out_path}")
    except Exception as e:
        logging.error(f"Failed to write forecast reports to {args.output_file}: {e}")

    # Emit submission logs
    try:
        publish_flag = bool(getattr(template_bot, 'publish_reports_to_metaculus', False))
        skip_flag = bool(getattr(template_bot, 'skip_previously_forecasted_questions', True))
        for r in forecast_reports:
            q = getattr(r, 'question', None)
            url = getattr(q, 'page_url', None)
            already_forecasted = bool(getattr(q, 'already_forecasted', False)) if q else None
            if not publish_flag:
                logging.info(f"Submission Skipped: publish disabled | url={url}")
                continue
            if already_forecasted and skip_flag:
                logging.info(f"Submission Skipped: already_forecasted and skip enabled | url={url}")
            else:
                logging.info(f"Submission Attempt: posting forecast | url={url}")
    except Exception as e:
        logging.warning(f"Could not emit submission-intent logs: {e}")


if __name__ == "__main__":
    main()
