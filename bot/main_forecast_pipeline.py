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
from typing import Literal, Optional, Any
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
    ApiFilter,
    QuestionState,
)

from bot.agent.utils import ExaClient, SerperClient, TavilyClient, SonarClient, BraveClient, reset_token_usage, get_token_usage
from bot.agent.agent_experiment import run_ensemble_forecast
from bot.coherence import enforce_sum_to_one

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

    async def _make_prediction(
        self,
        question: MetaculusQuestion,
        research: str,
    ) -> ReasonedPrediction[Any]:
        """
        Override base prediction wrapper.

        forecasting_tools currently validates intermediate prediction objects as
        ReasonedPrediction. For non-binary types we therefore wrap the concrete
        prediction payload inside ReasonedPrediction(prediction_value=...).
        """
        notepad = await self._get_notepad(question)
        notepad.total_predictions_attempted += 1

        if isinstance(question, BinaryQuestion):
            return await self._run_forecast_on_binary(question, research)
        if isinstance(question, MultipleChoiceQuestion):
            mc_prediction = await self._run_forecast_on_multiple_choice(question, research)
            return ReasonedPrediction(
                prediction_value=mc_prediction,
                reasoning="Fallback multiple-choice distribution generated by TemplateForecaster.",
            )
        if isinstance(question, NumericQuestion):
            numeric_prediction = await self._run_forecast_on_numeric(question, research)
            return ReasonedPrediction(
                prediction_value=numeric_prediction,
                reasoning="Fallback numeric distribution generated by TemplateForecaster.",
            )
        raise ValueError(f"Unknown or unsupported question type: {type(question)}")

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
            
            final_prob = float(result["final_probability"])
            # Metaculus forecast endpoint rejects extreme values (<0.01 or >0.99).
            # Clamp here so a single model/pathological output doesn't make the whole
            # question fail to post.
            final_prob = max(0.01, min(0.99, final_prob))
            # Use full reasoning instead of just summary so complete analysis is posted
            full_reasoning = result.get("full_reasoning", result["summary_text"])
            
            return ReasonedPrediction(
                prediction_value=final_prob,
                reasoning=full_reasoning + "\n\n---\n*Generated by Multi-Model Agent Ensemble*"
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
        options = [opt.strip() for opt in question.options if opt and opt.strip()]
        if not options:
            raise ValueError("Multiple choice question has no options.")

        try:
            result = await run_ensemble_forecast(
                question=question.question_text,
                models=None,
                publish_to_metaculus=False,
                question_type="multiple_choice",
                options=options,
            )
            mc_probs = result.get("mc_probabilities")
            if not isinstance(mc_probs, dict):
                raise ValueError("Missing mc_probabilities from ensemble output")
            normalized = enforce_sum_to_one({str(k): float(v) for k, v in mc_probs.items()})
            predicted_options = [
                {"option_name": option, "probability": float(normalized.get(option, 0.0))}
                for option in options
            ]
            logger.info(
                "Generated model-based multiple-choice forecast "
                f"(parse_failure_rate={result.get('parsing_failure_rate')})."
            )
            return PredictedOptionList(predicted_options=predicted_options)
        except Exception as e:
            logger.warning(f"MC model path failed, using fallback uniform distribution: {e}")
            base_probability = 1.0 / len(options)
            predicted_options = [
                {"option_name": option, "probability": base_probability}
                for option in options
            ]
            if len(predicted_options) > 1:
                running_sum = sum(item["probability"] for item in predicted_options[:-1])
                predicted_options[-1]["probability"] = max(0.0, min(1.0, 1.0 - running_sum))
            return PredictedOptionList(predicted_options=predicted_options)

    async def _run_forecast_on_numeric(
        self,
        question: NumericQuestion,
        research: str,
    ) -> NumericDistribution:
        lower = float(question.lower_bound)
        upper = float(question.upper_bound)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"Numeric bounds must be finite. lower={lower}, upper={upper}")
        if upper <= lower:
            raise ValueError(f"Numeric bounds invalid. lower={lower}, upper={upper}")

        unit = str(getattr(question, "unit", "") or getattr(question, "units", "") or "")
        try:
            result = await run_ensemble_forecast(
                question=question.question_text,
                models=None,
                publish_to_metaculus=False,
                question_type="numeric",
                lower_bound=lower,
                upper_bound=upper,
                unit=unit,
            )
            pct = result.get("numeric_percentiles")
            if not isinstance(pct, dict):
                raise ValueError("Missing numeric_percentiles from ensemble output")
            points = []
            for pctl in [10, 25, 50, 75, 90]:
                value = float(pct.get(pctl, pct.get(str(pctl), lower)))
                value = max(lower, min(upper, value))
                points.append({"value": value, "percentile": pctl / 100.0})
            # monotonic repair in case external coercion introduced small violations
            for i in range(1, len(points)):
                if points[i]["value"] < points[i - 1]["value"]:
                    points[i]["value"] = points[i - 1]["value"]
            logger.info(
                "Generated model-based numeric forecast "
                f"(parse_failure_rate={result.get('parsing_failure_rate')})."
            )
            return NumericDistribution(
                declared_percentiles=points,
                open_upper_bound=bool(question.open_upper_bound),
                open_lower_bound=bool(question.open_lower_bound),
                upper_bound=upper,
                lower_bound=lower,
                zero_point=question.zero_point,
            )
        except Exception as e:
            logger.warning(f"Numeric model path failed, using fallback bounded distribution: {e}")
            span = upper - lower
            value_fracs = [0.15, 0.35, 0.50, 0.65, 0.85]
            percentile_points = [0.10, 0.25, 0.50, 0.75, 0.90]
            values = [lower + (span * frac) for frac in value_fracs]
            min_step = max(span * 1e-6, 1e-9)
            for i in range(1, len(values)):
                if values[i] <= values[i - 1]:
                    values[i] = values[i - 1] + min_step
            if not question.open_lower_bound:
                values[0] = max(values[0], lower + min_step)
            if not question.open_upper_bound:
                values[-1] = min(values[-1], upper - min_step)
            for i in range(1, len(values)):
                if values[i] <= values[i - 1]:
                    values[i] = values[i - 1] + min_step
            return NumericDistribution(
                declared_percentiles=[
                    {"value": value, "percentile": percentile}
                    for value, percentile in zip(values, percentile_points)
                ],
                open_upper_bound=bool(question.open_upper_bound),
                open_lower_bound=bool(question.open_lower_bound),
                upper_bound=upper,
                lower_bound=lower,
                zero_point=question.zero_point,
            )

    async def scan_tournament(
        self,
        tournament_id: int | str,
        question_types: Literal["all", "binary"] = "all",
        inter_question_delay_seconds: float = 2.0,
        max_open_questions: int = 0,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """
        Smart scan: Fetch OPEN and UPCOMING questions.
        - Open: Forecast and submit (if not already done).
        - Upcoming: Log for visibility (preparation step).
        """
        logger.info(f"Starting Smart Scan for tournament {tournament_id}...")
        self._last_scan_stats = {
            "fetched_total": 0,
            "open_total": 0,
            "upcoming_total": 0,
            "supported_open_total": 0,
            "skipped_unsupported_total": 0,
            "skipped_already_forecasted_total": 0,
            "processed_open_total": 0,
        }
        
        # 1. Fetch Open and Upcoming
        api_filter = ApiFilter(
            allowed_tournaments=[tournament_id],
            allowed_statuses=["open", "upcoming"],
        )
        questions = await MetaculusApi.get_questions_matching_filter(api_filter)
        logger.info(f"Fetched {len(questions)} questions (Open + Upcoming)")
        
        open_qs = []
        upcoming_qs = []
        
        for q in questions:
            if q.state == QuestionState.OPEN:
                open_qs.append(q)
            elif q.state == QuestionState.UPCOMING:
                upcoming_qs.append(q)
        self._last_scan_stats.update(
            {
                "fetched_total": len(questions),
                "open_total": len(open_qs),
                "upcoming_total": len(upcoming_qs),
            }
        )
        
        # 2. Process Upcoming (Visibility)
        if upcoming_qs:
            logger.info("--------------------------------------------------")
            logger.info(f"FOUND {len(upcoming_qs)} UPCOMING QUESTIONS:")
            for q in upcoming_qs:
                opens_at = q.open_time.strftime('%Y-%m-%d %H:%M UTC') if q.open_time else "Unknown"
                logger.info(f" - [UPCOMING] {q.question_text}")
                logger.info(f"   Opens: {opens_at} | URL: {q.page_url}")
            logger.info("--------------------------------------------------")
            
        # 3. Process Open (Forecast)
        if open_qs:
            supported_classes = (BinaryQuestion, MultipleChoiceQuestion, NumericQuestion)
            if question_types == "binary":
                supported_open_qs = [q for q in open_qs if isinstance(q, BinaryQuestion)]
                support_label = "supported binary"
            else:
                supported_open_qs = [q for q in open_qs if isinstance(q, supported_classes)]
                support_label = "supported types"

            skipped = len(open_qs) - len(supported_open_qs)
            self._last_scan_stats["supported_open_total"] = len(supported_open_qs)
            self._last_scan_stats["skipped_unsupported_total"] = skipped
            if skipped:
                type_counts: dict[str, int] = {}
                for q in open_qs:
                    if q not in supported_open_qs:
                        t = getattr(q, "question_type", None) or q.__class__.__name__
                        type_counts[t] = type_counts.get(t, 0) + 1
                logger.warning(
                    f"Skipping {skipped} OPEN questions (unsupported for mode={question_types}). "
                    f"Type counts: {type_counts}"
                )

            logger.info(
                f"Found {len(open_qs)} OPEN questions ({len(supported_open_qs)} {support_label}). "
                "Proceeding to forecast..."
            )
            skipped_already_forecasted = 0
            if self.skip_previously_forecasted_questions:
                remaining_open_qs = []
                for q in supported_open_qs:
                    if bool(getattr(q, "already_forecasted", False)):
                        skipped_already_forecasted += 1
                        continue
                    remaining_open_qs.append(q)
                supported_open_qs = remaining_open_qs
                self._last_scan_stats["skipped_already_forecasted_total"] = skipped_already_forecasted
                if skipped_already_forecasted:
                    logger.info(
                        f"Skipping {skipped_already_forecasted} previously forecasted questions "
                        "before processing queue construction."
                    )
            logger.info(
                f"Processable OPEN questions after skip filter: {len(supported_open_qs)}"
            )
            if not supported_open_qs:
                self._last_scan_stats["processed_open_total"] = 0
                return []
            if max_open_questions > 0:
                supported_open_qs = supported_open_qs[:max_open_questions]
                logger.info(
                    f"Applying max_open_questions={max_open_questions}. "
                    f"Processing {len(supported_open_qs)} question(s) this run."
                )
            self._last_scan_stats["processed_open_total"] = len(supported_open_qs)
            reports: list[Any] = []
            delay = max(0.0, float(inter_question_delay_seconds))
            for idx, question in enumerate(supported_open_qs, start=1):
                logger.info(
                    f"Processing question {idx}/{len(supported_open_qs)} | url={getattr(question, 'page_url', None)}"
                )
                per_question_reports = await self.forecast_questions(
                    [question], return_exceptions=return_exceptions
                )
                reports.extend(per_question_reports)
                if idx < len(supported_open_qs) and delay > 0:
                    await asyncio.sleep(delay)
            return reports
        else:
            logger.info("No OPEN questions found to forecast.")
            self._last_scan_stats["processed_open_total"] = 0
            return []



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
        "--tournament",
        type=str,
        default=None,
        help="Tournament ID or slug (default: main AI competition). Examples: 'minibench', 'aibq2'"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./forecastingoutput/forecast_reports.json",
        help="Path to save forecast reports"
    )
    parser.add_argument(
        "--tournament-id",
        type=str,
        default=os.getenv("TOURNAMENT_ID"),
        help=(
            "Tournament ID or slug for tournament mode. "
            "Defaults to TOURNAMENT_ID env var, else forecasting_tools default."
        ),
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=int(os.getenv("MIN_QUESTIONS", "0")),
        help="Fail run if fewer than this many questions are returned (tournament mode).",
    )
    parser.add_argument(
        "--min-post-attempts",
        type=int,
        default=int(os.getenv("MIN_POST_ATTEMPTS", "0")),
        help="Fail run if fewer than this many post attempts are made.",
    )
    parser.add_argument(
        "--question-types",
        choices=["all", "binary"],
        default=os.getenv("QUESTION_TYPES", "all"),
        help="Question types to process in tournament mode.",
    )
    parser.add_argument(
        "--inter-question-delay-seconds",
        type=float,
        default=float(os.getenv("INTER_QUESTION_DELAY_SECONDS", "2.0")),
        help="Delay between tournament questions to reduce API rate-limit errors.",
    )
    parser.add_argument(
        "--max-open-questions",
        type=int,
        default=int(os.getenv("MAX_OPEN_QUESTIONS", "0")),
        help="Optional cap for number of OPEN questions processed in tournament mode (0 = no cap).",
    )

    args = parser.parse_args()
    run_mode = args.mode

    # Initialize bot
    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # 3 models, but treated as 1 ensemble
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        # Disable fragile internal saver from forecasting_tools; we persist via --output-file below.
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    # Startup banner
    logger.info(
        f"RUN MODE={run_mode} | publish={template_bot.publish_reports_to_metaculus} | "
        f"OPENROUTER={bool(os.getenv('OPENROUTER_API_KEY'))} | "
        f"METACULUS={bool(os.getenv('METACULUS_TOKEN'))} | "
        f"TAVILY={bool(os.getenv('TAVILY_API_KEY'))}"
    )

    selected_tournament_id = (
        args.tournament_id
        or args.tournament
        or MetaculusApi.CURRENT_AI_COMPETITION_ID
    )
    logger.info(f"TOURNAMENT_ID={selected_tournament_id}")

    # Apply CLI overrides
    if getattr(args, "force_repost", False):
        template_bot.skip_previously_forecasted_questions = False

    # Run based on mode
    if run_mode == "tournament":
        logger.info(
            f"Targeting tournament: {selected_tournament_id} | question_types={args.question_types}"
        )
        forecast_reports = asyncio.run(
            template_bot.scan_tournament(
                selected_tournament_id,
                question_types=args.question_types,
                inter_question_delay_seconds=args.inter_question_delay_seconds,
                max_open_questions=args.max_open_questions,
                return_exceptions=True,
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

    # Run-level diagnostics and guardrails
    publish_flag = bool(getattr(template_bot, 'publish_reports_to_metaculus', False))
    skip_flag = bool(getattr(template_bot, 'skip_previously_forecasted_questions', True))
    scan_stats = (
        dict(getattr(template_bot, "_last_scan_stats", {}))
        if run_mode == "tournament"
        else {}
    )

    successful_reports = [r for r in forecast_reports if not isinstance(r, BaseException)]
    errored_reports = [r for r in forecast_reports if isinstance(r, BaseException)]

    question_count = len(successful_reports)
    retrieved_from_api_count = int(scan_stats.get("fetched_total", question_count) or 0)
    processable_open_count = int(scan_stats.get("processed_open_total", question_count) or 0)
    skipped_from_scan = int(scan_stats.get("skipped_already_forecasted_total", 0) or 0)
    post_attempt_count = 0
    skipped_previously_forecasted_count = 0
    skipped_publish_disabled_count = 0

    for r in successful_reports:
        q = getattr(r, 'question', None)
        already_forecasted = bool(getattr(q, 'already_forecasted', False)) if q else False
        if not publish_flag:
            skipped_publish_disabled_count += 1
        elif already_forecasted and skip_flag:
            skipped_previously_forecasted_count += 1
        else:
            post_attempt_count += 1
    skipped_previously_forecasted_count += skipped_from_scan

    logging.info(f"Retrieved {retrieved_from_api_count} questions from tournament")
    logging.info(
        "Run Stats: "
        f"retrieved_from_api={retrieved_from_api_count} | "
        f"processable_open={processable_open_count} | "
        f"total_reports={len(forecast_reports)} | "
        f"success={question_count} | "
        f"errors={len(errored_reports)} | "
        f"post_attempts={post_attempt_count} | "
        f"skipped_already_forecasted={skipped_previously_forecasted_count} | "
        f"skipped_publish_disabled={skipped_publish_disabled_count}"
    )

    if run_mode == "tournament" and args.min_questions > 0 and retrieved_from_api_count < args.min_questions:
        raise SystemExit(
            f"Retrieved only {retrieved_from_api_count} questions, below required minimum {args.min_questions}."
        )

    if args.min_post_attempts > 0 and post_attempt_count < args.min_post_attempts:
        if run_mode == "tournament":
            if processable_open_count == 0:
                logging.info(
                    "No processable open questions remained after type/skip filters; "
                    "post-attempt minimum waived."
                )
            elif skipped_previously_forecasted_count >= processable_open_count:
                logging.info(
                    "All processable open questions were already forecasted; "
                    "post-attempt minimum waived."
                )
            else:
                raise SystemExit(
                    f"Only {post_attempt_count} post attempts, below required minimum {args.min_post_attempts}."
                )
        else:
            raise SystemExit(
                f"Only {post_attempt_count} post attempts, below required minimum {args.min_post_attempts}."
            )

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
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
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
        for r in successful_reports:
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
        for err in errored_reports:
            logging.error(f"Forecast report failure: {err}")
    except Exception as e:
        logging.warning(f"Could not emit submission-intent logs: {e}")


if __name__ == "__main__":
    main()
