from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.core.schemas import EvidenceBundle, EvidenceItem, Question
from src.core.utils import stable_hash, utc_now


@dataclass
class ForecastRunResult:
    p_ens: float
    p_agents: dict[str, float]
    model_versions: dict[str, str]
    evidence_bundle: EvidenceBundle
    cost_estimate: float | None
    latency: float | None
    forecast_context: dict[str, Any]
    rationale_markdown: str


def _quality_to_trust(quality: str | None) -> float:
    q = (quality or "").lower()
    if q == "high":
        return 0.9
    if q == "low":
        return 0.3
    return 0.6


def _build_evidence_bundle(items: list[dict[str, Any]]) -> EvidenceBundle:
    evidence_items: list[EvidenceItem] = []
    now = utc_now()
    for idx, item in enumerate(items, start=1):
        snippet = item.get("snippet") or item.get("content") or ""
        evidence_items.append(
            EvidenceItem(
                url=str(item.get("url", "")),
                retrieved_at=now,
                snippet_hash=stable_hash(snippet),
                trust_score=_quality_to_trust(item.get("quality")),
                rank=idx,
                snippet=snippet[:500] if snippet else None,
            )
        )
    bundle_id = str(uuid.uuid4())
    archived_hashes = [stable_hash((item.snippet or "") + item.url) for item in evidence_items]
    return EvidenceBundle(bundle_id=bundle_id, items=evidence_items, archived_text_hashes=archived_hashes)


def _dry_run_probability(question: Question) -> float:
    h = stable_hash(question.title + "|" + question.source + "|" + question.source_id)
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return min(max(bucket, 0.01), 0.99)


def run_forecast_for_question(
    question: Question,
    dry_run: bool = False,
    apply_calibration: bool = False,
    calibrator: tuple[str, dict[str, Any]] | None = None,
) -> ForecastRunResult:
    start = time.perf_counter()
    rationale_markdown = ""
    if dry_run:
        p_ens = _dry_run_probability(question)
        p_agents = {"dryrun-agent-a": min(max(p_ens - 0.05, 0.01), 0.99), "dryrun-agent-b": min(max(p_ens + 0.05, 0.01), 0.99)}
        model_versions = {"dryrun-agent-a": "dryrun/v1", "dryrun-agent-b": "dryrun/v1"}
        evidence = _build_evidence_bundle(
            [
                {"url": "https://example.com/source-a", "snippet": question.title, "quality": "medium"},
                {"url": "https://example.com/source-b", "snippet": question.description, "quality": "medium"},
            ]
        )
        rationale_markdown = (
            "# Dry-Run Forecast Rationale\n\n"
            "This forecast was generated in dry-run mode using deterministic hashing.\n\n"
            f"- Ensemble probability: `{p_ens:.4f}`\n"
            f"- Agent probabilities: `{p_agents}`\n"
        )
    else:
        from bot.agent.agent_experiment import run_ensemble_forecast

        prompt_text = f"{question.title}\n\n{question.description}".strip()
        raw = asyncio.run(
            run_ensemble_forecast(
                question=prompt_text,
                publish_to_metaculus=False,
                use_react=True,
            )
        )
        p_ens = float(raw["final_probability"])
        summary_text = str(raw.get("summary_text", "")).strip()
        full_reasoning = str(raw.get("full_reasoning", "")).strip()
        p_agents: dict[str, float] = {}
        model_versions: dict[str, str] = {}
        source_items: list[dict[str, Any]] = []
        for item in raw.get("individual_results", []):
            config = item.get("config", {})
            result = item.get("result")
            label = str(config.get("label", config.get("name", "agent")))
            p_agents[label] = float(getattr(result, "probability", p_ens))
            model_versions[label] = str(config.get("name", "unknown"))
            for source in getattr(result, "sources", []) or []:
                source_items.append(source)
        evidence = _build_evidence_bundle(source_items)
        rationale_markdown = (
            "# Forecast Rationale\n\n"
            "## Ensemble Summary\n\n"
            f"{summary_text or 'No summary provided.'}\n\n"
            "## Full Reasoning\n\n"
            f"{full_reasoning or 'No detailed reasoning provided.'}\n"
        )

    calibrator_version = None
    if apply_calibration and calibrator:
        from src.core.updater import apply_calibrator

        calibrator_version = calibrator[0]
        p_ens = apply_calibrator(calibrator[1], p_ens)

    latency = time.perf_counter() - start
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_template_version": "league-v1",
        "forecast_pipeline_version": "v1",
        "dry_run": dry_run,
        "apply_calibration": apply_calibration,
        "calibrator_version": calibrator_version,
    }
    return ForecastRunResult(
        p_ens=p_ens,
        p_agents=p_agents,
        model_versions=model_versions,
        evidence_bundle=evidence,
        cost_estimate=None,
        latency=latency,
        forecast_context=context,
        rationale_markdown=rationale_markdown,
    )
