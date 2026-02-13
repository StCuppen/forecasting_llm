from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlparse

from bot.aggregation import AggregatedForecast


@dataclass
class EvidenceItem:
    content: str
    source_url: str
    source_name: str
    retrieved_at: datetime
    published_at: Optional[datetime]
    relevance_score: float
    is_primary_source: bool


@dataclass
class SpecLockResult:
    confidence: float
    canonical_text: str


@dataclass
class GateReport:
    publishable: bool
    confidence_class: str
    gate_score: float
    spec_confidence: float
    temporal_relevance: float
    evidence_sufficiency: float
    model_agreement: float
    action: str
    reasons: List[str]
    evidence_count: int = 0
    distinct_sources: int = 0
    primary_sources: int = 0
    mean_relevance: float = 0.0
    freshness_days: float = 999.0


def _as_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def compute_temporal_relevance(evidence: List[EvidenceItem], question_deadline: Optional[datetime]) -> float:
    if not evidence:
        return 0.0
    now = datetime.now(timezone.utc)
    latest = max((_as_utc(item.published_at) or _as_utc(item.retrieved_at) for item in evidence if _as_utc(item.retrieved_at)), default=None)
    if latest is None:
        return 0.2

    age_days = max(0.0, (now - latest).total_seconds() / 86400.0)
    freshness = max(0.0, 1.0 - min(age_days / 90.0, 1.0))

    deadline = _as_utc(question_deadline)
    if deadline is None:
        return freshness

    days_to_deadline = (deadline - now).total_seconds() / 86400.0
    if days_to_deadline <= 30 and age_days > 30:
        freshness *= 0.5
    elif days_to_deadline <= 7 and age_days > 7:
        freshness *= 0.35
    return max(0.0, min(1.0, freshness))


def compute_evidence_sufficiency(evidence: List[EvidenceItem]) -> float:
    if not evidence:
        return 0.0
    now = datetime.now(timezone.utc)
    distinct_domains = {
        urlparse(item.source_url).netloc.lower()
        for item in evidence
        if item.source_url
    }
    primary_count = sum(1 for item in evidence if item.is_primary_source)
    direct_count = sum(1 for item in evidence if item.relevance_score >= 0.6)
    recent_count = 0
    for item in evidence:
        ts = _as_utc(item.published_at) or _as_utc(item.retrieved_at)
        if ts and (now - ts) <= timedelta(days=30):
            recent_count += 1

    source_score = min(1.0, len(distinct_domains) / 3.0)
    primary_score = min(1.0, primary_count / 2.0)
    direct_score = min(1.0, direct_count / 3.0)
    recent_score = min(1.0, recent_count / 2.0)
    raw_score = max(0.0, min(1.0, 0.35 * source_score + 0.25 * primary_score + 0.2 * direct_score + 0.2 * recent_score))

    # Hard floors
    if primary_count == 0:
        raw_score = min(raw_score, 0.5)
    mean_rel = sum(item.relevance_score for item in evidence) / len(evidence)
    if mean_rel < 0.7:
        raw_score = min(raw_score, 0.6)

    return raw_score


def _build_reasons(
    spec_score: float,
    temporal_score: float,
    evidence_score: float,
    agreement_score: float,
    action: str,
    evidence_count: int = 0,
    primary_sources: int = 0,
    mean_relevance: float = 0.0,
    n_runs: int = 1,
) -> List[str]:
    reasons: List[str] = []
    if evidence_count < 3:
        reasons.append(f"HARD FLOOR: Only {evidence_count} evidence items (minimum 3 required). Forcing abstain.")
    if primary_sources == 0 and evidence_count > 0:
        reasons.append("HARD FLOOR: 0 primary sources detected. Evidence sufficiency capped at 0.5.")
    if mean_relevance < 0.7 and evidence_count >= 1:
        reasons.append(f"HARD FLOOR: Mean relevance {mean_relevance:.2f} < 0.70. Evidence sufficiency capped at 0.6.")
    if n_runs <= 1:
        reasons.append("WARNING: Single model run -- agreement metric set to neutral 0.5.")
    if spec_score < 0.5:
        reasons.append("Spec lock confidence is low.")
    if temporal_score < 0.5:
        reasons.append("Evidence may be stale versus the resolution horizon.")
    if evidence_score < 0.5:
        reasons.append("Evidence sufficiency is weak (few/diverse/direct sources).")
    if agreement_score < 0.5 and n_runs > 1:
        reasons.append("Model disagreement is high.")
    if not reasons:
        reasons.append("Gate checks passed with adequate evidence and agreement.")
    reasons.append(f"Gate action: {action}.")
    return reasons


def evaluate_publish_gate(
    forecast: AggregatedForecast,
    evidence: List[EvidenceItem],
    spec_lock: SpecLockResult,
    question_deadline: Optional[datetime],
    publish_threshold: float = 0.6,
    low_conf_threshold: float = 0.35,
) -> GateReport:
    spec_score = max(0.0, min(1.0, spec_lock.confidence))
    temporal_score = compute_temporal_relevance(evidence, question_deadline)
    evidence_score = compute_evidence_sufficiency(evidence)
    agreement_score = max(0.0, min(1.0, 1 - forecast.dispersion * 5))
    # Neutral agreement when only 1 run (tautological 1.0 is meaningless)
    if forecast.n_runs <= 1:
        agreement_score = 0.5
    now = datetime.now(timezone.utc)
    ages = []
    distinct_sources = set()
    primary_sources = 0
    relevance_vals: list[float] = []
    for item in evidence:
        distinct_sources.add(urlparse(item.source_url).netloc.lower())
        if item.is_primary_source:
            primary_sources += 1
        relevance_vals.append(float(item.relevance_score))
        ts = _as_utc(item.published_at) or _as_utc(item.retrieved_at)
        if ts is not None:
            ages.append(max(0.0, (now - ts).total_seconds() / 86400.0))
    freshness_days = min(ages) if ages else 999.0
    mean_relevance = sum(relevance_vals) / len(relevance_vals) if relevance_vals else 0.0
    gate_score = (
        0.2 * spec_score
        + 0.3 * temporal_score
        + 0.3 * evidence_score
        + 0.2 * agreement_score
    )
    # Hard floor: minimum evidence count
    if len(evidence) < 3:
        action = "abstain"
    elif gate_score >= publish_threshold:
        action = "publish"
    elif gate_score >= low_conf_threshold:
        action = "publish_low_confidence"
    else:
        action = "abstain"

    return GateReport(
        publishable=(action != "abstain"),
        confidence_class=forecast.confidence_class,
        gate_score=gate_score,
        spec_confidence=spec_score,
        temporal_relevance=temporal_score,
        evidence_sufficiency=evidence_score,
        model_agreement=agreement_score,
        action=action,
        reasons=_build_reasons(
            spec_score=spec_score,
            temporal_score=temporal_score,
            evidence_score=evidence_score,
            agreement_score=agreement_score,
            action=action,
            evidence_count=len(evidence),
            primary_sources=primary_sources,
            mean_relevance=mean_relevance,
            n_runs=forecast.n_runs,
        ),
        evidence_count=len(evidence),
        distinct_sources=len([x for x in distinct_sources if x]),
        primary_sources=primary_sources,
        mean_relevance=mean_relevance,
        freshness_days=freshness_days,
    )


def shrink_probability(p: float, gate_score: float, base_rate: Optional[float]) -> float:
    anchor = 0.5 if base_rate is None else max(0.01, min(0.99, base_rate))
    alpha = max(0.0, min(1.0, gate_score))
    return max(0.01, min(0.99, alpha * p + (1 - alpha) * anchor))
